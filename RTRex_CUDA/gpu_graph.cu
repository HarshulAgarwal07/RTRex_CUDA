#include "gpu_graph.cuh"
#include <cstdio>

// ============================================================
// Kernel to build partnerMap on GPU
// Each thread handles one directed edge, looks up the reverse edge
// ============================================================
__global__ void buildPartnerMapKernel(
    EdgeIdx*        d_partnerMap,
    const EdgeIdx*  d_offsets,
    const VertexIdx* d_nbors,
    VertexIdx        nVertices,
    EdgeIdx          nEdges)
{
    EdgeIdx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nEdges) return;

    // Find which vertex this edge belongs to
    // Linear scan (small overhead for startup, run once)
    VertexIdx u = 0;
    // Binary search for the source vertex of edge i
    // d_offsets is sorted increasing, find the vertex whose offset range contains i
    VertexIdx lo = 0, hi = nVertices - 1;
    while (lo <= hi) {
        VertexIdx mid = (lo + hi) / 2;
        if (d_offsets[mid] <= i && i < d_offsets[mid + 1]) {
            u = mid;
            break;
        }
        if (d_offsets[mid] > i)
            hi = mid - 1;
        else
            lo = mid + 1;
    }

    VertexIdx v = d_nbors[i];

    // Look up (v, u) in v's neighbor list
    EdgeIdx loc = getEdgeBinaryDevice(d_offsets, d_nbors, nVertices, v, u);
    d_partnerMap[i] = loc;
}

// ============================================================
// Transfer graph from host CSR to device
// ============================================================
void transferGraphToDevice(
    const EdgeIdx*   h_offsets,
    const VertexIdx* h_nbors,
    VertexIdx        nVertices,
    EdgeIdx          nEdges,
    DeviceGraph*     dg)
{
    dg->nVertices = nVertices;
    dg->nEdges = nEdges;

    printf("[GPU Graph] Allocating device memory for graph with %" PRId64
           " vertices, %" PRId64 " edges\n", nVertices, nEdges);

    // Allocate device CSR arrays
    CUDA_CHECK(cudaMalloc(&dg->d_offsets, (nVertices + 1) * sizeof(EdgeIdx)));
    CUDA_CHECK(cudaMalloc(&dg->d_nbors, nEdges * sizeof(VertexIdx)));
    CUDA_CHECK(cudaMalloc(&dg->d_partnerMap, nEdges * sizeof(EdgeIdx)));

    // Copy CSR data to device
    CUDA_CHECK(cudaMemcpy(dg->d_offsets, h_offsets,
                          (nVertices + 1) * sizeof(EdgeIdx),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg->d_nbors, h_nbors,
                          nEdges * sizeof(VertexIdx),
                          cudaMemcpyHostToDevice));

    // Build partnerMap on GPU
    EdgeIdx nBlocks = (nEdges + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    buildPartnerMapKernel<<<nBlocks, THREADS_PER_BLOCK>>>(
        dg->d_partnerMap, dg->d_offsets, dg->d_nbors, nVertices, nEdges);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute and store degrees on device
    CUDA_CHECK(cudaMalloc(&dg->d_degree, nVertices * sizeof(Count)));
    // Allocate a temporary host degree array
    Count* h_degree = new Count[nVertices];
    for (VertexIdx i = 0; i < nVertices; i++) {
        h_degree[i] = h_offsets[i + 1] - h_offsets[i];
    }
    CUDA_CHECK(cudaMemcpy(dg->d_degree, h_degree,
                          nVertices * sizeof(Count),
                          cudaMemcpyHostToDevice));
    delete[] h_degree;

    printf("[GPU Graph] Graph transfer complete\n");
}

// ============================================================
// Allocate device-side state arrays
// ============================================================
void allocateDeviceState(DeviceGraph* dg)
{
    EdgeIdx nEdges = dg->nEdges;
    VertexIdx nVertices = dg->nVertices;

    printf("[GPU Graph] Allocating state arrays\n");

    // Per-edge state
    CUDA_CHECK(cudaMalloc(&dg->d_edgeStatus, nEdges * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&dg->d_perEdge, nEdges * sizeof(double)));

    // Initialize edgeStatus to all 'Y' (alive)
    CUDA_CHECK(cudaMemset(dg->d_edgeStatus, (int)'Y', nEdges * sizeof(char)));
    // Initialize perEdge to 0.0
    CUDA_CHECK(cudaMemset(dg->d_perEdge, 0, nEdges * sizeof(double)));

    // Per-vertex triangle weight
    CUDA_CHECK(cudaMalloc(&dg->d_perVertex, nVertices * sizeof(double)));
    CUDA_CHECK(cudaMemset(dg->d_perVertex, 0, nVertices * sizeof(double)));

    // Extraction state
    CUDA_CHECK(cudaMalloc(&dg->d_clustered, nVertices * sizeof(bool)));
    CUDA_CHECK(cudaMemset(dg->d_clustered, 0, nVertices * sizeof(bool)));

    CUDA_CHECK(cudaMalloc(&dg->d_nbdBitMap, nVertices * sizeof(bool)));

    // Candidate accumulation arrays for extraction
    CUDA_CHECK(cudaMalloc(&dg->d_candidateWgt, nVertices * sizeof(double)));
    CUDA_CHECK(cudaMemset(dg->d_candidateWgt, 0, nVertices * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&dg->d_candidateFlag, nVertices * sizeof(bool)));

    // Work counter for cleaning rounds
    CUDA_CHECK(cudaMalloc(&dg->d_hasStack, sizeof(int)));

    dg->totalTriangles = 0;

    printf("[GPU Graph] State arrays allocated\n");
}

// ============================================================
// Free all GPU resources
// ============================================================
void freeDeviceGraph(DeviceGraph* dg)
{
    if (dg->d_offsets)     { cudaFree(dg->d_offsets); dg->d_offsets = nullptr; }
    if (dg->d_nbors)       { cudaFree(dg->d_nbors); dg->d_nbors = nullptr; }
    if (dg->d_partnerMap)  { cudaFree(dg->d_partnerMap); dg->d_partnerMap = nullptr; }
    if (dg->d_degree)      { cudaFree(dg->d_degree); dg->d_degree = nullptr; }
    if (dg->d_edgeStatus)  { cudaFree(dg->d_edgeStatus); dg->d_edgeStatus = nullptr; }
    if (dg->d_perEdge)     { cudaFree(dg->d_perEdge); dg->d_perEdge = nullptr; }
    if (dg->d_perVertex)   { cudaFree(dg->d_perVertex); dg->d_perVertex = nullptr; }
    if (dg->d_clustered)   { cudaFree(dg->d_clustered); dg->d_clustered = nullptr; }
    if (dg->d_nbdBitMap)   { cudaFree(dg->d_nbdBitMap); dg->d_nbdBitMap = nullptr; }
    if (dg->d_candidateWgt) { cudaFree(dg->d_candidateWgt); dg->d_candidateWgt = nullptr; }
    if (dg->d_candidateFlag) { cudaFree(dg->d_candidateFlag); dg->d_candidateFlag = nullptr; }
    if (dg->d_hasStack)    { cudaFree(dg->d_hasStack); dg->d_hasStack = nullptr; }
}

// ============================================================
// Sync per-edge and per-vertex weights device → host
// ============================================================
void syncWeightsDeviceToHost(
    const DeviceGraph* dg,
    double* h_perEdge,
    double* h_perVertex,
    Count*  h_totalTriangles)
{
    CUDA_CHECK(cudaMemcpy(h_perEdge, dg->d_perEdge,
                          dg->nEdges * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_perVertex, dg->d_perVertex,
                          dg->nVertices * sizeof(double),
                          cudaMemcpyDeviceToHost));
    *h_totalTriangles = dg->totalTriangles;
}

// ============================================================
// Sync edge status device → host
// ============================================================
void syncEdgeStatusDeviceToHost(const DeviceGraph* dg, char* h_edgeStatus)
{
    CUDA_CHECK(cudaMemcpy(h_edgeStatus, dg->d_edgeStatus,
                          dg->nEdges * sizeof(char),
                          cudaMemcpyDeviceToHost));
}

// ============================================================
// Sync clustered bitmap device → host
// ============================================================
void syncClusteredDeviceToHost(const DeviceGraph* dg, bool* h_clustered)
{
    CUDA_CHECK(cudaMemcpy(h_clustered, dg->d_clustered,
                          dg->nVertices * sizeof(bool),
                          cudaMemcpyDeviceToHost));
}
