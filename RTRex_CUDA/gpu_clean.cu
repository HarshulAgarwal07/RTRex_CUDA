#include "gpu_clean.cuh"
#include <cstdio>
#include <chrono>

// ============================================================
// Device helper: find source vertex u for edge at index i
// ============================================================
__device__ VertexIdx findSrcVertex(
    const EdgeIdx* __restrict__ d_offsets,
    VertexIdx nVertices,
    EdgeIdx i)
{
    VertexIdx lo = 0, hi = nVertices - 1;
    while (lo <= hi) {
        VertexIdx mid = (lo + hi) / 2;
        if (d_offsets[mid] <= i && i < d_offsets[mid + 1])
            return mid;
        if (d_offsets[mid] > i)
            hi = mid - 1;
        else
            lo = mid + 1;
    }
    return 0;
}

// ============================================================
// Kernel: Identify edges below cleaning threshold → mark 'S'
// ============================================================
__global__ void markLowWeightEdgesKernel(
    const EdgeIdx*   __restrict__ d_offsets,
    const VertexIdx* __restrict__ d_nbors,
    const Count*     __restrict__ d_degree,
    const double*    __restrict__ d_perEdge,
    int* __restrict__ d_edgeStatus,
    const EdgeIdx*   __restrict__ d_partnerMap,
    double           eps,
    VertexIdx        nVertices,
    EdgeIdx          nEdges)
{
    EdgeIdx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nEdges) return;
    if (d_edgeStatus[i] != EDGE_ALIVE) return;

    VertexIdx u = findSrcVertex(d_offsets, nVertices, i);
    VertexIdx v = d_nbors[i];
    Count degu = d_degree[u];
    Count degv = d_degree[v];
    double edgeWt = 1.0 / (double)(degu * degv);

    if (d_perEdge[i] < eps * edgeWt) {
        int old = atomicExch(&d_edgeStatus[i], EDGE_STACK);
        if (old == EDGE_ALIVE) {
            atomicExch(&d_edgeStatus[d_partnerMap[i]], EDGE_STACK);
        }
    }
}

// ============================================================
// Kernel: Process all 'S' edges — remove triangles, cascade
//
// Tiebreaker: process triangle only if current edge has
// smallest directed index among the 3 triangle edges.
// ============================================================
__global__ void processStackEdgesKernel(
    const EdgeIdx*   __restrict__ d_offsets,
    const VertexIdx* __restrict__ d_nbors,
    const Count*     __restrict__ d_degree,
    double*          __restrict__ d_perEdge,
    double*          __restrict__ d_perVertex,
    int* __restrict__ d_edgeStatus,
    const EdgeIdx*   __restrict__ d_partnerMap,
    VertexIdx        nVertices,
    EdgeIdx          nEdges)
{
    EdgeIdx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nEdges) return;
    if (d_edgeStatus[i] != EDGE_STACK) return;

    VertexIdx u = findSrcVertex(d_offsets, nVertices, i);
    VertexIdx v = d_nbors[i];
    Count degu = d_degree[u];
    Count degv = d_degree[v];

    VertexIdx lower  = (degu < degv) ? u : v;
    VertexIdx higher = (degu < degv) ? v : u;

    EdgeIdx loStart = d_offsets[lower];
    EdgeIdx loEnd   = d_offsets[lower + 1];

    for (EdgeIdx j = loStart; j < loEnd; j++) {
        if (d_edgeStatus[j] == EDGE_DEAD) continue;

        VertexIdx w = d_nbors[j];
        if (w == u || w == v) continue;

        EdgeIdx loc_hw = getEdgeBinaryDevice(d_offsets, d_nbors,
                                              nVertices, higher, w);
        if (loc_hw == -1) continue;
        if (d_edgeStatus[loc_hw] == EDGE_DEAD) continue;

        // Triangle (u,v,w). Edges at i, j, loc_hw.
        // Only process if i is smallest (avoid double-counting)
        if (i >= j || i >= loc_hw) continue;

        Count degw = d_degree[w];
        double wgt = 1.0 / (double)(degu * degv * degw);

        EdgeIdx pi  = d_partnerMap[i];
        EdgeIdx pj  = d_partnerMap[j];
        EdgeIdx phw = d_partnerMap[loc_hw];

        atomicAdd(&d_perEdge[i],      -wgt);
        atomicAdd(&d_perEdge[pi],     -wgt);
        atomicAdd(&d_perEdge[j],      -wgt);
        atomicAdd(&d_perEdge[pj],     -wgt);
        atomicAdd(&d_perEdge[loc_hw], -wgt);
        atomicAdd(&d_perEdge[phw],    -wgt);

        atomicAdd(&d_perVertex[u], -wgt);
        atomicAdd(&d_perVertex[v], -wgt);
        atomicAdd(&d_perVertex[w], -wgt);
    }

    // Mark as dead
    d_edgeStatus[i] = EDGE_DEAD;
    d_edgeStatus[d_partnerMap[i]] = EDGE_DEAD;
}

// ============================================================
// Kernel: Check if any 'S' edges remain
// ============================================================
__global__ void checkStackKernel(
    const int* __restrict__ d_edgeStatus,
    int*        __restrict__ d_hasStack,
    EdgeIdx     nEdges)
{
    EdgeIdx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nEdges) return;
    if (d_edgeStatus[i] == EDGE_STACK) {
        *d_hasStack = 1;
    }
}

// ============================================================
// Shared bulk cleaning loop
// ============================================================
static void runBulkCleanLoop(
    DeviceGraph* dg,
    double eps,
    int maxRounds,
    const char* label)
{
    VertexIdx nVertices = dg->nVertices;
    EdgeIdx nEdges = dg->nEdges;
    EdgeIdx nBlocksE = (nEdges + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (nBlocksE > 65535) nBlocksE = 65535;

    int round = 0;
    while (round < maxRounds) {
        round++;

        // Phase A: identify below-threshold edges
        markLowWeightEdgesKernel<<<nBlocksE, THREADS_PER_BLOCK>>>(
            dg->d_offsets, dg->d_nbors, dg->d_degree,
            dg->d_perEdge, dg->d_edgeStatus, dg->d_partnerMap,
            eps, nVertices, nEdges);
        CUDA_CHECK(cudaGetLastError());

        // Check if anything was marked
        CUDA_CHECK(cudaMemset(dg->d_hasStack, 0, sizeof(int)));
        checkStackKernel<<<nBlocksE, THREADS_PER_BLOCK>>>(
            dg->d_edgeStatus, dg->d_hasStack, nEdges);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int hasStack;
        CUDA_CHECK(cudaMemcpy(&hasStack, dg->d_hasStack, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (hasStack == 0) break;

        // Phase B: process deletions
        processStackEdgesKernel<<<nBlocksE, THREADS_PER_BLOCK>>>(
            dg->d_offsets, dg->d_nbors, dg->d_degree,
            dg->d_perEdge, dg->d_perVertex, dg->d_edgeStatus,
            dg->d_partnerMap, nVertices, nEdges);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (round % 20 == 0 && label[0] == 'e')
            printf("  [clean %s] round %d...\n", label, round);
    }

    if (round >= maxRounds) {
        printf("[GPU Clean %s] WARNING: max rounds (%d) reached\n", label, maxRounds);
    } else if (round > 0 && label[0] == 'e') {
        printf("  [clean %s] converged in %d rounds\n", label, round);
    }
}

// ============================================================
// gpuInitialClean
// ============================================================
void gpuInitialClean(DeviceGraph* dg, double eps, int maxRounds)
{
    printf("[GPU Clean] Initial clean (eps=%.3f)\n", eps);
    auto t0 = std::chrono::high_resolution_clock::now();
    runBulkCleanLoop(dg, eps, maxRounds, "init");
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("[GPU Clean] Done in %.3f s\n",
           std::chrono::duration<double>(t1 - t0).count());
}

// ============================================================
// Kernel: Mark specific edges for deletion
// ============================================================
__global__ void markGivenEdgesKernel(
    const Pair*      __restrict__ d_toDelete,
    const EdgeIdx*   __restrict__ d_offsets,
    const VertexIdx* __restrict__ d_nbors,
    int* __restrict__ d_edgeStatus,
    const EdgeIdx*   __restrict__ d_partnerMap,
    VertexIdx        nVertices,
    EdgeIdx          nToDelete)
{
    EdgeIdx idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nToDelete) return;

    VertexIdx u = d_toDelete[idx].first;
    VertexIdx v = d_toDelete[idx].second;

    EdgeIdx loc = getEdgeBinaryDevice(d_offsets, d_nbors, nVertices, u, v);
    if (loc == -1) return;
    if (d_edgeStatus[loc] == EDGE_ALIVE) {
        d_edgeStatus[loc] = EDGE_STACK;
        d_edgeStatus[d_partnerMap[loc]] = EDGE_STACK;
    }
}

// ============================================================
// gpuDeleteAndClean
// ============================================================
void gpuDeleteAndClean(DeviceGraph* dg, const Pair* d_toDelete,
                       EdgeIdx nToDelete, double eps)
{
    if (nToDelete > 0) {
        EdgeIdx nBlocks = (nToDelete + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (nBlocks > 65535) nBlocks = 65535;

        markGivenEdgesKernel<<<nBlocks, THREADS_PER_BLOCK>>>(
            d_toDelete, dg->d_offsets, dg->d_nbors,
            dg->d_edgeStatus, dg->d_partnerMap,
            dg->nVertices, nToDelete);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    runBulkCleanLoop(dg, eps, 50, "extract");
}
