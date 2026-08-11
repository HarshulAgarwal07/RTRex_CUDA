#include "gpu_triangle_count.cuh"
#include <cstdio>
#include <chrono>

// ============================================================
// GPU kernel: Count weighted triangles (commonNbr equivalent)
//
// Grid: one block per vertex u
// Threads within block process u's outgoing edges in parallel
// ============================================================
// To avoid massive load imbalance (some vertices have degree 100K+
// while most have degree <10), we use a dynamic work assignment:
// each thread processes multiple edges using strided access.
// ============================================================
__global__ void triangleCountKernel(
    const EdgeIdx*   __restrict__ d_offsets,
    const VertexIdx* __restrict__ d_nbors,
    const Count*     __restrict__ d_degree,
    double*          __restrict__ d_perEdge,
    double*          __restrict__ d_perVertex,
    const EdgeIdx*   __restrict__ d_partnerMap,
    unsigned long long* __restrict__ d_totalTriangles,
    VertexIdx        nVertices,
    EdgeIdx          nEdges)
{
    VertexIdx u = blockIdx.x;
    if (u >= nVertices) return;

    Count degu = d_degree[u];
    EdgeIdx start = d_offsets[u];
    EdgeIdx end = d_offsets[u + 1];

    // Each thread processes a subset of u's edges (strided loop)
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (EdgeIdx i = start + tid; i < end; i += stride) {
        VertexIdx v = d_nbors[i];

        // Tiebreaking: only process (u, v) if u < v
        if (u > v) continue;

        Count degv = d_degree[v];

        // Determine lower/higher degree vertex for efficient intersection
        VertexIdx lower  = (degu < degv) ? u : v;
        VertexIdx higher = (degu < degv) ? v : u;

        // Iterate over neighbors of lower
        EdgeIdx loStart = d_offsets[lower];
        EdgeIdx loEnd   = d_offsets[lower + 1];

        for (EdgeIdx j = loStart; j < loEnd; j++) {
            VertexIdx w = d_nbors[j];

            // Only consider w if w has the largest ID
            // (ensures each triangle counted exactly once)
            if (w <= u || w <= v) continue;

            // Binary search for w in higher's neighbor list
            EdgeIdx loc = getEdgeBinaryDevice(d_offsets, d_nbors,
                                              nVertices, higher, w);
            if (loc == -1) continue;

            // Triangle (u, v, w) found
            Count degw = d_degree[w];
            double wgt = 1.0 / (double)(degu * degv * degw);

            // Increment total triangle count
            atomicAdd(d_totalTriangles, 1ULL);

            // Update per-edge weights for all 6 directed edge copies
            // Edge (u,v) at index i
            atomicAdd(&d_perEdge[i], wgt);
            atomicAdd(&d_perEdge[d_partnerMap[i]], wgt);

            // Edge (lower,w) at index j
            atomicAdd(&d_perEdge[j], wgt);
            atomicAdd(&d_perEdge[d_partnerMap[j]], wgt);

            // Edge (higher,w) at index loc
            atomicAdd(&d_perEdge[loc], wgt);
            atomicAdd(&d_perEdge[d_partnerMap[loc]], wgt);

            // Update per-vertex weights
            atomicAdd(&d_perVertex[u], wgt);
            atomicAdd(&d_perVertex[v], wgt);
            atomicAdd(&d_perVertex[w], wgt);
        }
    }
}

// ============================================================
// Host wrapper
// ============================================================
void gpuCommonNbr(DeviceGraph* dg)
{
    printf("[GPU Triangle Count] Starting weighted triangle enumeration\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Reset total triangle count on device
    unsigned long long* d_totalTriangles;
    CUDA_CHECK(cudaMalloc(&d_totalTriangles, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_totalTriangles, 0, sizeof(unsigned long long)));

    VertexIdx nVertices = dg->nVertices;
    // Cap blocks at a reasonable maximum
    VertexIdx nBlocks = nVertices;
    if (nBlocks > 65535) nBlocks = 65535;

    printf("[GPU Triangle Count] Launching %" PRId64 " blocks x %d threads\n",
           nBlocks, THREADS_PER_BLOCK);

    triangleCountKernel<<<nBlocks, THREADS_PER_BLOCK>>>(
        dg->d_offsets,
        dg->d_nbors,
        dg->d_degree,
        dg->d_perEdge,
        dg->d_perVertex,
        dg->d_partnerMap,
        d_totalTriangles,
        dg->nVertices,
        dg->nEdges);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy total triangle count back
    unsigned long long tmpCount;
    CUDA_CHECK(cudaMemcpy(&tmpCount, d_totalTriangles,
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    dg->totalTriangles = (Count)tmpCount;
    CUDA_CHECK(cudaFree(d_totalTriangles));

    auto stop = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(stop - start).count();

    printf("[GPU Triangle Count] Found %" PRId64 " triangles in %.3f s\n",
           dg->totalTriangles, elapsed);
}
