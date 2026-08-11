#include "gpu_extract.cuh"
#include <cstdio>
#include <algorithm>
#include <chrono>

// ============================================================
// Kernel: Build neighborhood bitmap for seed u
// ============================================================
__global__ void buildNbdBitMapKernel(
    const EdgeIdx*   __restrict__ d_offsets,
    const VertexIdx* __restrict__ d_nbors,
    const int* __restrict__ d_edgeStatus,
    bool*            __restrict__ d_nbdBitMap,
    VertexIdx        u,
    Count            degu)
{
    EdgeIdx tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= degu) return;

    EdgeIdx loc = d_offsets[u] + tid;
    if (d_edgeStatus[loc] == EDGE_ALIVE) {
        d_nbdBitMap[d_nbors[loc]] = true;
    }
}

// ============================================================
// Kernel: Clear neighborhood bitmap
// ============================================================
__global__ void clearNbdBitMapKernel(
    const EdgeIdx*   __restrict__ d_offsets,
    const VertexIdx* __restrict__ d_nbors,
    const int* __restrict__ d_edgeStatus,
    bool*            __restrict__ d_nbdBitMap,
    VertexIdx        u,
    Count            degu)
{
    EdgeIdx tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= degu) return;

    EdgeIdx loc = d_offsets[u] + tid;
    if (d_edgeStatus[loc] == EDGE_ALIVE) {
        d_nbdBitMap[d_nbors[loc]] = false;
    }
}

// ============================================================
// Kernel: Two-hop triangle enumeration for seed vertex u
//
// Parallelized over edges of u (strided access).
// Each active thread processes one neighbor v.
// ============================================================
__global__ void extractInnerKernel(
    const EdgeIdx*   __restrict__ d_offsets,
    const VertexIdx* __restrict__ d_nbors,
    const int* __restrict__ d_edgeStatus,
    const Count*     __restrict__ d_degree,
    const bool*      __restrict__ d_nbdBitMap,
    const bool*      __restrict__ d_clustered,
    double*          __restrict__ d_candidateWgt,
    bool*            __restrict__ d_candidateFlag,
    double*          __restrict__ d_internalWgt,
    double*          __restrict__ d_potentialWgt,
    VertexIdx        u,
    Count            degu,
    double           eps,
    VertexIdx        nVertices)
{
    EdgeIdx startU = d_offsets[u];
    EdgeIdx endU   = d_offsets[u + 1];

    // Strided loop: each thread handles multiple v positions
    for (EdgeIdx vPos = startU + blockIdx.x * blockDim.x + threadIdx.x;
         vPos < endU;
         vPos += gridDim.x * blockDim.x)
    {
        if (d_edgeStatus[vPos] != EDGE_ALIVE) continue;

        VertexIdx v = d_nbors[vPos];
        Count degv = d_degree[v];

        // Degree filter
        if ((double)degv > (double)degu / eps) continue;

        // For each subsequent neighbor w of u
        for (EdgeIdx wPos = vPos + 1; wPos < endU; wPos++) {
            if (d_edgeStatus[wPos] != EDGE_ALIVE) continue;

            VertexIdx w = d_nbors[wPos];
            Count degw = d_degree[w];

            if ((double)degw > (double)degu / eps) continue;

            // Check if (v,w) is an edge → triangle (u,v,w)
            EdgeIdx loc_vw = getEdgeBinaryDevice(d_offsets, d_nbors,
                                                  nVertices, v, w);
            if (loc_vw == -1) continue;
            if (d_edgeStatus[loc_vw] == EDGE_DEAD) continue;

            // Triangle (u,v,w) found
            double triWgt = 1.0 / (double)(degu * degv * degw);
            atomicAdd(d_internalWgt, triWgt);

            // Enumerate triangles (v,w,x)
            VertexIdx lower  = (degv < degw) ? v : w;
            VertexIdx higher = (degv < degw) ? w : v;

            EdgeIdx loStart = d_offsets[lower];
            EdgeIdx loEnd   = d_offsets[lower + 1];

            for (EdgeIdx ell = loStart; ell < loEnd; ell++) {
                if (d_edgeStatus[ell] == EDGE_DEAD) continue;

                VertexIdx x = d_nbors[ell];
                if (x == lower || x == higher || x == u) continue;

                EdgeIdx loc_hx = getEdgeBinaryDevice(d_offsets, d_nbors,
                                                     nVertices, higher, x);
                if (loc_hx == -1) continue;
                if (d_edgeStatus[loc_hx] == EDGE_DEAD) continue;

                // Triangle (v,w,x) found
                Count degx = d_degree[x];
                double wgt = 1.0 / (double)(degx * degv * degw);

                if (d_nbdBitMap[x]) {
                    // x is also a neighbor of u — add to internalWgt
                    // Tiebreaking: only if x is the highest ID
                    if (x > v && x > w) {
                        atomicAdd(d_internalWgt, wgt);
                    }
                    continue;
                }

                // x is a two-hop candidate
                if (d_clustered[x]) continue;

                bool oldFlag = atomicExch(&d_candidateFlag[x], true);
                if (!oldFlag) {
                    atomicAdd(d_potentialWgt, wgt);
                }
                atomicAdd(&d_candidateWgt[x], wgt);
            }
        }
    }
}

// ============================================================
// Host wrapper: gpuExtractInner
//
// Returns true if seed u has at least one alive neighbor.
// ============================================================
bool gpuExtractInner(
    DeviceGraph* dg,
    VertexIdx    u,
    double       eps,
    VertexIdx*   h_candidates,
    double*      h_candidateWgt,
    VertexIdx    maxCandidates,
    VertexIdx*   nCandidates,
    double*      internalWgt,
    double*      potentialWgt)
{
    VertexIdx nVertices = dg->nVertices;

    // Get degree from device
    Count h_degu;
    CUDA_CHECK(cudaMemcpy(&h_degu, &dg->d_degree[u], sizeof(Count),
                          cudaMemcpyDeviceToHost));

    if (h_degu == 0) return false;

    // Step 1: Build nbdBitMap
    {
        EdgeIdx nBlocks = (h_degu + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (nBlocks > 65535) nBlocks = 65535;

        buildNbdBitMapKernel<<<nBlocks, THREADS_PER_BLOCK>>>(
            dg->d_offsets, dg->d_nbors, dg->d_edgeStatus,
            dg->d_nbdBitMap, u, h_degu);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Step 2: Allocate per-call GPU scalars for internalWgt and potentialWgt
    double *d_internalWgt, *d_potentialWgt;
    CUDA_CHECK(cudaMalloc(&d_internalWgt, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_potentialWgt, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_internalWgt, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_potentialWgt, 0, sizeof(double)));

    // Reset candidate arrays
    CUDA_CHECK(cudaMemset(dg->d_candidateWgt, 0, nVertices * sizeof(double)));
    CUDA_CHECK(cudaMemset(dg->d_candidateFlag, 0, nVertices * sizeof(bool)));

    // Step 3: Launch extraction kernel
    {
        // Use enough blocks for coverage; cap at 65535
        EdgeIdx nBlocks = (h_degu + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (nBlocks > 65535) nBlocks = 65535;

        extractInnerKernel<<<nBlocks, THREADS_PER_BLOCK>>>(
            dg->d_offsets, dg->d_nbors, dg->d_edgeStatus,
            dg->d_degree, dg->d_nbdBitMap, dg->d_clustered,
            dg->d_candidateWgt, dg->d_candidateFlag,
            d_internalWgt, d_potentialWgt,
            u, h_degu, eps, nVertices);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Step 4: Clear nbdBitMap
    {
        EdgeIdx nBlocks = (h_degu + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (nBlocks > 65535) nBlocks = 65535;

        clearNbdBitMapKernel<<<nBlocks, THREADS_PER_BLOCK>>>(
            dg->d_offsets, dg->d_nbors, dg->d_edgeStatus,
            dg->d_nbdBitMap, u, h_degu);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Step 5: Transfer results to host
    CUDA_CHECK(cudaMemcpy(internalWgt, d_internalWgt, sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(potentialWgt, d_potentialWgt, sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_internalWgt));
    CUDA_CHECK(cudaFree(d_potentialWgt));

    // Step 6: Extract candidates from device
    // Transfer candidateFlag to host, scan for set bits
    bool* h_candidateFlag = new bool[nVertices];
    double* h_candidateWgtFull = new double[nVertices];
    CUDA_CHECK(cudaMemcpy(h_candidateFlag, dg->d_candidateFlag,
                          nVertices * sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_candidateWgtFull, dg->d_candidateWgt,
                          nVertices * sizeof(double), cudaMemcpyDeviceToHost));

    *nCandidates = 0;
    for (VertexIdx x = 0; x < nVertices && *nCandidates < maxCandidates; x++) {
        if (h_candidateFlag[x]) {
            h_candidates[*nCandidates] = x;
            h_candidateWgt[*nCandidates] = h_candidateWgtFull[x];
            (*nCandidates)++;
        }
    }

    delete[] h_candidateFlag;
    delete[] h_candidateWgtFull;

    return true;
}

// ============================================================
// Kernel: Mark cluster edges for deletion
// ============================================================
__global__ void markClusterEdgesKernel(
    const VertexIdx* __restrict__ d_clusterVerts,
    VertexIdx  nClusterVerts,
    const EdgeIdx*   __restrict__ d_offsets,
    const VertexIdx* __restrict__ d_nbors,
    int* __restrict__ d_edgeStatus,
    const EdgeIdx*   __restrict__ d_partnerMap,
    Pair*      __restrict__ d_toDelete,
    EdgeIdx*   __restrict__ d_nToDelete,
    EdgeIdx    maxToDelete)
{
    VertexIdx idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nClusterVerts) return;

    VertexIdx z = d_clusterVerts[idx];
    EdgeIdx start = d_offsets[z];
    EdgeIdx end   = d_offsets[z + 1];

    for (EdgeIdx j = start; j < end; j++) {
        if (d_edgeStatus[j] != EDGE_ALIVE) continue;

        int old = atomicExch(&d_edgeStatus[j], EDGE_STACK);
        if (old == EDGE_ALIVE) {
            atomicExch(&d_edgeStatus[d_partnerMap[j]], EDGE_STACK);

            EdgeIdx pos = atomicAdd(d_nToDelete, (EdgeIdx)1);
            if (pos < maxToDelete) {
                d_toDelete[pos].first  = z;
                d_toDelete[pos].second = d_nbors[j];
            }
        }
    }
}

// ============================================================
// Host wrapper: gpuMarkClusterEdges
// ============================================================
void gpuMarkClusterEdges(
    DeviceGraph* dg,
    const VertexIdx* d_clusterVerts,
    VertexIdx  nClusterVerts,
    Pair*      d_toDelete,
    EdgeIdx*   nToDelete,
    EdgeIdx    maxToDelete)
{
    if (nClusterVerts == 0) return;

    CUDA_CHECK(cudaMemset(nToDelete, 0, sizeof(EdgeIdx)));

    EdgeIdx nBlocks = (nClusterVerts + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (nBlocks > 65535) nBlocks = 65535;

    markClusterEdgesKernel<<<nBlocks, THREADS_PER_BLOCK>>>(
        d_clusterVerts, nClusterVerts,
        dg->d_offsets, dg->d_nbors,
        dg->d_edgeStatus, dg->d_partnerMap,
        d_toDelete, nToDelete, maxToDelete);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
