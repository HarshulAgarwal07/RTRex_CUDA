// ============================================================
// RTRExtractor CUDA — Main Driver
//
// GPU-accelerated implementation of the RTRExtractor algorithm
// for dense subgraph discovery via triangle-rich sets.
// ============================================================

#include "gpu_graph.cuh"
#include "gpu_triangle_count.cuh"
#include "gpu_clean.cuh"
#include "gpu_extract.cuh"

// Include Escape library for graph loading and output
#include "../RTRex_Sequential/Escape/GraphIO.h"
#include "../RTRex_Sequential/Escape/Graph.h"
#include "../RTRex_Sequential/Escape/ClusterStructures.h"

#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <unordered_set>
#include <set>
#include <cstdio>
#include <inttypes.h>
#include <chrono>

using namespace std;
using namespace Escape;

// Pair is defined in both gpu_common.cuh (global) and Escape::Pair.
// The GPU .cu files use the global one; disambiguate here.
using ::Pair;

// Inline populateStats — avoids including Decomposition.h (pulls in OpenMP)
static void populateStats(const CGraph* g, Cluster* clus)
{
    clus->nVertices = clus->vertices.size();
    clus->nEdges = 0;
    clus->cut = 0;
    for (VertexIdx u : clus->vertices) {
        for (EdgeIdx i = g->offsets[u]; i < g->offsets[u + 1]; i++) {
            VertexIdx v = g->nbors[i];
            if (clus->vertices.find(v) != clus->vertices.end())
                clus->nEdges++;
            else
                clus->cut++;
        }
    }
    clus->nEdges = clus->nEdges / 2;
}

// ============================================================
// Print decomposition (same format as sequential version)
// ============================================================
void printDecomposition(
    const vector<Cluster>& decomposition,
    const CGraph* cg,
    const string& name)
{
    cout << "\n--------------------------------------------\n";
    cout << "Generating details of " << name << " decomposition\n";

    FILE* f = fopen((name + "-decomposition.txt").c_str(), "w");
    if (!f) { printf("Could not write decomposition output\n"); return; }
    FILE* fs = fopen((name + "-stats.txt").c_str(), "w");
    if (!fs) { printf("Could not write stats output\n"); fclose(f); return; }

    Count* freq = new Count[cg->nVertices + 1]();
    Count maxsize = 0, totalClustered = 0, totalInternal = 0;

    vector<Cluster> sorted = decomposition;
    sort(sorted.begin(), sorted.end(),
         [](const Cluster& a, const Cluster& b) { return a.nVertices > b.nVertices; });

    FILE* fcut = fopen((name + "-cut.txt").c_str(), "w");
    for (const auto& clus : sorted) {
        if (clus.vertices.empty()) continue;
        for (VertexIdx v : clus.vertices) fprintf(f, "%" PRId64 " ", v);
        fprintf(f, "\n");
        if (clus.nVertices > maxsize) maxsize = clus.nVertices;
        freq[clus.nVertices]++;
        if (clus.nVertices > 1) {
            double density = 2.0 * clus.nEdges / ((double)clus.nVertices * (clus.nVertices - 1));
            totalClustered += clus.nVertices;
            totalInternal += clus.nEdges;
            fprintf(fs, "%" PRId64 " %.2f\n", clus.nVertices, density);
        }
        for (VertexIdx vert : clus.vertices) {
            for (EdgeIdx j = cg->offsets[vert]; j < cg->offsets[vert + 1]; j++) {
                VertexIdx nbr = cg->nbors[j];
                if (nbr < vert) continue;
                if (clus.vertices.find(nbr) == clus.vertices.end())
                    fprintf(fcut, "%" PRId64 " %" PRId64 "\n", vert, nbr);
            }
        }
    }
    printf("Non-trivial clustered vertices: %" PRId64 "\n", totalClustered);
    printf("Edges inside clusters: %" PRId64 "\n", totalInternal);

    FILE* ffreq = fopen((name + "-freq.txt").c_str(), "w");
    for (Count i = 0; i <= maxsize; i++)
        fprintf(ffreq, "Size = %" PRId64 ", freq = %" PRId64 ", total vertices = %" PRId64 "\n",
                i, freq[i], i * freq[i]);
    delete[] freq;
    fclose(fs); fclose(ffreq); fclose(fcut); fclose(f);
}

static inline Count hostDegree(const EdgeIdx* offsets, VertexIdx v) {
    return offsets[v + 1] - offsets[v];
}

// ============================================================
// GPU kernel: mark vertex list as clustered
// ============================================================
__global__ void markVerticesClusteredKernel(
    const VertexIdx* __restrict__ d_vertices,
    Count nVertices,
    bool* __restrict__ d_clustered)
{
    Count idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nVertices) return;
    d_clustered[d_vertices[idx]] = true;
}

// ============================================================
// GPU-Accelerated disjointExtract
// ============================================================
vector<Cluster> disjointExtractCUDA(
    DeviceGraph* dg,
    const CGraph* cg,
    double eps,
    map<VertexIdx, VertexIdx>& cluster_map)
{
    printf("\n[Extraction] Starting GPU-accelerated extraction\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    VertexIdx nVertices = dg->nVertices;
    EdgeIdx   nEdges   = dg->nEdges;
    const EdgeIdx* h_offsets = cg->offsets;

    // Sort vertices by degree (increasing)
    vector<pair<VertexIdx, Count>> degInfo;
    degInfo.reserve(nVertices);
    for (VertexIdx i = 0; i < nVertices; i++)
        degInfo.emplace_back(i, hostDegree(h_offsets, i));
    sort(degInfo.begin(), degInfo.end(),
         [](const auto& a, const auto& b) {
             if (a.second != b.second) return a.second < b.second;
             return a.first < b.first;
         });

    bool* h_clustered = new bool[nVertices]();
    for (VertexIdx i = 0; i < nVertices; i++) cluster_map[i] = -1;

    VertexIdx* h_candidates  = new VertexIdx[nVertices];
    double*    h_candidateWgt = new double[nVertices];

    // Cached host copies — synced only when GPU weights change
    int*    h_edgeStatus = new int[nEdges];
    double* h_perVertex  = new double[nVertices];

    VertexIdx* d_clusterVerts;
    CUDA_CHECK(cudaMalloc(&d_clusterVerts, nVertices * sizeof(VertexIdx)));

    EdgeIdx maxToDelete = nEdges;
    ::Pair* d_toDelete;
    unsigned long long* d_nToDelete;
    CUDA_CHECK(cudaMalloc(&d_toDelete, maxToDelete * sizeof(::Pair)));
    CUDA_CHECK(cudaMalloc(&d_nToDelete, sizeof(unsigned long long)));

    // Sync initial state from GPU once
    syncEdgeStatusDeviceToHost(dg, h_edgeStatus);
    { Count dummy; syncWeightsDeviceToHost(dg, nullptr, h_perVertex, &dummy); }

    vector<Cluster> decomposition;
    int nClusters = 0;

    // ============================================================
    // Main extraction loop
    // ============================================================
    for (VertexIdx idx = 0; idx < nVertices; idx++) {
        VertexIdx u = degInfo[idx].first;
        if (h_clustered[u]) continue;

        // Filter alive neighbors from CACHED edgeStatus
        vector<VertexIdx> aliveNbd;
        for (EdgeIdx j = cg->offsets[u]; j < cg->offsets[u + 1]; j++) {
            if (h_edgeStatus[j] == EDGE_ALIVE)
                aliveNbd.push_back(cg->nbors[j]);
        }

        // No alive edges → deferred singleton (handled in batch at end)
        if (aliveNbd.empty()) continue;

        // --- Build cluster ---
        Cluster clus;
        clus.vertices.insert(u);
        vector<VertexIdx> allClusterVerts = {u};
        h_clustered[u] = true;
        cluster_map[u] = nClusters;

        for (VertexIdx v : aliveNbd) {
            clus.vertices.insert(v);
            allClusterVerts.push_back(v);
            h_clustered[v] = true;
            cluster_map[v] = nClusters;
        }

        // Mark neighbors as clustered on GPU (single kernel, no sync)
        CUDA_CHECK(cudaMemcpy(d_clusterVerts, aliveNbd.data(),
                              aliveNbd.size() * sizeof(VertexIdx),
                              cudaMemcpyHostToDevice));
        {
            VertexIdx nB = (aliveNbd.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            if (nB > 65535) nB = 65535;
            markVerticesClusteredKernel<<<nB, THREADS_PER_BLOCK>>>(
                d_clusterVerts, aliveNbd.size(), dg->d_clustered);
            CUDA_CHECK(cudaGetLastError());
        }

        // --- GPU extraction inner loop ---
        double internalWgt = 0, potentialWgt = 0;
        VertexIdx nCandidates = 0;
        gpuExtractInner(dg, u, eps, h_candidates, h_candidateWgt,
                        nVertices, &nCandidates, &internalWgt, &potentialWgt);

        // --- Ratio optimization (CPU) ---
        if (nCandidates > 0) {
            vector<pair<VertexIdx, double>> candPairs;
            candPairs.reserve(nCandidates);
            for (VertexIdx c = 0; c < nCandidates; c++)
                candPairs.emplace_back(h_candidates[c], h_candidateWgt[c]);
            sort(candPairs.begin(), candPairs.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });

            double ratio = internalWgt / (internalWgt + potentialWgt);
            int maxind = -1;
            double numSum = 0, denSum = 0;
            for (int c = 0; c < (int)candPairs.size(); c++) {
                numSum += candPairs[c].second;
                denSum += h_perVertex[candPairs[c].first];
                double nr = (internalWgt + numSum) / (internalWgt + potentialWgt + denSum);
                if (nr > ratio) { ratio = nr; maxind = c; }
            }

            for (int c = 0; c <= maxind; c++) {
                VertexIdx x = candPairs[c].first;
                clus.vertices.insert(x);
                allClusterVerts.push_back(x);
                h_clustered[x] = true;
                cluster_map[x] = nClusters;
            }

            if (maxind >= 0) {
                CUDA_CHECK(cudaMemcpy(d_clusterVerts,
                                      allClusterVerts.data() + 1 + aliveNbd.size(),
                                      (maxind + 1) * sizeof(VertexIdx),
                                      cudaMemcpyHostToDevice));
                VertexIdx nB = ((maxind + 1) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                if (nB > 65535) nB = 65535;
                markVerticesClusteredKernel<<<nB, THREADS_PER_BLOCK>>>(
                    d_clusterVerts, maxind + 1, dg->d_clustered);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // --- Finalize cluster ---
        populateStats(cg, &clus);
        decomposition.push_back(clus);
        nClusters++;

        // --- Delete cluster edges + clean on GPU ---
        CUDA_CHECK(cudaMemcpy(d_clusterVerts, allClusterVerts.data(),
                              allClusterVerts.size() * sizeof(VertexIdx),
                              cudaMemcpyHostToDevice));
        gpuMarkClusterEdges(dg, d_clusterVerts, allClusterVerts.size(),
                            d_toDelete, d_nToDelete, maxToDelete);

        unsigned long long nDel;
        CUDA_CHECK(cudaMemcpy(&nDel, d_nToDelete, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        if (nDel > 0)
            gpuDeleteAndClean(dg, d_toDelete, (EdgeIdx)nDel, eps);

        // Sync cached copies after GPU weight changes
        syncEdgeStatusDeviceToHost(dg, h_edgeStatus);
        { Count dummy; syncWeightsDeviceToHost(dg, nullptr, h_perVertex, &dummy); }

        printf("  Cluster %d: seed=%" PRId64 " size=%zu\n",
               nClusters, u, clus.nVertices);
    }

    // ============================================================
    // Batch: remaining unclustered → singletons (single GPU kernel)
    // ============================================================
    vector<VertexIdx> singletons;
    for (VertexIdx v = 0; v < nVertices; v++)
        if (!h_clustered[v]) singletons.push_back(v);

    if (!singletons.empty()) {
        CUDA_CHECK(cudaMemcpy(d_clusterVerts, singletons.data(),
                              singletons.size() * sizeof(VertexIdx),
                              cudaMemcpyHostToDevice));
        VertexIdx nB = (singletons.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (nB > 65535) nB = 65535;
        markVerticesClusteredKernel<<<nB, THREADS_PER_BLOCK>>>(
            d_clusterVerts, singletons.size(), dg->d_clustered);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        for (VertexIdx v : singletons) {
            Cluster sing;
            sing.vertices.insert(v);
            sing.nVertices = 1;
            sing.nEdges = 0;
            sing.cut = hostDegree(h_offsets, v);
            decomposition.push_back(sing);
            cluster_map[v] = nClusters++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    printf("[Extraction] %d clusters in %.3f s\n", nClusters,
           std::chrono::duration<double>(t1 - t0).count());

    delete[] h_clustered;
    delete[] h_candidates;
    delete[] h_candidateWgt;
    delete[] h_perVertex;
    delete[] h_edgeStatus;
    CUDA_CHECK(cudaFree(d_clusterVerts));
    CUDA_CHECK(cudaFree(d_toDelete));
    CUDA_CHECK(cudaFree(d_nToDelete));

    return decomposition;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[])
{
    if (argc <= 3) {
        printf("Usage: %s <graph_file> <output_name> <eps> [gpu_device]\n", argv[0]);
        return 1;
    }

    string graphPath(argv[1]), name(argv[2]);
    double eps = atof(argv[3]);
    int deviceId = (argc >= 5) ? atoi(argv[4]) : 0;

    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    printf("[Main] GPU: %s (CC %d.%d, %.1f GB VRAM)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    Graph g;
    if (loadGraph(graphPath.c_str(), g, 1, IOFormat::escape)) {
        printf("Error: Failed to load graph\n");
        return 1;
    }
    printf("[Main] Loaded: %" PRId64 " vertices, %" PRId64 " edges\n",
           g.nVertices, g.nEdges / 2);

    CGraph cg = makeCSR(g);
    cg.sortById();
    cg.getPartnerMap();

    DeviceGraph dg;
    transferGraphToDevice(cg.offsets, cg.nbors, cg.nVertices, cg.nEdges, &dg);
    allocateDeviceState(&dg);

    gpuCommonNbr(&dg);
    gpuInitialClean(&dg, eps);

    map<VertexIdx, VertexIdx> cluster_map;
    vector<Cluster> decomposition = disjointExtractCUDA(&dg, &cg, eps, cluster_map);

    printDecomposition(decomposition, &cg, "RTRex-CUDA-" + name);

    freeDeviceGraph(&dg);
    delCGraph(cg);
    delGraph(g);

    printf("[Main] Done.\n");
    return 0;
}
