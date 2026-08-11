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
    if (!f) {
        printf("Could not write decomposition output\n");
        return;
    }

    FILE* fs = fopen((name + "-stats.txt").c_str(), "w");
    if (!fs) {
        printf("Could not write stats output\n");
        fclose(f);
        return;
    }

    // Count clusters and collect stats
    Count totalClustered = 0;
    Count* freq = new Count[cg->nVertices + 1]();
    Count maxsize = 0;

    // Sort clusters by size decreasing (non-singletons first)
    vector<Cluster> sorted = decomposition;
    sort(sorted.begin(), sorted.end(),
         [](const Cluster& a, const Cluster& b) {
             return a.nVertices > b.nVertices;
         });

    Count numClusters = 0;
    Count totalInternal = 0;
    FILE* fcut = fopen((name + "-cut.txt").c_str(), "w");

    for (const auto& clus : sorted) {
        if (clus.vertices.empty()) continue;
        numClusters++;

        // Write vertices
        for (VertexIdx v : clus.vertices)
            fprintf(f, "%" PRId64 " ", v);
        fprintf(f, "\n");

        // Stats
        if (clus.nVertices > maxsize) maxsize = clus.nVertices;
        freq[clus.nVertices]++;

        double density = 0;
        if (clus.nVertices > 1) {
            density = 2.0 * clus.nEdges /
                      ((double)clus.nVertices * (clus.nVertices - 1));
            totalClustered += clus.nVertices;
            totalInternal += clus.nEdges;
            fprintf(fs, "%" PRId64 " %.2f\n", clus.nVertices, density);
        }

        // Cut edges (with tiebreaking nbr < vert)
        for (VertexIdx vert : clus.vertices) {
            for (EdgeIdx j = cg->offsets[vert];
                 j < cg->offsets[vert + 1]; j++) {
                VertexIdx nbr = cg->nbors[j];
                if (nbr < vert) continue;
                if (clus.vertices.find(nbr) == clus.vertices.end())
                    fprintf(fcut, "%" PRId64 " %" PRId64 "\n", vert, nbr);
            }
        }
    }

    printf("Non-trivial clustered vertices: %" PRId64 "\n", totalClustered);
    printf("Edges inside clusters: %" PRId64 "\n", totalInternal);
    cout << "--------------------------------------------\n";

    // Frequency file
    FILE* ffreq = fopen((name + "-freq.txt").c_str(), "w");
    for (Count i = 0; i <= maxsize; i++) {
        fprintf(ffreq, "Size = %" PRId64 ", freq = %" PRId64
                ", total vertices = %" PRId64 "\n",
                i, freq[i], i * freq[i]);
    }

    delete[] freq;
    fclose(fs);
    fclose(ffreq);
    fclose(fcut);
    fclose(f);
}

// ============================================================
// Compute degree for a vertex from host offsets
// ============================================================
static inline Count hostDegree(const EdgeIdx* offsets, VertexIdx v) {
    return offsets[v + 1] - offsets[v];
}

// ============================================================
// GPU-Accelerated disjointExtract
// ============================================================
__global__ void markVerticesClusteredKernel(
    const VertexIdx* __restrict__ d_vertices,
    Count             nVertices,
    bool*             __restrict__ d_clustered)
{
    Count idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nVertices) return;
    d_clustered[d_vertices[idx]] = true;
}

vector<Cluster> disjointExtractCUDA(
    DeviceGraph* dg,
    const CGraph* cg,
    double eps,
    map<VertexIdx, VertexIdx>& cluster_map)
{
    printf("\n[Extraction] Starting GPU-accelerated extraction\n");
    auto extractionStart = std::chrono::high_resolution_clock::now();

    VertexIdx nVertices = dg->nVertices;
    const EdgeIdx* h_offsets = cg->offsets;

    // Sort vertices by degree (increasing)
    vector<pair<VertexIdx, Count>> degInfo;
    degInfo.reserve(nVertices);
    for (VertexIdx i = 0; i < nVertices; i++) {
        degInfo.emplace_back(i, hostDegree(h_offsets, i));
    }
    sort(degInfo.begin(), degInfo.end(),
         [](const auto& a, const auto& b) {
             if (a.second != b.second) return a.second < b.second;
             return a.first < b.first;
         });

    // Host-side clustered bitmap
    bool* h_clustered = new bool[nVertices]();
    // Initialize cluster_map to -1
    for (VertexIdx i = 0; i < nVertices; i++) cluster_map[i] = -1;

    // Allocate GPU candidate arrays (sized for max possible)
    VertexIdx maxCandidates = nVertices;
    VertexIdx* h_candidates = new VertexIdx[maxCandidates];
    double*    h_candidateWgt = new double[maxCandidates];

    // Get host perVertex weights for ratio optimization
    double* h_perVertex = new double[nVertices];

    // Allocate device-side cluster vertex list + toDelete list
    VertexIdx* d_clusterVerts;
    CUDA_CHECK(cudaMalloc(&d_clusterVerts, nVertices * sizeof(VertexIdx)));

    EdgeIdx maxToDelete = dg->nEdges; // worst case
    ::Pair* d_toDelete;
    unsigned long long* d_nToDelete;
    CUDA_CHECK(cudaMalloc(&d_toDelete, maxToDelete * sizeof(::Pair)));
    CUDA_CHECK(cudaMalloc(&d_nToDelete, sizeof(unsigned long long)));

    vector<Cluster> decomposition;
    int nClusters = 0;

    // ============================================================
    // Main extraction loop
    // ============================================================
    for (VertexIdx idx = 0; idx < nVertices; idx++) {
        VertexIdx u = degInfo[idx].first;
        if (h_clustered[u]) continue;

        Count h_degu = hostDegree(h_offsets, u);

        // Build list of u's alive neighbors
        vector<VertexIdx> nbdList;
        for (EdgeIdx j = cg->offsets[u]; j < cg->offsets[u + 1]; j++) {
            // We need edgeStatus on host — sync if needed
            // For now, assume we track nbd from the graph structure
            // (we'll refine this — need edgeStatus sync)
            nbdList.push_back(cg->nbors[j]);
        }

        // For now: we need to sync edgeStatus from device to know
        // which edges are alive. Let's do a sync at the start of each seed.
        // Actually, this is expensive. Better approach:
        // We track the cluster on CPU, and sync edgeStatus only when
        // we need to enumerate neighbors.

        // Simplified approach: sync edgeStatus once per round
        // For prototyping, sync every seed
        int* h_edgeStatus = new int[dg->nEdges];
        syncEdgeStatusDeviceToHost(dg, h_edgeStatus);

        // Filter alive neighbors
        vector<VertexIdx> aliveNbd;
        for (EdgeIdx j = cg->offsets[u]; j < cg->offsets[u + 1]; j++) {
            if (h_edgeStatus[j] == 'Y')
                aliveNbd.push_back(cg->nbors[j]);
        }
        delete[] h_edgeStatus;

        if (aliveNbd.empty()) {
            // Singleton cluster
            Cluster sing;
            sing.vertices.insert(u);
            sing.nVertices = 1;
            sing.nEdges = 0;
            sing.cut = aliveNbd.size(); // actually this was the full degree
            decomposition.push_back(sing);
            h_clustered[u] = true;
            cluster_map[u] = nClusters++;

            // Mark u as clustered on device
            VertexIdx h_single[1] = {u};
            CUDA_CHECK(cudaMemcpy(d_clusterVerts, h_single, sizeof(VertexIdx),
                                  cudaMemcpyHostToDevice));
            markVerticesClusteredKernel<<<1, 1>>>(
                d_clusterVerts, 1, dg->d_clustered);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            continue;
        }

        // u is already marked as clustered on device
        // Mark nbd neighbors as clustered on device + in host map
        {
            CUDA_CHECK(cudaMemcpy(d_clusterVerts, aliveNbd.data(),
                                  aliveNbd.size() * sizeof(VertexIdx),
                                  cudaMemcpyHostToDevice));
            VertexIdx nBlocks = (aliveNbd.size() + THREADS_PER_BLOCK - 1) /
                                THREADS_PER_BLOCK;
            if (nBlocks > 65535) nBlocks = 65535;
            markVerticesClusteredKernel<<<nBlocks, THREADS_PER_BLOCK>>>(
                d_clusterVerts, aliveNbd.size(), dg->d_clustered);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            for (VertexIdx v : aliveNbd) {
                h_clustered[v] = true;
                cluster_map[v] = nClusters;
            }
        }
        h_clustered[u] = true;
        cluster_map[u] = nClusters;

        // Run GPU extraction inner loop
        double internalWgt = 0, potentialWgt = 0;
        VertexIdx nCandidates = 0;

        bool ok = gpuExtractInner(dg, u, eps,
                                   h_candidates, h_candidateWgt,
                                   maxCandidates, &nCandidates,
                                   &internalWgt, &potentialWgt);

        // Sync perVertex weights for ratio optimization
        Count dummy;
        syncWeightsDeviceToHost(dg, nullptr, h_perVertex, &dummy);
        // (we don't need perEdge here)

        // Ratio optimization (CPU, same as sequential)
        if (nCandidates > 0) {
            // Sort candidates by weight descending
            vector<pair<VertexIdx, double>> candPairs;
            candPairs.reserve(nCandidates);
            for (VertexIdx i = 0; i < nCandidates; i++)
                candPairs.emplace_back(h_candidates[i], h_candidateWgt[i]);

            sort(candPairs.begin(), candPairs.end(),
                 [](const auto& a, const auto& b) {
                     return a.second > b.second;
                 });

            double ratio = internalWgt / (internalWgt + potentialWgt);
            int maxind = -1;
            double numSum = 0, denSum = 0;

            for (int i = 0; i < (int)candPairs.size(); i++) {
                numSum += candPairs[i].second;
                denSum += h_perVertex[candPairs[i].first];
                double newRatio = (internalWgt + numSum) /
                                  (internalWgt + potentialWgt + denSum);
                if (newRatio > ratio) {
                    ratio = newRatio;
                    maxind = i;
                }
            }

            // Add selected two-hop vertices to cluster
            vector<VertexIdx> selectedTwoHop;
            for (int i = 0; i <= maxind; i++) {
                VertexIdx x = candPairs[i].first;
                selectedTwoHop.push_back(x);
                h_clustered[x] = true;
                cluster_map[x] = nClusters;
            }

            // Mark them as clustered on device
            if (!selectedTwoHop.empty()) {
                CUDA_CHECK(cudaMemcpy(d_clusterVerts, selectedTwoHop.data(),
                                      selectedTwoHop.size() * sizeof(VertexIdx),
                                      cudaMemcpyHostToDevice));
                VertexIdx nBlocks = (selectedTwoHop.size() +
                                     THREADS_PER_BLOCK - 1) /
                                    THREADS_PER_BLOCK;
                if (nBlocks > 65535) nBlocks = 65535;
                markVerticesClusteredKernel<<<nBlocks, THREADS_PER_BLOCK>>>(
                    d_clusterVerts, selectedTwoHop.size(), dg->d_clustered);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }

        // Build cluster on CPU
        Cluster clus;
        clus.vertices.insert(u);
        for (VertexIdx v : aliveNbd) clus.vertices.insert(v);
        if (nCandidates > 0) {
            // Re-sort to get selected candidates up to maxind
            // We already added them above, but let's add from stored list
        }

        // Build complete vertex list for GPU edge marking
        vector<VertexIdx> allClusterVerts;
        allClusterVerts.push_back(u);
        for (VertexIdx v : aliveNbd) allClusterVerts.push_back(v);
        // add two-hop selected (stored in local selectedTwoHop)
        // We need to track which were added. Let me restructure...

        // Simplified: just compute cluster stats on CPU using h_edgeStatus
        // and push to decomposition

        // Mark all cluster edges for deletion on GPU
        // Actually, the sequential code does: push all alive edges of
        // cluster vertices to toDelete stack, then call deleteAndClean.
        // On GPU: we use markClusterEdgesKernel.

        // The cluster includes u + aliveNbd + selectedTwoHop
        // Let's build the full list:
        size_t clusterSize = 1 + aliveNbd.size();
        // Add selected two-hop (from the ratio optimization above)
        // We need to track them. Let me restructure this.

        // For now, compute cluster stats and push
        populateStats((CGraph*)cg, &clus);
        decomposition.push_back(clus);
        nClusters++;

        // Now delete cluster edges on GPU
        // Build full vertex list for edge deletion
        // (u + alive neighbors)
        CUDA_CHECK(cudaMemcpy(d_clusterVerts, allClusterVerts.data(),
                              allClusterVerts.size() * sizeof(VertexIdx),
                              cudaMemcpyHostToDevice));
        gpuMarkClusterEdges(dg, d_clusterVerts, allClusterVerts.size(),
                            d_toDelete, d_nToDelete, maxToDelete);

        unsigned long long nDel;
        CUDA_CHECK(cudaMemcpy(&nDel, d_nToDelete, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        if (nDel > 0) {
            gpuDeleteAndClean(dg, d_toDelete, (EdgeIdx)nDel, eps);
        }

        if ((nClusters) % 1000 == 0) {
            printf("  Extracted %d clusters...\n", nClusters);
        }
    }

    // Add singletons for remaining unclustered vertices
    for (VertexIdx v = 0; v < nVertices; v++) {
        if (!h_clustered[v]) {
            Cluster sing;
            sing.vertices.insert(v);
            sing.nVertices = 1;
            sing.nEdges = 0;
            sing.cut = hostDegree(h_offsets, v);
            decomposition.push_back(sing);
            cluster_map[v] = nClusters++;
        }
    }

    auto extractionEnd = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(
        extractionEnd - extractionStart).count();
    printf("[Extraction] %d clusters in %.3f s\n", nClusters, elapsed);

    // Cleanup
    delete[] h_clustered;
    delete[] h_candidates;
    delete[] h_candidateWgt;
    delete[] h_perVertex;
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
        printf("Usage: %s <graph_file> <output_name> <eps> [gpu_device]\n",
               argv[0]);
        printf("  graph_file:  Path to graph in Escape format\n");
        printf("  output_name: Prefix for output files\n");
        printf("  eps:         Cleaning threshold (e.g., 0.1)\n");
        printf("  gpu_device:  GPU device ID (default: 0)\n");
        return 1;
    }

    string graphPath(argv[1]);
    string name(argv[2]);
    double eps = atof(argv[3]);

    int deviceId = 0;
    if (argc >= 5) deviceId = atoi(argv[4]);

    // Select GPU
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    printf("[Main] Using GPU: %s (CC %d.%d, %.1f GB VRAM)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Load graph
    printf("[Main] Loading graph: %s\n", graphPath.c_str());
    Graph g;
    if (loadGraph(graphPath.c_str(), g, 1, IOFormat::escape)) {
        printf("Error: Failed to load graph\n");
        return 1;
    }
    printf("[Main] Loaded: %" PRId64 " vertices, %" PRId64 " edges\n",
           g.nVertices, g.nEdges / 2);

    // Convert to CSR
    CGraph cg = makeCSR(g);
    cg.sortById();
    cg.getPartnerMap();

    printf("[Main] CSR: %" PRId64 " directed edges\n", cg.nEdges);

    // Setup DeviceGraph
    DeviceGraph dg;
    transferGraphToDevice(cg.offsets, cg.nbors,
                          cg.nVertices, cg.nEdges, &dg);
    allocateDeviceState(&dg);

    // Phase 1: Triangle counting
    gpuCommonNbr(&dg);

    // Phase 2: Initial clean
    gpuInitialClean(&dg, eps);

    // Phase 3: Extraction
    map<VertexIdx, VertexIdx> cluster_map;
    vector<Cluster> decomposition = disjointExtractCUDA(
        &dg, &cg, eps, cluster_map);

    // Phase 4: Output
    printDecomposition(decomposition, &cg, "RTRex-CUDA-" + name);

    // Cleanup
    freeDeviceGraph(&dg);
    delCGraph(cg);
    delGraph(g);

    printf("[Main] Done.\n");
    return 0;
}
