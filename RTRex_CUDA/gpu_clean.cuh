#ifndef GPU_CLEAN_CUH
#define GPU_CLEAN_CUH

#include "gpu_graph.cuh"

// ============================================================
// Bulk-synchronous GPU edge cleaning
// ============================================================
// Replaces the sequential initialClean() + deleteAndClean()
// cascade with iterative GPU rounds.
//
// Algorithm:
//   Each round consists of multiple sub-rounds:
//     Sub-round:
//       1. Identify edges with perEdge < eps * edgeWeight → mark 'S'
//       2. Process all 'S' edges: remove triangles, subtract weights,
//          cascade to adjacent edges
//   Repeat until no edges are marked 'S' in a round.
//
// Parameters:
//   dg:       Device graph with pre-computed perEdge/perVertex weights
//   eps:      Cleaning threshold parameter
//   maxRounds: Maximum number of bulk rounds (safety limit)
//
// Output (modified in dg):
//   dg->d_edgeStatus[] — edges marked 'N' for deleted
//   dg->d_perEdge[]    — updated triangle weights
//   dg->d_perVertex[]  — updated vertex triangle weights
// ============================================================
void gpuInitialClean(DeviceGraph* dg, double eps, int maxRounds = 100);

// ============================================================
// GPU bulk deleteAndClean for a specific set of edges
// ============================================================
// Used during extraction: given a list of edges to delete,
// removes them and propagates cleaning.
//
// Parameters:
//   dg:           Device graph
//   d_toDelete:   Device array of edge pairs to delete
//   nToDelete:    Number of edges to delete
//   eps:          Cleaning threshold
// ============================================================
void gpuDeleteAndClean(DeviceGraph* dg, const Pair* d_toDelete,
                       EdgeIdx nToDelete, double eps);

#endif // GPU_CLEAN_CUH
