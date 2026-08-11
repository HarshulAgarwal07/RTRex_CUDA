#ifndef GPU_EXTRACT_CUH
#define GPU_EXTRACT_CUH

#include "gpu_graph.cuh"

// ============================================================
// GPU-Accelerated Extract Inner Loop
// ============================================================
// Given a seed vertex u, performs the two-hop triangle enumeration
// on GPU and returns candidate vertex weights for ratio optimization.
//
// This is called once per unclustered vertex during the extraction
// outer loop on CPU.
//
// Parameters:
//   dg:              Device graph (CSR + perEdge/perVertex/edgeStatus/clustered)
//   u:               Seed vertex
//   eps:             Cleaning threshold (for degree filtering)
//   h_candidates:    [OUT] Host array of candidate vertex IDs
//   h_candidateWgt:  [OUT] Host array of candidate triangle weights
//   maxCandidates:   Max size of output arrays
//   nCandidates:     [OUT] Number of valid candidates returned
//   internalWgt:     [OUT] Internal triangle weight of the cluster
//   potentialWgt:    [OUT] Sum of all candidate weights (before selection)
//
// Returns: true if cluster is non-trivial, false if no alive edges from u
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
    double*      potentialWgt);

// ============================================================
// Mark all incident edges of a cluster for deletion
// ============================================================
// Given a list of cluster vertices, marks all their alive
// incident edges as 'S' (on stack for deletion by gpuDeleteAndClean).
//
// Parameters:
//   dg:             Device graph
//   d_clusterVerts: Device array of cluster vertex IDs
//   nClusterVerts:  Number of vertices in the cluster
//   d_toDelete:     [OUT] Device array of (u,v) pairs to delete
//   nToDelete:      [OUT] Number of edge pairs written
//   maxToDelete:    Max size of d_toDelete
// ============================================================
void gpuMarkClusterEdges(
    DeviceGraph* dg,
    const VertexIdx* d_clusterVerts,
    VertexIdx  nClusterVerts,
    Pair*      d_toDelete,
    EdgeIdx*   nToDelete,
    EdgeIdx    maxToDelete);

#endif // GPU_EXTRACT_CUH
