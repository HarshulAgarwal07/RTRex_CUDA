#ifndef GPU_TRIANGLE_COUNT_CUH
#define GPU_TRIANGLE_COUNT_CUH

#include "gpu_graph.cuh"

// ============================================================
// GPU triangle counting (equivalent to sequential commonNbr)
// ============================================================
// Computes per-edge and per-vertex weighted triangle information.
// Triangle weight = 1 / (deg(u) * deg(v) * deg(w))
//
// Input:
//   dg: Device graph with CSR + degree arrays populated
//
// Output (modified in dg):
//   dg->d_perEdge[]   — weighted triangle count per directed edge
//   dg->d_perVertex[] — weighted triangle count per vertex
//   dg->totalTriangles — total number of triangles found
//
// This kernel produces BIT-IDENTICAL results to the sequential
// commonNbr function because atomicAdd for doubles is commutative.
// ============================================================
void gpuCommonNbr(DeviceGraph* dg);

#endif // GPU_TRIANGLE_COUNT_CUH
