#ifndef GPU_GRAPH_CUH
#define GPU_GRAPH_CUH

#include "gpu_common.cuh"

// ============================================================
// Device-side graph in CSR format
// ============================================================
struct DeviceGraph {
    VertexIdx nVertices;       // number of vertices
    EdgeIdx   nEdges;          // number of directed edges (2 * undirected edges)

    EdgeIdx*   d_offsets;      // length nVertices+1, CSR offsets
    VertexIdx* d_nbors;        // length nEdges, neighbor vertex IDs
    EdgeIdx*   d_partnerMap;   // length nEdges, partner edge index for undirected graphs

    // Per-vertex data
    Count*     d_degree;       // length nVertices, degree of each vertex

    // Per-edge state
    char*      d_edgeStatus;   // length nEdges: 'Y' alive, 'N' deleted, 'S' on stack
    double*    d_perEdge;      // length nEdges, triangle weight per directed edge

    // Per-vertex triangle weight
    double*    d_perVertex;    // length nVertices, total triangle weight per vertex

    // Extraction state
    bool*      d_clustered;    // length nVertices, whether vertex is already clustered
    bool*      d_nbdBitMap;    // length nVertices, temporary neighborhood bitmap

    // Candidate accumulation (for extraction inner loop)
    double*    d_candidateWgt;  // length nVertices, two-hop candidate weights
    bool*      d_candidateFlag; // length nVertices, set to true if vertex is a candidate

    // GPU-side work counters (per round)
    int*       d_hasStack;     // flag for edge stack non-empty check

    // Global triangle count
    Count      totalTriangles; // total number of triangles
};

// ============================================================
// Transfer graph from host CSR to device
// ============================================================
// Takes host-side CGraph (from Escape library) and copies data to GPU.
// Also builds partnerMap on device.
void transferGraphToDevice(
    const EdgeIdx*   h_offsets,
    const VertexIdx* h_nbors,
    VertexIdx        nVertices,
    EdgeIdx          nEdges,
    DeviceGraph*     dg);

// ============================================================
// Allocate device-side state arrays
// ============================================================
void allocateDeviceState(DeviceGraph* dg);

// ============================================================
// Free all GPU resources
// ============================================================
void freeDeviceGraph(DeviceGraph* dg);

// ============================================================
// Sync per-edge weights and per-vertex weights device → host
// ============================================================
void syncWeightsDeviceToHost(
    const DeviceGraph* dg,
    double* h_perEdge,
    double* h_perVertex,
    Count*  h_totalTriangles);

// ============================================================
// Sync edge status device → host
// ============================================================
void syncEdgeStatusDeviceToHost(const DeviceGraph* dg, char* h_edgeStatus);

// ============================================================
// Sync clustered bitmap device → host
// ============================================================
void syncClusteredDeviceToHost(const DeviceGraph* dg, bool* h_clustered);

#endif // GPU_GRAPH_CUH
