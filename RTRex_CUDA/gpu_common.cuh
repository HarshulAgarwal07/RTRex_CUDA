#ifndef GPU_COMMON_CUH
#define GPU_COMMON_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <inttypes.h>

// ============================================================
// Type definitions — match Escape library's types exactly
// ============================================================
typedef int64_t VertexIdx;
typedef int64_t EdgeIdx;
typedef int64_t Count;

// ============================================================
// CUDA kernel launch configuration
// ============================================================
const int THREADS_PER_BLOCK = 256;
const int WARP_SIZE = 32;

// ============================================================
// Edge status constants (must match sequential: 'Y', 'N', 'S')
// ============================================================
const char EDGE_ALIVE = 'Y';
const char EDGE_DEAD = 'N';
const char EDGE_STACK = 'S';

// ============================================================
// CUDA error checking macro
// ============================================================
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(1);                                                    \
        }                                                               \
    } while (0)

// ============================================================
// Pair type used for edge deletion worklist
// ============================================================
struct Pair {
    VertexIdx first;
    VertexIdx second;
};

// ============================================================
// Device functions for binary search in sorted neighbor lists
// ============================================================

// Binary search for 'val' in nbors[low .. high-1].
// Returns the index if found, or -1 if not found.
__device__ inline EdgeIdx binarySearchDevice(
    const VertexIdx* __restrict__ nbors,
    EdgeIdx low,
    EdgeIdx high,
    VertexIdx val)
{
    while (low <= high) {
        EdgeIdx mid = (low + high) / 2;
        VertexIdx midVal = nbors[mid];
        if (midVal == val)
            return mid;
        if (midVal > val)
            high = mid - 1;
        else
            low = mid + 1;
    }
    return -1;
}

// Look up edge (v1, v2) in the CSR graph using binary search.
// Assumes adjacency lists are sorted by vertex ID.
__device__ inline EdgeIdx getEdgeBinaryDevice(
    const EdgeIdx* __restrict__ offsets,
    const VertexIdx* __restrict__ nbors,
    VertexIdx nVertices,
    VertexIdx v1,
    VertexIdx v2)
{
    if (v1 >= nVertices)
        return -1;
    EdgeIdx low = offsets[v1];
    EdgeIdx high = offsets[v1 + 1] - 1;
    return binarySearchDevice(nbors, low, high, v2);
}

// ============================================================
// Atomic double operations (for pre-sm_60 compatibility)
// ============================================================
#if __CUDA_ARCH__ >= 600
    // atomicAdd for double is natively available on sm_60+
#else
__device__ inline double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

#endif // GPU_COMMON_CUH
