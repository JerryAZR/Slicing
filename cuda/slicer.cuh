#ifndef SLICER
#define SLICER

#include "triangle.cuh"
#include <thrust/device_vector.h>
#include <iostream>

#define LOG_THREADS 8
#define THREADS_PER_BLOCK (1 << LOG_THREADS)
#define MAX_TRUNK_SIZE	28
#define BATCH_SIZE  (4 * THREADS_PER_BLOCK)

// in mm
#define LOG_X 8
#define LOG_Y 7
#define X_LEN (1 << LOG_X)
#define Y_LEN (1 << LOG_Y)
#define HEIGHT 100
#define RESOLUTION 0.5 // Must be (negative) power of 2

// in pixels
#define NUM_LAYERS ((size_t)(HEIGHT / RESOLUTION))
// X_DIM must be at least as large as THREADS_PER_BLOCK
#define X_DIM ((size_t)(X_LEN / RESOLUTION))
#define Y_DIM ((size_t)(Y_LEN / RESOLUTION))

#define X_MIN ((long)(-1 * X_DIM / 2))
#define X_MAX ((long)(X_DIM / 2))
#define Y_MIN ((long)(-1 * Y_DIM / 2))
#define Y_MAX ((long)(Y_DIM / 2))

typedef int layer_t;

#define NONE ((layer_t)(-1))

// Sanity Check
static_assert(THREADS_PER_BLOCK <= X_DIM, "THREADS_PER_BLOCK may not be larger than X_DIM");
static_assert(!(X_DIM & (X_DIM-1)), "RESOLUTION must be some power of 2");

__global__ void pps(triangle* triangles, size_t num_triangles, bool* out);
// returns the layer of intersection
__device__ layer_t pixelRayIntersection(triangle t, int x, int y);
__device__ int getIntersectionTrunk(int x, int y, triangle* triangles, size_t num_triangles, layer_t* layers);
__device__ bool isInside(layer_t current, layer_t* trunk, size_t length);

__global__ void fps1(triangle* triangles, size_t num_triangles, layer_t* all_intersections, size_t* trunk_length, int* locks);
__global__ void fps2(layer_t* all_intersections, size_t* trunk_length);
__global__ void fps3(layer_t* sorted_intersections, size_t* trunk_length, bool* out);

__global__ void triangle_sort(triangle* triangles_global, size_t num_triangles, double* zmins_global, int* index_global);
__global__ void outputArray(triangle* triangles_global, size_t num_triangles, bool* out, int* index_global);
__device__ int pixelRayIntersectionNew(triangle t, int x, int y);
__device__ bool getIntersect(int x, int y, triangle* triangles, size_t num_triangles, size_t layer, int* index);
__device__ void getOutarray(int x, int y, triangle* triangles, size_t num_triangles, size_t layer, size_t outIdx, size_t flagIdx, bool* out, bool* flagArray, int* index);

__global__ void smallTriIntersection(triangle* tri_small, double* zMins, size_t num_small, bool* out);

__global__ void overlapSlicer(triangle* tri_small, double* zMins, size_t num_small, bool* out);
__global__ void layerExtraction(bool* out, layer_t start);
__host__ void GPUsort(triangle* tris_dev, size_t size, double* zMins);

#endif
