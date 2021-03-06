#ifndef SLICER
#define SLICER

#include "triangle.cuh"
#include <thrust/device_vector.h>
#include <iostream>

#define LOG_THREADS 7
#define THREADS_PER_BLOCK (1 << LOG_THREADS)
#define MAX_TRUNK_SIZE	28
#define BATCH_SIZE  (4 * THREADS_PER_BLOCK)
#define NUM_BLOCKS  256
#define TILE_WIDTH 128
#define BLOCK_HEIGHT 16
#define NUM_CPU_THREADS 16

// in mm
#define LOG_X 7
#define LOG_Y 7
#define X_LEN (1 << LOG_X)
#define Y_LEN (1 << LOG_Y)
#define HEIGHT 128
#define RESOLUTION 0.125 // Must be (negative) power of 2

// in pixels
#define NUM_LAYERS ((long)(HEIGHT / RESOLUTION))
// X_DIM must be at least as large as THREADS_PER_BLOCK
#define X_DIM ((long)(X_LEN / RESOLUTION))
#define Y_DIM ((long)(Y_LEN / RESOLUTION))

#define X_MIN ((long)(-1 * X_DIM / 2))
#define X_MAX ((long)(X_DIM / 2 - 1))
#define Y_MIN ((long)(-1 * Y_DIM / 2))
#define Y_MAX ((long)(Y_DIM / 2 - 1))

typedef int layer_t;

#define NONE ((layer_t)(-1))

// Sanity Check
static_assert(THREADS_PER_BLOCK <= X_DIM, "THREADS_PER_BLOCK may not be larger than X_DIM");
static_assert(!(X_DIM & (X_DIM-1)), "RESOLUTION must be some power of 2");

__global__ void pps(triangle* triangles, size_t num_triangles, bool* out);
__global__ void pps(triangle* triangles, size_t num_triangles, bool* out, unsigned base_layer);

// returns the layer of intersection
__device__ layer_t pixelRayIntersection(triangle t, int x, int y);
__device__ int getIntersectionTrunk(int x, int y, triangle* triangles, size_t num_triangles, layer_t* layers);
__device__ bool isInside(layer_t current, layer_t* trunk, size_t length);

__global__ void fps1(triangle* triangles, size_t num_triangles, layer_t* all_intersections, unsigned* trunk_length, int* locks);
__global__ void fps2(layer_t* all_intersections, unsigned* trunk_length);
__global__ void fps3(layer_t* sorted_intersections, unsigned* trunk_length, bool* out);

__global__ void triangle_sort(triangle* triangles_global, size_t num_triangles, double* zmins_global, int* index_global);
__global__ void outputArray(triangle* triangles_global, size_t num_triangles, bool* out, int* index_global);
__device__ int pixelRayIntersectionNew(triangle t, int x, int y);
__device__ bool getIntersect(int x, int y, triangle* triangles, size_t num_triangles, size_t layer, int* index);
__device__ void getOutarray(int x, int y, triangle* triangles, size_t num_triangles, size_t layer, size_t outIdx, size_t flagIdx, bool* out, bool* flagArray, int* index);

__global__ void smallTriIntersection(triangle* tri_small, double* zMins, size_t num_small, bool* out);

__global__ void overlapSlicer(triangle* tri_small, double* zMins, size_t num_small, bool* out);
__global__ void layerExtraction(bool* out, layer_t start);
__host__ void GPUsort(triangle* tris_dev, size_t size, double* zMins);

__global__ void rectTriIntersection(double* tri_global, size_t num_tri, bool* out);
__global__ void rectTriIntersection(double* tri_global, size_t num_tri, bool* out, unsigned base_layer);
__global__ void layerExtraction(bool* out);

__global__ void triangleSelect(triangle* in, triangle* out, unsigned in_length, unsigned* out_length, unsigned base_layer);
__global__ void pointSelect(double* in, double* out, unsigned in_length, unsigned* out_length, unsigned base_layer);

// Compression

#define MAX_FLIPS 32

__global__ void bbox_ints(bool* in, unsigned* out);
void bbox_ints_decompress(unsigned* in, bool* out);


#endif
