#ifndef SLICER
#define SLICER

#include "triangle.cuh"
#include <thrust/device_vector.h>

#define LOG_THREADS 8
#define THREADS_PER_BLOCK (1 << LOG_THREADS)
#define MAX_TRUNK_SIZE	24

// in mm
#define LOG_X 8
#define LOG_Y 7
#define X_LEN (1 << LOG_X)
#define Y_LEN (1 << LOG_Y)
#define HEIGHT 100
#define RESOLUTION 1

// in pixels
#define NUM_LAYERS (size_t)(HEIGHT / RESOLUTION)
#define X_DIM (size_t)(X_LEN / RESOLUTION)
#define Y_DIM (size_t)(Y_LEN / RESOLUTION)

#define X_MIN (long)(-1 * X_LEN / 2)
#define X_MAX (long)(X_LEN / 2)
#define Y_MIN (long)(-1 * Y_LEN / 2)
#define Y_MAX (long)(Y_LEN / 2)

typedef int layer_t;

#define NONE ((layer_t)(-1))

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

__global__ void largeTriIntersection(triangle* tri_large, size_t num_large, layer_t* intersections, size_t* trunk_length);
__global__ void smallTriIntersection(triangle* tri_small, double* zMins, 
    size_t num_small, layer_t* intersections_large, size_t* trunk_length, bool* out);
#endif
