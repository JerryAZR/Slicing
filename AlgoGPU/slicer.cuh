#ifndef SLICER
#define SLICER

#include "triangle.cuh"
#include <map>

#define THREADS_PER_BLOCK 256

// in mm
#define X_LEN 256
#define Y_LEN 128
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



__global__ void outputArray(triangle* triangles_global, size_t num_triangles, bool* out);
__device__ int pixelRayIntersection(triangle t, int x, int y);
__device__ bool getIntersect(int x, int y, triangle* triangles, size_t num_triangles, size_t layer);
__device__ void getOutarray(int x, int y, triangle* triangles, size_t num_triangles, size_t layer, size_t outIdx, size_t flagIdx, bool* out, bool* flagArray);
#endif
