#ifndef SLICER
#define SLICER

#include "triangle.cuh"

// in mm
#define X_LEN 200
#define Y_LEN 200
#define HEIGHT 20
#define RESOLUTION 1

// in pixels
#define NUM_LAYERS (size_t)(HEIGHT / RESOLUTION)
#define X_DIM (size_t)(X_LEN / RESOLUTION)
#define Y_DIM (size_t)(Y_LEN / RESOLUTION)

#define X_MIN (long)(-1 * X_LEN / 2)
#define X_MAX (long)(X_LEN / 2)
#define Y_MIN (long)(-1 * Y_LEN / 2)
#define Y_MAX (long)(Y_LEN / 2)

__global__ void pps(triangle* triangles, size_t num_triangles, bool* out);
// returns the layer of intersection
__device__ int pixelRayIntersection(triangle t, int x, int y);
__device__ int getIntersectionTrunk(int x, int y, triangle* triangles, size_t num_triangles, int* layers);

__global__ void fps1(triangle* triangles, size_t num_triangles, int* all_intersections, size_t* trunk_length, int* locks);
__global__ void fps2(int* all_intersections, size_t* trunk_length);
__global__ void fps3(int* sorted_intersections, size_t* trunk_length, bool* out);

#endif
