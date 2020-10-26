#ifndef SLICER
#define SLICER

#include "triangle.cuh"

#define THREADS_PER_BLOCK 256
#define MAX_TRUNK_SIZE	48

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

__global__ void pps(triangle* triangles, size_t num_triangles, bool* out);
// returns the layer of intersection
__device__ char pixelRayIntersection(triangle t, int x, int y);
__device__ int getIntersectionTrunk(int x, int y, triangle* triangles, size_t num_triangles, char* layers);
__device__ bool isInside(char current, char* trunk, size_t length);

__global__ void fps1(triangle* triangles, size_t num_triangles, char* all_intersections, size_t* trunk_length, int* locks);
__global__ void fps2(char* all_intersections, size_t* trunk_length);
__global__ void fps3(char* sorted_intersections, size_t* trunk_length, bool* out);

#endif
