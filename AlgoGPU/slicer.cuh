#ifndef SLICER
#define SLICER

#include "triangle.cuh"
#include <map>

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


std::multimap<int, triangle> sortTriangle(triangle* triangles, int num_triangles, std::multimap<int, triangle> bucket);
__global__ void outputArray(triangle* d_intersectTriangles, int* d_tMun, int* d_outArray, int* d_intersectArray, int* d_intersectArrayPre);
__device__ int pixelRayIntersection(triangle t, int x, int y);
__device__ void getIntersectionArray(int x, int y, triangle* triangles, int num_triangles, int layer, triangle* d_intersectArray, int x_idx, int y_idx);
__device__ void getOutputArray(int x, int y, triangle* triangles, int num_triangles, int layer, int* d_intersectArray, int* d_intersectArrayPre, int* d_outArray, int x_idx, int y_idx);
#endif
