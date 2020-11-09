#ifndef SLICER
#define SLICER

#include "triangle.cuh"
#include "slicer.cuh"

// returns the layer of intersection
__device__ char _pixelRayIntersection(triangle t, int x, int y);
__device__ bool _isInside(char current, char* trunk, size_t length);

__global__ void _fps1(triangle* triangles, size_t num_triangles, char* all_intersections, size_t* trunk_length, int* locks);
__global__ void _fps2(char* all_intersections, size_t* trunk_length);
__global__ void _fps3(char* sorted_intersections, size_t* trunk_length, bool* out);

#endif
