#ifndef GOLDEN
#define GOLDEN

#include "triangle.cuh"
#include "slicer.cuh"

// returns the layer of intersection
__host__ long checkOutput(triangle* triangles_dev, size_t num_triangles, bool* in);
__host__ void goldenModel(triangle* triangles_dev, size_t num_triangles, bool* out);
__device__ layer_t _pixelRayIntersection(triangle t, int x, int y);
__device__ bool _isInside(layer_t current, layer_t* trunk, size_t length);

__global__ void _fps1(triangle* triangles, size_t num_triangles, layer_t* all_intersections, size_t* trunk_length, int* locks);
__global__ void _fps2(layer_t* all_intersections, size_t* trunk_length);
__global__ void _fps3(layer_t* sorted_intersections, size_t* trunk_length, bool* out);

#endif
