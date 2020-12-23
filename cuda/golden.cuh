#ifndef GOLDEN
#define GOLDEN

#include "triangle.cuh"
#include "slicer.cuh"

// returns the layer of intersection
__host__ long checkOutput(triangle* triangles_dev, size_t num_triangles, bool* in);
__host__ void goldenModel(triangle* triangles_dev, size_t num_triangles, bool* out);

#endif
