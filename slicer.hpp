#ifndef SLICER
#define SLICER

#include "triangle.hpp"
#include <vector>

#define X_MIN -100
#define X_MAX 100
#define Y_MIN -100
#define Y_MAX 100
#define X_DIM (X_MAX - X_MIN + 1)
#define Y_DIM (Y_MAX - Y_MIN + 1)
#define NUM_LAYERS 100
#define RESOLUTION 1

// returns the layer of intersection
int pixelRayIntersection(triangle t, int x, int y);
int getIntersectionTrunk(int x, int y, triangle* triangles, int num_triangles, int* layers);
void pps(triangle* triangles, int num_triangles, int x_dim, int y_dim, int z_dim, bool* out, int id);
#endif
