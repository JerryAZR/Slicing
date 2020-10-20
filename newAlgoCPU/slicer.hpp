#ifndef SLICER
#define SLICER

#include "triangle.hpp"
#include <vector>
#include <map>
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
void outputArray(triangle* triangles, int num_triangles, int* outArray);
std::multimap<int, triangle> sortTriangle(triangle* triangles, int num_triangles, std::multimap<int, triangle> bucket);
#endif
