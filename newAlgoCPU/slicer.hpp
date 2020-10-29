#ifndef SLICER
#define SLICER

#include "triangle.hpp"
#include <vector>
#include <map>

//in mm
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

// returns the layer of intersection
int pixelRayIntersection(triangle t, int x, int y);
void outputArray(triangle* triangles, int num_triangles, bool* outArray);
std::multimap<int, triangle> sortTriangle(triangle* triangles, int num_triangles, std::multimap<int, triangle> bucket);
#endif

