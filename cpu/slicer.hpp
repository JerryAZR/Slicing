#ifndef SLICER
#define SLICER

#include "triangle.hpp"
#include <vector>

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


// returns the layer of intersection
int pixelRayIntersection(triangle t, int x, int y);
int getIntersectionTrunk(int x, int y, triangle* triangles, int num_triangles, int* layers);
bool isInside(int current, int* trunk, size_t length);

void pps(triangle* triangles_global, int num_triangles, bool* out, unsigned id);

void fps1(triangle* triangles, size_t num_triangles, int* all_intersections, size_t* trunk_length, int* locks, long id);
void fps2(int* all_intersections, size_t* trunk_length, long id);
void fps3(int* sorted_intersections, size_t* trunk_length, bool* out, long id);
#endif
