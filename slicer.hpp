#ifndef SLICER
#define SLICER

#include "triangle.hpp"
#include <vector>

#define X_MIN -100
#define X_MAX 100
#define Y_MIN -100
#define Y_MAX 100
#define NUM_LAYERS 100
#define RESOLUTION 1

// returns the layer of intersection
int pixelRayIntersection(triangle & t, int x, int y);
void getIntersectionTrunk(int x, int y, vector<triangle>& triangles, vector<int>& layers);
void pps(int x, int y, vector<triangle>& triangles, vector<int>& layers);

#endif
