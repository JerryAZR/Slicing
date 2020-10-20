#include "slicer.hpp"
#include <algorithm>
#include <iostream>
#include <map>
using std::cout;
using std::endl;

std::multimap<int, triangle> sortTriangle(triangle* triangles, int num_triangles,  std::multimap<int, triangle> bucket) {
	int zmin, zmax;
	for (int i = 0; i < num_triangles; i++) {
		zmin = std::min(std::min(triangles[i].p1.z, triangles[i].p2.z), triangles[i].p3.z);
		zmax = std::max(std::max(triangles[i].p1.z, triangles[i].p2.z), triangles[i].p3.z);
		zmin = std::ceil(zmin / RESOLUTION);
		zmax = std::ceil(zmax / RESOLUTION);
		for (int j = zmin; j < zmax; j++) {
			bucket.insert({ j, triangles[i] });
		}
	}
	return bucket;
}

void outputArray(triangle* triangles, int num_triangles, int* outArray) {
	triangle t;
	int intersectLayer;
	size_t outIdx, preIdx;
	std::multimap<int, triangle>::iterator itr;
	std::multimap<int, triangle> bucket;
	bucket = sortTriangle(triangles, num_triangles, bucket);
	//int intersectArray[X_DIM * Y_DIM * NUM_LAYERS];
	int* intersectArray = (int*)malloc(X_DIM * Y_DIM * NUM_LAYERS * sizeof(int));
	for (int layer = 0; layer < NUM_LAYERS; layer++) {
		for (int y = 0; y < Y_DIM; y++) {
			for (int x = 0; x < X_DIM; x++) {
				outIdx = layer * Y_DIM * X_DIM + y * X_DIM + x;
				
				//calculate whether this pixel intersected in this layer and stored to intersectArray
				for (itr = bucket.find(layer); itr != bucket.find(layer + 1); ++itr) {
					t = itr->second;
					intersectLayer = pixelRayIntersection(t, x, y);
					if (intersectLayer == layer) {
						*(intersectArray + outIdx) = 1;
					}
					else {
						*(intersectArray + outIdx) = 0;
					}
				}
				//output array
				if (layer == 0) {
					*(outArray+outIdx) = *(intersectArray + outIdx);
				}
				else {
					preIdx = (layer-1) * Y_DIM * X_DIM + y * X_DIM + x;
					if (*(intersectArray + outIdx) == 1 && *(intersectArray + preIdx) == 0) {
						*(outArray + outIdx) = 1;
					}
					else if (*(intersectArray + outIdx) == 0) {
						*(outArray + outIdx) = *(outArray + preIdx);
					}
					//check later
					else if (*(intersectArray + outIdx) == 1 && *(intersectArray + preIdx) == 1) {

					}	
				}
			}
		}

	}
	free(intersectArray);
}


int pixelRayIntersection(triangle t, int x, int y) {
	/*
	Let A, B, C be the 3 vertices of the given triangle
	Let S(x,y,z) be the intersection, where x,y are given
	We want to find some a, b such that AS = a*AB + b*AC
	If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.
	*/
	int layer;
	double x_d = x * RESOLUTION - t.p1.x;
	double y_d = y * RESOLUTION - t.p1.y;

	double x1 = t.p2.x - t.p1.x;
	double y1 = t.p2.y - t.p1.y;
	double z1 = t.p2.z - t.p1.z;

	double x2 = t.p3.x - t.p1.x;
	double y2 = t.p3.y - t.p1.y;
	double z2 = t.p3.z - t.p1.z;
	double a = (x_d * y2 - x2 * y_d) / (x1 * y2 - x2 * y1);
	double b = (x_d * y1 - x1 * y_d) / (x2 * y1 - x1 * y2);
	bool inside = (a >= 0) && (b >= 0) && (a + b <= 1);
	double intersection = (a * z1 + b * z2) + t.p1.z;
	// // divide by layer width
	layer = (intersection / RESOLUTION) * inside - (!inside);
	return layer;
}

