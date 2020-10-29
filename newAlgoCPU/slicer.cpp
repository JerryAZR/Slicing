
#include "slicer.hpp"
#include <algorithm>
#include <iostream>
#include <map>
using std::cout;
using std::endl;
#include <math.h>

std::multimap<int, triangle> sortTriangle(triangle* triangles, int num_triangles, std::multimap<int, triangle> bucket) {
	double zmin, zmax;
	int min, max;
	for (int i = 0; i < num_triangles; i++) {
		zmin = std::min(std::min(triangles[i].p1.z, triangles[i].p2.z), triangles[i].p3.z);
		zmax = std::max(std::max(triangles[i].p1.z, triangles[i].p2.z), triangles[i].p3.z);
		min = std::ceil(zmin);
		max = std::ceil(zmax)+1;
		//std::cout << min << max << std::endl;
		//std::cout << std::endl;
		for (int j = min; j < max; j++) {
			bucket.insert({ j, triangles[i] });
		}
	}
	return bucket;
}

void outputArray(triangle* triangles, int num_triangles, bool* outArray) {
	triangle t;
	int intersectLayer;
	bool intersect;
	int x_idx, y_idx;
	int outIdx, preIdx, flagIdx;
//	std::multimap<int, triangle>::iterator itr;
//	std::multimap<int, triangle> bucket;
//	bucket = sortTriangle(triangles, num_triangles, bucket);
	//bool intersectArray[X_DIM * Y_DIM * NUM_LAYERS];
	bool* flagArray = (bool*)malloc(X_DIM * Y_DIM * sizeof(bool));
	for (int layer = 0; layer < 2; layer++) {
		std::cout << layer << std::endl;
		//std::cout << "#intersect triangles:" << bucket.count(layer) << std::endl;
		for (int y = 0; y < Y_DIM; y++) {
			for (int x = 0; x < X_DIM; x++) {
				outIdx = layer * Y_DIM * X_DIM + y * X_DIM + x;
				flagIdx = y * X_DIM + x;
				x_idx = x - (X_DIM / 2);
				y_idx = y - (Y_DIM / 2);
				//calculate whether this pixel intersected in this layer and stored to intersectArray
//				for (itr = bucket.find(layer); itr != bucket.find(layer + 1); ++itr) {
//					t = itr->second;
				for (int i = 0; i < num_triangles; i++){					
					t = triangles[i];
					intersectLayer = pixelRayIntersection(t, x_idx, y_idx);
					
					if (intersectLayer == layer) {
						//(intersectArray + outIdx) = 1;
						intersect = true;
						//std::cout << "Intersect" << std::endl;
						break;
					}
					else {
						//(intersectArray + outIdx) = 0;
						intersect = false;
					}
				}
				//output array
				if (layer == 0) {
					//(outArray + outIdx) = *(intersectArray + outIdx);
					outArray[outIdx] = intersect || false;
					flagArray[flagIdx] = intersect ^ false;
				}
				else {
					outArray[outIdx] = intersect || flagArray[flagIdx];
					flagArray[flagIdx] = intersect ^ flagArray[flagIdx];
					/*
					preIdx = (layer - 1) * Y_DIM * X_DIM + y * X_DIM + x;
					if (*(intersectArray + outIdx) == true && *(intersectArray + preIdx) == false) {
						*(outArray + outIdx) = true;
					}
					else if (*(intersectArray + outIdx) == false) {
						*(outArray + outIdx) = *(outArray + preIdx);
					}
					//check later
					else if (*(intersectArray + outIdx) == true && *(intersectArray + preIdx) == true) {
						*(outArray + outIdx) = true;*/
					}
				}
				
			}
		}

	
	//free(flagArray);
}


int pixelRayIntersection(triangle t, int x, int y) {
	/*
	Let A, B, C be the 3 vertices of the given triangle
	Let S(x,y,z) be the intersection, where x,y are given
	We want to find some a, b such that AS = a*AB + b*AC
	If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.
	*/
	int layer;
	/*
	if ((x < t.p1.x && x < t.p2.x && x < t.p3.x)
		|| (x > t.p1.x && x > t.p2.x && x > t.p3.x)
		|| (y < t.p1.y && y < t.p2.y && y < t.p3.y)
		|| (y > t.p1.y && y > t.p2.y && y > t.p3.y)
		) return -1; */

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
	//printf("%d \n", layer);
	//std::cout << layer << std::endl;
	return layer;
}
