#include "slicer.cuh"
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <stdio.h>
#include <map>

std::multimap<int, triangle> sortTriangle(triangle* triangles, int num_triangles, std::multimap<int, triangle> bucket) {
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

//calculate output array of each layer
__global__
void outputArray(triangle* d_intersectTriangles, int* d_tMun, int* d_outArray, int* d_intersectArray, int* d_intersectArrayPre) {
    triangle* triangleCurrent = d_intersectTriangles;
    int* out = d_outArray;
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int x_idx = idx % X_DIM;
        int y_idx = idx / X_DIM;

        int x = x_idx - (X_DIM / 2);
        int y = y_idx - (Y_DIM / 2);

        triangle* triangles_global = triangleCurrent;
        // Copy triangles to shared memory
        // Each block has a shared memory storing some triangles.
        int tNum = d_tMun[layer];
        __shared__ triangle tri_base[256];
        triangle* triangles = (triangle*)tri_base;
        int num_iters = tNum / 256;

        int length = 0;
        int layers_local[NUM_LAYERS + 1];
        int* layers = &layers_local[0];
        for (int i = 0; i < num_iters; i++) {
            triangles[threadIdx.x] = triangleCurrent[threadIdx.x + (i * 256)];  //copy triangle to shared memory
            // Wait for other threads to complete;
            __syncthreads();
            if (y_idx < Y_DIM)
                getOutputArray(x, y, triangles, 256, layer, d_intersectArray, d_intersectArrayPre, out, x_idx, y_idx);
        }
        int remaining = tNum - (num_iters * 256);
        if (threadIdx.x < remaining) {
            triangles[threadIdx.x] = triangleCurrent[threadIdx.x + (num_iters * 256)];
            __syncthreads();
        }
        if (remaining) {
            if (y_idx < Y_DIM)
                getOutputArray(x, y, triangles, 256, layer, d_intersectArray, d_intersectArrayPre, out, x_idx, y_idx);
        }
        triangleCurrent += tNum;
        out += X_DIM * Y_DIM;
    }

    return;
}


__device__
int pixelRayIntersection(triangle t, int x, int y) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.
    return the layer of intersection, or -1 if none
    */

    if ((x < t.p1.x && x < t.p2.x && x < t.p3.x)
        || (x > t.p1.x && x > t.p2.x && x > t.p3.x)
        || (y < t.p1.y && y < t.p2.y && y < t.p3.y)
        || (y > t.p1.y && y > t.p2.y && y > t.p3.y)
        ) return -1;

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
    int layer = (intersection / RESOLUTION) * inside - (!inside);
    return layer;
}

__device__
void getIntersectionArray(int x, int y, triangle* triangles, int num_triangles, int layer, int* d_intersectArray, int x_idx, int y_idx) {

    for (int i = 0; i < num_triangles; i++) {
        int intersectLayer = pixelRayIntersection(triangles[i], x, y);
        if (intersectLayer == layer) {
            *(d_intersectArray + y_idx * X_DIM + x_idx) = 1;
            return;
        }
    }
    *(d_intersectArray + y_idx * X_DIM + x_idx) = 0;
    return;
}

__device__
void getOutputArray(int x, int y, triangle* triangles, int num_triangles, int layer, int* d_intersectArray, int* d_intersectArrayPre, int* d_outArray, int x_idx, int y_idx) {
    getIntersectionArray(x, y, triangles, num_triangles, layer, d_intersectArray, x_idx, y_idx);
    if (layer == 0) {
        d_outArray[layer * X_DIM * Y_DIM + y_idx * X_DIM + x_idx] = d_intersectArray[y_idx * X_DIM + x_idx];
    }
    else {
        if (d_intersectArray[y_idx * X_DIM + x_idx] == 1 && d_intersectArrayPre[y_idx * X_DIM + x_idx] == 0) {
            d_outArray[layer * X_DIM * Y_DIM + y_idx * X_DIM + x_idx] == 1;
        } 
        else if (d_intersectArray[y_idx * X_DIM + x_idx] == 0) {
            d_outArray[layer * X_DIM * Y_DIM + y_idx * X_DIM + x_idx] = d_outArray[(layer - 1) * X_DIM * Y_DIM + y_idx * X_DIM + x_idx];
        }
        else if (d_intersectArray[y_idx * X_DIM + x_idx] == 1 && d_intersectArrayPre[y_idx * X_DIM + x_idx] == 1) {
            d_outArray[layer * X_DIM * Y_DIM + y_idx * X_DIM + x_idx] == 1;
        }
    }
    d_intersectArrayPre = d_intersectArray;

    return;
}