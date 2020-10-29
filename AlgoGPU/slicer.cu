#include "slicer.cuh"
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <stdio.h>
#include <map>

//calculate output array of each layer
__global__
void outputArray(triangle* triangles_global, size_t num_triangles, bool* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int x_idx = idx % X_DIM;
    int y_idx = idx / X_DIM;

    int x = x_idx - (X_DIM / 2);
    int y = y_idx - (Y_DIM / 2);

    int outIdx, flagIdx;
    bool flagArray[X_DIM * Y_DIM];
    __shared__ triangle tri_base[THREADS_PER_BLOCK];
    triangle* triangles = (triangle*)tri_base;
    size_t num_iters = num_triangles / THREADS_PER_BLOCK;
    /*
    for (int layer = 0; layer < 2; layer++) {
        outIdx = layer * X_DIM * Y_DIM + y_idx * X_DIM + x_idx;
        flagIdx = y_idx * X_DIM + x_idx;
        bool intersect = false;

        for (int i = 0; i < num_triangles; i++) {
            int intersectLayer = pixelRayIntersection(triangles_global[i], x, y);
            if (intersectLayer == layer) {
                intersect = true;
                break;
            }
            else {
                intersect = false;
            }
        }
        
        if (layer == 0) {
            out[outIdx] = intersect || false;
            flagArray[flagIdx] = intersect ^ false;
        }
        else {
            out[outIdx] = intersect || flagArray[flagIdx];
            flagArray[flagIdx] = intersect ^ flagArray[flagIdx];
        }
    }
    */
    ///*
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        outIdx = layer * X_DIM * Y_DIM + y_idx * X_DIM + x_idx;
        flagIdx = y_idx * X_DIM + x_idx;
        for (size_t i = 0; i < num_iters; i++) {
            triangles[threadIdx.x] = triangles_global[threadIdx.x + (i * THREADS_PER_BLOCK)];
            // Wait for other threads to complete;
            __syncthreads();
            if (y_idx < Y_DIM) {
                //for (int layer = 0; layer < 2; layer++) {
                //    outIdx = layer * X_DIM * Y_DIM + y_idx * X_DIM + x_idx;
                //    flagIdx = y_idx * X_DIM + x_idx;
                    getOutarray(x, y, triangles, THREADS_PER_BLOCK, layer, outIdx, flagIdx, out, flagArray);
                //}
            }
        }
        size_t remaining = num_triangles - (num_iters * THREADS_PER_BLOCK);
        if (threadIdx.x < remaining) {
            triangles[threadIdx.x] = triangles_global[threadIdx.x + (num_iters * THREADS_PER_BLOCK)];
        }
        if (remaining) {
            __syncthreads();
            if (y_idx < Y_DIM) {
                //for (int layer = 0; layer < 2; layer++) {
                //    outIdx = layer * X_DIM * Y_DIM + y_idx * X_DIM + x_idx;
                //    flagIdx = y_idx * X_DIM + x_idx;
                getOutarray(x, y, triangles, remaining, layer, outIdx, flagIdx, out, flagArray);
                //}
            }
        }
    }
    //*/
    
}

__device__ __forceinline__
int pixelRayIntersection(triangle t, int x, int y) {
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
bool getIntersect(int x, int y, triangle* triangles, size_t num_triangles, size_t layer) {
    bool intersect;
    for (int i = 0; i < num_triangles; i++) {
        int intersectLayer = pixelRayIntersection(triangles[i], x, y);
        if (intersectLayer == layer) {
            intersect = true;
            return intersect;
        }
        else {
            intersect = false;
        }
    }
    return intersect;
}



__device__
void getOutarray(int x, int y, triangle* triangles, size_t num_triangles, size_t layer, size_t outIdx, size_t flagIdx, bool* out, bool* flagArray) {
    bool intersect;
    intersect = getIntersect(x, y, triangles, num_triangles, layer);

    if (layer == 0) {
        out[outIdx] = intersect || false;
        flagArray[flagIdx] = intersect ^ false;
    }
    else {
        out[outIdx] = intersect || flagArray[flagIdx];
        flagArray[flagIdx] = intersect ^ flagArray[flagIdx];
    }
}
