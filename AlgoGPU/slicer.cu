#include "slicer.cuh"
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <stdio.h>
#include <map>
#include <math.h>
#include <stdio.h>

__global__ 
void triangle_sort(triangle* triangles_global, size_t num_triangles, double* zmins_global, int* index_global) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_triangles) return;
        zmins_global[idx] = fmin(fmin(triangles_global[idx].p1.z, triangles_global[idx].p2.z), triangles_global[idx].p3.z);
        index_global[idx] = &(triangles_global[idx]) - triangles_global;

    //thrust::sort_by_key(thrust::device, zmins_global, zmins_global + num_triangles, index_global);
}

//calculate output array of each layer
__global__
void outputArray(triangle* triangles_global, size_t num_triangles, bool* out, int* index_global) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int x_idx = idx % X_DIM;
    int y_idx = idx / X_DIM;

    int x = x_idx - (X_DIM / 2);
    int y = y_idx - (Y_DIM / 2);

    int outIdx, flagIdx;
    bool flagArray[X_DIM * Y_DIM];

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        outIdx = layer * X_DIM * Y_DIM + y_idx * X_DIM + x_idx;
        flagIdx = y_idx * X_DIM + x_idx;
        getOutarray(x, y, triangles_global, num_triangles, layer, outIdx, flagIdx, out, flagArray, index_global);
    }

    /*
    __shared__ triangle tri_base[THREADS_PER_BLOCK];
    triangle* triangles = (triangle*)tri_base;
    size_t num_iters = num_triangles / (THREADS_PER_BLOCK);

    __shared__ int index_base[THREADS_PER_BLOCK];
    int* index = (int*)index_base;

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        outIdx = layer * X_DIM * Y_DIM + y_idx * X_DIM + x_idx;
        flagIdx = y_idx * X_DIM + x_idx;

        //getOutarray(x, y, triangles_global, THREADS_PER_BLOCK, layer, outIdx, flagIdx, out, flagArray, index_global);
        
        for (size_t i = 0; i < num_iters; i++) {
            index[threadIdx.x] = index_global[threadIdx.x + (i * THREADS_PER_BLOCK)];
            triangles[threadIdx.x] = triangles_global[index[threadIdx.x]];

            // Wait for other threads to complete;
            __syncthreads();
            if (y_idx < Y_DIM) {
                    getOutarray(x, y, triangles, THREADS_PER_BLOCK, layer, outIdx, flagIdx, out, flagArray, index);
            }
        }
        size_t remaining = num_triangles - (num_iters * THREADS_PER_BLOCK);
        if (threadIdx.x < remaining) {
            //triangles[threadIdx.x] = triangles_global[threadIdx.x + (num_iters * THREADS_PER_BLOCK)];
            index[threadIdx.x] = index_global[threadIdx.x + (num_iters * THREADS_PER_BLOCK)];
            triangles[threadIdx.x] = triangles_global[index[threadIdx.x]];
        }
        if (remaining) {
            __syncthreads();
            if (y_idx < Y_DIM) {
                getOutarray(x, y, triangles, remaining, layer, outIdx, flagIdx, out, flagArray, index);
            }
        }
        
    }*/
    
    
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
bool getIntersect(int x, int y, triangle* triangles, size_t num_triangles, size_t layer, int* index) {
    bool intersect = false;
    double zmin;
    int idx;
    for (int i = 0; i < num_triangles; i++) {
        idx = index[i];
        //if (layer == 0) printf("%d\n", idx);
        //idx = i;
        zmin = fmin(fmin(triangles[idx].p1.z, triangles[idx].p2.z), triangles[idx].p3.z);
        if (zmin > layer) {
            return intersect;
        }
        else {
            int intersectLayer = pixelRayIntersection(triangles[idx], x, y);
            if (intersectLayer == layer) {
                intersect = true;
                return intersect;
            }
            else {
                intersect = false;
            }
        }
        
    }
    return intersect;
}



__device__
void getOutarray(int x, int y, triangle* triangles, size_t num_triangles, size_t layer, size_t outIdx, size_t flagIdx, bool* out, bool* flagArray, int* index) {
    bool intersect;
    intersect = getIntersect(x, y, triangles, num_triangles, layer, index);

    if (layer == 0) {
        out[outIdx] = intersect || false;
        flagArray[flagIdx] = intersect ^ false;
    }
    else {
        out[outIdx] = intersect || flagArray[flagIdx];
        flagArray[flagIdx] = intersect ^ flagArray[flagIdx];
    }
}
