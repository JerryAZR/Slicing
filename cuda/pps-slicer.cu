#include "slicer.cuh"
#include <thrust/sort.h>
#include <thrust/functional.h>

__global__
void pps(triangle* triangles_global, size_t num_triangles, bool* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("starting thread %d\n", idx);
    int y_idx = idx / X_DIM;
    // if (y >= Y_DIM) return;
    int x_idx = idx % X_DIM;
    int x = x_idx - (X_DIM / 2);
    int y = y_idx - (Y_DIM / 2);

    // Copy triangles to shared memory
    // Each block has a shared memory storing some triangles.
    __shared__ triangle tri_base[THREADS_PER_BLOCK];
    triangle* triangles = (triangle*) tri_base;
    size_t num_iters = num_triangles / THREADS_PER_BLOCK;
    int length = 0;
    __shared__ layer_t layers_shared[THREADS_PER_BLOCK][MAX_TRUNK_SIZE];
    layer_t* layers = &layers_shared[threadIdx.x][0];
    for (size_t i = 0; i < num_iters; i++) {
        triangles[threadIdx.x] = triangles_global[threadIdx.x + (i * THREADS_PER_BLOCK)];
        // Wait for other threads to complete;
        __syncthreads();
        if (y_idx < Y_DIM) {
            length += getIntersectionTrunk(x, y, triangles, THREADS_PER_BLOCK, layers);
            layers = &layers_shared[threadIdx.x][length]; // update pointer value
        }
        __syncthreads();
    }
    size_t remaining = num_triangles - (num_iters * THREADS_PER_BLOCK);
    if (threadIdx.x < remaining) {
        triangles[threadIdx.x] = triangles_global[threadIdx.x + (num_iters * THREADS_PER_BLOCK)];
    }
    __syncthreads();
    if (remaining) {
        if (y_idx < Y_DIM) {
            length += getIntersectionTrunk(x, y, triangles, remaining, layers);
            layers = &layers_shared[threadIdx.x][length]; // update pointer value
        }
    }

    if (y_idx >= Y_DIM) return;
    layers = &layers_shared[threadIdx.x][0]; // reset to beginning

    thrust::sort(thrust::device, &layers[0], &layers[length]);
    layers[length] = NUM_LAYERS;

    bool flag = false;
    int layerIdx = 0;
    for (layer_t z = 0; z < NUM_LAYERS; /*z++*/) {
        // If intersect
        bool intersect = (z == layers[layerIdx]);
        out[z*Y_DIM*X_DIM + y_idx*X_DIM + x_idx] = intersect || flag;
        flag = intersect ^ flag;
        if (intersect) layerIdx ++;
        else z++;
    }
}

/**
 * pixelRayIntersection: helper function, computes the intersection of given triangle and pixel ray
 * Inputs:
 *      t -- input triangle
 *      x, y -- coordinates of the input pixel ray
 * Returns:
 *      The layer on which they intersect, or -1 if no intersection
 */
__device__ __forceinline__
layer_t pixelRayIntersection(triangle t, int x, int y) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.
    */

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
    bool inside = (a >= 0) && (b >= 0) && (a+b <= 1);
    double intersection = (a * z1 + b * z2) + t.p1.z;
    // // divide by layer width
    layer_t layer = inside ? (intersection / RESOLUTION) : -1;
    return layer;
}

/**
 * get the array of intersections of a given pixel ray
 */
 __device__
 int getIntersectionTrunk(int x, int y, triangle* triangles, size_t num_triangles, layer_t* layers) {
     int idx = 0;
 
     for (int i = 0; i < num_triangles; i++) {
         layer_t layer = pixelRayIntersection(triangles[i], x, y);
         if (layer != -1) {
             layers[idx] = layer;
             idx++;
         }
     }
     return idx;
 }

