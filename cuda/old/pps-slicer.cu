#include "slicer.cuh"
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <stdio.h>

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
    __shared__ layer_t layers_shared[THREADS_PER_BLOCK][MAX_TRUNK_SIZE+1];
    layer_t* layers = &layers_shared[threadIdx.x][0];
    for (size_t i = 0; i < num_iters; i++) {
        triangles[threadIdx.x] = triangles_global[threadIdx.x + (i * THREADS_PER_BLOCK)];
        // Wait for other threads to complete;
        __syncthreads();
        if (y_idx < Y_DIM) {
            for (size_t tri_idx = 0; tri_idx < THREADS_PER_BLOCK; tri_idx++) {
                layer_t intersection = pixelRayIntersection(triangles[tri_idx], x, y);
                if (intersection != NONE) {
                    layers[length] = intersection;
                    length++;
                }
            }
        }
        __syncthreads();
    }
    size_t remaining = num_triangles - (num_iters * THREADS_PER_BLOCK);
    if (threadIdx.x < remaining) {
        triangles[threadIdx.x] = triangles_global[threadIdx.x + (num_iters * THREADS_PER_BLOCK)];
    }
    __syncthreads();
    if (remaining && y_idx < Y_DIM) {
        for (size_t tri_idx = 0; tri_idx < remaining; tri_idx++) {
            layer_t intersection = pixelRayIntersection(triangles[tri_idx], x, y);
            if (intersection != NONE) {
                layers[length] = intersection;
                length++;
            }
        }
    }

    if (y_idx >= Y_DIM) return;

    thrust::sort(thrust::device, &layers[0], &layers[length]);
    layers[length] = NUM_LAYERS;
    if (length > MAX_TRUNK_SIZE) 
        printf("Error: Too many intersections.\n \
                Please increase MAX_TRUNK_SIZE in slicer.cuh and recompile.\n");

    bool flag = false;
    int layerIdx = 0;
    for (layer_t z = 0; z < NUM_LAYERS; z++) {
        // If intersect
        while (layers[layerIdx] < z) layerIdx++;
        bool intersect = (z == layers[layerIdx]);
        flag = (bool) (layerIdx & 1);
        out[z*Y_DIM*X_DIM + y_idx*X_DIM + x_idx] = intersect || flag;
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

    double x_max = max(t.p1.x, max(t.p2.x, t.p3.x));
    double x_min = min(t.p1.x, min(t.p2.x, t.p3.x));
    double y_max = max(t.p1.y, max(t.p2.y, t.p3.y));
    double y_min = min(t.p1.y, min(t.p2.y, t.p3.y));

    double x_pos = x * RESOLUTION;
    double y_pos = y * RESOLUTION;
    if ((x_pos < x_min) || (x_pos > x_max) || (y_pos < y_min) || (y_pos > y_max)) return NONE;

    double x_d = x_pos - t.p1.x;
    double y_d = y_pos - t.p1.y;

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
    layer_t layer = inside ? (intersection / RESOLUTION) : NONE;
    return layer;
}
