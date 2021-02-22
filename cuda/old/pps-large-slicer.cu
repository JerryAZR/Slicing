#include "slicer.cuh"
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <stdio.h>

#define YNONE INT_MIN

__device__ __forceinline__
int pixelRayIntersectionY(triangle t, int x, int z);

__global__
void pps(triangle* triangles_global, size_t num_triangles, bool* out, unsigned base_layer) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("starting thread %d\n", idx);
    int z_idx = idx / X_DIM;
    // if (y >= Y_DIM) return;
    int x_idx = idx % X_DIM;
    int x = x_idx - (X_DIM / 2);
    int z = z_idx + base_layer;

    // Copy triangles to shared memory
    // Each block has a shared memory storing some triangles.
    __shared__ triangle tri_base[THREADS_PER_BLOCK];
    triangle* triangles = (triangle*) tri_base;
    size_t num_iters = num_triangles / THREADS_PER_BLOCK;
    int length = 0;
    int yints[MAX_TRUNK_SIZE+1];
    for (size_t i = 0; i < num_iters; i++) {
        triangles[threadIdx.x] = triangles_global[threadIdx.x + (i * THREADS_PER_BLOCK)];
        // Wait for other threads to complete;
        __syncthreads();
        if (z < NUM_LAYERS) {
            for (size_t tri_idx = 0; tri_idx < THREADS_PER_BLOCK; tri_idx++) {
                int intersection = pixelRayIntersectionY(triangles[tri_idx], x, z);
                if (intersection != YNONE) {
                    yints[length] = intersection;
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
    if (remaining && z < NUM_LAYERS) {
        for (size_t tri_idx = 0; tri_idx < remaining; tri_idx++) {
            int intersection = pixelRayIntersectionY(triangles[tri_idx], x, z);
            if (intersection != YNONE) {
                yints[length] = intersection;
                length++;
            }
        }
    }

    if (z >= NUM_LAYERS) return;

    thrust::sort(thrust::device, &yints[0], &yints[length]);
    yints[length] = Y_MAX;
    if (length > MAX_TRUNK_SIZE) 
        printf("Error: Too many intersections.\n \
                Please increase MAX_TRUNK_SIZE in slicer.cuh and recompile.\n");

    bool flag = false;
    int layerIdx = 0;
    for (int y = Y_MIN; y < Y_MAX; y++) {
        // If intersect
        while (yints[layerIdx] < y) layerIdx++;
        bool intersect = (y == yints[layerIdx]);
        flag = (bool) (layerIdx & 1);
        unsigned y_idx = y - Y_MIN;
        out[z_idx*Y_DIM*X_DIM + y_idx*X_DIM + x_idx] = intersect || flag;
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
int pixelRayIntersectionY(triangle t, int x, int z) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.
    */

    double x_max = max(t.p1.x, max(t.p2.x, t.p3.x));
    double x_min = min(t.p1.x, min(t.p2.x, t.p3.x));
    double z_max = max(t.p1.z, max(t.p2.z, t.p3.z));
    double z_min = min(t.p1.z, min(t.p2.z, t.p3.z));

    double x_pos = x * RESOLUTION;
    double z_pos = z * RESOLUTION;
    if ((x_pos < x_min) || (x_pos > x_max) || (z_pos < z_min) || (z_pos > z_max)) return YNONE;

    double x_d = x_pos - t.p1.x;
    double z_d = z_pos - t.p1.z;

    double x1 = t.p2.x - t.p1.x;
    double y1 = t.p2.y - t.p1.y;
    double z1 = t.p2.z - t.p1.z;

    double x2 = t.p3.x - t.p1.x;
    double y2 = t.p3.y - t.p1.y;
    double z2 = t.p3.z - t.p1.z;

    double a = (x_d * z2 - x2 * z_d) / (x1 * z2 - x2 * z1);
    double b = (x_d * z1 - x1 * z_d) / (x2 * z1 - x1 * z2);
    bool inside = (a >= 0) && (b >= 0) && (a+b <= 1);
    double intersection = (a * y1 + b * y2) + t.p1.y;
    // // divide by layer width
    return inside ? (intersection / RESOLUTION) : YNONE;
}

__global__ 
void triangleSelect(triangle* in, triangle* out, unsigned in_length,
                    unsigned* out_length, unsigned base_layer)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t total_threads = blockDim.x * gridDim.x;
    double min_height = base_layer * RESOLUTION;
    double max_height = (base_layer + BLOCK_HEIGHT) * RESOLUTION;
    while (idx < in_length) {
        triangle t = in[idx];
        idx += total_threads;
        double z_min = min(t.p1.z, min(t.p2.z, t.p3.z));
        if (z_min > max_height) continue;
        double z_max = max(t.p1.z, max(t.p2.z, t.p3.z));
        if (z_max < min_height) continue;
        size_t curr_length = atomicAdd(out_length, 1);
        out[curr_length] = t;
    }
}
    
