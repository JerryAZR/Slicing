#include "slicer.cuh"
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <stdio.h>

#define XNONE INT_MIN

__device__ __forceinline__
int pixelRayIntersectionX(triangle t, int y, int z);

__global__
void pps(triangle* triangles_global, size_t num_triangles, bool* out, unsigned base_layer) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("starting thread %d\n", idx);
    int z_idx = idx / Y_DIM;
    // if (y >= Y_DIM) return;
    int y_idx = idx % Y_DIM;
    int y = y_idx - (Y_DIM / 2);
    int z = z_idx + base_layer;

    // Copy triangles to shared memory
    // Each block has a shared memory storing some triangles.
    __shared__ triangle tri_base[THREADS_PER_BLOCK];
    triangle* triangles = (triangle*) tri_base;
    size_t num_iters = num_triangles / THREADS_PER_BLOCK;
    int length = 0;
    int xints[MAX_TRUNK_SIZE+1];
    for (size_t i = 0; i < num_iters; i++) {
        triangles[threadIdx.x] = triangles_global[threadIdx.x + (i * THREADS_PER_BLOCK)];
        // Wait for other threads to complete;
        __syncthreads();
        if (z < NUM_LAYERS) {
            for (size_t tri_idx = 0; tri_idx < THREADS_PER_BLOCK; tri_idx++) {
                int intersection = pixelRayIntersectionX(triangles[tri_idx], y, z);
                if (intersection != XNONE) {
                    xints[length] = intersection;
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
            int intersection = pixelRayIntersectionX(triangles[tri_idx], y, z);
            if (intersection != XNONE) {
                xints[length] = intersection;
                length++;
            }
        }
    }

    if (z >= NUM_LAYERS) return;

    thrust::sort(thrust::device, &xints[0], &xints[length]);
    xints[length] = X_MAX;
    if (length > MAX_TRUNK_SIZE) 
        printf("Error: Too many intersections.\n \
                Please increase MAX_TRUNK_SIZE in slicer.cuh and recompile.\n");

    bool flag = false;
    int layerIdx = 0;
    for (int x = X_MIN; x < X_MAX; x++) {
        // If intersect
        while (xints[layerIdx] < x) layerIdx++;
        bool intersect = (x == xints[layerIdx]);
        flag = (bool) (layerIdx & 1);
        unsigned x_idx = x - X_MIN;
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
int pixelRayIntersectionX(triangle t, int y, int z) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.
    */

    double y_max = max(t.p1.y, max(t.p2.y, t.p3.y));
    double y_min = min(t.p1.y, min(t.p2.y, t.p3.y));
    double z_max = max(t.p1.z, max(t.p2.z, t.p3.z));
    double z_min = min(t.p1.z, min(t.p2.z, t.p3.z));

    double y_pos = y * RESOLUTION;
    double z_pos = z * RESOLUTION;
    if ((y_pos < y_min) || (y_pos > y_max) || (z_pos < z_min) || (z_pos > z_max)) return XNONE;

    double y_d = y_pos - t.p1.y;
    double z_d = z_pos - t.p1.z;

    double x1 = t.p2.x - t.p1.x;
    double y1 = t.p2.y - t.p1.y;
    double z1 = t.p2.z - t.p1.z;

    double x2 = t.p3.x - t.p1.x;
    double y2 = t.p3.y - t.p1.y;
    double z2 = t.p3.z - t.p1.z;

    double a = (y_d * z2 - y2 * z_d) / (y1 * z2 - y2 * z1);
    double b = (y_d * z1 - y1 * z_d) / (y2 * z1 - y1 * z2);
    bool inside = (a >= 0) && (b >= 0) && (a+b <= 1);
    double intersection = (a * x1 + b * x2) + t.p1.x;
    // // divide by layer width
    return inside ? (intersection / RESOLUTION) : XNONE;
}

__global__ 
void triangleSelect(triangle* in, triangle* out, unsigned in_length,
                    unsigned* out_length, unsigned base_layer)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t total_threads = blockDim.x * gridDim.x;
    double min_height = base_layer * RESOLUTION;
    size_t max_layers = (base_layer + PPS_BLOCK_HEIGHT);
    double max_height = max_layers * RESOLUTION;
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
    
