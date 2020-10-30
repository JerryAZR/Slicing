#include "slicer.cuh"
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <stdio.h>

__global__
void pps(triangle* triangles_global, size_t num_triangles, bool* out) {
    unsigned layers_per_thread = NUM_LAYERS / THREADS_PER_BLOCK;
    unsigned remaining_layers = NUM_LAYERS % THREADS_PER_BLOCK;
    unsigned prev_layers = threadIdx.x * layers_per_thread 
        + ((threadIdx.x < remaining_layers) ? threadIdx.x : remaining_layers);
    unsigned total_layers = layers_per_thread + (threadIdx.x < remaining_layers);

    __shared__ char layers_shared[NUM_LAYERS];
    char* layers_init = &layers_shared[prev_layers];
    for (int i = 0; i < total_layers; i++) {
        layers_init[i] = 0;
    }
    __syncthreads();    

    int y_idx = blockIdx.x % Y_DIM;
    int x_idx = blockIdx.x / Y_DIM;
    int x = x_idx - (X_DIM / 2);
    int y = y_idx - (Y_DIM / 2);
    char intersection;

    unsigned triangles_per_thread = num_triangles / THREADS_PER_BLOCK;
    unsigned remaining_triangles = num_triangles % THREADS_PER_BLOCK;
    unsigned prev = threadIdx.x * triangles_per_thread;
    unsigned total = triangles_per_thread + (threadIdx.x < remaining_triangles);
    prev += (threadIdx.x < remaining_triangles) ? threadIdx.x : remaining_triangles;

    triangle* triangles = triangles_global;
    // Each block has a shared memory storing some triangles.
    // nvcc plz unroll this loop
    for (size_t i = threadIdx.x; i < num_triangles; i+=THREADS_PER_BLOCK) {
        intersection = pixelRayIntersection(triangles[i],x,y);
        if (intersection != -1) {
            layers_shared[intersection] = 1;
        }
    }
    __syncthreads();

    bool flag = (bool)(1 & thrust::count(thrust::device, &layers_shared[0], layers_init, 1));
    bool* out_begin = out + blockIdx.x * NUM_LAYERS;
    for (unsigned z = threadIdx.x; z < NUM_LAYERS; z+=THREADS_PER_BLOCK) {
        out_begin[z] = ((bool) layers_shared[z]) || flag;
        flag = flag ^ ((bool) layers_shared[z]);
    }
}

__global__
void mfps1(triangle* triangles, size_t num_triangles, char* all_intersections) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tri_idx = idx / (X_DIM * Y_DIM / PIXELS_PER_THREAD);
    if (tri_idx >= num_triangles) return;
    int y_idx = (idx - (tri_idx * X_DIM * Y_DIM / PIXELS_PER_THREAD)) / (X_DIM / PIXELS_PER_THREAD);
    int x_group = (idx - (tri_idx * X_DIM * Y_DIM / PIXELS_PER_THREAD)) % (X_DIM / PIXELS_PER_THREAD);
    int x_idx, x, y;
    char intersection;
    triangle triangle_shared = triangles[tri_idx];
    y = y_idx - (Y_DIM / 2);
    char* layers = all_intersections + y_idx * X_DIM * NUM_LAYERS + x_group * NUM_LAYERS * PIXELS_PER_THREAD;

    for (x_idx = 0; x_idx < PIXELS_PER_THREAD; x_idx++) {
        x = x_group * PIXELS_PER_THREAD + x_idx - (X_DIM / 2);
        intersection = pixelRayIntersection(triangle_shared, x, y);
        if(-1 != intersection)
            layers[x_idx * NUM_LAYERS + intersection] = 1;
    }
}

__global__
void mfps2(char* sorted_intersections, bool* out) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    int z_idx = idx / (X_DIM * Y_DIM);
    if (z_idx >= NUM_LAYERS) return;
    int y_idx = (idx - (z_idx * X_DIM * Y_DIM)) / X_DIM;
    int x_idx = (idx - (z_idx * X_DIM * Y_DIM)) % X_DIM;

    char* intersection_trunk = sorted_intersections + y_idx * X_DIM * NUM_LAYERS + x_idx * NUM_LAYERS;
    out[idx] = intersection_trunk[z_idx] | (1 & thrust::count(thrust::device, intersection_trunk, intersection_trunk + z_idx, 1));
}

__device__ __forceinline__
char pixelRayIntersection(triangle t, int x, int y) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.

    return the layer of intersection, or -1 if none
    */
/*
    double x_pos = x * RESOLUTION;
    double y_pos = y * RESOLUTION;

    if (   ((x_pos < t.p1.x) && (x_pos < t.p2.x) && (x_pos < t.p3.x))
        || ((x_pos > t.p1.x) && (x_pos > t.p2.x) && (x_pos > t.p3.x))
        || ((y_pos < t.p1.y) && (y_pos < t.p2.y) && (y_pos < t.p3.y))
        || ((y_pos > t.p1.y) && (y_pos > t.p2.y) && (y_pos > t.p3.y))
    ) return -1;
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
    char layer = inside ? (intersection / RESOLUTION) : -1;
    return layer;
}

__device__
int getIntersectionTrunk(int x, int y, triangle* triangles, size_t num_triangles, char* layers) {
    int idx = 0;

    for (int i = 0; i < num_triangles; i++) {
        char layer = pixelRayIntersection(triangles[i], x, y);
        if (layer != -1) {
            layers[idx] = layer;
            idx++;
        }
    }
    return idx;
}

__device__
bool isInside(char current, char* trunk, size_t length) {
    size_t startIdx = 0;
    size_t endIdx = length;
    size_t mid;
    bool goLeft;

    // perform binary search
    while (startIdx < endIdx) {
        mid = (startIdx + endIdx) / 2;
        if (trunk[mid] == current) return true;
        goLeft = trunk[mid] > current;
        startIdx = goLeft ? startIdx : (mid + 1);
        endIdx = goLeft ? mid : endIdx;
    }

    return (bool)(startIdx & 1);
}
