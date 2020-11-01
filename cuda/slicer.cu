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
    __shared__ char layers_shared[THREADS_PER_BLOCK][NUM_LAYERS+1];
    char* layers = &layers_shared[threadIdx.x][0];
    for (size_t i = 0; i < num_iters; i++) {
        triangles[threadIdx.x] = triangles_global[threadIdx.x + (i * THREADS_PER_BLOCK)];
        // Wait for other threads to complete;
        __syncthreads();
        if (y_idx < Y_DIM)
        length += getIntersectionTrunk(x, y, triangles, THREADS_PER_BLOCK, layers);
        layers = &layers_shared[threadIdx.x][length]; // update pointer value
    }
    size_t remaining = num_triangles - (num_iters * THREADS_PER_BLOCK);
    if (threadIdx.x < remaining) {
        triangles[threadIdx.x] = triangles_global[threadIdx.x + (num_iters * THREADS_PER_BLOCK)];
    }
    if (remaining) {
        __syncthreads();
        if (y_idx < Y_DIM)
        length += getIntersectionTrunk(x, y, triangles, remaining, layers);
        layers = &layers_shared[threadIdx.x][length]; // update pointer value
    }

    if (y_idx >= Y_DIM) return;
    layers = &layers_shared[threadIdx.x][0]; // reset to beginning

    thrust::sort(thrust::device, &layers[0], &layers[length]);
    layers[length] = NUM_LAYERS;

    bool flag = false;
    int layerIdx = 0;
    for (char z = 0; z < NUM_LAYERS; z++) {
        // If intersect
        bool intersect = (z == layers[layerIdx]);
        out[z*Y_DIM*X_DIM + y_idx*X_DIM + x_idx] = intersect || flag;
        flag = intersect ^ flag;
        layerIdx += intersect;
    }
}

__global__
void fps1(triangle* triangles, size_t num_triangles, char* all_intersections, size_t* trunk_length, int* locks) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tri_idx = idx >> (LOG_X + LOG_Y);
    // if (tri_idx >= num_triangles) return;

    // copy 1 triangle to the shared memory -- That's all we need on this block
    __shared__  triangle triangles_shared;
    __shared__  double x_max, x_min, y_max, y_min;
    if (threadIdx.x == 0) {
        triangles_shared = triangles[tri_idx];
        thrust::maximum<double> max;
        thrust::minimum<double> min;
        x_max = max(triangles_shared.p1.x, max(triangles_shared.p2.x, triangles_shared.p3.x));
        x_min = min(triangles_shared.p1.x, min(triangles_shared.p2.x, triangles_shared.p3.x));
        y_max = max(triangles_shared.p1.y, max(triangles_shared.p2.y, triangles_shared.p3.y));
        y_min = min(triangles_shared.p1.y, min(triangles_shared.p2.y, triangles_shared.p3.y));
    }
    __syncthreads();

    int y_idx = (idx - (tri_idx << (LOG_X + LOG_Y))) >> LOG_X;
    int x_idx = (idx - (tri_idx << (LOG_X + LOG_Y))) & (X_DIM-1);
    int x = x_idx - (X_DIM >> 1);
    int y = y_idx - (Y_DIM >> 1);

    double x_pos = x * RESOLUTION;
    double y_pos = y * RESOLUTION;
    bool notInRect = (x_pos < x_min) || (x_pos > x_max) || (y_pos < y_min) || (y_pos > y_max);

    char* layers = all_intersections + y_idx * X_DIM * NUM_LAYERS + x_idx * NUM_LAYERS;
    int* lock = locks + y_idx * X_DIM + x_idx;
    size_t* length = trunk_length + y_idx * X_DIM + x_idx;
    char intersection = notInRect ? -1 : pixelRayIntersection(triangles_shared, x, y);
    bool run = (intersection != -1);
    while (run) {
        if(atomicCAS(lock, 0, 1) == 0) {
            layers[length[0]] = intersection;
            length[0]++;
            run = false;
            atomicExch(lock, 0);
        }
    }
}

__global__
void fps2(char* all_intersections, size_t* trunk_length) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= X_DIM * Y_DIM) return;
    size_t length = trunk_length[idx];
    char* curr_trunk = all_intersections + (idx * NUM_LAYERS);
    thrust::sort(thrust::device, curr_trunk, curr_trunk + length);
}

__global__
void fps3(char* sorted_intersections, size_t* trunk_length, bool* out) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    int z_idx = idx / (X_DIM * Y_DIM);
    if (z_idx >= NUM_LAYERS) return;
    int y_idx = (idx - (z_idx * X_DIM * Y_DIM)) / X_DIM;
    int x_idx = (idx - (z_idx * X_DIM * Y_DIM)) % X_DIM;

    size_t length = trunk_length[y_idx * X_DIM + x_idx];
    char* intersection_trunk = sorted_intersections + y_idx * X_DIM * NUM_LAYERS + x_idx * NUM_LAYERS;
    out[idx] = isInside(z_idx, intersection_trunk, length);
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
