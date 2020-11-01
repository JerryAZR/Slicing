#include "slicer.cuh"
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/count.h>
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
void mfps(triangle* triangles, size_t num_triangles, char* all_intersections, size_t* trunk_length, int* locks) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tri_idx = idx / (X_DIM * Y_DIM / PIXELS_PER_THREAD);
    if (tri_idx >= num_triangles) return;

    __shared__ double x_max, x_min, y_max, y_min;

    __shared__ triangle triangles_shared;
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

    int y_idx = (idx - (tri_idx * X_DIM * Y_DIM / PIXELS_PER_THREAD)) / (X_DIM / PIXELS_PER_THREAD);
    int x_idx, x, y;
    y = y_idx - (Y_DIM / 2);
    double x_pos, y_pos;
    int x_base;
    y_pos = y * RESOLUTION;
    if (y_pos < y_min || y_pos > y_max) return;

    int x_group = (idx - (tri_idx * X_DIM * Y_DIM / PIXELS_PER_THREAD)) % (X_DIM / PIXELS_PER_THREAD);
    char intersections[PIXELS_PER_THREAD];
    x_base = x_group * PIXELS_PER_THREAD - (X_DIM >> 1);
    for (x_idx = 0; x_idx < PIXELS_PER_THREAD; x_idx++) {
        x = x_base + x_idx;
        x_pos = x * RESOLUTION;
        intersections[x_idx] = (x_pos < x_min || x_pos > x_max) ? -1 : pixelRayIntersection(triangles_shared, x, y);
    }

    char* layers = all_intersections + y_idx * X_DIM * NUM_LAYERS + x_group * NUM_LAYERS * PIXELS_PER_THREAD;
    int* lock = locks + y_idx * (X_DIM / PIXELS_PER_THREAD) + x_group;
    size_t* length = trunk_length + (y_idx * X_DIM) + (x_group * PIXELS_PER_THREAD);
    bool run = (PIXELS_PER_THREAD + thrust::reduce(thrust::device, &intersections[0], &intersections[0] + PIXELS_PER_THREAD) > 0);
    while (run) {
        if(atomicCAS(lock, 0, 1) == 0) {
            for (x_idx = 0; x_idx < PIXELS_PER_THREAD; x_idx++) {
                layers[x_idx * NUM_LAYERS + length[x_idx]] = intersections[x_idx];
                length[x_idx] = (intersections[x_idx] == -1) ? length[x_idx] : length[x_idx] + 1;
            }
            atomicExch(lock, 0);
            run = false;
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
//    return t.p1.z < 0 ? (char)t.p1.z : -1;

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
