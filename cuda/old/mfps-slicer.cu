#include "slicer.cuh"
#include <thrust/sort.h>
#include <thrust/functional.h>

__global__
void fps1(triangle* triangles, size_t num_triangles, layer_t* all_intersections, size_t* trunk_length, int* locks) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    size_t tri_group = idx / (X_DIM * Y_DIM);
    size_t tri_idx = tri_group << LOG_THREADS;
    triangle* tri_base = triangles + tri_idx;

    __shared__  triangle triangles_shared[THREADS_PER_BLOCK];
    __shared__  double x_max[THREADS_PER_BLOCK];
    __shared__  double x_min[THREADS_PER_BLOCK];
    // Assumption: X_DIM is divisible by THREADS_PER_BLOCK;
    // so that all pixels on a block have the same y value.
    __shared__  bool y_notInside[THREADS_PER_BLOCK];
    __shared__  layer_t layers_shared[THREADS_PER_BLOCK][MAX_TRUNK_SIZE];

    thrust::maximum<double> max;
    thrust::minimum<double> min;
    thrust::minimum<int> minInt;

    int y_idx = (idx / X_DIM) & (Y_DIM-1);
    int y = y_idx - (Y_DIM >> 1);
    double y_pos = y * RESOLUTION;

    if (threadIdx.x + tri_idx < num_triangles) {
        triangle t = tri_base[threadIdx.x];
        triangles_shared[threadIdx.x] = t;
        x_max[threadIdx.x] = max(t.p1.x, max(t.p2.x, t.p3.x));
        x_min[threadIdx.x] = min(t.p1.x, min(t.p2.x, t.p3.x));
        double y_max = max(t.p1.y, max(t.p2.y, t.p3.y));
        double y_min = min(t.p1.y, min(t.p2.y, t.p3.y));
        y_notInside[threadIdx.x] = (y_pos < y_min) || (y_pos > y_max);
    }
    __syncthreads();

    int x_idx = idx & (X_DIM-1);
    int x = x_idx - (X_DIM >> 1);
    double x_pos = x * RESOLUTION;

    size_t length_local = 0;
    int total_work = minInt(THREADS_PER_BLOCK, num_triangles - tri_idx);
    layer_t* layers_local = &layers_shared[threadIdx.x][0];

    for (int i = 0; i < THREADS_PER_BLOCK; i++) {
        if (i < total_work) {
            bool notInRect = (y_notInside[i] || (x_pos < x_min[i]) || (x_pos > x_max[i]));
            layer_t intersection = notInRect ? NONE : pixelRayIntersection(triangles_shared[i], x, y);
            if (intersection != NONE) {
                layers_local[length_local] = intersection;
                length_local++;
            }
        }
    }
    bool run = (length_local > 0);

    size_t* length = trunk_length + y_idx * X_DIM + x_idx;
    int* lock = locks + y_idx * X_DIM + x_idx;
    while (run) {
        if(atomicCAS(lock, 0, 1) == 0) {
            layer_t* layers = all_intersections + y_idx * X_DIM * NUM_LAYERS + x_idx * NUM_LAYERS + length[0];
            for (int i = 0; i < length_local; i++) {
                layers[i] = layers_local[i];
            }
            length[0] += length_local;
            run = false;
            atomicExch(lock, 0);
        }
    }
}

__global__
void fps2(layer_t* all_intersections, size_t* trunk_length) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= X_DIM * Y_DIM) return;
    size_t length = trunk_length[idx];
    layer_t* curr_trunk = all_intersections + (idx * NUM_LAYERS);
    thrust::sort(thrust::device, curr_trunk, curr_trunk + length);
}

__global__
void fps3(layer_t* sorted_intersections, size_t* trunk_length, bool* out) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    int z_idx = idx / (X_DIM * Y_DIM);
    if (z_idx >= NUM_LAYERS) return;
    int y_idx = (idx - (z_idx * X_DIM * Y_DIM)) / X_DIM;
    int x_idx = (idx - (z_idx * X_DIM * Y_DIM)) % X_DIM;

    size_t length = trunk_length[y_idx * X_DIM + x_idx];
    layer_t* intersection_trunk = sorted_intersections + y_idx * X_DIM * NUM_LAYERS + x_idx * NUM_LAYERS;
    out[idx] = isInside(z_idx, intersection_trunk, length);
}

__device__ __forceinline__
layer_t pixelRayIntersection(triangle t, int x, int y) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.
    return the layer of intersection, or -1 if none
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
    layer_t layer = inside ? (intersection / RESOLUTION) : NONE;
    return layer;
}

__device__
bool isInside(layer_t current, layer_t* trunk, size_t length) {
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
