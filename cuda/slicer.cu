#include "slicer.cuh"
#include <thrust/sort.h>
#include <thrust/binary_search.h>
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
    __shared__ triangle tri_base[256];
    triangle* triangles = (triangle*) tri_base;
    size_t num_iters = num_triangles / 256;
    int length = 0;
    int layers_local[NUM_LAYERS+1];
    int* layers = &layers_local[0];
    for (size_t i = 0; i < num_iters; i++) {
        triangles[threadIdx.x] = triangles_global[threadIdx.x + (i * 256)];
        // Wait for other threads to complete;
        __syncthreads();
        if (y_idx < Y_DIM)
        length += getIntersectionTrunk(x, y, triangles, 256, layers);
        layers = &layers_local[length]; // update pointer value
    }
    size_t remaining = num_triangles - (num_iters * 256);
    if (threadIdx.x < remaining) {
        triangles[threadIdx.x] = triangles_global[threadIdx.x + (num_iters * 256)];
        __syncthreads();
    }
    if (remaining) {
        if (y_idx < Y_DIM)
        length += getIntersectionTrunk(x, y, triangles, remaining, layers);
        layers = &layers_local[length]; // update pointer value
    }

    if (y_idx >= Y_DIM) return;
    layers = &layers_local[0]; // reset to beginning

    thrust::sort(thrust::device, &layers[0], &layers[length]);
    layers[length] = NUM_LAYERS;

    bool flag = false;
    int layerIdx = 0;
    for (int z = 0; z < NUM_LAYERS; z++) {
        // If intersect
        bool intersect = (z == layers[layerIdx]);
        out[z*Y_DIM*X_DIM + y_idx*X_DIM + x_idx] = intersect || flag;
        flag = intersect ^ flag;
        layerIdx += intersect;
    }
}

__global__
void fps1(triangle* triangles, size_t num_triangles, int* all_intersections, size_t* trunk_length, int* locks) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tri_idx = idx / (X_DIM * Y_DIM);
    if (tri_idx >= num_triangles) return;
    int y_idx = (idx - (tri_idx * X_DIM * Y_DIM)) / X_DIM;
    int x_idx = (idx - (tri_idx * X_DIM * Y_DIM)) % X_DIM;

    int x = x_idx - (X_DIM / 2);
    int y = y_idx - (Y_DIM / 2);

    // all_intersections[y][x][layer]
    int* layers = all_intersections + y_idx * X_DIM * NUM_LAYERS + x_idx * NUM_LAYERS;
    int* lock = locks + y_idx * X_DIM + x_idx;
    size_t* length = trunk_length + y_idx * X_DIM + x_idx;
    int intersection = pixelRayIntersection(triangles[tri_idx], x, y);
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
void fps2(int* all_intersections, size_t* trunk_length) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= X_DIM * Y_DIM) return;
    size_t length = trunk_length[idx];
    int* curr_trunk = all_intersections + (idx * NUM_LAYERS);
    thrust::sort(thrust::device, curr_trunk, curr_trunk + length);
}

struct lessThan
{
  __host__ __device__ lessThan(int x) : target(x) {}
  __host__ __device__ bool operator()(const int& curr) { return curr < target; }
  int target;
};

__global__
void fps3(int* sorted_intersections, size_t* trunk_length, bool* out) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    int z_idx = idx / (X_DIM * Y_DIM);
    if (z_idx >= NUM_LAYERS) return;
    int y_idx = (idx - (z_idx * X_DIM * Y_DIM)) / X_DIM;
    int x_idx = (idx - (z_idx * X_DIM * Y_DIM)) % X_DIM;

    size_t length = trunk_length[y_idx * X_DIM + x_idx];
    int* intersection_trunk = sorted_intersections + y_idx * X_DIM * NUM_LAYERS + x_idx * NUM_LAYERS;
    bool edge = 0 < thrust::count(thrust::device, intersection_trunk, intersection_trunk + length, z_idx);
    bool inside = (bool) (1 & thrust::count_if(thrust::device, intersection_trunk, intersection_trunk + length, lessThan(z_idx)));
    out[idx] = inside || edge;
}

__device__
int pixelRayIntersection(triangle t, int x, int y) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.

    return the layer of intersection, or -1 if none
    */

    if (   (x < t.p1.x && x < t.p2.x && x < t.p3.x)
        || (x > t.p1.x && x > t.p2.x && x > t.p3.x)
        || (y < t.p1.y && y < t.p2.y && y < t.p3.y)
        || (y > t.p1.y && y > t.p2.y && y > t.p3.y)
    ) return -1;

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
    int layer = (intersection / RESOLUTION) * inside - (!inside);
    return layer;
}

__device__
int getIntersectionTrunk(int x, int y, triangle* triangles, size_t num_triangles, int* layers) {
    int idx = 0;
    for (int i = 0; i < num_triangles; i++) {
        int layer = pixelRayIntersection(triangles[i], x, y);
        if (layer != -1) {
            layers[idx] = layer;
            idx++;
        }
    }
    return idx;
}
