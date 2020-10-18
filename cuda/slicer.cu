#include "slicer.cuh"
#include <thrust/sort.h>
#include <stdio.h>

__global__
void pps(triangle* triangles, int num_triangles, bool* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("starting thread %d\n", idx);
    int y = idx / X_DIM;
    if (y >= Y_DIM) return;
    int x = idx % X_DIM - (X_DIM / 2);
    y = y - (Y_DIM / 2);

    int layers[NUM_LAYERS];
    int length = getIntersectionTrunk(x, y, triangles, num_triangles, &layers[0]);

    thrust::sort(thrust::device, layers, layers+length);
    if (length == 0) layers[0] = NUM_LAYERS;

    bool flag = false;
    int layerIdx = 0;
    for (int z = 0; z < NUM_LAYERS; z++) {
        // If intersect
        int x_idx = x + (X_DIM / 2);
        int y_idx = y + (Y_DIM / 2);
        // std::cout << "(z,y,x) = " << z << ", " << y_idx << ", " << x_idx << std::endl;
        bool intersect = (z == layers[layerIdx]);
        out[z*Y_DIM*X_DIM + y_idx*X_DIM + x_idx] = intersect || flag;
        flag = intersect ^ flag;
        layerIdx += intersect;
    }
    // printf("exiting thread %d\n", idx);
}

__global__
void fps1(triangle* triangles, int num_triangles, bool* out) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t tri_idx = idx / (X_DIM * Y_DIM);
    if (tri_idx >= num_triangles) return;
    int y_idx = (idx - (tri_idx * X_DIM * Y_DIM)) / X_DIM;
    int x_idx = (idx - (tri_idx * X_DIM * Y_DIM)) % X_DIM;

    int x = x_idx - (X_DIM / 2);
    int y = y_idx - (Y_DIM / 2);

    // __shared__ int layers[256][NUM_LAYERS];
    // __shared__ bool locks[256] = {false};
}

__device__
int pixelRayIntersection(triangle t, int x, int y) {
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
    int layer = (intersection / RESOLUTION) * inside - (!inside);
    return layer;
}

__device__
int getIntersectionTrunk(int x, int y, triangle* triangles, int num_triangles, int* layers) {
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
