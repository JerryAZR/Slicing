#include "slicer.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>

using std::cout;
using std::endl;

void pps(triangle* triangles_global, int num_triangles, bool* out, unsigned id) {
    unsigned idx = id;
    int y_idx = idx / X_DIM;
    int x_idx = idx % X_DIM;
    int x = x_idx - (X_DIM / 2);
    int y = y_idx - (Y_DIM / 2);

    // Copy triangles to shared memory
    // Each block has a shared memory storing some triangles.
    triangle* triangles;
    size_t num_iters = num_triangles / 256;
    int length = 0;
    int layers_local[NUM_LAYERS+1];
    int* layers = &layers_local[0];
    for (size_t i = 0; i < num_iters; i++) {
        triangles = triangles_global + i * 256;
        // Wait for other threads to complete;
        if (y_idx < Y_DIM)
        length += getIntersectionTrunk(x, y, triangles, 256, layers);
        layers = &layers_local[length]; // update pointer value
    }
    size_t remaining = num_triangles - (num_iters * 256);
    triangles = triangles_global + (num_iters * 256);

    if (remaining) {
        if (y_idx < Y_DIM)
        length += getIntersectionTrunk(x, y, triangles, remaining, layers);
        layers = &layers_local[length]; // update pointer value
    }

    if (y_idx >= Y_DIM) return;

    layers = &layers_local[0]; // reset to beginning
    std::sort(layers, layers+length);
    layers[0] = layers[0] * (length >= 1) + NUM_LAYERS * (length == 0);
    bool flag = false;
    bool intersect;
    size_t layerIdx = 0;
    size_t outIdx;
    for (int z = 0; z < NUM_LAYERS; z++) {
        // If intersect
        intersect = (z == layers[layerIdx]);
        outIdx = z*Y_DIM*X_DIM + y_idx*X_DIM + x_idx;
        out[outIdx] = intersect || flag;
        flag = intersect ^ flag;
        if (intersect)
        layerIdx++;
    }
}

int atomicCAS(int* ptr, int cmp, int val) {
    int temp = *ptr;
    if (temp == cmp) *ptr = val;
    return (temp == cmp) ? temp : val;
}

int atomicExch(int* ptr, int val) {
    int temp = *ptr;
    *ptr = val;
    return temp;
}
void fps1(triangle* triangles, size_t num_triangles, int* all_intersections, size_t* trunk_length, int* locks, long id) {
    size_t idx = id;
    size_t tri_idx = idx / (X_DIM * Y_DIM / PIXELS_PER_THREAD);
    if (tri_idx >= num_triangles) return;
    int y_idx = (idx - (tri_idx * X_DIM * Y_DIM / PIXELS_PER_THREAD)) / (X_DIM / PIXELS_PER_THREAD);
    int x_group = (idx - (tri_idx * X_DIM * Y_DIM / PIXELS_PER_THREAD)) % (X_DIM / PIXELS_PER_THREAD);
    int x_idx, x, y;
    char intersections[PIXELS_PER_THREAD];
    triangle triangles_shared = triangles[tri_idx];
    y = y_idx - (Y_DIM / 2);

    for (x_idx = 0; x_idx < PIXELS_PER_THREAD; x_idx++) {
        x = x_group * PIXELS_PER_THREAD + x_idx - (X_DIM / 2);
        intersections[x_idx] = pixelRayIntersection(triangles_shared, x, y);
    }

    int* layers = all_intersections + y_idx * X_DIM * NUM_LAYERS + x_group * NUM_LAYERS * PIXELS_PER_THREAD;
    int* lock = locks + y_idx * (X_DIM / PIXELS_PER_THREAD) + x_group;
    size_t* length = trunk_length + (y_idx * X_DIM) + (x_group * PIXELS_PER_THREAD);
    bool run = (PIXELS_PER_THREAD + std::reduce(intersections, intersections + PIXELS_PER_THREAD) > 0);
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


void fps2(int* all_intersections, size_t* trunk_length, long id) {
    size_t idx = id;
    if (idx >= X_DIM * Y_DIM) return;
    size_t length = trunk_length[idx];
    int* curr_trunk = all_intersections + (idx * NUM_LAYERS);
    std::sort(curr_trunk, curr_trunk + length);
}

struct lessThan
{
    lessThan(int x) : target(x) {}
    bool operator()(const int& curr) { return curr < target; }
    int target;
};

void fps3(int* sorted_intersections, size_t* trunk_length, bool* out, long id) {
    size_t idx = id;
    int z_idx = idx / (X_DIM * Y_DIM);
    if (z_idx >= NUM_LAYERS) return;
    int y_idx = (idx - (z_idx * X_DIM * Y_DIM)) / X_DIM;
    int x_idx = (idx - (z_idx * X_DIM * Y_DIM)) % X_DIM;

    size_t length = trunk_length[y_idx * X_DIM + x_idx];
    int* intersection_trunk = sorted_intersections + y_idx * X_DIM * NUM_LAYERS + x_idx * NUM_LAYERS;
    bool inside = (bool) (1 & std::count_if(intersection_trunk, intersection_trunk + length, lessThan(z_idx)));
    bool edge = std::binary_search(intersection_trunk, intersection_trunk + length, z_idx);
    out[idx] = inside || edge;
    if ((inside || edge) != isInside(z_idx, intersection_trunk, length)) {
        std::cout << "result mismatch" << std::endl;
    }
}
int pixelRayIntersection(triangle t, int x, int y) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.

    return the layer of intersection, or -1 if none
    */

    // quick check
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
    // divide by layer width
    int layer = (intersection / RESOLUTION) * inside - (!inside);
    return layer;
}

int getIntersectionTrunk(int x, int y, triangle* triangles, int num_triangles, int* layers) {
    int idx = 0;
    for (int i = 0; i < num_triangles; i++) {
        int layer = pixelRayIntersection(triangles[i], x, y);
        layers[idx] = layer;
        idx += (layer != -1);
    }
    return idx;
}

bool isInside(int current, int* trunk, size_t length) {
    size_t startIdx = 0;
    size_t endIdx = length;
    size_t mid;
    bool goLeft, goRight;

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
