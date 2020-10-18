#include "slicer.hpp"
#include <algorithm>
#include <iostream>

using std::cout;
using std::endl;

void pps(triangle* triangles, int num_triangles, bool* out, unsigned id) {
    unsigned idx = id;
    int y = idx / X_DIM;
    if (y >= Y_DIM) return;
    int x = idx - (y*X_DIM) - (X_DIM / 2);
    y = y - (Y_DIM / 2);

    int layers[NUM_LAYERS];
    int length = getIntersectionTrunk(x, y, triangles, num_triangles, &layers[0]);
    std::sort(layers, layers+length);
    layers[0] = layers[0] * (length >= 1) + NUM_LAYERS * (length == 0);
    bool flag = false;
    bool intersect;
    size_t layerIdx = 0;
    size_t outIdx;
    int x_idx = x + (X_DIM / 2);
    int y_idx = y + (Y_DIM / 2);
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
