#include "slicer.hpp"
#include <algorithm>
#include <iostream>
#include <climits>
#include <cmath>
#include <thread>
#include <cstring>
#include <chrono>
#include <assert.h>
#define NOW (std::chrono::high_resolution_clock::now())
typedef std::chrono::time_point<std::chrono::high_resolution_clock> chrono_t;

using std::vector;
using std::min;
using std::max;

constexpr double min3(double a, double b, double c) { return min(a, min(b, c)); }
constexpr double max3(double a, double b, double c) { return max(a, max(b, c)); }
int pixelRayIntersection(triangle t, int y, int z);

void bbox_cpu(vector<triangle> tris, vector<vector<unsigned>>& out_compressed, size_t base_z) {
    for (auto it = tris.begin(); it != tris.end(); it++) {
        triangle t = *it;
        // Find Bounding Box
        long yMin = ceil(min3(t.p1.y, t.p2.y, t.p3.y) / RESOLUTION);
        long zMin = ceil(min3(t.p1.z, t.p2.z, t.p3.z) / RESOLUTION);
        long yMax = floor(max3(t.p1.y, t.p2.y, t.p3.y) / RESOLUTION);
        long zMax = floor(max3(t.p1.z, t.p2.z, t.p3.z) / RESOLUTION);
        // Make sure the bounds are inside the supported space
        yMax = min(yMax, Y_MAX);
        yMin = max(yMin, Y_MIN);
        long zMax_ub = min(NUM_LAYERS-1, (base_z+BBOX_BLOCK_HEIGHT-1));
        zMax = min(zMax, zMax_ub);
        zMin = max(zMin, (long)base_z);
        if (yMax < yMin || zMax < zMin) continue;

        // iterate over all pixels inside the bounding box
        for (long y = yMin; y <= yMax; y++) {
            for (long z = zMin; z <= zMax; z++) {
                int curr_intersection = pixelRayIntersection(t, y, z);
                if (curr_intersection >= X_MIN && curr_intersection <= X_MAX) {
                    long y_idx = y + (Y_DIM >> 1);
                    unsigned x_idx = curr_intersection + (X_DIM >> 1);
                    size_t trunk_idx = (z - base_z)*Y_DIM + y_idx;
                    out_compressed[trunk_idx].push_back(x_idx);
                }
            }
        }
    }
}

void trunk_compress(vector<vector<unsigned>>& out_compressed) {
    for (auto it = out_compressed.begin(); it != out_compressed.end(); it++) {
        vector<unsigned> trunk = *it; // make a copy
        vector<unsigned> out;
        std::sort(trunk.begin(), trunk.end());

        // Manually process the first intersection to avoid problems
        unsigned length = trunk.size();
        // assert(length & 1 == 0);
        trunk.push_back(X_DIM);
        out.push_back(trunk[0]);
        unsigned prev_idx = trunk[0];
        unsigned i = 0;

        while (i < length) {
            // Find the next run of 1's
            i++;
            while ((trunk[i] - trunk[i-1] <= 1 || i & 1 == 1) && i < length) {
                i++;
            }
            unsigned run_1s = trunk[i-1] - prev_idx + 1;
            unsigned run_0s = (i == length) ?
                    X_DIM - trunk[i-1] - 1 : trunk[i] - trunk[i-1] - 1;
            prev_idx = trunk[i];
            out.push_back(run_1s);
            out.push_back(run_0s);
        }
        *it = out;
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
int pixelRayIntersection(triangle t, int y, int z) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.
    */
    double y_pos = y * RESOLUTION;
    double z_pos = z * RESOLUTION;

    double y_d = y_pos - t.p1.y;
    double z_d = z_pos - t.p1.z;

    double xx1 = t.p2.x - t.p1.x;
    double yy1 = t.p2.y - t.p1.y;
    double zz1 = t.p2.z - t.p1.z;

    double xx2 = t.p3.x - t.p1.x;
    double yy2 = t.p3.y - t.p1.y;
    double zz2 = t.p3.z - t.p1.z;
    double a = (y_d * zz2 - yy2 * z_d) / (yy1 * zz2 - yy2 * zz1);
    double b = (y_d * zz1 - yy1 * z_d) / (yy2 * zz1 - yy1 * zz2);
    bool inside = (a >= 0) && (b >= 0) && (a+b <= 1);
    double intersection = (a * xx1 + b * xx2) + t.p1.x;
    // // divide by layer width
    int layer = inside ? (intersection / RESOLUTION) : INT_MIN;
    return layer;
}

// single thread ver
void rleDecodeSt(vector<vector<unsigned>>& in, bool* out, const bool* out_end) {
    size_t i = 0;
    for (auto it = in.begin(); it != in.end(); it++, i++) {
        auto & in_base = *it;
        bool* out_base = out + (i*X_DIM);
        if (out_base == out_end) break;
        bool inside = false;
        unsigned start = 0;
        unsigned length;
        for (auto it2 = in_base.begin(); it2 != in_base.end(); it2++) {
            length = *it2;
            memset(out_base+start, inside, length);
            inside = !inside;
            start += length;
        }
        assert(start == X_DIM);
    }
}
