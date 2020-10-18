#include "slicer.hpp"

int pixelRayIntersection(triangle & t, int x, int y) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.
    */
    double x_d = x * RESOLUTION - t.p1.x;
    double y_d = y * RESOLUTION - t.p1.y;
    v3 vect1 = t.p2 - t.p1;
    v3 vect2 = t.p3 - t.p1;
    double a = (x_d * vect2.y - vect2.x * y_d) / (vect1.x * vect2.y - vect2.x * vect1.y);
    double b = (x_d * vect1.y - vect1.x * y_d) / (vect2.x * vect1.y - vect1.x * vect2.y);
    bool inside = (a >= 0) && (b >= 0) && (a+b <= 1);
    double intersection = (a * vect1.z + b * vect2.z) + t.p1.z;
    // divide by layer width
    int layer = (intersection / RESOLUTION) * inside - (!inside);
    return layer;
}

void getIntersectionTrunk(int x, int y, vector<triangle>& triangles, vector<int>& layers) {
    for (auto tri : triangles) {
        int layer = pixelRayIntersection(tri, x, y);
        if (layer != -1) {
            layers.push_back(layer);
        }
    }
}