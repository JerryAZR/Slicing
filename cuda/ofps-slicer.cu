#include "slicer.cuh"
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <stdio.h>

/**
 * fps1: First stage of slicing -- Ray Triangle Intersection
 * Inputs: 
 *      triangles -- array of all triangles
 *      num_triangles -- length of the triangle array
 *      locks -- array of locks (used in atomic memory access)
 * Outputs:
 *      all_intersections -- array of all intersections
 *      trunk_length -- number of intersections of each pixel ray
 */
__global__
void fps1(triangle* triangles, size_t num_triangles, layer_t* all_intersections, size_t* trunk_length, int* locks) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    size_t tri_idx = idx / (X_DIM * Y_DIM);
    // if (tri_idx >= num_triangles) return;

    // copy 1 triangle to the shared memory -- That's all we need on this block
    __shared__  triangle triangle_shared;
    __shared__  double x_max, x_min;
    __shared__  bool y_notInside;

    int y_idx = (idx / X_DIM) & (Y_DIM-1);
    int y = y_idx - (Y_DIM >> 1);
    double y_pos = y * RESOLUTION;

    if (threadIdx.x == 0) {
        // copy the triangle to shared memory
        triangle_shared = triangles[tri_idx];
        // compute x_min, x_max of the triangle, store results in shared memory
        thrust::maximum<double> max;
        thrust::minimum<double> min;
        x_max = max(triangle_shared.p1.x, max(triangle_shared.p2.x, triangle_shared.p3.x));
        x_min = min(triangle_shared.p1.x, min(triangle_shared.p2.x, triangle_shared.p3.x));
        // check if current y value is inside the triangle
        // All threads (pixels) on this block have the same y value,
        // so this condition only needs to be checked once.
        double y_max = max(triangle_shared.p1.y, max(triangle_shared.p2.y, triangle_shared.p3.y));
        double y_min = min(triangle_shared.p1.y, min(triangle_shared.p2.y, triangle_shared.p3.y));
        y_notInside = (y_pos < y_min) || (y_pos > y_max);
    }
    __syncthreads();

    if (y_notInside) return;

    int x_idx = idx & (X_DIM-1);
    int x = x_idx - (X_DIM >> 1);
    double x_pos = x * RESOLUTION;

    bool notInRect = (x_pos < x_min) || (x_pos > x_max);

    layer_t* layers = all_intersections + y_idx * X_DIM * MAX_TRUNK_SIZE + x_idx * MAX_TRUNK_SIZE;
    int* lock = locks + y_idx * X_DIM + x_idx;
    size_t* length = trunk_length + y_idx * X_DIM + x_idx;
    // if current pixel is not in the rectangle defined by x_min/max and y_min/max,
    // there cannot be an intersection
    layer_t intersection = notInRect ? (layer_t)(-1) : pixelRayIntersection(triangle_shared, x, y);
    bool run = (intersection != (layer_t)(-1));

    while (run) {
        if(atomicCAS(lock, 0, 1) == 0) {
            layers[length[0]] = intersection;
            length[0]++;
            run = false;
            atomicExch(lock, 0);
        }
    }
}
 
 /**
  * fps2: second stage of slicing -- trunk sorting
  * Inputs:
  *      all_intersections -- array of intersections computed in fps1
  *      trunk_length -- number of intersections of each pixel ray
  * Outputs:
  *      all_intersections -- sorting will be performed in-place
  */
 __global__
 void fps2(layer_t* all_intersections, size_t* trunk_length) {
     size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
     if (idx >= X_DIM * Y_DIM) return;
     size_t length = trunk_length[idx];
     layer_t* curr_trunk = all_intersections + (idx * MAX_TRUNK_SIZE);
     thrust::sort(thrust::device, curr_trunk, curr_trunk + length);
 }
 
 /**
  * fps3: third stage of slicing: layer extractions
  * Inputs:
  *      sorted_intersections -- sorted array of intersections
  *      trunk_length -- number of intersections of each pixel ray
  * Outputs:
  *      out -- Z*X*Y array representing the sliced model. A cell is True
  *             if it is inside the model, False if not.
  */
 __global__
 void fps3(layer_t* sorted_intersections, size_t* trunk_length, bool* out) {
     size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
     int z_idx = idx / (X_DIM * Y_DIM);
     if (z_idx >= NUM_LAYERS) return;
     int y_idx = (idx - (z_idx * X_DIM * Y_DIM)) / X_DIM;
     int x_idx = (idx - (z_idx * X_DIM * Y_DIM)) & (X_DIM - 1);
 
     size_t length = trunk_length[y_idx * X_DIM + x_idx];
     layer_t* intersection_trunk = sorted_intersections + y_idx * X_DIM * MAX_TRUNK_SIZE + x_idx * MAX_TRUNK_SIZE;
     out[idx] = isInside(z_idx, intersection_trunk, length);
 }

/**
 * pixelRayIntersection: helper function, computes the intersection of given triangle and pixel ray
 * Inputs:
 *      t -- input triangle
 *      x, y -- coordinates of the input pixel ray
 * Returns:
 *      The layer on which they intersect, or -1 if no intersection
 */
__device__ __forceinline__
layer_t pixelRayIntersection(triangle t, int x, int y) {
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
    layer_t layer = inside ? (intersection / RESOLUTION) : (layer_t)(-1);
    return layer;
}

/**
 * isInside: given an array of intersection, check if the current pixel is inside the model
 * Inputs:
 *      current -- z value of current pixel
 *      trunk -- intersection array of current pixel ray
 *      length -- length of intersection array (trunk)
 * Returns:
 *      True if current pixel is inside the model, False if not
 */
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
 
 