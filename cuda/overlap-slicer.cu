#include "slicer.cuh"
#include "triangle.cuh"
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/copy.h>

// Declare local helper functions
__device__ __forceinline__ void toNextLayer(layer_t* intersections_large_local, 
    size_t trunk_length_local, layer_t & curr_layer, bool & isInside, char* out_local);

__device__ __forceinline__ double min3(double a, double b, double c);
__device__ __forceinline__ double max3(double a, double b, double c);

__global__ 
void overlapSlicer(triangle* tri_small, double* zMins, size_t num_small, bool* out) {

    // out[y][x][z]
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    int x_idx = idx & (X_DIM-1);
    int y_idx = idx / X_DIM;
    int x = x_idx - (X_DIM >> 1);
    int y = y_idx - (Y_DIM >> 1);

    __shared__ triangle tri_base[THREADS_PER_BLOCK];
    // __shared__ double zMins_base[THREADS_PER_BLOCK];
    // __shared__ double xMin[THREADS_PER_BLOCK];
    // __shared__ double xMax[THREADS_PER_BLOCK];
    __shared__ bool yNotInside[THREADS_PER_BLOCK];

    // Use local array. Mapped to registers if NUM_LAYERS is small
    char out_local[NUM_LAYERS];
    char* out_ptr = (char*)(out + idx);

    for (size_t i = 0; i < NUM_LAYERS; i++) {
        out_local[i] = out_ptr[i*X_DIM*Y_DIM];
    }

    // layer_t curr_layer = curr_layers[idx];
    // layer_t* intersections_large_local = intersections_large + idx * NUM_LAYERS;
    // This flag only applies to pixels that are not intersections.
    // bool isInside = false;

    size_t num_iters = num_small / THREADS_PER_BLOCK;

    double y_pos = y * RESOLUTION;
    // double x_pos = x * RESOLUTION;

    for (size_t i = 0; i < num_iters; i++) {
        triangle t = tri_small[threadIdx.x + (i * THREADS_PER_BLOCK)];
        tri_base[threadIdx.x] = t;
        // zMins_base[threadIdx.x] = zMins[threadIdx.x + (i * THREADS_PER_BLOCK)];
        double yMin = min3(t.p1.y, t.p2.y, t.p3.y);
        double yMax = max3(t.p1.y, t.p2.y, t.p3.y);
        yNotInside[threadIdx.x] = (y_pos < yMin) || (y_pos > yMax);
        // Wait for other threads to complete;
        __syncthreads();
        if (y_idx < Y_DIM) {
            for (size_t tri_idx = 0; tri_idx < THREADS_PER_BLOCK; tri_idx++) {
                layer_t curr_intersection = yNotInside[tri_idx] ? NONE : pixelRayIntersection(tri_base[tri_idx], x, y);
                if (curr_intersection >= 0 && curr_intersection < NUM_LAYERS) out_local[curr_intersection]++;
            }
            // Move to the next triangle-layer pair that intersects
            // Add 1 to curr_layer when comparing to avoid rounding issues.
            // while ((curr_layer+1)*RESOLUTION < zMins_base[THREADS_PER_BLOCK-1] && curr_layer < NUM_LAYERS) {
            //     toNextLayer(intersections_large_local, trunk_length_local, curr_layer, isInside, out_local);
            // }
        }
        __syncthreads();
    }

    size_t remaining = num_small - (num_iters * THREADS_PER_BLOCK);

    if (threadIdx.x < remaining) {
        triangle t = tri_small[threadIdx.x + (num_iters * THREADS_PER_BLOCK)];
        tri_base[threadIdx.x] = t;
        // zMins_base[threadIdx.x] = zMins[threadIdx.x + (num_iters * THREADS_PER_BLOCK)];
        double yMin = min3(t.p1.y, t.p2.y, t.p3.y);
        double yMax = max3(t.p1.y, t.p2.y, t.p3.y);
        yNotInside[threadIdx.x] = (y_pos < yMin) || (y_pos > yMax);
    }
    __syncthreads();
    if (remaining) {
        if (y_idx < Y_DIM) {
            for (size_t tri_idx = 0; tri_idx < remaining; tri_idx++) {
                layer_t curr_intersection = yNotInside[tri_idx] ? NONE : pixelRayIntersection(tri_base[tri_idx], x, y);
                if (curr_intersection >= 0 && curr_intersection < NUM_LAYERS) out_local[curr_intersection]++;
            }
        }
    }

    // Process the remaining layers
    // while (curr_layer < NUM_LAYERS) {
    //     toNextLayer(intersections_large_local, trunk_length_local, curr_layer, isInside, out_local);
    // }
    // thrust::copy(thrust::device, &out_local[0], &out_local[0] + NUM_LAYERS, out_ptr);

    for (size_t i = 0; i < NUM_LAYERS; i++) {
        out_ptr[i*X_DIM*Y_DIM] = out_local[i];
    }
}

__global__
void layerExtraction(bool* out, layer_t start) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    bool isInside = false;
    char* out_ptr = (char*) (out + idx);
    char intersection_count;
    for (size_t i = start; i < NUM_LAYERS; i++) {
        intersection_count = out_ptr[i*X_DIM*Y_DIM];
        bool flip = (bool)(intersection_count & 1);
        bool intersect = (intersection_count > 0);
        out_ptr[i*X_DIM*Y_DIM] = (char) (isInside || intersect);
        isInside = isInside ^ flip;
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
 * get the array of intersections of a given pixel ray
 */
__device__
int getIntersectionTrunk(int x, int y, triangle* triangles, size_t num_triangles, layer_t* layers) {
    int idx = 0;

    for (int i = 0; i < num_triangles; i++) {
        layer_t layer = pixelRayIntersection(triangles[i], x, y);
        if (layer != -1) {
            layers[idx] = layer;
            idx++;
        }
    }
    return idx;
}

__device__ __forceinline__
void extractLayer(layer_t curr_layer, bool & isInside, char* out_local) {
    char total_intersections = out_local[curr_layer];
    bool flip = (bool) (total_intersections & 1);
    bool intersect = (total_intersections > 0);
    out_local[curr_layer] = (char) (isInside || intersect);
    isInside = isInside ^ flip;
}

__device__ __forceinline__
double min3(double a, double b, double c) {
    thrust::minimum<double> min;
    return min(a, min(b, c));
}

__device__ __forceinline__
double max3(double a, double b, double c) {
    thrust::maximum<double> max;
    return max(a, max(b, c));
}
