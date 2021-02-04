#include "slicer.cuh"
#include "triangle.cuh"
#include <thrust/functional.h>

__device__ __forceinline__ void triangleCopy(void* src, void* dest, int id);
__device__ __forceinline__ double min3(double a, double b, double c);
__device__ __forceinline__ double max3(double a, double b, double c);
__device__ __forceinline__ char atomicAdd(char* address, char val);
__device__ __forceinline__ layer_t pixelRayIntersection_point(double x1, double y1, double z1,
    double x2, double y2, double z2, double x3, double y3, double z3, int x, int y);

__global__ void rectTriIntersection(double* tri_global, size_t num_tri, bool* out) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    size_t num_per_thread = num_tri / (NUM_BLOCKS << LOG_THREADS) + 1;
    size_t base_idx = idx;

    double* x1_base = tri_global;
    double* y1_base = tri_global + num_tri;
    double* z1_base = tri_global + 2*num_tri;
    double* x2_base = tri_global + 3*num_tri;
    double* y2_base = tri_global + 4*num_tri;
    double* z2_base = tri_global + 5*num_tri;
    double* x3_base = tri_global + 6*num_tri;
    double* y3_base = tri_global + 7*num_tri;
    double* z3_base = tri_global + 8*num_tri;

    // iterate over all triangles assigned to this thread.
    for (size_t i = 0; i < num_per_thread; i++) {
        // Compute bounding box
        if (base_idx >= num_tri) break;
        double x1 = x1_base[base_idx];
        double y1 = y1_base[base_idx];
        double z1 = z1_base[base_idx];
        double x2 = x2_base[base_idx];
        double y2 = y2_base[base_idx];
        double z2 = z2_base[base_idx];
        double x3 = x3_base[base_idx];
        double y3 = y3_base[base_idx];
        double z3 = z3_base[base_idx];
        
        long xMin = (long)(min3(x1, x2, x3) / RESOLUTION);
        long yMin = (long)(min3(y1, y2, y3) / RESOLUTION);
        long xMax = __double2ll_ru(max3(x1, x2, x3) / RESOLUTION);
        long yMax = __double2ll_ru(max3(y1, y2, y3) / RESOLUTION);
        base_idx += (NUM_BLOCKS << LOG_THREADS);
        // Make sure the bounds are inside the supported space
        xMax = min(xMax, X_MAX);
        xMin = max(xMin, X_MIN);
        yMax = min(yMax, Y_MAX);
        yMin = max(yMin, Y_MIN);
        if (xMax < xMin || yMax < yMin) continue;
        // iterate over all pixels inside the bounding box
        // Will likely cause (lots of) wrap divergence, but we'll deal with that later
        int x = xMin;
        int y = yMin;
        while (y <= yMax) {
            layer_t curr_intersection = 
                pixelRayIntersection_point(x1, y1, z1, x2, y2, z2, x3, y3, z3, x, y);
            if (curr_intersection >= 0 && curr_intersection < NUM_LAYERS) {
                // Found a valid intersection
                int x_idx = x + (X_DIM >> 1);
                int y_idx = y + (Y_DIM >> 1);
                char* temp_ptr = (char*) (out + curr_intersection*X_DIM*Y_DIM + y_idx*X_DIM + x_idx);
                atomicAdd(temp_ptr, 1);
            }
            // update coords
            bool nextLine = (x == xMax);
            y += (int)nextLine;
            x = nextLine ? xMin : (x+1);
        }
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
layer_t pixelRayIntersection_point(double x1, double y1, double z1,
    double x2, double y2, double z2, double x3, double y3, double z3, int x, int y) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.
    */

    double x_pos = x * RESOLUTION;
    double y_pos = y * RESOLUTION;

    // double x_max = max3(x1, x2, x3);
    // double x_min = min3(x1, x2, x3);

    // if (x_pos < x_min || x_pos > x_max) return NONE;

    double x_d = x_pos - x1;
    double y_d = y_pos - y1;

    double xx1 = x2 - x1;
    double yy1 = y2 - y1;
    double zz1 = z2 - z1;

    double xx2 = x3 - x1;
    double yy2 = y3 - y1;
    double zz2 = z3 - z1;
    double a = (x_d * yy2 - xx2 * y_d) / (xx1 * yy2 - xx2 * yy1);
    double b = (x_d * yy1 - xx1 * y_d) / (xx2 * yy1 - xx1 * yy2);
    bool inside = (a >= 0) && (b >= 0) && (a+b <= 1);
    double intersection = (a * zz1 + b * zz2) + z1;
    // // divide by layer width
    layer_t layer = inside ? (intersection / RESOLUTION) : (layer_t)(-1);
    return layer;
}
 
// Copy (THREADS_PER_BLOCK) triangles from src to dest
// Achieves 100% memory efficiency
__device__ __forceinline__
void triangleCopy(void* src, void* dest, int id) {
    copy_unit_t* src_ptr = (copy_unit_t*) src;
    copy_unit_t* dest_ptr = (copy_unit_t*) dest;

    #pragma unroll
    for (int d = 0; d < unit_per_tri; d++) {
        size_t offset = d * THREADS_PER_BLOCK;
        dest_ptr[id + offset] = src_ptr[id + offset];
    }
}

__device__ __forceinline__
double min3(double a, double b, double c) {
    // thrust::minimum<double> min;
    return min(a, min(b, c));
}

__device__ __forceinline__
double max3(double a, double b, double c) {
    // thrust::maximum<double> max;
    return max(a, max(b, c));
}

__device__ __forceinline__
char atomicAdd(char* address, char val) {
    // *address = *address + val;
    // return 0;
    size_t addr_offset = (size_t) address & 3;
    auto* base_address = (unsigned int*) ((size_t) address - addr_offset);
    unsigned int long_val = (unsigned int) val << (8 * addr_offset);
    unsigned int long_old = atomicAdd(base_address, long_val);

    // Overflow check. skipped for simplicity.
    // if (addr_offset == 3) {
    //     return (char) (long_old >> 24);
    // } else {
    //     // bits that represent the char value within long_val
    //     unsigned int mask = 0x000000ff << (8 * addr_offset);
    //     unsigned int masked_old = long_old & mask;
    //     // isolate the bits that represent the char value within long_old, add the long_val to that,
    //     // then re-isolate by excluding bits that represent the char value
    //     unsigned int overflow = (masked_old + long_val) & ~mask;
    //     if (overflow) {
    //         atomicSub(base_address, overflow);
    //     }
    //     return (char) (masked_old >> 8 * addr_offset);
    // }

    return (char) ((long_old >> 8 * addr_offset) & 0xff);
}
