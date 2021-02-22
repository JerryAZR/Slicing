#include "slicer.cuh"
#include "triangle.cuh"
#include <thrust/functional.h>
#include <thread>

__device__ __forceinline__ void triangleCopy(void* src, void* dest, int id);
__device__ __forceinline__ double min3(double a, double b, double c);
__device__ __forceinline__ double max3(double a, double b, double c);
__device__ __forceinline__ char atomicAdd(char* address, char val);
__device__ __forceinline__ int pixelRayIntersection_point(double x1, double y1, double z1,
    double x2, double y2, double z2, double x3, double y3, double z3, int x, int y);

__global__ void rectTriIntersection(double* tri_global, size_t num_tri, bool* out, unsigned base_layer) {
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
        
        long xMin = __double2ll_ru(min3(x1, x2, x3) / RESOLUTION);
        long zMin = __double2ll_ru(min3(z1, z2, z3) / RESOLUTION);
        long xMax = __double2ll_rd(max3(x1, x2, x3) / RESOLUTION);
        long zMax = __double2ll_rd(max3(z1, z2, z3) / RESOLUTION);
        base_idx += (NUM_BLOCKS << LOG_THREADS);
        // Make sure the bounds are inside the supported space
        xMax = min(xMax, X_MAX);
        xMin = max(xMin, X_MIN);
        long zMax_ub = min(NUM_LAYERS-1, (long)(base_layer+BLOCK_HEIGHT-1));
        zMax = min(zMax, zMax_ub);
        zMin = max(zMin, (long)(base_layer));
        if (xMax < xMin || zMax < zMin) continue;
        // iterate over all pixels inside the bounding box
        // Will likely cause (lots of) wrap divergence, but we'll deal with that later
        int x = xMin;
        int z = zMin;
        while (z <= zMax) {
            int curr_intersection = 
                pixelRayIntersection_point(x1, y1, z1, x2, y2, z2, x3, y3, z3, x, z);
            if (curr_intersection >= Y_MIN && curr_intersection <= Y_MAX) {
                // Found a valid intersection
                int x_idx = x + (X_DIM >> 1);
                int y_idx = curr_intersection + (Y_DIM >> 1);
                // replace by adding current intersection to trunk
                char* temp_ptr = (char*) (out + (z-base_layer)*X_DIM*Y_DIM + y_idx*X_DIM + x_idx);
                atomicAdd(temp_ptr, 1);
            }
            // update coords
            bool nextLine = (x == xMax);
            z += (int)nextLine;
            x = nextLine ? xMin : (x+1);
        }
    }
}

// Output is the indices at which the state flips
__global__ void bbox_ints(bool* in, unsigned* out) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    size_t x_idx = idx % X_DIM;
    size_t z_idx = idx / X_DIM;
    bool isInside = false;
    char* in_ptr = (char*) (in);
    char intersection_count;
    bool curr = false;
    bool prev = false;
    unsigned out_length = 0;
    unsigned* out_base = out + (idx * MAX_FLIPS);

    for (size_t y = 0; y < Y_DIM; y++) {
        prev = curr;
        size_t full_idx = z_idx*X_DIM*Y_DIM + y*X_DIM + x_idx;
        intersection_count = in_ptr[full_idx];
        bool flip = (bool)(intersection_count & 1);
        bool intersect = (intersection_count > 0);
        curr = (isInside || intersect);
        isInside = isInside ^ flip;
        if (prev != curr) {
            out_base[out_length] = y;
            out_length++;
        }
    }
    if (out_length > MAX_FLIPS) printf("Too many flips.\nPlease change MAX_FLIPS and try again.\n");
    else if (out_length < MAX_FLIPS) out_base[out_length] = Y_DIM;
}


// single thread ver
void bbox_ints_decompress_st(unsigned* in, bool* out, unsigned nlayers) {
    for (unsigned z = 0; z < nlayers; z++) {
        for (unsigned x = 0; x < X_DIM; x++) {
            unsigned* in_base = in + (z*X_DIM*MAX_FLIPS + x*MAX_FLIPS);
            bool* out_base = out + (z*Y_DIM*X_DIM + x);
            unsigned flip_idx = 0;
            bool inside = false;
            for (unsigned y = 0; y < Y_DIM; y++) {
                if (in_base[flip_idx] == y) {
                    inside = !inside;
                    flip_idx = (flip_idx + 1) & (MAX_FLIPS - 1);
                }
                out_base[y*X_DIM] = inside;
            }
        }
    }
}

void bbox_ints_decompress(unsigned* in, bool* out) {
    unsigned num_per_thread = (NUM_LAYERS + NUM_CPU_THREADS - 1) / NUM_CPU_THREADS;
    std::thread threads[NUM_CPU_THREADS];
    size_t in_offset = 0;
    size_t out_offset = 0;
    for (unsigned i = 0; i < NUM_CPU_THREADS-1; i++) {
        unsigned* thread_in = in + in_offset;
        bool* thread_out = out + out_offset;
        threads[i] = std::thread(bbox_ints_decompress_st, thread_in, thread_out, num_per_thread);
        in_offset += (num_per_thread*X_DIM*MAX_FLIPS);
        out_offset += (num_per_thread*X_DIM*Y_DIM);
    }
    unsigned remaining = NUM_LAYERS - ((NUM_CPU_THREADS-1)*num_per_thread);
    unsigned* thread_in = in + in_offset;
    bool* thread_out = out + out_offset;
    threads[NUM_CPU_THREADS-1] = std::thread(bbox_ints_decompress_st, thread_in, thread_out, remaining);
    for (unsigned i = 0; i < NUM_CPU_THREADS; i++) {
        threads[i].join();
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
int pixelRayIntersection_point(double x1, double y1, double z1,
    double x2, double y2, double z2, double x3, double y3, double z3, int x, int z) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.
    */

    double x_pos = x * RESOLUTION;
    double z_pos = z * RESOLUTION;

    // double x_max = max3(x1, x2, x3);
    // double x_min = min3(x1, x2, x3);

    // if (x_pos < x_min || x_pos > x_max) return NONE;

    double x_d = x_pos - x1;
    double z_d = z_pos - z1;

    double xx1 = x2 - x1;
    double yy1 = y2 - y1;
    double zz1 = z2 - z1;

    double xx2 = x3 - x1;
    double yy2 = y3 - y1;
    double zz2 = z3 - z1;
    double a = (x_d * zz2 - xx2 * z_d) / (xx1 * zz2 - xx2 * zz1);
    double b = (x_d * zz1 - xx1 * z_d) / (xx2 * zz1 - xx1 * zz2);
    bool inside = (a >= 0) && (b >= 0) && (a+b <= 1);
    double intersection = (a * yy1 + b * yy2) + y1;
    // // divide by layer width
    int layer = inside ? (intersection / RESOLUTION) : INT_MIN;
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
