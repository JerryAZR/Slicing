#include "slicer.cuh"
#include "triangle.cuh"
#include <thrust/functional.h>
#include <thread>
#include <chrono>
#define NOW (std::chrono::high_resolution_clock::now())
typedef std::chrono::time_point<std::chrono::high_resolution_clock> chrono_t;

__device__ __forceinline__ void triangleCopy(void* src, void* dest, int id);
__device__ __forceinline__ double min3(double a, double b, double c);
__device__ __forceinline__ double max3(double a, double b, double c);
__device__ __forceinline__ char atomicAdd(char* address, char val);
__device__ __forceinline__ int pixelRayIntersection_point(double x1, double y1, double z1,
    double x2, double y2, double z2, double x3, double y3, double z3, int x, int y);

__global__ void rectTriIntersection(double* tri_global, size_t num_tri, unsigned* trunks, unsigned* trunk_length, unsigned base_layer) {
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
        
        long yMin = __double2ll_ru(min3(y1, y2, y3) / RESOLUTION);
        long zMin = __double2ll_ru(min3(z1, z2, z3) / RESOLUTION);
        long yMax = __double2ll_rd(max3(y1, y2, y3) / RESOLUTION);
        long zMax = __double2ll_rd(max3(z1, z2, z3) / RESOLUTION);
        base_idx += (NUM_BLOCKS << LOG_THREADS);
        // Make sure the bounds are inside the supported space
        yMax = min(yMax, Y_MAX);
        yMin = max(yMin, Y_MIN);
        long zMax_ub = min(NUM_LAYERS-1, (long)(base_layer+BLOCK_HEIGHT-1));
        zMax = min(zMax, zMax_ub);
        zMin = max(zMin, (long)(base_layer));
        if (yMax < yMin || zMax < zMin) continue;
        // iterate over all pixels inside the bounding box
        // Will likely cause (lots of) wrap divergence, but we'll deal with that later
        int y = yMin;
        int z = zMin;
        while (z <= zMax) {
            int curr_intersection = 
                pixelRayIntersection_point(x1, y1, z1, x2, y2, z2, x3, y3, z3, y, z);
            if (curr_intersection >= X_MIN && curr_intersection <= X_MAX) {
                // Found a valid intersection
                int y_idx = y + (Y_DIM >> 1);
                unsigned x_idx = curr_intersection + (X_DIM >> 1);
                // Add current intersection to trunk
                unsigned* trunk_base = trunks + (z-base_layer)*Y_DIM*MAX_TRUNK_SIZE + y_idx;
                unsigned* length_address = trunk_length + (z-base_layer)*Y_DIM + y_idx;
                unsigned curr_length = atomicAdd(length_address, 1);
                // Need to check if out of range
                if (curr_length >= MAX_TRUNK_SIZE) 
                    printf("Error: Too many intersections.\n \
                            Please increase MAX_TRUNK_SIZE in slicer.cuh and recompile.\n");
                trunk_base[curr_length*Y_DIM] = x_idx;
            }
            // update coords
            bool nextLine = (y == yMax);
            z += (int)nextLine;
            y = nextLine ? yMin : (y+1);
        }
    }
}

__global__ void trunk_compress(unsigned* trunks, unsigned* trunk_length, unsigned* out) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    size_t y_idx = idx % Y_DIM;
    size_t z_idx = idx / Y_DIM;
    unsigned length = trunk_length[idx];
    unsigned* trunk_base = out + idx*MAX_TRUNK_SIZE;
    unsigned out_length = 0;
    unsigned prev_idx = 0;

    unsigned input_trunk[MAX_TRUNK_SIZE];
    for (unsigned i = 0; i < length; i++) {
        input_trunk[i] = *(trunks + z_idx*MAX_TRUNK_SIZE*Y_DIM + i*Y_DIM + y_idx);
    }
    __syncthreads();
    thrust::sort(thrust::device, input_trunk, input_trunk + length);
    if (length < MAX_TRUNK_SIZE) input_trunk[length] = X_DIM;

    unsigned i = 0;
    // Manually process the first intersection to avoid problems
    trunk_base[out_length++] = input_trunk[0];
    prev_idx = input_trunk[0];
    i = 0;

    while (i < length) {
        // Find the next run of 1's
        i++;
        while ((input_trunk[i] - input_trunk[i-1] <= 1 || i & 1 == 1) && i < length) {
            i++;
        }
        unsigned run_1s = input_trunk[i-1] - prev_idx + 1;
        unsigned run_0s = (i == length) ?
                X_DIM - input_trunk[i-1] - 1 : input_trunk[i] - input_trunk[i-1] - 1;
        prev_idx = input_trunk[i];
        trunk_base[out_length++] = run_1s;
        trunk_base[out_length++] = run_0s;
    }
    if (out_length < MAX_TRUNK_SIZE) trunk_base[out_length] = 0;
}

// single thread ver
void bbox_ints_decompress_st(unsigned* in, bool* out, unsigned nlayers) {
    for (unsigned z = 0; z < nlayers; z++) {
        for (unsigned y = 0; y < Y_DIM; y++) {
            // if (z == 1) printf("STarting layer %d, row %d.\n", z, y);
            unsigned* in_base = in + (z*Y_DIM*MAX_TRUNK_SIZE + y*MAX_TRUNK_SIZE);
            bool* out_base = out + (z*Y_DIM*X_DIM + y*X_DIM);
            bool inside = false;
            unsigned start = 0;
            unsigned length;
            for (unsigned idx = 0; in_base[idx] != 0; idx++) {
                length = in_base[idx];
                memset(out_base+start, inside, length);
                inside = !inside;
                start += length;
            }
        }
    }
}

// Returns the running time
double bbox_ints_decompress(unsigned* in, bool* out, unsigned nlayers) {
    chrono_t start = NOW;
    
    unsigned num_per_thread = (nlayers + NUM_CPU_THREADS - 1) / NUM_CPU_THREADS;
    std::thread threads[NUM_CPU_THREADS];
    size_t in_offset = 0;
    size_t out_offset = 0;
    for (unsigned i = 0; i < NUM_CPU_THREADS-1; i++) {
        unsigned* thread_in = in + in_offset;
        bool* thread_out = out + out_offset;
        threads[i] = std::thread(bbox_ints_decompress_st, thread_in, thread_out, num_per_thread);
        in_offset += (num_per_thread*X_DIM*MAX_TRUNK_SIZE);
        out_offset += (num_per_thread*X_DIM*Y_DIM);
    }
    unsigned remaining = nlayers - ((NUM_CPU_THREADS-1)*num_per_thread);
    unsigned* thread_in = in + in_offset;
    bool* thread_out = out + out_offset;
    threads[NUM_CPU_THREADS-1] = std::thread(bbox_ints_decompress_st, thread_in, thread_out, remaining);
    for (unsigned i = 0; i < NUM_CPU_THREADS; i++) {
        threads[i].join();
    }

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(NOW - start);
    double ms = (double)duration.count() / 1e6;
    return ms;
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
    double x2, double y2, double z2, double x3, double y3, double z3, int y, int z) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.
    */

    double y_pos = y * RESOLUTION;
    double z_pos = z * RESOLUTION;

    double y_d = y_pos - y1;
    double z_d = z_pos - z1;

    double xx1 = x2 - x1;
    double yy1 = y2 - y1;
    double zz1 = z2 - z1;

    double xx2 = x3 - x1;
    double yy2 = y3 - y1;
    double zz2 = z3 - z1;
    double a = (y_d * zz2 - yy2 * z_d) / (yy1 * zz2 - yy2 * zz1);
    double b = (y_d * zz1 - yy1 * z_d) / (yy2 * zz1 - yy1 * zz2);
    bool inside = (a >= 0) && (b >= 0) && (a+b <= 1);
    double intersection = (a * xx1 + b * xx2) + x1;
    // // divide by layer width
    int layer = inside ? (intersection / RESOLUTION) : INT_MIN;
    return layer;
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
