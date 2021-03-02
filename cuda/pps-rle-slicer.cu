#include "slicer.cuh"
#include "triangle.cuh"
#include <thrust/functional.h>
#include <thread>
#include <chrono>
#define NOW (std::chrono::high_resolution_clock::now())
typedef std::chrono::time_point<std::chrono::high_resolution_clock> chrono_t;
#define XNONE INT_MIN

__device__ __forceinline__
int pixelRayIntersectionX(triangle t, int y, int z);
__device__ unsigned bubblesort(unsigned* start, unsigned length, unsigned step);

__global__
void pps(triangle* triangles_global, size_t num_triangles, unsigned* trunks, unsigned* trunk_length, unsigned base_layer) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("starting thread %d\n", idx);
    int z_idx = idx / Y_DIM;
    // if (y >= Y_DIM) return;
    int y_idx = idx % Y_DIM;
    int y = y_idx - (Y_DIM / 2);
    int z = z_idx + base_layer;

    // Copy triangles to shared memory
    // Each block has a shared memory storing some triangles.
    __shared__ triangle tri_base[THREADS_PER_BLOCK];
    triangle* triangles = (triangle*) tri_base;
    size_t num_iters = num_triangles / THREADS_PER_BLOCK;
    int length = 0;
    unsigned* xints = trunks + z_idx*MAX_TRUNK_SIZE*Y_DIM + y_idx;
    for (size_t i = 0; i < num_iters; i++) {
        triangles[threadIdx.x] = triangles_global[threadIdx.x + (i * THREADS_PER_BLOCK)];
        // Wait for other threads to complete;
        __syncthreads();
        if (z < NUM_LAYERS) {
            for (size_t tri_idx = 0; tri_idx < THREADS_PER_BLOCK; tri_idx++) {
                int intersection = pixelRayIntersectionX(triangles[tri_idx], y, z);
                if (intersection >= X_MIN && intersection <= X_MAX) {
                    xints[length*Y_DIM] = intersection - X_MIN;
                    length++;
                }
            }
        }
        __syncthreads();
    }
    size_t remaining = num_triangles - (num_iters * THREADS_PER_BLOCK);
    if (threadIdx.x < remaining) {
        triangles[threadIdx.x] = triangles_global[threadIdx.x + (num_iters * THREADS_PER_BLOCK)];
    }
    __syncthreads();
    if (remaining && z < NUM_LAYERS) {
        for (size_t tri_idx = 0; tri_idx < remaining; tri_idx++) {
            int intersection = pixelRayIntersectionX(triangles[tri_idx], y, z);
            if (intersection >= X_MIN && intersection <= X_MAX) {
                xints[length*Y_DIM] = intersection - X_MIN;
                length++;
            }
        }
    }
    trunk_length[idx] = length;
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
int pixelRayIntersectionX(triangle t, int y, int z) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.
    */

    double y_max = max(t.p1.y, max(t.p2.y, t.p3.y));
    double y_min = min(t.p1.y, min(t.p2.y, t.p3.y));
    double z_max = max(t.p1.z, max(t.p2.z, t.p3.z));
    double z_min = min(t.p1.z, min(t.p2.z, t.p3.z));

    double y_pos = y * RESOLUTION;
    double z_pos = z * RESOLUTION;
    if ((y_pos < y_min) || (y_pos > y_max) || (z_pos < z_min) || (z_pos > z_max)) return XNONE;

    double y_d = y_pos - t.p1.y;
    double z_d = z_pos - t.p1.z;

    double x1 = t.p2.x - t.p1.x;
    double y1 = t.p2.y - t.p1.y;
    double z1 = t.p2.z - t.p1.z;

    double x2 = t.p3.x - t.p1.x;
    double y2 = t.p3.y - t.p1.y;
    double z2 = t.p3.z - t.p1.z;

    double a = (y_d * z2 - y2 * z_d) / (y1 * z2 - y2 * z1);
    double b = (y_d * z1 - y1 * z_d) / (y2 * z1 - y1 * z2);
    bool inside = (a >= 0) && (b >= 0) && (a+b <= 1);
    double intersection = (a * x1 + b * x2) + t.p1.x;
    // // divide by layer width
    return inside ? (intersection / RESOLUTION) : XNONE;
}

__global__ 
void triangleSelect(triangle* in, triangle* out, unsigned in_length,
                    unsigned* out_length, unsigned base_layer)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t total_threads = blockDim.x * gridDim.x;
    double min_height = base_layer * RESOLUTION;
    size_t max_layers = (base_layer + PPS_BLOCK_HEIGHT);
    double max_height = max_layers * RESOLUTION;
    while (idx < in_length) {
        triangle t = in[idx];
        idx += total_threads;
        double z_min = min(t.p1.z, min(t.p2.z, t.p3.z));
        if (z_min > max_height) continue;
        double z_max = max(t.p1.z, max(t.p2.z, t.p3.z));
        if (z_max < min_height) continue;
        size_t curr_length = atomicAdd(out_length, 1);
        out[curr_length] = t;
    }
}
    
__global__ void trunk_compress(unsigned* trunks, unsigned* trunk_length, unsigned* out, unsigned max_length) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    size_t y_idx = idx % Y_DIM;
    size_t z_idx = idx / Y_DIM;
    unsigned length = trunk_length[idx];
    unsigned* trunk_base = out + idx*max_length;
    unsigned* input_trunk = trunks + z_idx*MAX_TRUNK_SIZE*Y_DIM + y_idx;
    unsigned out_length = 0;
    unsigned prev_idx = 0;

    bubblesort(input_trunk, length, Y_DIM);
    input_trunk[length*Y_DIM] = X_DIM;

    unsigned i = 0;
    // Manually process the first intersection to avoid problems
    trunk_base[out_length++] = input_trunk[0];
    prev_idx = input_trunk[0];
    i = 0;

    while (i < length) {
        // Find the next run of 1's
        i++;
        while ((input_trunk[i*Y_DIM] - input_trunk[(i-1)*Y_DIM] <= 1 || i & 1 == 1) && i < length) {
            i++;
        }
        __syncwarp();
        unsigned run_1s = input_trunk[(i-1)*Y_DIM] - prev_idx + 1;
        unsigned run_0s = (i == length) ?
                X_DIM - input_trunk[(i-1)*Y_DIM] - 1 : input_trunk[i*Y_DIM] - input_trunk[(i-1)*Y_DIM] - 1;
        prev_idx = input_trunk[i*Y_DIM];
        trunk_base[out_length++] = run_1s;
        trunk_base[out_length++] = run_0s;
    }
    trunk_base[out_length] = 0;
}

// single thread ver
void rleDecodeSt(unsigned* in, bool* out, unsigned num_trunks, unsigned max_length) {
    for (unsigned y = 0; y < num_trunks; y++) {
        unsigned* in_base = in + (y*max_length);
        bool* out_base = out + (y*X_DIM);
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

// Returns the running time
double rleDecode(unsigned* in, bool* out, unsigned nlayers, unsigned max_length) {
    chrono_t start = NOW;
    
    unsigned num_trunks = nlayers * Y_DIM;
    unsigned num_per_thread = (num_trunks + NUM_CPU_THREADS - 1) / NUM_CPU_THREADS;
    std::thread threads[NUM_CPU_THREADS];
    size_t in_offset = 0;
    size_t out_offset = 0;
    for (unsigned i = 0; i < NUM_CPU_THREADS-1; i++) {
        unsigned* thread_in = in + in_offset;
        bool* thread_out = out + out_offset;
        threads[i] = std::thread(rleDecodeSt, thread_in, thread_out, num_per_thread, max_length);
        in_offset += (num_per_thread*max_length);
        out_offset += (num_per_thread*X_DIM);
    }
    unsigned remaining = num_trunks - ((NUM_CPU_THREADS-1)*num_per_thread);
    unsigned* thread_in = in + in_offset;
    bool* thread_out = out + out_offset;
    threads[NUM_CPU_THREADS-1] = std::thread(rleDecodeSt, thread_in, thread_out, remaining, max_length);
    for (unsigned i = 0; i < NUM_CPU_THREADS; i++) {
        threads[i].join();
    }

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(NOW - start);
    double ms = (double)duration.count() / 1e6;
    return ms;
}

__device__ unsigned bubblesort(unsigned* start, unsigned length, unsigned step) {
    for (unsigned unsorted = length; unsorted > 1; unsorted--) {
        for (unsigned i = 1; i < unsorted; i++) {
            unsigned curr_idx = i*step;
            unsigned prev_idx = curr_idx-step;
            unsigned curr = start[curr_idx];
            unsigned prev = start[prev_idx];
            if (curr < prev) {
                start[curr_idx] = prev;
                start[prev_idx] = curr;
            }
        }
    }
    return length;
}
