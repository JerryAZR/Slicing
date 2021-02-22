#include "slicer.cuh"
#include "triangle.cuh"
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/copy.h>

// Declare local helper functions
__device__ __forceinline__ void toNextLayer(layer_t & curr_layer, bool & isInside, char* out_local);

__device__ __forceinline__ void triangleCopy(void* src, void* dest, int id);
__device__ __forceinline__ double min3(double a, double b, double c);
__device__ __forceinline__ double max3(double a, double b, double c);

__global__
void smallTriIntersection(triangle* tri_small, double* zMins, size_t num_small, bool* out) {

    // out[y][x][z]
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    int x_idx = idx & (X_DIM-1);
    int y_idx = idx / X_DIM;
    int x = x_idx - (X_DIM >> 1);
    int y = y_idx - (Y_DIM >> 1);

    __shared__ triangle tri_base[THREADS_PER_BLOCK];
    __shared__ double zMins_base[THREADS_PER_BLOCK];
    // __shared__ double xMin[THREADS_PER_BLOCK];
    // __shared__ double xMax[THREADS_PER_BLOCK];
    __shared__ bool yNotInside[THREADS_PER_BLOCK];

    // Use local array. Mapped to registers if NUM_LAYERS is small
    char out_local[NUM_LAYERS] = {0};
    char* out_ptr = (char*)(out + idx);
    layer_t curr_layer = 0;
    // This flag only applies to pixels that are not intersections.
    bool isInside = false;

    size_t num_iters = num_small / THREADS_PER_BLOCK;

    double y_pos = y * RESOLUTION;
    // double x_pos = x * RESOLUTION;

    for (size_t i = 0; i < num_iters; i++) {
        //triangle t = tri_small[threadIdx.x + (i * THREADS_PER_BLOCK)];
        //tri_base[threadIdx.x] = t;
        size_t global_offset = i * THREADS_PER_BLOCK;// * unit_per_tri;
        triangleCopy(tri_small + global_offset, tri_base, threadIdx.x);
        __syncthreads();
        triangle t = tri_base[threadIdx.x];

        zMins_base[threadIdx.x] = zMins[threadIdx.x + (i * THREADS_PER_BLOCK)];
        double yMin = min3(t.p1.y, t.p2.y, t.p3.y);
        double yMax = max3(t.p1.y, t.p2.y, t.p3.y);
        yNotInside[threadIdx.x] = (y_pos < yMin) || (y_pos > yMax);
        // Wait for other threads to complete;
        __syncthreads();
        if (y_idx < Y_DIM) {
            for (size_t tri_idx = 0; tri_idx < THREADS_PER_BLOCK; tri_idx++) {
                if (yNotInside[tri_idx]) continue;
                layer_t curr_intersection = pixelRayIntersection(tri_base[tri_idx], x, y);
                if (curr_intersection >= 0 && curr_intersection < NUM_LAYERS) out_local[curr_intersection]++;
            }
            // Move to the next triangle-layer pair that intersects
            // Add 1 to curr_layer when comparing to avoid rounding issues.
            while ((curr_layer+1)*RESOLUTION < zMins_base[THREADS_PER_BLOCK-1] && curr_layer < NUM_LAYERS) {
                toNextLayer(curr_layer, isInside, out_local);
            }
        }
        __syncthreads();
    }

    size_t remaining = num_small - (num_iters * THREADS_PER_BLOCK);

    if (threadIdx.x < remaining) {
        triangle t = tri_small[threadIdx.x + (num_iters * THREADS_PER_BLOCK)];
        tri_base[threadIdx.x] = t;

        zMins_base[threadIdx.x] = zMins[threadIdx.x + (num_iters * THREADS_PER_BLOCK)];
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
    while (curr_layer < NUM_LAYERS) {
        toNextLayer(curr_layer, isInside, out_local);
    }
    // thrust::copy(thrust::device, &out_local[0], &out_local[0] + NUM_LAYERS, out_ptr);

    for (size_t i = 0; i < NUM_LAYERS; i++) {
        out_ptr[i*X_DIM*Y_DIM] = out_local[i];
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
    double x_max = max3(t.p1.x, t.p2.x, t.p3.x);
    double x_min = min3(t.p1.x, t.p2.x, t.p3.x);
    double x_pos = x * RESOLUTION;
    double y_pos = y * RESOLUTION;

    if (x_pos < x_min || x_pos > x_max) return NONE;

    double x_d = x_pos - t.p1.x;
    double y_d = y_pos - t.p1.y;

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

__global__
void getZMin(triangle* tris, size_t size, double* zMins) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    thrust::minimum<double> min;
    zMins[i] = min(tris[i].p1.z, min(tris[i].p2.z, tris[i].p3.z));
}

__host__
void GPUsort(triangle* tris_dev, size_t size, double* zMins) {    
    int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    getZMin<<<num_blocks, THREADS_PER_BLOCK>>>(tris_dev, size, zMins);
    cudaDeviceSynchronize();

    thrust::sort_by_key(thrust::device, zMins, zMins + size, tris_dev);
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
void toNextLayer(layer_t & curr_layer, bool & isInside, char* out_local) {
    char total_intersections = out_local[curr_layer];
    bool flip = (bool) (total_intersections & 1);
    bool intersect = (total_intersections > 0);
    out_local[curr_layer] = (char) (isInside || intersect);
    isInside = isInside ^ flip;
    curr_layer++;
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
