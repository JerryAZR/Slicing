#include "golden.cuh"
#include "slicer.cuh"
#include "triangle.cuh"
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <stdio.h>
#include <fstream>

// Declare local helper functions
__global__ void _RTIntersection(triangle* tri_small, size_t num_small, bool* out);
__global__ void _layerExtraction(bool* out);
__device__ __forceinline__ void _toNextLayer(layer_t* intersections_large_local, 
    size_t trunk_length_local, layer_t & curr_layer, bool & isInside, char* out_local);
__device__ __forceinline__ layer_t _pixelRayIntersection(triangle t, int x, int y);
__device__ __forceinline__ void _extractLayer(layer_t curr_layer, bool & isInside, char* out_local);
__device__ __forceinline__ double _min3(double a, double b, double c);
__device__ __forceinline__ double _max3(double a, double b, double c);


long checkOutput(triangle* triangles_dev, size_t num_triangles, bool* in) {
    bool* expected = (bool*) malloc(NUM_LAYERS * X_DIM * Y_DIM * sizeof(bool));
    std::cout << "executing golden model" << std::endl;
    goldenModel(triangles_dev, num_triangles, &expected[0]);
    long size = NUM_LAYERS * Y_DIM * X_DIM;
    long diff = 0;
    long inside = 0;
    long real = 0;
    std::cout << "comparing results" << std::endl;
    for (int i = 0; i < size; i++) {
        inside += expected[i];
        real += in[i];
        diff += (expected[i] != in[i]);
    }

    std::ofstream outfile;
    outfile.open("expected.txt");
    for (int z = 0; z < NUM_LAYERS; z++) {
        for (int y = Y_DIM-1; y >= 0; y--) {
            for (int x = 0; x < X_DIM; x++) {
                if (expected[z*X_DIM*Y_DIM + y*X_DIM + x]) outfile << "XX";
                else outfile << "  ";
            }
            outfile << "\n";
        }
        outfile << "\n\n";
    }
    outfile.close();

    free(expected);
    std::cout << inside << " pixels are inside the model." << std::endl;
    std::cout << real << " pixels are inside the actual output model." << std::endl;
    std::cout << diff << " pixels are different in the expected and actual output." << std::endl;
    return diff;
}

void goldenModel(triangle* triangles_dev, size_t num_triangles, bool* out) {
    size_t threadsPerBlock = THREADS_PER_BLOCK;
    size_t blocksPerGrid;
    bool* all_dev;
    size_t size = NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool);
    cudaMalloc(&all_dev, size);

    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    blocksPerGrid = (Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;
    _RTIntersection<<<blocksPerGrid, threadsPerBlock>>>(triangles_dev, num_triangles, all_dev);
    cudaDeviceSynchronize();
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    blocksPerGrid = (X_DIM * Y_DIM + threadsPerBlock - 1) / threadsPerBlock;
    _layerExtraction<<<blocksPerGrid, threadsPerBlock>>>(all_dev);
    cudaDeviceSynchronize();
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    cudaMemcpy(out, all_dev, size, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    cudaFree(all_dev);
}

__global__ 
void _RTIntersection(triangle* tri_small, size_t num_small, bool* out) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    int x_idx = idx & (X_DIM-1);
    int y_idx = idx / X_DIM;
    int x = x_idx - (X_DIM >> 1);
    int y = y_idx - (Y_DIM >> 1);

    __shared__ triangle tri_base[THREADS_PER_BLOCK];
    __shared__ bool yNotInside[THREADS_PER_BLOCK];

    // Use local array. Mapped to registers if NUM_LAYERS is small
    char out_local[NUM_LAYERS] = {0};
    char* out_ptr = (char*)(out + idx);

    size_t num_iters = num_small / THREADS_PER_BLOCK;

    double y_pos = y * RESOLUTION;
    // double x_pos = x * RESOLUTION;

    for (size_t i = 0; i < num_iters; i++) {
        triangle t = tri_small[threadIdx.x + (i * THREADS_PER_BLOCK)];
        tri_base[threadIdx.x] = t;
        double yMin = _min3(t.p1.y, t.p2.y, t.p3.y);
        double yMax = _max3(t.p1.y, t.p2.y, t.p3.y);
        yNotInside[threadIdx.x] = (y_pos < yMin) || (y_pos > yMax);
        // Wait for other threads to complete;
        __syncthreads();
        if (y_idx < Y_DIM) {
            for (size_t tri_idx = 0; tri_idx < THREADS_PER_BLOCK; tri_idx++) {
                if (yNotInside[tri_idx]) continue;
                layer_t curr_intersection = _pixelRayIntersection(tri_base[tri_idx], x, y);
                if (curr_intersection >= 0 && curr_intersection < NUM_LAYERS) out_local[curr_intersection]++;
            }
        }
        __syncthreads();
    }

    size_t remaining = num_small - (num_iters * THREADS_PER_BLOCK);

    if (threadIdx.x < remaining) {
        triangle t = tri_small[threadIdx.x + (num_iters * THREADS_PER_BLOCK)];
        tri_base[threadIdx.x] = t;
        double yMin = _min3(t.p1.y, t.p2.y, t.p3.y);
        double yMax = _max3(t.p1.y, t.p2.y, t.p3.y);
        yNotInside[threadIdx.x] = (y_pos < yMin) || (y_pos > yMax);
    }
    __syncthreads();
    if (remaining) {
        if (y_idx < Y_DIM) {
            for (size_t tri_idx = 0; tri_idx < remaining; tri_idx++) {
                if (yNotInside[tri_idx]) continue;
                layer_t curr_intersection = _pixelRayIntersection(tri_base[tri_idx], x, y);
                if (curr_intersection >= 0 && curr_intersection < NUM_LAYERS) out_local[curr_intersection]++;
            }
        }
    }

    for (size_t i = 0; i < NUM_LAYERS; i++) {
        out_ptr[i*X_DIM*Y_DIM] = out_local[i];
    }
}

__global__
void _layerExtraction(bool* out) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    bool isInside = false;
    char* out_ptr = (char*) (out + idx);
    char intersection_count;
    for (size_t i = 0; i < NUM_LAYERS; i++) {
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
layer_t _pixelRayIntersection(triangle t, int x, int y) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.
    */

    double x_max = _max3(t.p1.x, t.p2.x, t.p3.x);
    double x_min = _min3(t.p1.x, t.p2.x, t.p3.x);
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

__device__ __forceinline__
void _extractLayer(layer_t curr_layer, bool & isInside, char* out_local) {
    char total_intersections = out_local[curr_layer];
    bool flip = (bool) (total_intersections & 1);
    bool intersect = (total_intersections > 0);
    out_local[curr_layer] = (char) (isInside || intersect);
    isInside = isInside ^ flip;
}

__device__ __forceinline__
double _min3(double a, double b, double c) {
    thrust::minimum<double> min;
    return min(a, min(b, c));
}

__device__ __forceinline__
double _max3(double a, double b, double c) {
    thrust::maximum<double> max;
    return max(a, max(b, c));
}
