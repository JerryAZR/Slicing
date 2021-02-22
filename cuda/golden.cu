#include "golden.cuh"
#include "slicer.cuh"
#include "triangle.cuh"
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <stdio.h>
#include <fstream>

#define GOLDEN_BLOCK_SIZE 256
#define GOLDEN_GRID_SIZE 256

// Declare local helper functions
__global__ void _RTIntersection(triangle* tri_small, size_t num_small, bool* out);
__global__ void _layerExtraction(bool* out);
__device__ __forceinline__ void _toNextLayer(layer_t* intersections_large_local, 
    size_t trunk_length_local, layer_t & curr_layer, bool & isInside, char* out_local);
__device__ __forceinline__ layer_t _pixelRayIntersection(triangle t, int x, int y);
__device__ __forceinline__ void _extractLayer(layer_t curr_layer, bool & isInside, char* out_local);
__device__ __forceinline__ double _min3(double a, double b, double c);
__device__ __forceinline__ double _max3(double a, double b, double c);
__device__ __forceinline__ char _atomicAdd(char* address, char val);


long checkOutput(triangle* triangles_dev, size_t num_triangles, bool* in) {
    bool* expected = (bool*) malloc(NUM_LAYERS * X_DIM * Y_DIM * sizeof(bool));
    std::cout << "executing golden model" << std::endl;
    goldenModel(triangles_dev, num_triangles, &expected[0]);
    size_t size = NUM_LAYERS * Y_DIM * X_DIM;
    long diff = 0;
    long inside = 0;
    long real = 0;
    std::cout << "comparing results" << std::endl;
    for (size_t i = 0; i < size; i++) {
        inside += expected[i];
        real += in[i];
        diff += (expected[i] != in[i]);
    }

    // std::ofstream outfile;
    // outfile.open("expected.txt");
    // for (int z = 0; z < NUM_LAYERS; z++) {
    //     for (int y = Y_DIM-1; y >= 0; y--) {
    //         for (int x = 0; x < X_DIM; x++) {
    //             if (expected[z*X_DIM*Y_DIM + y*X_DIM + x]) outfile << "XX";
    //             else outfile << "  ";
    //         }
    //         outfile << "\n";
    //     }
    //     outfile << "\n\n";
    // }
    // outfile.close();

    free(expected);
    std::cout << inside << " pixels are inside the model." << std::endl;
    std::cout << real << " pixels are inside the actual output model." << std::endl;
    std::cout << diff << " pixels are different in the expected and actual output." << std::endl;
    return diff;
}

void goldenModel(triangle* triangles_dev, size_t num_triangles, bool* out) {
    size_t threadsPerBlock = GOLDEN_BLOCK_SIZE;
    size_t blocksPerGrid;
    bool* all_dev;
    size_t size = NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool);
    cudaMalloc(&all_dev, size);

    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    _RTIntersection<<<GOLDEN_GRID_SIZE, GOLDEN_BLOCK_SIZE>>>
        (triangles_dev, num_triangles, all_dev);
    cudaDeviceSynchronize();
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) 
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
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
void _RTIntersection(triangle* tri_base, size_t num_tri, bool* out) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    size_t num_per_thread = num_tri / (GOLDEN_BLOCK_SIZE * GOLDEN_GRID_SIZE) + 1;
    size_t base_idx = idx;

    // iterate over all triangles assigned to this thread.
    for (size_t i = 0; i < num_per_thread; i++) {
        // Compute bounding box
        if (base_idx >= num_tri) break;
        triangle t = tri_base[base_idx];
        double x1 = t.p1.x;
        double y1 = t.p1.y;
        // double z1 = t.p1.z;
        double x2 = t.p2.x;
        double y2 = t.p2.y;
        // double z2 = t.p2.z;
        double x3 = t.p3.x;
        double y3 = t.p3.y;
        // double z3 = t.p3.z;
        
        long xMin = __double2ll_ru(_min3(x1, x2, x3) / RESOLUTION);
        long yMin = __double2ll_ru(_min3(y1, y2, y3) / RESOLUTION);
        long xMax = __double2ll_rd(_max3(x1, x2, x3) / RESOLUTION);
        long yMax = __double2ll_rd(_max3(y1, y2, y3) / RESOLUTION);
        base_idx += (GOLDEN_BLOCK_SIZE * GOLDEN_GRID_SIZE);
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
                _pixelRayIntersection(t, x, y);
            if (curr_intersection >= 0 && curr_intersection < NUM_LAYERS) {
                // Found a valid intersection
                int x_idx = x + (X_DIM >> 1);
                int y_idx = y + (Y_DIM >> 1);
                char* temp_ptr = (char*) (out + curr_intersection*X_DIM*Y_DIM + y_idx*X_DIM + x_idx);
                _atomicAdd(temp_ptr, 1);
            }
            // update coords
            bool nextLine = (x == xMax);
            y += (int)nextLine;
            x = nextLine ? xMin : (x+1);
        }
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

__device__ __forceinline__
char _atomicAdd(char* address, char val) {
    // *address = *address + val;
    // return 0;
    size_t addr_offset = (size_t) address & 3;
    auto* base_address = (unsigned int*) ((size_t) address - addr_offset);
    unsigned int long_val = (unsigned int) val << (8 * addr_offset);
    unsigned int long_old = atomicAdd(base_address, long_val);

    // Overflow check. skipped for simplicity.

    return (char) ((long_old >> 8 * addr_offset) & 0xff);
}
