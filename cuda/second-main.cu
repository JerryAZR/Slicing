#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include "triangle.cuh"
#include "slicer.cuh"
#include "golden.cuh"

using std::pair;

int main(int argc, char* argv[]) {
    std::string stl_file_name;
    std::vector<triangle> small_tri;

    if (argc == 2) {
        stl_file_name = argv[1];
    } else if (argc > 2) {
        std::cout << "ERROR: Too many command line arguments" << std::endl;
        return 0;
    } else {
        std::cout << "ERROR: Too few command line arguments" << std::endl;
        return 0;
    }

    read_stl(stl_file_name, small_tri);
    size_t num_small = small_tri.size();

    triangle* small_tri_dev;
    cudaMalloc(&small_tri_dev, num_small * sizeof(triangle));
    cudaMemcpy(small_tri_dev, small_tri.data(), num_small * sizeof(triangle), cudaMemcpyHostToDevice);

    layer_t* intersections_large;
    cudaMalloc(&intersections_large, Y_DIM * X_DIM * NUM_LAYERS * sizeof(layer_t));

    size_t* trunk_length;
    cudaMalloc(&trunk_length, Y_DIM * X_DIM * sizeof(size_t));

    // out[y][x][z]
    bool* out = (bool*)malloc(NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool));
    bool* out_dev;
    cudaMalloc(&out_dev, Y_DIM * X_DIM * NUM_LAYERS * sizeof(bool));
    cudaMemset(out_dev, 0, Y_DIM * X_DIM * NUM_LAYERS * sizeof(bool));

    double* z_mins_dev;
    cudaMalloc(&z_mins_dev, num_small * sizeof(double));

    int threadsPerBlock = THREADS_PER_BLOCK;
    int numBlocks;

    numBlocks = (Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;

    GPUsort(small_tri_dev, num_small, z_mins_dev);

    cudaDeviceSynchronize();

    smallTriIntersection<<<numBlocks, threadsPerBlock>>>(small_tri_dev, z_mins_dev, num_small, out_dev);

    cudaDeviceSynchronize();

    cudaMemcpy(out, out_dev, Y_DIM * X_DIM * NUM_LAYERS * sizeof(bool), cudaMemcpyDeviceToHost);

    free(out);
    cudaFree(small_tri_dev);
    cudaFree(z_mins_dev);
    cudaFree(out_dev);

    return 0;
}