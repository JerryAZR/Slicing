#include <iostream>
#include <string>
#include "triangle.cuh"
#include "slicer.cuh"
#include <vector>


int main(int argc, char* argv[]) {
    std::string stl_file_name;
    std::vector<triangle> triangles;

    if (argc == 2) {
        stl_file_name = argv[1];
    } else if (argc > 2) {
        std::cout << "ERROR: Too many command line arguments" << std::endl;
    }

    read_stl(stl_file_name,triangles);
    int num_triangles = triangles.size();
    triangle* triangles_dev;
    // all[z][y][x]
    bool all[NUM_LAYERS][Y_DIM][X_DIM];
    bool* all_dev;
    size_t size = NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool);
    cudaMalloc(&all_dev, size);
    cudaMalloc(&triangles_dev, num_triangles * sizeof(triangle));
    cudaMemcpy(all_dev, &all[0][0][0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(triangles_dev, triangles.data(), num_triangles * sizeof(triangle), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_LAYERS * Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;

    pps<<<blocksPerGrid, threadsPerBlock>>>(&triangles_dev[0], triangles.size(), all_dev);
    cudaDeviceSynchronize();

    // Copy result from device memory to host memory
    cudaMemcpy(&all[0][0][0], all_dev, size, cudaMemcpyDeviceToHost);

    cudaFree(all_dev);
    cudaFree(triangles_dev);

    // Visualize
    for (int y = 200; y > 0; y--) {
        for (int x = 25; x < 175; x++) {
            if (all[8][y][x]) std::cout << "x";
            else std::cout << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}