#include <iostream>
#include <string>
#include "triangle.cuh"
#include "slicer.cuh"
#include "golden.cuh"
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
    //cudaMemcpy(all_dev, &all[0][0][0], size, cudaMemcpyHostToDevice); // unnecessary
    cudaMemcpy(triangles_dev, triangles.data(), num_triangles * sizeof(triangle), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid;

    blocksPerGrid = (Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;
    pps<<<blocksPerGrid, threadsPerBlock>>>(&triangles_dev[0], num_triangles, all_dev);
    cudaDeviceSynchronize();
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    // Copy result from device memory to host memory
    cudaMemcpy(&all[0][0][0], all_dev, size, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    checkOutput(triangles_dev, num_triangles, &all[0][0][0]);
    cudaFree(all_dev);
    cudaFree(triangles_dev);

    // for (int z = 0; z < NUM_LAYERS; z++) {
    //     for (int y = Y_DIM; y > 0; y--) {
    //         for (int x = 0; x < X_DIM; x++) {
    //             if (all[z][y][x]) std::cout << "XX";
    //             else std::cout << "  ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl << std::endl;
    // }

    return 0;
}
