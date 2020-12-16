#include <iostream>
#include <string>
#include "triangle.cuh"
#include "slicer.cuh"
#include "golden.cuh"
#include <vector>
#include <chrono>
#define NOW (std::chrono::high_resolution_clock::now())

typedef std::chrono::time_point<std::chrono::high_resolution_clock> chrono_t;

void timer_checkpoint(chrono_t & checkpoint) {
    chrono_t end = NOW;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - checkpoint);
    std::cout << duration.count() << "ms" << std::endl;
    checkpoint = end;
}

int main(int argc, char* argv[]) {
    std::string stl_file_name;
    std::vector<triangle> triangles;

    if (argc == 2) {
        stl_file_name = argv[1];
    } else if (argc > 2) {
        std::cout << "ERROR: Too many command line arguments" << std::endl;
    }

    chrono_t start = NOW;

    read_stl(stl_file_name,triangles);

    std::cout << "Reading STL file...                   ";
    timer_checkpoint(start);
    std::cout << "Allocating device memory...           ";

    int num_triangles = triangles.size();
    triangle* triangles_dev;
    // all[z][y][x]
    bool* all = (bool*)malloc(NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool));
    bool* all_dev;
    size_t size = NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool);
    cudaMalloc(&all_dev, size);
    cudaMalloc(&triangles_dev, num_triangles * sizeof(triangle));
    cudaMemcpy(triangles_dev, triangles.data(), num_triangles * sizeof(triangle), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid;

    blocksPerGrid = (Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;
    timer_checkpoint(start);
    std::cout << "Running pps kernel...                 ";
    pps<<<blocksPerGrid, threadsPerBlock>>>(&triangles_dev[0], num_triangles, all_dev);
    cudaDeviceSynchronize();
    timer_checkpoint(start);
    std::cout << "Copying memory contents...            ";
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy result from device memory to host memory
    cudaMemcpy(all, all_dev, size, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    timer_checkpoint(start);
    checkOutput(triangles_dev, num_triangles, all);
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

    free(all);

    return 0;
}
