#include <iostream>
#include <string>
#include "triangle.cuh"
#include "slicer.cuh"
#include "golden.cuh"
#include <vector>
#include <chrono>
#define NOW (std::chrono::high_resolution_clock::now())

typedef std::chrono::time_point<std::chrono::high_resolution_clock> chrono_t;

void checkCudaError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

void timer_checkpoint(chrono_t & checkpoint) {
#ifdef TEST
    chrono_t end = NOW;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - checkpoint);
    std::cout << duration.count() << "ms" << std::endl;
    checkpoint = end;
#else
    std::cout << std::endl;
#endif
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
#ifdef TEST 
    // Allocate all reauired memory
    size_t size = NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool);
#else 
    // Allocation just enough memory for profiling
    size_t size = BLOCK_HEIGHT * Y_DIM * X_DIM * sizeof(bool);
#endif
    bool* all = (bool*)malloc(size);
    bool* all_dev;
    cudaMalloc(&all_dev, BLOCK_HEIGHT * Y_DIM * X_DIM * sizeof(bool));
    cudaMalloc(&triangles_dev, num_triangles * sizeof(triangle));
    cudaMemcpy(triangles_dev, triangles.data(), num_triangles * sizeof(triangle), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid;

    blocksPerGrid = (BLOCK_HEIGHT * X_DIM + threadsPerBlock - 1) / threadsPerBlock;
    timer_checkpoint(start);
    std::cout << "Running pps kernel...                 ";
    for (unsigned layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx += BLOCK_HEIGHT) {
        pps<<<blocksPerGrid, threadsPerBlock>>>(&triangles_dev[0], num_triangles, all_dev, layer_idx);
        cudaDeviceSynchronize();
        checkCudaError();
        size_t copy_size = (layer_idx + BLOCK_HEIGHT) < NUM_LAYERS ? BLOCK_HEIGHT : NUM_LAYERS - layer_idx;
        copy_size = copy_size * X_DIM * Y_DIM * sizeof(bool);
    #ifdef TEST
        bool* host_addr = &all[X_DIM*Y_DIM*layer_idx];
    #else
        bool* host_addr = &all[0];
    #endif
        cudaMemcpy(host_addr, all_dev, copy_size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        checkCudaError();
    }
    
    timer_checkpoint(start);
    cudaFree(all_dev);

#ifdef TEST
    checkOutput(triangles_dev, num_triangles, all);
    // for (int z = 0; z < NUM_LAYERS; z++) {
    //     for (int y = Y_DIM; y > 0; y--) {
    //         for (int x = 0; x < X_DIM; x++) {
    //             if (all[z*Y_DIM*X_DIM + y*X_DIM + x]) std::cout << "XX";
    //             else std::cout << "  ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl << std::endl;
    // }
#endif
    cudaFree(triangles_dev);
    free(all);

    return 0;
}
