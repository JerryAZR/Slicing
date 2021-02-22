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
    triangle* triangles_host = (triangle*) triangles.data();
    // all[z][y][x]
    bool* all = (bool*)malloc(NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool));
    bool* all_dev;
    size_t size = NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool);
    double* zMins;
    cudaMalloc(&all_dev, size);
    cudaMalloc(&triangles_dev, BATCH_SIZE * sizeof(triangle));
    cudaMalloc(&zMins, BATCH_SIZE * sizeof(double));

    size_t num_iters = num_triangles / BATCH_SIZE;
    size_t remaining = num_triangles % BATCH_SIZE;

    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid;

    blocksPerGrid = (Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;
    timer_checkpoint(start);
    std::cout << "Running kernel...                     ";
    for (size_t i = 0; i < num_iters; i++) {
        GPUsort(triangles_dev, BATCH_SIZE, zMins);
        cudaMemcpy(triangles_dev, triangles_host + i * BATCH_SIZE, BATCH_SIZE * sizeof(triangle), cudaMemcpyHostToDevice);
        overlapSlicer<<<blocksPerGrid, threadsPerBlock>>>(triangles_dev, zMins, BATCH_SIZE, all_dev);
        // cudaDeviceSynchronize();
    }
    if (remaining) {
        cudaMemcpy(triangles_dev, triangles_host + num_iters * BATCH_SIZE, remaining * sizeof(triangle), cudaMemcpyHostToDevice);
        overlapSlicer<<<blocksPerGrid, threadsPerBlock>>>(triangles_dev, nullptr, remaining, all_dev);
        // cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    timer_checkpoint(start);
    std::cout << "Layer Extraction...                   ";
    layerExtraction<<<blocksPerGrid, threadsPerBlock>>>(all_dev, 0);
    cudaDeviceSynchronize();
    timer_checkpoint(start);
    std::cout << "Copying memory contents...            ";
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    // Copy result from device memory to host memory
    cudaMemcpy(all, all_dev, size, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    timer_checkpoint(start);

    cudaFree(triangles_dev);
    cudaMalloc(&triangles_dev, num_triangles * sizeof(triangle));
    cudaMemcpy(triangles_dev, triangles_host, num_triangles * sizeof(triangle), cudaMemcpyHostToDevice);

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
