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
#ifdef TEST
    chrono_t end = NOW;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - checkpoint);
    std::cout << duration.count() << "ms" << std::endl;
    checkpoint = end;
#else
    std::cout << std::endl;
#endif
}

void checkCudaError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    std::string stl_file_name;
    std::vector<triangle> triangles;
    std::vector<std::vector<double>> point_array(9);

    if (argc == 2) {
        stl_file_name = argv[1];
    } else if (argc > 2) {
        std::cout << "ERROR: Too many command line arguments" << std::endl;
    }

    chrono_t start = NOW;

    load_point_array(stl_file_name, point_array, triangles);

    std::cout << "Reading STL file...                   ";
    timer_checkpoint(start);
    std::cout << "Allocating device memory...           ";

    int num_triangles = triangles.size();
    triangle* triangles_dev;
    double* points_dev;
    // all[z][y][x]
#ifdef TEST
    bool* all = (bool*)malloc(NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool));
#else
    bool* all = (bool*)malloc(BLOCK_HEIGHT * Y_DIM * X_DIM * sizeof(bool));
#endif
    bool* all_dev;
    size_t size = BLOCK_HEIGHT * Y_DIM * X_DIM * sizeof(bool);
    cudaMalloc(&all_dev, size);
    cudaMemset(all_dev, 0, size);
    unsigned* flips_dev;
    cudaMalloc(&flips_dev, BLOCK_HEIGHT * X_DIM * MAX_FLIPS * sizeof(unsigned));
#ifdef TEST
    unsigned* flips_host = (unsigned*)malloc(NUM_LAYERS * MAX_FLIPS * X_DIM * sizeof(unsigned));
#else
    unsigned* flips_host = (unsigned*)malloc(BLOCK_HEIGHT * MAX_FLIPS * X_DIM * sizeof(unsigned));
#endif
    cudaMalloc(&triangles_dev, num_triangles * sizeof(triangle));
    cudaMemcpy(triangles_dev, triangles.data(), num_triangles * sizeof(triangle), cudaMemcpyHostToDevice);
    
    cudaMalloc(&points_dev, num_triangles * sizeof(triangle));
    size_t temp_offset = 0;
    for (int i = 0; i < 9; i++) {
        cudaMemcpy(points_dev + temp_offset, point_array[i].data(),
                    num_triangles * sizeof(double), cudaMemcpyHostToDevice);
        temp_offset += num_triangles;
    }
    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    timer_checkpoint(start);
    std::cout << "Running 1st kernel...                 ";
    for (unsigned layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx += BLOCK_HEIGHT) {
        rectTriIntersection<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(points_dev, num_triangles, all_dev, layer_idx);
        cudaDeviceSynchronize();
        checkCudaError();
        size_t blocksPerGrid = (X_DIM * BLOCK_HEIGHT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        bbox_ints<<<blocksPerGrid, THREADS_PER_BLOCK>>>(all_dev, flips_dev);
        cudaDeviceSynchronize();
        checkCudaError();
        size_t copy_size = (layer_idx + BLOCK_HEIGHT) < NUM_LAYERS ? BLOCK_HEIGHT : NUM_LAYERS - layer_idx;
        copy_size = copy_size * X_DIM * MAX_FLIPS * sizeof(unsigned);
    #ifdef TEST
        unsigned* flips_addr = &flips_host[X_DIM*MAX_FLIPS*layer_idx];
    #else
        unsigned* flips_addr = &flips_host[0];
    #endif
        cudaMemcpy(flips_addr, flips_dev, copy_size, cudaMemcpyDeviceToHost);
        cudaMemset(all_dev, 0, size);
        cudaDeviceSynchronize();
        checkCudaError();
    }

    timer_checkpoint(start);
    cudaFree(all_dev);
    cudaFree(points_dev);
    cudaFree(flips_dev);

#ifdef TEST
    std::cout << "Decompressing...                 ";
    bbox_ints_decompress(flips_host, all);
    timer_checkpoint(start);
    checkOutput(triangles_dev, num_triangles, all);
    // for (int z = 0; z < NUM_LAYERS; z++) {
    //     for (int y = Y_DIM; y > 0; y--) {
    //         for (int x = 0; x < X_DIM; x++) {
    //             if (all[z*X_DIM*Y_DIM + y*X_DIM + x]) std::cout << "XX";
    //             else std::cout << "  ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl << std::endl;
    // }
#endif
    cudaFree(triangles_dev);
    free(all);
    free(flips_host);

    return 0;
}
