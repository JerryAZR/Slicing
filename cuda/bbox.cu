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
    bool* all = (bool*)malloc(NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool));
    bool* all_dev;
    size_t size = NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool);
    cudaMalloc(&all_dev, size);
    cudaMemset(all_dev, 0, size);
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
    rectTriIntersection<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(points_dev, num_triangles, all_dev);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    timer_checkpoint(start);
    std::cout << "Running 2nd kernel...                 ";
    size_t blocksPerGrid = (X_DIM * Y_DIM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    layerExtraction<<<blocksPerGrid, THREADS_PER_BLOCK>>>(all_dev, 0);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    timer_checkpoint(start);
    std::cout << "Copying memory contents...            ";
    // Copy result from device memory to host memory
    cudaMemcpy(all, all_dev, size, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    timer_checkpoint(start);
#ifdef TEST
    checkOutput(triangles_dev, num_triangles, all);
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
#endif
    cudaFree(all_dev);
    cudaFree(triangles_dev);
    cudaFree(points_dev);
    free(all);

    return 0;
}
