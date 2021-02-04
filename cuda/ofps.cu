#include <iostream>
#include <fstream>
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

    if (argc == 2) {
        stl_file_name = argv[1];
    } else if (argc > 2) {
        std::cout << "ERROR: Too many command line arguments" << std::endl;
    }

    chrono_t start = NOW;

    read_stl(stl_file_name,triangles);
    int num_triangles = triangles.size();

    std::cout << "Reading STL file...                   ";
    timer_checkpoint(start);
    std::cout << "Allocating device memory...           ";

    triangle* triangles_dev;
    // all[z][y][x]
    bool* all = (bool*)malloc(NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool));
    bool* all_dev;
    size_t size = NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool);
    cudaMalloc(&all_dev, size);
    cudaMalloc(&triangles_dev, num_triangles * sizeof(triangle));
    cudaMemcpy(triangles_dev, triangles.data(), num_triangles * sizeof(triangle), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    long threadsPerBlock = THREADS_PER_BLOCK;
    long blocksPerGrid;

    layer_t* all_intersections;
    cudaMalloc(&all_intersections, Y_DIM * X_DIM * MAX_TRUNK_SIZE * sizeof(layer_t));
    unsigned* trunk_length;
    cudaMalloc(&trunk_length, Y_DIM * X_DIM * sizeof(unsigned));
    cudaMemset(trunk_length, 0, Y_DIM * X_DIM * sizeof(unsigned));
    int* locks;
    cudaMalloc(&locks, Y_DIM * X_DIM * sizeof(int));
    cudaMemset(locks, 0, Y_DIM * X_DIM * sizeof(int));

    timer_checkpoint(start);
    std::cout << "Stage 1: Ray Triangle Intersection    ";

    blocksPerGrid = (num_triangles * Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;
    fps1<<<blocksPerGrid, threadsPerBlock>>>(&triangles_dev[0], num_triangles, all_intersections, trunk_length, locks);
    cudaDeviceSynchronize();

    timer_checkpoint(start);
    std::cout << "Stage 2: Trunk Sorting                ";

    blocksPerGrid = (X_DIM * Y_DIM + threadsPerBlock - 1) / threadsPerBlock;
    fps2<<<blocksPerGrid, threadsPerBlock>>>(all_intersections, trunk_length);
    cudaDeviceSynchronize();

    timer_checkpoint(start);
    std::cout << "Stage 3: Layer Extraction             ";

    blocksPerGrid = (X_DIM * Y_DIM * NUM_LAYERS + threadsPerBlock - 1) / threadsPerBlock;
    fps3<<<blocksPerGrid, threadsPerBlock>>>(all_intersections, trunk_length, all_dev);
    cudaDeviceSynchronize();

    timer_checkpoint(start);
    std::cout << "Copying memory contents...            ";

    cudaFree(all_intersections);
    cudaFree(trunk_length);
    cudaFree(locks);
    // Copy result from device memory to host memory
    cudaMemcpy(all, all_dev, size, cudaMemcpyDeviceToHost);
    timer_checkpoint(start);

    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    std::cout << "begin verification" << std::endl;
#ifdef TEST
    checkOutput(triangles_dev, num_triangles, all);
    // std::ofstream outfile;
    // outfile.open("layers.txt");
    // for (int z = 0; z < NUM_LAYERS; z++) {
    //     for (int y = Y_DIM-1; y >= 0; y--) {
    //         for (int x = 0; x < X_DIM; x++) {
    //             if (all[z*Y_DIM*X_DIM + y*X_DIM + x]) outfile << "XX";
    //             else outfile << "  ";
    //         }
    //         outfile << "\n";
    //     }
    //     outfile << "\n\n";
    // }
    // outfile.close();
#endif
    cudaFree(all_dev);
    cudaFree(triangles_dev);
    free(all);

    return 0;
}
