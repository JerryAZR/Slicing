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

    // switch (sanityCheck()) {
    //     case 1:
    //         std::cerr << "THREADS_PER_BLOCK may not be larger than X_DIM" << std::endl;
    //         std::cerr << "Current THREADS_PER_BLOCK: " << THREADS_PER_BLOCK;
    //         std::cerr << "X_DIM: " << X_DIM << std::endl; 
    //         return 0;
    //     case 2:
    //         std::cerr << "RESOLUTION must be some (positive/negative) power of 2" << std::endl;
    //         std::cerr << "Current RESOULTION: " << RESOLUTION << std::endl;
    //         return 0;
    // }

    read_stl(stl_file_name,triangles);
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
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    size_t threadsPerBlock = THREADS_PER_BLOCK;
    size_t blocksPerGrid;

    layer_t* all_intersections;
    cudaMalloc(&all_intersections, Y_DIM * X_DIM * MAX_TRUNK_SIZE * sizeof(layer_t));
    size_t* trunk_length;
    cudaMalloc(&trunk_length, Y_DIM * X_DIM * sizeof(size_t));
    cudaMemset(trunk_length, 0, Y_DIM * X_DIM * sizeof(size_t));
    int* locks;
    cudaMalloc(&locks, Y_DIM * X_DIM * sizeof(int));
    cudaMemset(locks, 0, Y_DIM * X_DIM * sizeof(int));

    blocksPerGrid = (num_triangles * Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;
    fps1<<<blocksPerGrid, threadsPerBlock>>>(&triangles_dev[0], num_triangles, all_intersections, trunk_length, locks);
    cudaDeviceSynchronize();

    blocksPerGrid = (X_DIM * Y_DIM + threadsPerBlock - 1) / threadsPerBlock;
    fps2<<<blocksPerGrid, threadsPerBlock>>>(all_intersections, trunk_length);
    cudaDeviceSynchronize();

    blocksPerGrid = (X_DIM * Y_DIM * NUM_LAYERS + threadsPerBlock - 1) / threadsPerBlock;
    fps3<<<blocksPerGrid, threadsPerBlock>>>(all_intersections, trunk_length, all_dev);
    cudaDeviceSynchronize();

    cudaFree(all_intersections);
    cudaFree(trunk_length);
    cudaFree(locks);
    // Copy result from device memory to host memory
    cudaMemcpy(all, all_dev, size, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    free(all);

    return 0;
}
