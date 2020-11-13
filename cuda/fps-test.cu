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

    char* all_intersections;
    cudaMalloc(&all_intersections, Y_DIM * X_DIM * NUM_LAYERS * sizeof(char));
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
    cudaMemcpy(&all[0][0][0], all_dev, size, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    std::cout << "begin verification" << std::endl;
    checkOutput(triangles_dev, num_triangles, &all[0][0][0]);
    cudaFree(all_dev);
    cudaFree(triangles_dev);
    long diff = 0;
    for (int z = 0; z < NUM_LAYERS; z++) {
        for (int y = 0; y < Y_DIM; y++) {
            for (int x = 0; x < X_DIM; x++) {
                diff += (expected[z][y][x] != all[z][y][x]);
            }
        }
    }
    std::cout << "Diff: " << diff << " pixel(s)." << std::endl;

    return 0;
}
