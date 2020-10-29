#include <iostream>
#include <string>
#include "triangle.cuh"
#include "slicer.cuh"
#include <vector>
#include <map>

#define BLOCK_SIZE 16

int main(int argc, char* argv[]) {
    std::string stl_file_name;
    std::vector<triangle> triangles;

    if (argc == 2) {
        stl_file_name = argv[1];
    }
    else if (argc > 2) {
        std::cout << "ERROR: Too many command line arguments" << std::endl;
    }

    read_stl(stl_file_name, triangles);
    int num_triangles = triangles.size();

    //load from host to device

    bool* outArray = (bool*)malloc(X_DIM * Y_DIM * NUM_LAYERS * sizeof(bool));
    bool* d_outArray;
    cudaMalloc(&d_outArray, X_DIM * Y_DIM * NUM_LAYERS * sizeof(bool));

    triangle* d_triangles;
    cudaMalloc(&d_triangles, num_triangles * sizeof(triangle));
    cudaMemcpy(d_triangles, triangles.data(), num_triangles * sizeof(triangle), cudaMemcpyHostToDevice);

    //bool* flagArray;
    //cudaMalloc(&flagArray, X_DIM * Y_DIM * sizeof(bool));

    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error 1: " << cudaGetErrorString(err) << std::endl;

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid;
    blocksPerGrid = (Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;

    outputArray << <blocksPerGrid, threadsPerBlock >> > (d_triangles, num_triangles, d_outArray);

    cudaDeviceSynchronize();
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error 2: " << cudaGetErrorString(err) << std::endl;
    // Copy result from device memory to host memory
    cudaMemcpy(outArray, d_outArray, X_DIM * Y_DIM * NUM_LAYERS * sizeof(bool), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error 3: " << cudaGetErrorString(err) << std::endl;

    cudaFree(d_outArray);
    //cudaFree(flagArray);
    cudaFree(d_triangles);
    //cudaFree(d_intersectArray);
    //cudaFree(d_intersectArrayPre);

    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error 4: " << cudaGetErrorString(err) << std::endl;
/*
    for (int y = Y_DIM; y > 0; y--) {
        for (int x = 0; x < X_DIM; x++) {
            if (outArray[0 * Y_DIM * X_DIM + y * X_DIM + x]) std::cout << "x";
            else std::cout << " ";
        }
        std::cout << std::endl;
    }
*/
    return 0;
}
