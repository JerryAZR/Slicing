#include <iostream>
#include <string>
#include "triangle.cuh"
#include "slicer.cuh"
#include <vector>
#include <map>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "golden.cuh"

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

    double* zmins;
    cudaMalloc(&zmins, num_triangles * sizeof(double));

    int* h_index = (int*)malloc(num_triangles * sizeof(int));
    int* index;
    cudaMalloc(&index, num_triangles * sizeof(int));


    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error 1: " << cudaGetErrorString(err) << std::endl;

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid;

    blocksPerGrid = (num_triangles + threadsPerBlock - 1) / threadsPerBlock;
    triangle_sort << <blocksPerGrid, threadsPerBlock >> > (d_triangles, num_triangles, zmins, index);
    cudaDeviceSynchronize();
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error 2: " << cudaGetErrorString(err) << std::endl;

    thrust::device_ptr<double> d_zmins(zmins);
    thrust::device_ptr<int> d_index(index);

    thrust::sort_by_key(d_zmins, d_zmins + num_triangles, d_index);
    index = thrust::raw_pointer_cast(&d_index[0]);

    cudaDeviceSynchronize();
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error 3: " << cudaGetErrorString(err) << std::endl;


    blocksPerGrid = (Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;
    outputArray << <blocksPerGrid, threadsPerBlock >> > (d_triangles, num_triangles, d_outArray, index);

    cudaDeviceSynchronize();
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error 4: " << cudaGetErrorString(err) << std::endl;
    // Copy result from device memory to host memory
    cudaMemcpy(outArray, d_outArray, X_DIM * Y_DIM * NUM_LAYERS * sizeof(bool), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error 5: " << cudaGetErrorString(err) << std::endl;

    err = cudaGetLastError();  // add
    if (err != cudaSuccess) std::cout << "CUDA error 6: " << cudaGetErrorString(err) << std::endl;
    /*
    std::cout << "begin verification" << std::endl;
    checkOutput(d_triangles, num_triangles, outArray);
    */
    cudaFree(d_outArray);
    cudaFree(d_triangles);
    cudaFree(zmins);
    cudaFree(index);

    return 0;
}