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

    //presort triangles
    std::multimap<int, triangle>::iterator itr;
    std::multimap<int, triangle> bucket;
    bucket = sortTriangle(triangles.data(), num_triangles, bucket);

    //load from host to device
    triangle t;
    int* outArray = (int*)malloc(X_DIM * Y_DIM * NUM_LAYERS * sizeof(int));

    triangle* intersectTriangles = (triangle*)malloc(num_triangles * sizeof(triangle));
    triangle* d_intersectTriangles;
    cudaMalloc(&d_intersectTriangles, num_triangles * sizeof(triangle));

    int tNum[NUM_LAYERS];
    int* d_tNum;
    cudaMalloc(&d_tNum, NUM_LAYERS * sizeof(int));

    //transform map to array
    int i = 0;
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        //copy triangles of each layer to device
        tNum[layer] = bucket.count(layer);
        for (itr = bucket.find(layer); itr != bucket.find(layer + 1); ++itr) {
            t = itr->second;
            *(intersectTriangles + i) = t;
            i++;
        }
    }

    cudaMemcpy(d_intersectTriangles, intersectTriangles, num_triangles * sizeof(triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tNum, tNum, NUM_LAYERS * sizeof(int), cudaMemcpyHostToDevice);

    int* d_outArray;
    cudaMalloc(&d_outArray, X_DIM * Y_DIM * NUM_LAYERS * sizeof(int));

    int* d_intersectArray;
    cudaMalloc(&d_intersectArray, X_DIM * Y_DIM * sizeof(int));

    int* d_intersectArrayPre;
    cudaMalloc(&d_intersectArrayPre, X_DIM * Y_DIM * sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid;
    blocksPerGrid = (Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;

    outputArray << <blocksPerGrid, threadsPerBlock >> > (d_intersectTriangles, d_tNum, d_outArray, d_intersectArray, d_intersectArrayPre);

    cudaDeviceSynchronize();

    // Copy result from device memory to host memory
    cudaMemcpy(outArray, d_outArray, X_DIM * Y_DIM * NUM_LAYERS * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_outArray);
    cudaFree(d_intersectTriangles);
    cudaFree(d_tNum);
    cudaFree(d_intersectArray);
    cudaFree(d_intersectArrayPre);

    return 0;
}
