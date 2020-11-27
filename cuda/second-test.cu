#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <utility>
#include "triangle.cuh"
#include "slicer.cuh"
#include "golden.cuh"

using std::pair;

int main(int argc, char* argv[]) {
    std::string stl_file_name;
    std::vector<triangle> small_tri;
    std::vector<triangle> large_tri;
    std::vector<double> z_mins_vect;
    // std::chrono::time_point<std::chrono::system_clock> mid, end;

    if (argc == 2) {
        stl_file_name = argv[1];
    } else if (argc > 2) {
        std::cout << "ERROR: Too many command line arguments" << std::endl;
        return 0;
    } else {
        std::cout << "ERROR: Too few command line arguments" << std::endl;
        return 0;
    }

    preprocess_stl(stl_file_name, small_tri, large_tri, z_mins_vect);
    size_t num_small = small_tri.size();
    size_t num_large = large_tri.size();

    triangle* small_tri_dev;
    cudaMalloc(&small_tri_dev, num_small * sizeof(triangle));
    cudaMemcpy(small_tri_dev, small_tri.data(), num_small * sizeof(triangle), cudaMemcpyHostToDevice);

    triangle* large_tri_dev;
    cudaMalloc(&large_tri_dev, num_large * sizeof(triangle));
    cudaMemcpy(large_tri_dev, large_tri.data(), num_large * sizeof(triangle), cudaMemcpyHostToDevice);

    layer_t* intersections_large;
    cudaMalloc(&intersections_large, Y_DIM * X_DIM * NUM_LAYERS * sizeof(layer_t));

    size_t* trunk_length;
    cudaMalloc(&trunk_length, Y_DIM * X_DIM * sizeof(size_t));

    bool out[Y_DIM][X_DIM][NUM_LAYERS];
    bool* out_dev;
    cudaMalloc(&out_dev, Y_DIM * X_DIM * NUM_LAYERS * sizeof(bool));
    cudaMemset(out_dev, 0, Y_DIM * X_DIM * NUM_LAYERS * sizeof(bool));

    double* z_mins_dev;
    cudaMalloc(&z_mins_dev, num_small * sizeof(double));
    cudaMemcpy(z_mins_dev, z_mins_vect.data(), num_small * sizeof(double), cudaMemcpyHostToDevice);

    // start = std::chrono::system_clock::now();

    // mid = std::chrono::system_clock::now();

    int threadsPerBlock = THREADS_PER_BLOCK;
    int numBlocks;

    numBlocks = (Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;

    largeTriIntersection<<<numBlocks, threadsPerBlock>>>(large_tri_dev, num_large, intersections_large, trunk_length);
    thrust::sort_by_key(thrust::device, z_mins_dev, z_mins_dev + num_small, small_tri_dev);

    cudaDeviceSynchronize();

    smallTriIntersection<<<numBlocks, threadsPerBlock>>>(small_tri_dev, z_mins_dev, num_small, intersections_large, trunk_length, out_dev);

    cudaDeviceSynchronize();

    cudaMemcpy(&out[0][0][0], out_dev, Y_DIM * X_DIM * NUM_LAYERS * sizeof(bool), cudaMemcpyDeviceToHost);

    // timer end
    // end = std::chrono::system_clock::now();

    // std::chrono::duration<double> std_seconds = mid - start;
    // std::chrono::duration<double> thrust_seconds = end - mid;

    // std::cout << "std algo: " << std_seconds.count() << "s" << std::endl;
    // std::cout << "thrust algo: " << thrust_seconds.count() << "s" << std::endl;

    bool out_reshaped[NUM_LAYERS][Y_DIM][X_DIM];

    for (int z = 0; z < NUM_LAYERS; z++) {
        for (int y = Y_DIM-1; y >= 0; y--) {
            for (int x = 0; x < X_DIM; x++) {
                // if (out[y][x][z]) std::cout << "XX";
                // else std::cout << "  ";

                out_reshaped[z][y][x] = out[y][x][z];
            }
            // std::cout << std::endl;
        }
        // std::cout << std::endl << std::endl;
    }

    small_tri.insert(small_tri.end(), large_tri.begin(), large_tri.end());
    size_t num_triangles = small_tri.size();
    triangle* triangles_dev;
    cudaMalloc(&triangles_dev, num_triangles * sizeof(triangle));
    cudaMemcpy(triangles_dev, small_tri.data(), num_triangles * sizeof(triangle), cudaMemcpyHostToDevice);
    checkOutput(triangles_dev, num_triangles, &out_reshaped[0][0][0]);

    cudaFree(large_tri_dev);
    cudaFree(small_tri_dev);
    cudaFree(intersections_large);
    cudaFree(trunk_length);

    return 0;
}