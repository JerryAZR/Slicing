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

#define NOW (std::chrono::system_clock::now())

typedef std::chrono::time_point<std::chrono::system_clock> chrono_t;

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
    std::vector<triangle> small_tri;
    chrono_t start;

    if (argc == 2) {
        stl_file_name = argv[1];
    } else if (argc > 2) {
        std::cout << "ERROR: Too many command line arguments" << std::endl;
        return 0;
    } else {
        std::cout << "ERROR: Too few command line arguments" << std::endl;
        return 0;
    }

    start = NOW;
    
    read_stl(stl_file_name, small_tri);
    size_t num_small = small_tri.size();

    std::cout << "Reading STL file...           ";
    timer_checkpoint(start);
    std::cout << "Allocating device memory...   ";

    triangle* small_tri_dev;
    cudaMalloc(&small_tri_dev, num_small * sizeof(triangle));
    cudaMemcpy(small_tri_dev, small_tri.data(), num_small * sizeof(triangle), cudaMemcpyHostToDevice);

    layer_t* intersections_large;
    cudaMalloc(&intersections_large, Y_DIM * X_DIM * NUM_LAYERS * sizeof(layer_t));

    size_t* trunk_length;
    cudaMalloc(&trunk_length, Y_DIM * X_DIM * sizeof(size_t));

    // out[z][y][x]
    bool* out = (bool*)malloc(NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool));
    bool* out_dev;
    cudaMalloc(&out_dev, Y_DIM * X_DIM * NUM_LAYERS * sizeof(bool));
    cudaMemset(out_dev, 0, Y_DIM * X_DIM * NUM_LAYERS * sizeof(bool));

    double* z_mins_dev;
    cudaMalloc(&z_mins_dev, num_small * sizeof(double));

    timer_checkpoint(start);
    std::cout << "Sorting triangles...          ";

    int threadsPerBlock = THREADS_PER_BLOCK;
    int numBlocks;

    numBlocks = (Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;

    GPUsort(small_tri_dev, num_small, z_mins_dev);

    cudaDeviceSynchronize();
    timer_checkpoint(start);
    std::cout << "Processing sorted triangles...";

    smallTriIntersection<<<numBlocks, threadsPerBlock>>>(small_tri_dev, z_mins_dev, num_small, out_dev);

    cudaDeviceSynchronize();
    timer_checkpoint(start);
    std::cout << "Copying memory contents...    ";

    cudaMemcpy(out, out_dev, Y_DIM * X_DIM * NUM_LAYERS * sizeof(bool), cudaMemcpyDeviceToHost);

    timer_checkpoint(start);

    // for (int z = 0; z < NUM_LAYERS; z++) {
    //     for (int y = Y_DIM-1; y >= 0; y--) {
    //         for (int x = 0; x < X_DIM; x++) {
    //             if (out[z][y][x]) std::cout << "XX";
    //             else std::cout << "  ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl << std::endl;
    // }

#ifdef TEST
    checkOutput(small_tri_dev, num_small, out);
#endif

    free(out);
    cudaFree(small_tri_dev);
    cudaFree(out_dev);
    cudaFree(z_mins_dev);

    return 0;
}