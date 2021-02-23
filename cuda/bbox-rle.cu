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
    unsigned* trunks_dev;
    cudaMalloc(&trunks_dev, BLOCK_HEIGHT * Y_DIM * MAX_TRUNK_SIZE * sizeof(unsigned));
    unsigned* trunk_length;
    cudaMalloc(&trunk_length, BLOCK_HEIGHT * Y_DIM * sizeof(unsigned));
    cudaMemset(trunk_length, 0, BLOCK_HEIGHT * Y_DIM * sizeof(unsigned));

#ifdef TEST
    unsigned* trunks_host = (unsigned*)malloc(NUM_LAYERS * MAX_TRUNK_SIZE * Y_DIM * sizeof(unsigned));
#else
    unsigned* trunks_host = (unsigned*)malloc(BLOCK_HEIGHT * MAX_TRUNK_SIZE * Y_DIM * sizeof(unsigned));
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
        rectTriIntersection<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>
            (points_dev, num_triangles, trunks_dev, trunk_length, layer_idx);
        cudaDeviceSynchronize();
        checkCudaError();
        size_t blocksPerGrid = (X_DIM * BLOCK_HEIGHT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        trunk_compress<<<blocksPerGrid, THREADS_PER_BLOCK>>>(trunks_dev, trunk_length);
        cudaDeviceSynchronize();
        checkCudaError();
        size_t copy_size = (layer_idx + BLOCK_HEIGHT) < NUM_LAYERS ? BLOCK_HEIGHT : NUM_LAYERS - layer_idx;
        copy_size = copy_size * Y_DIM * MAX_TRUNK_SIZE * sizeof(unsigned);
    #ifdef TEST
        unsigned* trunks_addr = &trunks_host[Y_DIM*MAX_TRUNK_SIZE*layer_idx];
    #else
        unsigned* trunks_addr = &trunks_host[0];
    #endif
        cudaMemcpy(trunks_addr, trunks_dev, copy_size, cudaMemcpyDeviceToHost);
        cudaMemset(trunk_length, 0, BLOCK_HEIGHT * Y_DIM * sizeof(unsigned));
        cudaDeviceSynchronize();
        checkCudaError();
    }

    timer_checkpoint(start);
    cudaFree(trunk_length);
    cudaFree(points_dev);
    cudaFree(trunks_dev);

#ifdef TEST
    std::cout << "Decompressing...                 ";
    bbox_ints_decompress(trunks_host, all);
    timer_checkpoint(start);
    checkOutput(triangles_dev, num_triangles, all);

    // std::ofstream outfile;
    // std::cout << "Writing to output file...                 ";
    // outfile.open("out.txt");
    // for (int z = 0; z < NUM_LAYERS; z++) {
    //     for (int y = Y_DIM-1; y >= 0; y--) {
    //         for (int x = 0; x < X_DIM; x++) {
    //             if (all[z*X_DIM*Y_DIM + y*X_DIM + x]) outfile << "XX";
    //             else outfile << "  ";
    //         }
    //         outfile << "\n";
    //     }
    //     outfile << "\n\n";
    // }
    // outfile.close();
#endif
    cudaFree(triangles_dev);
    free(all);
    free(trunks_host);

    return 0;
}
