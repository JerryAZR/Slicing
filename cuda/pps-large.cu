#include <iostream>
#include <string>
#include "triangle.cuh"
#include "slicer.cuh"
#include "golden.cuh"
#include <vector>
#include <chrono>
#include <fstream>
#include "bitmap.cuh"
#define NOW (std::chrono::high_resolution_clock::now())

typedef std::chrono::time_point<std::chrono::high_resolution_clock> chrono_t;

void checkCudaError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

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

    std::cout << "Reading STL file...                   ";
    timer_checkpoint(start);
    std::cout << "Allocating device memory...           ";

    int num_triangles = triangles.size();
    triangle* triangles_dev, * triangles_selected;
    // all[z][y][x]
#ifdef TEST 
    // Allocate all reauired memory
    size_t size = NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool);
#else 
    // Allocation just enough memory for profiling
    size_t size = PPS_BLOCK_HEIGHT * Y_DIM * X_DIM * sizeof(bool);
#endif
    bool* all = (bool*)malloc(size);
    bool* all_dev;
    cudaMalloc(&all_dev, PPS_BLOCK_HEIGHT * Y_DIM * X_DIM * sizeof(bool));
    cudaMalloc(&triangles_dev, num_triangles * sizeof(triangle));
    cudaMalloc(&triangles_selected, num_triangles * sizeof(triangle));
    cudaMemcpy(triangles_dev, triangles.data(), num_triangles * sizeof(triangle), cudaMemcpyHostToDevice);

    unsigned* out_length_d, out_length_h;
    cudaMalloc(&out_length_d, sizeof(unsigned));

    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid;

    blocksPerGrid = (PPS_BLOCK_HEIGHT * X_DIM + threadsPerBlock - 1) / threadsPerBlock;
    timer_checkpoint(start);
    std::cout << "Running pps kernel...                 ";
    for (unsigned layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx += PPS_BLOCK_HEIGHT) {
        cudaMemset(out_length_d, 0, sizeof(unsigned));
        checkCudaError();
        triangleSelect<<<128,128>>>(triangles_dev, triangles_selected, num_triangles, out_length_d, layer_idx);
        checkCudaError();
        cudaMemcpy(&out_length_h, out_length_d, sizeof(unsigned), cudaMemcpyDeviceToHost);
        checkCudaError();
        pps<<<blocksPerGrid, threadsPerBlock>>>(triangles_selected, out_length_h, all_dev, layer_idx);
        checkCudaError();
        size_t copy_size = (layer_idx + PPS_BLOCK_HEIGHT) < NUM_LAYERS ? PPS_BLOCK_HEIGHT : NUM_LAYERS - layer_idx;
        copy_size = copy_size * X_DIM * Y_DIM * sizeof(bool);
    #ifdef TEST
        bool* host_addr = &all[X_DIM*Y_DIM*layer_idx];
    #else
        bool* host_addr = &all[0];
    #endif
        cudaMemcpy(host_addr, all_dev, copy_size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        checkCudaError();
    }

    
    timer_checkpoint(start);
    cudaFree(all_dev);
    cudaFree(triangles_selected);
    cudaFree(out_length_d);

#ifdef TEST
    checkOutput(triangles_dev, num_triangles, all);
#if (WRITE_BMP == 1)
    Pixel black = BLACK;
    Pixel white = WHITE;
    const char outDir[] = "bmp";
    char fname[128];
    for (int z = 0; z < NUM_LAYERS; z++) {
        sprintf(fname, "%s/layer_%d.bmp", outDir, z);
        std::ofstream outfile(fname, std::ios::out | std::ios::binary);
        // Write BMP header
        BmpHeader header;
        header.setDim(X_DIM, Y_DIM);
        header.setRes(RESOLUTION);
        outfile.write((char*)&header, HEADER_SIZE);
        
        for (int y = 0; y < Y_DIM; y++) {
            for (int x = 0; x < X_DIM; x++) {
                if (all[z*X_DIM*Y_DIM + y*X_DIM + x])
                    outfile.write((char*) &black, 3);
                else
                    outfile.write((char*) &white, 3);
            }
        }
        std::cout << "Writing to output file...  "<< z+1 << "/" << NUM_LAYERS << "\r";
        outfile.close();
    }
    std::cout << std::endl;
#endif
#endif
    cudaFree(triangles_dev);
    free(all);

    return 0;
}
