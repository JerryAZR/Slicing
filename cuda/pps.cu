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
    triangle* triangles_dev;
    // all[z][y][x]
    bool* all = (bool*)malloc(NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool));
    bool* all_dev;
    size_t size = NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool);
    cudaMalloc(&all_dev, size);
    cudaMalloc(&triangles_dev, num_triangles * sizeof(triangle));
    cudaMemcpy(triangles_dev, triangles.data(), num_triangles * sizeof(triangle), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();  // add
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid;

    blocksPerGrid = (Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;
    timer_checkpoint(start);
    std::cout << "Running pps kernel...                 ";
    pps<<<blocksPerGrid, threadsPerBlock>>>(&triangles_dev[0], num_triangles, all_dev);
    cudaDeviceSynchronize();
    timer_checkpoint(start);
    std::cout << "Copying memory contents...            ";
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy result from device memory to host memory
    cudaMemcpy(all, all_dev, size, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();  // add
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    timer_checkpoint(start);
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
    cudaFree(all_dev);
    cudaFree(triangles_dev);
    free(all);

    return 0;
}
