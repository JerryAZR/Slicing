#include <iostream>
#include <fstream>
#include <string>
#include "triangle.cuh"
#include "slicer.cuh"
#include "golden.cuh"
#include <vector>
#include <chrono>
#include "bitmap.cuh"
#define NOW (std::chrono::high_resolution_clock::now())

typedef std::chrono::time_point<std::chrono::high_resolution_clock> chrono_t;

void timer_checkpoint(chrono_t & checkpoint) {
#ifdef TEST
    chrono_t end = NOW;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - checkpoint);
    std::cout << (double)duration.count()/1000 << "ms" << std::endl;
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
    std::cout << "Block height is " << PPS_BLOCK_HEIGHT << std::endl;

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
    triangle* triangles_dev, * triangles_selected;;
    double* points_dev;
    // all[z][y][x]
#if (COMPRESSION_ONLY == 0)
    double decompression_time = 0.0;
#ifdef TEST
    bool* all = (bool*)malloc(NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool));
#else
    bool* all = (bool*)malloc(PPS_BLOCK_HEIGHT * Y_DIM * X_DIM * sizeof(bool));
#endif
#endif
    unsigned* trunks_dev;
    cudaMalloc(&trunks_dev, PPS_BLOCK_HEIGHT * Y_DIM * MAX_TRUNK_SIZE * sizeof(unsigned));
    unsigned* trunks_out;
    cudaMalloc(&trunks_out, PPS_BLOCK_HEIGHT * Y_DIM * MAX_TRUNK_SIZE * sizeof(unsigned));
    unsigned* trunk_length;
    cudaMalloc(&trunk_length, PPS_BLOCK_HEIGHT * Y_DIM * sizeof(unsigned));
    cudaMemset(trunk_length, 0, PPS_BLOCK_HEIGHT * Y_DIM * sizeof(unsigned));

#ifdef TEST
    unsigned* trunks_host = (unsigned*)malloc(NUM_LAYERS * MAX_TRUNK_SIZE * Y_DIM * sizeof(unsigned));
#else
    unsigned* trunks_host = (unsigned*)malloc(PPS_BLOCK_HEIGHT * MAX_TRUNK_SIZE * Y_DIM * sizeof(unsigned));
#endif
    cudaMalloc(&triangles_dev, num_triangles * sizeof(triangle));
    cudaMalloc(&triangles_selected, num_triangles * sizeof(triangle));
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
    std::cout << "Slicing...                            ";
    unsigned* out_length_d, out_length_h;
    cudaMalloc(&out_length_d, sizeof(unsigned));

    size_t blocksPerGrid = (Y_DIM * PPS_BLOCK_HEIGHT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    for (unsigned layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx += PPS_BLOCK_HEIGHT) {
        cudaMemset(out_length_d, 0, sizeof(unsigned));
        checkCudaError();
        triangleSelect<<<128,128>>>(triangles_dev, triangles_selected, num_triangles, out_length_d, layer_idx);
        checkCudaError();
        cudaMemcpy(&out_length_h, out_length_d, sizeof(unsigned), cudaMemcpyDeviceToHost);
        checkCudaError();
        pps<<<blocksPerGrid, THREADS_PER_BLOCK>>>
            (triangles_selected, out_length_h, trunks_dev, trunk_length, layer_idx);
        cudaDeviceSynchronize();
        checkCudaError();
        unsigned max_length = thrust::reduce(thrust::device,
            trunk_length, trunk_length+Y_DIM*PPS_BLOCK_HEIGHT, 0, thrust::maximum<unsigned>());
        max_length += 2; // Max number of runs + zero terminate
        if (max_length > MAX_TRUNK_SIZE) {
            std::cout << "too many intersections" << std::endl; return 0;
        }
        trunk_compress<<<blocksPerGrid, THREADS_PER_BLOCK>>>(trunks_dev, trunk_length, trunks_out, max_length);
        cudaDeviceSynchronize();
        checkCudaError();
        size_t copy_layers = (layer_idx + PPS_BLOCK_HEIGHT) < NUM_LAYERS ? PPS_BLOCK_HEIGHT : NUM_LAYERS - layer_idx;
        size_t copy_size = copy_layers * Y_DIM * max_length * sizeof(unsigned);
        unsigned* trunks_addr = &trunks_host[0];
        cudaMemcpy(trunks_addr, trunks_out, copy_size, cudaMemcpyDeviceToHost);
        cudaMemset(trunk_length, 0, PPS_BLOCK_HEIGHT * Y_DIM * sizeof(unsigned));
        cudaDeviceSynchronize();
        checkCudaError();
    #if (COMPRESSION_ONLY == 0)
    #ifdef TEST
        bool* out_addr = &all[layer_idx*X_DIM*Y_DIM];
    #else
        bool* out_addr = &all[0];
    #endif
        decompression_time += rleDecode(trunks_addr, out_addr, copy_layers, max_length);
    #endif
    }

    timer_checkpoint(start);
    cudaFree(trunk_length);
    cudaFree(points_dev);
    cudaFree(trunks_dev);
    cudaFree(trunks_out);

#if (COMPRESSION_ONLY == 0)
    std::cout << "Total decompression time: " << decompression_time << "ms" << std::endl;
#ifdef TEST
    // checkOutput(triangles_dev, num_triangles, all);
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
    free(all);
#endif
#endif

    cudaFree(triangles_dev);
    free(trunks_host);

    return 0;
}
