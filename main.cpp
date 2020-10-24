#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include "triangle.hpp"
#include "slicer.hpp"

#define TEST_PPS 0

int main(int argc, char* argv[]) {
    std::cout << "beginning of program" << std::endl;
    std::string stl_file_name;
    std::vector<triangle> triangles;

    if (argc == 2) {
        stl_file_name = argv[1];
    } else if (argc > 2) {
        std::cout << "ERROR: Too many command line arguments" << std::endl;
    }

    read_stl(stl_file_name,triangles);
    // all[z][y][x]
    bool all[X_DIM * Y_DIM * NUM_LAYERS];
#if(TEST_PPS == 1)
    for (int x = 0; x < X_DIM; x++) {
        for (int y = 0; y < Y_DIM; y++) {
            pps(triangles.data(), triangles.size(), &all[0], y*X_DIM + x);
        }
    }
#else
    std::cout << "Allocating memory" << std::endl;
    int* all_temp = (int*)malloc(X_DIM * Y_DIM * NUM_LAYERS * sizeof(int));
    size_t* trunk_length = (size_t*)calloc(X_DIM * Y_DIM , sizeof(size_t));
    int* locks = (int*)calloc(X_DIM * Y_DIM , sizeof(int));
    long id = 0;
    std::cout << "beginning part 1" << std::endl;
    for (int x = 0; x < X_DIM; x++) {
        for (int y = 0; y < Y_DIM; y++) {
            for (int tri = 0; tri < triangles.size(); tri++) {
                fps1(triangles.data(), triangles.size(), &all_temp[0], &trunk_length[0], &locks[0], id);
                id++;
            }
        }
    }
    std::cout << "beginning part 2" << std::endl;
    id = 0;
    for (int x = 0; x < X_DIM; x++) {
        for (int y = 0; y < Y_DIM; y++) {
            fps2(&all_temp[0], &trunk_length[0], id);
            id++;
        }
    }
    std::cout << "beginning part 3" << std::endl;
    id = 0;
    for (int x = 0; x < X_DIM; x++) {
        for (int y = 0; y < Y_DIM; y++) {
            for (int z = 0; z < NUM_LAYERS; z++) {
                fps3(&all_temp[0], &trunk_length[0], &all[0], id);
                id++;
            }
        }
    }
    free(all_temp);
    free(trunk_length);
    free(locks);
#endif
    // Visualize
    for (int y = Y_DIM; y > 0; y--) {
        for (int x = 0; x < X_DIM; x++) {
            if (all[y*X_DIM + x]) std::cout << "x";
            else std::cout << " ";
        }
        std::cout << std::endl;
    }
}
