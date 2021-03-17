#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include "triangle.hpp"
#include "slicer.hpp"

using std::min;

#define TEST_PPS 1

int main(int argc, char* argv[]) {
    std::cout << "beginning of program" << std::endl;
    std::string stl_file_name;
    vector<triangle> triangles;

    if (argc == 2) {
        stl_file_name = argv[1];
    } else if (argc > 2) {
        std::cout << "ERROR: Too many command line arguments" << std::endl;
    }

    read_stl(stl_file_name,triangles);
#ifdef TEST
    // all[z][y][x]
    bool* all = (bool*)malloc(NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool));
    const bool* out_end = all + (NUM_LAYERS * Y_DIM * X_DIM);
#endif
    vector<vector<unsigned>> out_compressed(Y_DIM*NUM_LAYERS*BBOX_BLOCK_HEIGHT, vector<unsigned>());
    for (size_t layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx += BBOX_BLOCK_HEIGHT) {
        bbox_cpu(triangles, out_compressed, layer_idx);
        trunk_compress(out_compressed);
#ifdef TEST
        bool* out_addr = &all[layer_idx*X_DIM*Y_DIM];
        rleDecodeSt(out_compressed, out_addr, out_end);
#endif
        out_compressed = vector<vector<unsigned>>(Y_DIM*NUM_LAYERS*BBOX_BLOCK_HEIGHT, vector<unsigned>());
    }
#ifdef TEST
    std::cout << "Writing to output file...                 ";
    for (int z = 0; z < NUM_LAYERS; z++) {
        for (int y = Y_DIM-1; y >= 0; y--) {
            for (int x = 0; x < X_DIM; x++) {
                if (all[z*X_DIM*Y_DIM + y*X_DIM + x]) std::cout << "XX";
                else std::cout << "  ";
            }
            std::cout << "\n";
        }
        std::cout << "\n\n";
    }
    free(all);
#endif

    return 0;
}
