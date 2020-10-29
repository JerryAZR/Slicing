#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "triangle.hpp"
#include "slicer.hpp"


int main(int argc, char* argv[]) {
    std::string stl_file_name;
    std::vector<triangle> triangles;

    if (argc == 2) {
        stl_file_name = argv[1];
        std::cout << stl_file_name << std::endl;
    }
    else if (argc > 2) {
        std::cout << "ERROR: Too many command line arguments" << std::endl;
    }

    read_stl(stl_file_name, triangles);

    // outArray[z][y][x]
    //int outArray[X_DIM * Y_DIM * NUM_LAYERS] = { 0 };
    //bool outArray[X_DIM * Y_DIM * NUM_LAYERS]; 
    bool* outArray = (bool*)malloc(X_DIM * Y_DIM * NUM_LAYERS * sizeof(bool));
    outputArray(triangles.data(), triangles.size(), &outArray[0]);

    // Visualize
    std::cout<<"printing:"<<std::endl;
    for (int y = Y_DIM; y > 0; y--) {
        for (int x = 0; x < X_DIM; x++) {
            if (outArray[1 * X_DIM * Y_DIM + y * X_DIM + x]) std::cout << "x";
            else 			std::cout << " ";
        }
        std::cout << std::endl;
    }
    free(outArray);
    return 0;
}
