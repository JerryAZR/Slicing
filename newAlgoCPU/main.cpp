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
    } else if (argc > 2) {
        std::cout << "ERROR: Too many command line arguments" << std::endl;
    }

    read_stl(stl_file_name,triangles);

    // outArray[z][y][x]
	//int outArray[X_DIM * Y_DIM * NUM_LAYERS] = { 0 };
	int* outArray = (int*)malloc(X_DIM * Y_DIM * NUM_LAYERS * sizeof(int));
	outputArray(triangles.data(), triangles.size(), outArray);

    // Visualize
    for (int y = 200; y > 0; y--) {
        for (int x = 25; x < 175; x++) {
            if (*(outArray + 8*X_DIM*Y_DIM + y*X_DIM + x)) std::cout << "x";
            else 			std::cout << " ";
        }
        std::cout << std::endl;
    }
	free(outArray);
}