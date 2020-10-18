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

    // all[z][y][x]
    bool all[100][201][201];
    
    for (int x = X_MIN; x <= X_MAX; x++) {
        for (int y = Y_MIN; y <= Y_MAX; y++) {

            // std::cout << "Step 1: get all intersections" << std::endl;
            std::vector<int> layers;
            getIntersectionTrunk(x, y, triangles, layers);

            // std::cout << "Step 2: trunk sorting" << std::endl;
            std::sort(layers.begin(),layers.end());

            // Step 3: layer extraction
            // std::cout << "Step 3: layer extraction" << std::endl;
            bool flag = false;
            int layerIdx = 0;
            layers.push_back(NUM_LAYERS);
            for (int z = 0; z < NUM_LAYERS; z++) {
                // If intersect
                int x_idx = x + 100;
                int y_idx = y + 100;
                // std::cout << "(z,y,x) = " << z << ", " << y_idx << ", " << x_idx << std::endl;
                bool intersect = (z == layers[layerIdx]);
                all[z][y_idx][x_idx] = intersect || flag;
                flag = intersect ^ flag;
                layerIdx += intersect;
            }
        }
    }

    // Visualize
    for (int y = 200; y > 0; y--) {
        for (int x = 50; x <150> 0; x++) {
            if (all[8][y][x]) std::cout << "x";
            else std::cout << " ";
        }
        std::cout << std::endl;
    }
}