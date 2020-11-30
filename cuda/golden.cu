#include "golden.cuh"
#include "slicer.cuh"
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <stdio.h>
#include <fstream>

long checkOutput(triangle* triangles_dev, size_t num_triangles, bool* in) {
    bool* expected = (bool*) malloc(NUM_LAYERS * X_DIM * Y_DIM * sizeof(bool));
    std::cout << "executing golden model" << std::endl;
    goldenModel(triangles_dev, num_triangles, &expected[0]);
    long size = NUM_LAYERS * Y_DIM * X_DIM;
    long diff = 0;
    long inside = 0;
    std::cout << "comparing results" << std::endl;
    for (int i = 0; i < size; i++) {
        inside += expected[i];
        diff += (expected[i] != in[i]);
    }

    std::ofstream outfile;
    outfile.open("expected.txt");
    for (int z = 0; z < NUM_LAYERS; z++) {
        for (int y = Y_DIM-1; y >= 0; y--) {
            for (int x = 0; x < X_DIM; x++) {
                if (expected[z*X_DIM*Y_DIM + y*X_DIM + x]) outfile << "XX";
                else outfile << "  ";
            }
            outfile << "\n";
        }
        outfile << "\n\n";
    }
    outfile.close();

    free(expected);
    std::cout << inside << " pixels are inside the model." << std::endl;
    std::cout << diff << " pixels are different in the expected and actual output." << std::endl;
    return diff;
}

void goldenModel(triangle* triangles_dev, size_t num_triangles, bool* out) {
    size_t threadsPerBlock = THREADS_PER_BLOCK;
    size_t blocksPerGrid;
    bool* all_dev;
    size_t size = NUM_LAYERS * Y_DIM * X_DIM * sizeof(bool);
    cudaMalloc(&all_dev, size);
    layer_t* all_intersections;
    cudaMalloc(&all_intersections, Y_DIM * X_DIM * NUM_LAYERS * sizeof(layer_t));
    size_t* trunk_length;
    cudaMalloc(&trunk_length, Y_DIM * X_DIM * sizeof(size_t));
    cudaMemset(trunk_length, 0, Y_DIM * X_DIM * sizeof(size_t));
    int* locks;
    cudaMalloc(&locks, Y_DIM * X_DIM * sizeof(int));
    cudaMemset(locks, 0, Y_DIM * X_DIM * sizeof(int));

    blocksPerGrid = (num_triangles * Y_DIM * X_DIM + threadsPerBlock - 1) / threadsPerBlock;
    _fps1<<<blocksPerGrid, threadsPerBlock>>>(&triangles_dev[0], num_triangles, all_intersections, trunk_length, locks);
    cudaDeviceSynchronize();
    blocksPerGrid = (X_DIM * Y_DIM + threadsPerBlock - 1) / threadsPerBlock;
    _fps2<<<blocksPerGrid, threadsPerBlock>>>(all_intersections, trunk_length);
    cudaDeviceSynchronize();
    blocksPerGrid = (X_DIM * Y_DIM * NUM_LAYERS + threadsPerBlock - 1) / threadsPerBlock;
    _fps3<<<blocksPerGrid, threadsPerBlock>>>(all_intersections, trunk_length, all_dev);
    cudaDeviceSynchronize();

    cudaFree(all_intersections);
    cudaFree(trunk_length);
    cudaFree(locks);

    cudaMemcpy(out, all_dev, size, cudaMemcpyDeviceToHost);

    cudaFree(all_dev);
}

__global__
void _fps1(triangle* triangles, size_t num_triangles, layer_t* all_intersections, size_t* trunk_length, int* locks) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    size_t tri_idx = idx / (X_DIM * Y_DIM);
    // if (tri_idx >= num_triangles) return;

    // copy 1 triangle to the shared memory -- That's all we need on this block
    __shared__  triangle triangles_shared;
    __shared__  double x_max, x_min, y_max, y_min;
    if (threadIdx.x == 0) {
        triangles_shared = triangles[tri_idx];
        thrust::maximum<double> max;
        thrust::minimum<double> min;
        x_max = max(triangles_shared.p1.x, max(triangles_shared.p2.x, triangles_shared.p3.x));
        x_min = min(triangles_shared.p1.x, min(triangles_shared.p2.x, triangles_shared.p3.x));
        y_max = max(triangles_shared.p1.y, max(triangles_shared.p2.y, triangles_shared.p3.y));
        y_min = min(triangles_shared.p1.y, min(triangles_shared.p2.y, triangles_shared.p3.y));
    }
    __syncthreads();

    int y_idx = (idx - (tri_idx * (X_DIM * Y_DIM))) / X_DIM;
    int x_idx = (idx - (tri_idx * (X_DIM * Y_DIM))) % X_DIM;
    int x = x_idx - (X_DIM >> 1);
    int y = y_idx - (Y_DIM >> 1);

    double x_pos = x * RESOLUTION;
    double y_pos = y * RESOLUTION;
    bool notInRect = (x_pos < x_min) || (x_pos > x_max) || (y_pos < y_min) || (y_pos > y_max);

    layer_t* layers = all_intersections + y_idx * X_DIM * NUM_LAYERS + x_idx * NUM_LAYERS;
    int* lock = locks + y_idx * X_DIM + x_idx;
    size_t* length = trunk_length + y_idx * X_DIM + x_idx;
    layer_t intersection = notInRect ? -1 : _pixelRayIntersection(triangles_shared, x, y);
    bool run = (intersection != -1);
    while (run) {
        if(atomicCAS(lock, 0, 1) == 0) {
            layers[length[0]] = intersection;
            length[0]++;
            run = false;
            atomicExch(lock, 0);
        }
    }
}

__global__
void _fps2(layer_t* all_intersections, size_t* trunk_length) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    if (idx >= X_DIM * Y_DIM) return;
    size_t length = trunk_length[idx];
    layer_t* curr_trunk = all_intersections + (idx * NUM_LAYERS);
    thrust::sort(thrust::device, curr_trunk, curr_trunk + length);
}

__global__
void _fps3(layer_t* sorted_intersections, size_t* trunk_length, bool* out) {
    size_t idx = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
    int z_idx = idx / (X_DIM * Y_DIM);
    if (z_idx >= NUM_LAYERS) return;
    int y_idx = (idx - (z_idx * X_DIM * Y_DIM)) / X_DIM;
    int x_idx = (idx - (z_idx * X_DIM * Y_DIM)) % X_DIM;

    size_t length = trunk_length[y_idx * X_DIM + x_idx];
    layer_t* intersection_trunk = sorted_intersections + y_idx * X_DIM * NUM_LAYERS + x_idx * NUM_LAYERS;
    out[idx] = _isInside(z_idx, intersection_trunk, length);
}

__device__ __forceinline__
layer_t _pixelRayIntersection(triangle t, int x, int y) {
    /*
    Let A, B, C be the 3 vertices of the given triangle
    Let S(x,y,z) be the intersection, where x,y are given
    We want to find some a, b such that AS = a*AB + b*AC
    If a >= 0, b >= 0, and a+b <= 1, S is a valid intersection.

    return the layer of intersection, or -1 if none
    */
    double x_d = x * RESOLUTION - t.p1.x;
    double y_d = y * RESOLUTION - t.p1.y;

    double x1 = t.p2.x - t.p1.x;
    double y1 = t.p2.y - t.p1.y;
    double z1 = t.p2.z - t.p1.z;

    double x2 = t.p3.x - t.p1.x;
    double y2 = t.p3.y - t.p1.y;
    double z2 = t.p3.z - t.p1.z;
    double a = (x_d * y2 - x2 * y_d) / (x1 * y2 - x2 * y1);
    double b = (x_d * y1 - x1 * y_d) / (x2 * y1 - x1 * y2);
    bool inside = (a >= 0) && (b >= 0) && (a+b <= 1);
    double intersection = (a * z1 + b * z2) + t.p1.z;
    // // divide by layer width
    layer_t layer = inside ? (intersection / RESOLUTION) : (layer_t)(-1);
    return layer;
}

__device__
bool _isInside(layer_t current, layer_t* trunk, size_t length) {
    size_t startIdx = 0;
    size_t endIdx = length;
    size_t mid;
    bool goLeft;

    // perform binary search
    while (startIdx < endIdx) {
        mid = (startIdx + endIdx) / 2;
        if (trunk[mid] == current) return true;
        goLeft = trunk[mid] > current;
        startIdx = goLeft ? startIdx : (mid + 1);
        endIdx = goLeft ? mid : endIdx;
    }

    return (bool)(startIdx & 1);
}
