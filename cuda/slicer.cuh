#ifndef SLICER
#define SLICER

#define COMPRESSION_ONLY 0

#include "triangle.cuh"
#include <thrust/device_vector.h>
#include <iostream>

#define LOG_THREADS 7
#define THREADS_PER_BLOCK (1 << LOG_THREADS)
#define MAX_TRUNK_SIZE	64
#define NUM_BLOCKS  256
#define NUM_CPU_THREADS 16
#define RECTS_PER_LAYER 1024
#define MAX_MEM_SIZE (1 << 30)

// in mm
// Power of 2 recommended for better performance
#define X_LEN 128
#define Y_LEN 128
#define HEIGHT 128
#define RESOLUTION 0.125

// in pixels
#define NUM_LAYERS ((long)(HEIGHT / RESOLUTION))
// X_DIM must be at least as large as THREADS_PER_BLOCK
#define X_DIM ((long)(X_LEN / RESOLUTION))
#define Y_DIM ((long)(Y_LEN / RESOLUTION))

#define X_MIN ((long)(-1 * X_DIM / 2))
#define X_MAX ((long)(X_DIM / 2 - 1))
#define Y_MIN ((long)(-1 * Y_DIM / 2))
#define Y_MAX ((long)(Y_DIM / 2 - 1))

#define MAX_WORDS ((size_t)(1<<31)) // 2GB

#if (COMPRESSION_ONLY)
#define PPS_BLOCK_HEIGHT (min((long)32, MAX_WORDS/(MAX_TRUNK_SIZE*Y_DIM))) // smaller is better
#define BBOX_BLOCK_HEIGHT (min(NUM_LAYERS, MAX_WORDS/(MAX_TRUNK_SIZE*Y_DIM))) // larger is better
static_assert(MAX_WORDS/(MAX_TRUNK_SIZE*Y_DIM), "Layer size too large.\n");
#else
#define PPS_BLOCK_HEIGHT (min((long)32, MAX_WORDS/(X_DIM*Y_DIM))) // smaller is better
#define BBOX_BLOCK_HEIGHT (min(NUM_LAYERS, MAX_WORDS/(X_DIM*Y_DIM))) // larger is better
static_assert(MAX_WORDS/(X_DIM*Y_DIM), "Layer size too large.\n");
#endif

typedef int layer_t;

#define NONE ((layer_t)(-1))

// Sanity Check

__global__ void pps(triangle* triangles, size_t num_triangles, bool* out);
__global__ void pps(triangle* triangles, size_t num_triangles, bool* out, unsigned base_layer);
__global__ void pps(triangle* triangles, size_t num_triangles, unsigned* trunks, unsigned base_layer);

// returns the layer of intersection
__device__ layer_t pixelRayIntersection(triangle t, int x, int y);

__global__ void rectTriIntersection(double* tri_global, size_t num_tri, bool* out);
__global__ void rectTriIntersection(double* tri_global, size_t num_tri, unsigned* trunks, unsigned* trunk_length, unsigned base_layer);
__global__ void rectTriIntersection(double* tri_global, size_t num_tri, bool* out, unsigned base_layer);
__global__ void layerExtraction(bool* out);
__global__ void rectEncoding(bool* in, unsigned* out, unsigned* length);
void rectEncodingCPU(bool* in, unsigned* out, unsigned* length);
__global__ void trunk_compress(unsigned* trunks, unsigned* trunk_length, unsigned* out);

__global__ void triangleSelect(triangle* in, triangle* out, unsigned in_length, unsigned* out_length, unsigned base_layer);
__global__ void pointSelect(double* in, double* out, unsigned in_length, unsigned* out_length, unsigned base_layer);

// Compression

__global__ void bbox_ints(bool* in, unsigned* out);
void bbox_rect_decode(unsigned* in, bool* out, unsigned* length);
double rleDecode(unsigned* in, bool* out, unsigned nlayers);


#endif
