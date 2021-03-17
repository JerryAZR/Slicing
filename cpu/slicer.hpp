#ifndef SLICER
#define SLICER

#include "triangle.hpp"
#include <vector>

// in mm
#define X_LEN 128
#define Y_LEN 128
#define HEIGHT 128
#define RESOLUTION 0.2

// in pixels
#define NUM_LAYERS (size_t)(HEIGHT / RESOLUTION)
#define X_DIM (size_t)(X_LEN / RESOLUTION)
#define Y_DIM (size_t)(Y_LEN / RESOLUTION)

#define X_MIN (long)(-1 * X_LEN / 2)
#define X_MAX (long)(X_LEN / 2)
#define Y_MIN (long)(-1 * Y_LEN / 2)
#define Y_MAX (long)(Y_LEN / 2)

#define MAX_TRUNK_SIZE	64
#define MAX_WORDS ((size_t)(1<<29)) // 2GB
#define BBOX_BLOCK_HEIGHT (min(NUM_LAYERS, MAX_WORDS/(MAX_TRUNK_SIZE*Y_DIM))) // larger is better
static_assert(MAX_WORDS/(MAX_TRUNK_SIZE*Y_DIM), "Layer size too large.\n");

void bbox_cpu(std::vector<triangle> tri, std::vector<std::vector<unsigned>>& out_compressed, size_t base_z);
void trunk_compress(vector<vector<unsigned>>& out_compressed);
void rleDecodeSt(vector<vector<unsigned>>& in, bool* out, const bool* out_end);

#endif
