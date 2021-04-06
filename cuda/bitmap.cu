#include "bitmap.cuh"

void BmpHeader::setDim(int32_t w, int32_t h) {
    width = w;
    height = h;
    sizeOfBitmapFile = HEADER_SIZE + w * h * 3; // Each pixel takes 3 bytes
}

void BmpHeader::setRes(double mmPerPixel) {
    horizontalResolution = (int32_t)(1000/mmPerPixel);
    verticalResolution = (int32_t)(1000/mmPerPixel);
}

