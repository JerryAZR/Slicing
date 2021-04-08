/**
 * @file bitmap.h
 */

#ifndef BITMAP_H_
#define BITMAP_H_

#include <cstdint>

#define HEADER_SIZE 54

struct __attribute__((packed))BmpHeader {
    char bitmapSignatureBytes[2] = {'B', 'M'};
    uint32_t sizeOfBitmapFile = 0; // Need to add size to it
    uint32_t reservedBytes = 0;
    uint32_t pixelDataOffset = HEADER_SIZE;

	uint32_t sizeOfThisHeader = HEADER_SIZE - 14;
    int32_t width = 0; // in pixels
    int32_t height = 0; // in pixels
    uint16_t numberOfColorPlanes = 1; // must be 1
    uint16_t colorDepth = 24;
    uint32_t compressionMethod = 0;
    uint32_t rawBitmapDataSize = 0; // generally ignored
    int32_t horizontalResolution = 0; // in pixel per meter
    int32_t verticalResolution = 0; // in pixel per meter
    uint32_t colorTableEntries = 0;
    uint32_t importantColors = 0;

	void setDim(int32_t width, int32_t height);
	void setRes(double mmPerPixel);
};

struct Pixel {
    uint8_t blue = 0;
    uint8_t green = 0;
    uint8_t red = 0;
};


#define BLACK (Pixel{0,0,0})
#define WHITE (Pixel{255,255,255})

#endif
