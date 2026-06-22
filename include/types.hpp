#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define NUM_RGB_CHANNELS 3

struct rgb_image_t
{
    int width;
    int height;
    std::vector<uint8_t> data;
};

struct rgb_image_d_t
{
    int width;
    int height;
    uchar3* data = nullptr;
};

struct grayscale_image_t
{
    int width;
    int height;
    std::vector<uint8_t> data;
};

typedef uint8_t brightness_t;
