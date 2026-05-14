#pragma once
#include <cstdint>
#include <vector>

#define NUM_RGB_CHANNELS 3

struct rgb_image_t
{
    int width;
    int height;
    std::vector<uint8_t> data;
};

struct grayscale_image_t
{
    int width;
    int height;
    std::vector<uint8_t> data;
};

typedef uint8_t brightness_t;

struct color_t
{
    brightness_t r, g, b;
};

void setPixel(rgb_image_t image, int x, int y, color_t color);
color_t getPixel(rgb_image_t image, int x, int y);

void setPixel(grayscale_image_t image, int x, int y, brightness_t value);
brightness_t getPixel(grayscale_image_t image, int x, int y);
