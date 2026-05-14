#pragma once
#include <cstdint>

#define NUM_RGB_CHANNELS 3

typedef struct {
    int width;
    int height;
    uint8_t* data;
} rgb_image_t;

typedef struct {
    int width;
    int height;
    uint8_t* data;
} grayscale_image_t;

typedef uint8_t brightness_t;

typedef struct {
    brightness_t r, g, b;
} color_t;


void setPixel(rgb_image_t image, int x, int y, color_t color);
color_t getPixel(rgb_image_t image, int x, int y);

void setPixel(grayscale_image_t image, int x, int y, brightness_t value);
brightness_t getPixel(grayscale_image_t image, int x, int y);
