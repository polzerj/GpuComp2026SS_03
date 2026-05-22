#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <string>
#include <stdexcept>
#include <iostream>

#include "types.hpp"
#include "preprocessing.cuh"
#include "piece_segmentation.cuh"
#include "ccl.cuh"

rgb_image_t read_image(const std::string &filePath)
{
    int width, height, channels;

    uint8_t *imgData = stbi_load(filePath.c_str(), &width, &height, &channels, 3);

    if (!imgData)
    {
        throw std::runtime_error("Failed to load image: " + filePath);
    }

    rgb_image_t result;
    result.width = width;
    result.height = height;
    result.data = std::vector<uint8_t>(imgData, imgData + width * height * 3);

    return result;
}

grayscale_image_t read_grayscale_image(const std::string &filePath)
{
    int width, height, channels;

    // The last argument '1' tells stb_image to force the output to 1 channel (grayscale)
    uint8_t *imgData = stbi_load(filePath.c_str(), &width, &height, &channels, 1);

    if (!imgData)
    {
        throw std::runtime_error("Failed to load image: " + filePath);
    }

    grayscale_image_t result;
    result.width = width;
    result.height = height;
    
    // Size is now width * height * 1
    result.data.assign(imgData, imgData + width * height);

    stbi_image_free(imgData);

    return result;
}

grayscale_image_t convert_image(const rgb_image_t &rgbImage)
{
    grayscale_image_t grayImage;
    grayImage.width = rgbImage.width;
    grayImage.height = rgbImage.height;
    grayImage.data.resize(rgbImage.width * rgbImage.height);

    for (int i = 0; i < rgbImage.width * rgbImage.height; ++i)
    {
        uint8_t r = rgbImage.data[i * 3];
        uint8_t g = rgbImage.data[i * 3 + 1];
        uint8_t b = rgbImage.data[i * 3 + 2];
        grayImage.data[i] = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
    }

    return grayImage;
}

int save_gray_image(const grayscale_image_t &grayImage, const std::string &filename){

    int channels = 1;
    int stride = grayImage.width * channels;
    
    return stbi_write_png(filename.c_str(), grayImage.width, grayImage.height, channels, grayImage.data.data(), stride);
}

void create_test_pattern(){

    int tileSize = 20; 

    grayscale_image_t image;
    image.height= 800;
    image.width = 800;
    image.data.resize(image.width * image.height);

    // Generate Checkerboard Pattern
    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            
            int tileX = x / tileSize;
            int tileY = y / tileSize;

            if ((tileX + tileY) % 2 == 0) {
                image.data[y * image.width + x] = 255; // White square
            } else {
                image.data[y * image.width + x] = 0;   // Black square
            }
        }
    }

    save_gray_image(image,"../data/processed/test_image.png");
}

int main()
{
    grayscale_image_t image = preprocessing("image.png");

    image = convert_image(read_image("../data/processed/binary.png"));

    std::cout << "Image loaded and converted to grayscale with size " << image.width << "x" << image.height << "." << std::endl;

    piece_segmentation(image);

    create_test_pattern();

    grayscale_image_t grayImage_cleaned = read_grayscale_image("../data/processed/cleaned.png");

    //set up host labels
    uint32_t  *global_labels_host = (uint32_t*)malloc(grayImage_cleaned.width* grayImage_cleaned.height * sizeof(unsigned int));
    playne_ccl(global_labels_host, grayImage_cleaned);
    
    grayscale_image_t output_image_playne = visualize_cll(global_labels_host, grayImage_cleaned);
    save_gray_image(grayImage_cleaned, "../data/processed/beforeplayne.png" );
    save_gray_image(output_image_playne, "../data/processed/playned.png");


    return 0;
}
