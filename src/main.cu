#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <string>
#include <stdexcept>
#include <iostream>
#include <chrono>

#include "types.hpp"
#include "preprocessing.cuh"
#include "piece_segmentation.cuh"
#include "ccl.cuh"
#include "pointerjumping.cuh"
#include "sort_distances.cuh"

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
    std:: cout << "[Preprocessing] starting... " << std::endl;
    auto preprocessing_time_start = std::chrono::high_resolution_clock::now();
    grayscale_image_t image = preprocessing("image.png");
    auto preprocessing_time_end = std::chrono::high_resolution_clock::now();

    auto preprocessing_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(preprocessing_time_end - preprocessing_time_start);

    std::cout << "Preprocessing time: "
              << preprocessing_duration_ms.count()
              << " milliseconds\n\n";

    image = convert_image(read_image("../data/processed/binarymultiple.png"));


    std::cout << "Image loaded and converted to grayscale with size " << image.width << "x" << image.height << "." << std::endl;
    std:: cout << "[Erosion/Dilation] starting... " << std::endl;

    auto piece_segmentation_time_start = std::chrono::high_resolution_clock::now();
    piece_segmentation(image); // black-white enlargement
    auto piece_segmentation_time_end = std::chrono::high_resolution_clock::now();

    std:: cout << "[Erosion/Dilation] done " << std::endl;
    auto piece_segmentation_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(piece_segmentation_time_end - piece_segmentation_time_start);
    std::cout << "Piece segmentation time: "
              << piece_segmentation_duration_ms.count()
              << " milliseconds\n\n";

    save_gray_image(image,"../data/processed/after_denoising.png");
    


    create_test_pattern();
    std:: cout << "[CCL] starting... " << std::endl;
    grayscale_image_t grayImage_cleaned = image;

    //set up host labels
    auto playne_time_start = std::chrono::high_resolution_clock::now();
    uint32_t  *global_labels_host = (uint32_t*)malloc(grayImage_cleaned.width* grayImage_cleaned.height * sizeof(unsigned int));
    playne_ccl(global_labels_host, grayImage_cleaned);
    auto playne_time_end = std::chrono::high_resolution_clock::now();

    std:: cout << "[CCL] done" << std::endl;
    auto playne_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(playne_time_end - playne_time_start);
    std::cout << "Playne time: "
              << playne_duration_ms.count()
              << " milliseconds\n\n";

    grayscale_image_t output_image_playne = visualize_cll(global_labels_host, grayImage_cleaned);
    save_gray_image(output_image_playne, "../data/processed/playned.png");

    ptr_jmp_out_t arrays;
    std:: cout << "[Pointer jumping] starting... " << std::endl;

    auto pointer_jumping_time_start = std::chrono::high_resolution_clock::now();
    arrays.boundaries = (uint32_t*)malloc(grayImage_cleaned.width* grayImage_cleaned.height * sizeof(unsigned int));
    pointer_jumping(global_labels_host,arrays , grayImage_cleaned.width,grayImage_cleaned.height);
    auto pointer_jumping_time_end = std::chrono::high_resolution_clock::now();

    std:: cout << "[Pointer jumping] done" << std::endl;
    auto pointer_jumping_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pointer_jumping_time_end - pointer_jumping_time_start);

    std::cout << "Pointer jumping time: "
              << pointer_jumping_duration_ms.count()
              << " milliseconds\n\n";
    grayscale_image_t boundary_image = visualize_cll(arrays.boundaries, grayImage_cleaned);
    save_gray_image(boundary_image, "../data/processed/extracted_boundaries.png");

    uint32_t* d_image_data_in;
    int num_pixels = grayImage_cleaned.height * grayImage_cleaned.width;
    size_t size = num_pixels * sizeof(brightness_t);
    cudaMalloc(&d_image_data_in, size);
    cudaMemcpy(d_image_data_in, grayImage_cleaned.data.data(), size, cudaMemcpyHostToDevice);


    thrust::host_vector<Centroid> centroids = compute_centroids(d_image_data_in, grayImage_cleaned.width, grayImage_cleaned.height);
    return 0;
}
