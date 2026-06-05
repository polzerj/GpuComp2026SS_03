#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// define which formats to export
#define EXPORT_AS_PNG (0)
#define EXPORT_AS_JPG (1)
#define EXPORT_AS_BMP (0)

#include <string>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>

#include "timer.hpp"
#include "types.hpp"
#include "preprocessing.cuh"
// #include "preproccesing_old.cuh"
#include "piece_segmentation.cuh"
#include "ccl.cuh"
#include "pointerjumping.cuh"
#include "sort_distances.cuh"
#include "signal_smoothing.cuh"
#include "corner_detection.cuh"
#include "classification.cuh"
#include "visualizer.cuh"

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

int save_gray_image(const grayscale_image_t &grayImage, const std::string &filename)
{

    int channels = 1;
    int stride = grayImage.width * channels;

    return stbi_write_png(filename.c_str(), grayImage.width, grayImage.height, channels, grayImage.data.data(), stride);
}

void create_test_pattern()
{

    int tileSize = 20;

    grayscale_image_t image;
    image.height = 800;
    image.width = 800;
    image.data.resize(image.width * image.height);

    // Generate Checkerboard Pattern
    for (int y = 0; y < image.height; y++)
    {
        for (int x = 0; x < image.width; x++)
        {

            int tileX = x / tileSize;
            int tileY = y / tileSize;

            if ((tileX + tileY) % 2 == 0)
            {
                image.data[y * image.width + x] = 255; // White square
            }
            else
            {
                image.data[y * image.width + x] = 0; // Black square
            }
        }
    }

    save_gray_image(image, "../data/processed/test_image.png");
}

void save_contour_signal_csv(
    const thrust::host_vector<ContourSignalPoint> &points,
    const std::string &filename)
{
    std::ofstream file(filename);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open CSV file: " + filename);
    }

    file << "label,rank,x,y,distance\n";

    for (const auto &p : points)
    {
        file << p.label << ","
             << p.rank << ","
             << p.x << ","
             << p.y << ","
             << p.distance << "\n";
    }
}

void save_contour_signal_csv(
    const thrust::host_vector<ContourSignalPointExt> &points,
    const std::string &filename)
{
    std::ofstream file(filename);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open CSV file: " + filename);
    }

    file << "label,rank,x,y,distance,piece_length\n";

    for (const auto &p : points)
    {
        file << p.label << ","
             << p.rank << ","
             << p.x << ","
             << p.y << ","
             << p.distance << ","
             << p.piece_length << "\n";
    }
}

grayscale_image_t visualize_corners(
    const grayscale_image_t &base,
    const thrust::host_vector<ContourSignalPointExt> &points,
    const thrust::host_vector<PieceCorners> &corners)
{
    grayscale_image_t out = base;

    // darken original image so corner markers are visible
    for (auto &px : out.data)
    {
        px = px / 3;
    }

    for (const auto &piece : corners)
    {
        if (!piece.valid)
        {
            printf("Piece not valid\n");
            continue;
        }

        for (uint32_t idx : piece.point_indices)
        {
            const auto &p = points[idx];

            int cx = static_cast<int>(p.x);
            int cy = static_cast<int>(p.y);

            // draw white square marker
            for (int dy = -10; dy <= 10; ++dy)
            {
                for (int dx = -10; dx <= 10; ++dx)
                {
                    int x = cx + dx;
                    int y = cy + dy;

                    if (x >= 0 && x < out.width &&
                        y >= 0 && y < out.height)
                    {
                        out.data[y * out.width + x] = 255;
                    }
                }
            }
        }
    }

    return out;
}

int save_rgb_image(const rgb_image_t &image, const std::string &filename)
{
    int channels = 3;
    int stride = image.width * channels;
    int status = 1;

#if (EXPORT_AS_PNG)
    {
        status &= stbi_write_png(
            filename.c_str(),
            image.width,
            image.height,
            channels,
            image.data.data(),
            stride);
    }
#endif

#if (EXPORT_AS_JPG)
    status &= stbi_write_jpg(
        filename.c_str(),
        image.width,
        image.height,
        channels,
        image.data.data(),
        stride);
#endif

#if (EXPORT_AS_BMP)
    status &= stbi_write_bmp(
        filename.c_str(),
        image.width,
        image.height,
        channels,
        image.data.data());
#endif

#if (!(EXPORT_AS_PNG || EXPORT_AS_JPG || EXPORT_AS_BMP))
    throw std::runtime_error("No valid export format specified");
#endif

    return status;
}

__global__ void helloFromDevice()
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    return;
}

int main(int argc, char *argv[])
{
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    GLOBAL_TIMER_START(full_execution);

    std::string image_path = "../data/1_p2.jpg";
    if (argc == 2)
    {
        std::cout << "Using provided image path: " << argv[1] << "\n";
        image_path = argv[1];
    }
    else if (argc > 2)
    {
        std::cerr << "Usage: " << argv[0] << " [image_path]\n";
        return 1;
    }
    else
    {
        std::cout << "No image path provided, using default: " << image_path << "\n";
    }

    DEBUG_TIMER_START(hello_from_device);
    helloFromDevice<<<2, 4>>>();
    cudaDeviceSynchronize();
    DEBUG_TIMER_STOP(hello_from_device, "Hello from Device");

    uint8_t *d_binary_image = nullptr;
    DEBUG_TIMER_START(preprocessing);
    rgb_image_t original_image = preprocessing(image_path, d_binary_image);
    DEBUG_TIMER_STOP(preprocessing, "Preprocessing");

    // save_gray_image(image, "debug_preprocessed.png");

    // image = convert_image(read_image("../data/processed/binarymultiple.png"));

    std::cout << "Image loaded and converted to grayscale with size " << original_image.width << "x" << original_image.height << "." << std::endl;

    uint32_t image_width = original_image.width;
    uint32_t image_height = original_image.height;
    uint32_t pixel_count = image_width * image_height;

    //------------------------------------------------------
    // PIECE SEGMENTATION
    //------------------------------------------------------
    DEBUG_TIMER_START(piece_segmentation);
    uint8_t *piece_seg_out = piece_segmentation(d_binary_image, image_width, image_height, 5); // black-white enlargement
    DEBUG_TIMER_STOP(piece_segmentation, "Piece Segmentation");

    cudaFree(d_binary_image); // not needed anymore, allocated in preprocessing and used in piece segmentation

    // cudaMemcpy(image.data.data(),piece_seg_out , image.width*image.height, cudaMemcpyDeviceToHost);
    // save_gray_image(image,"../data/processed/after_denoising.png");

    //------------------------------------------------------
    // CONNECTED COMPONENT LABELING
    //------------------------------------------------------
    DEBUG_TIMER_START(ccl);
    uint32_t *global_labels_d = playne_ccl(piece_seg_out, image_width, image_height);
    DEBUG_TIMER_STOP(ccl, "CCL");

    // grayscale_image_t& grayImage_cleaned = image;
    //  uint32_t  *global_labels_host = (uint32_t*)malloc(grayImage_cleaned.width* grayImage_cleaned.height * sizeof(unsigned int));
    //  cudaMemcpy(global_labels_host, global_labels_d, grayImage_cleaned.width* grayImage_cleaned.height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //  grayscale_image_t output_image_playne = visualize_cll(global_labels_host, grayImage_cleaned);
    //  save_gray_image(output_image_playne, "../data/processed/playned.png");

    cudaFree(piece_seg_out);

    //------------------------------------------------------
    // POINTER JUMPING
    //------------------------------------------------------
    DEBUG_TIMER_START(pointer_jumping);
    ptr_jmp_out_t out_d = pointer_jumping(global_labels_d, image_width, image_height);
    DEBUG_TIMER_STOP(pointer_jumping, "Pointer Jumping");

    // ptr_jmp_out_t arrays;
    // arrays.boundaries = (uint32_t*)malloc(pixel_count * sizeof(unsigned int));
    // arrays.successors = (uint32_t*)malloc(pixel_count * sizeof(unsigned int));
    // arrays.ranks = (uint32_t*)malloc(pixel_count * sizeof(unsigned int));

    // cudaMemcpy(arrays.boundaries, out_d.boundaries, pixel_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(arrays.successors, out_d.successors, pixel_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(arrays.ranks, out_d.ranks, pixel_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // free(arrays.boundaries);
    // free(arrays.ranks);
    // free(arrays.successors);

    // grayscale_image_t& grayImage_cleaned = image;
    // grayscale_image_t boundary_image = visualize_cll(arrays.boundaries, grayImage_cleaned);
    // save_gray_image(boundary_image, "../data/processed/extracted_boundaries.png");

    cudaFree(global_labels_d);

    //------------------------------------------------------
    // CENTROID CALUCLATION
    //------------------------------------------------------
    DEBUG_TIMER_START(centroid);
    thrust::device_vector<Centroid> d_centroids = compute_centroids(out_d.boundaries, image_width, image_height);
    DEBUG_TIMER_STOP(centroid, "Centroid Calculation");

    thrust::host_vector<Centroid> centroids = d_centroids; // we need this for classification

    //------------------------------------------------------
    // SORTING SIGNALS
    //------------------------------------------------------
    DEBUG_TIMER_START(sorting);
    thrust::device_vector<ContourSignalPoint> d_points = sort_distances(out_d.boundaries, out_d.ranks, d_centroids, image_width, image_height);
    DEBUG_TIMER_STOP(sorting, "Sorting Signals");

    // for (size_t i = 0; i < std::min<size_t>(h_points.size(), 100); ++i) {
    //     const auto& p = h_points[i];
    //     printf("label=%u rank=%u point=(%u,%u) dist=%f\n",
    //         p.label, p.rank, p.x, p.y, p.distance);
    // }

    thrust::host_vector<ContourSignalPoint> h_points = d_points; // this one is necessary for the steps after
    // save_contour_signal_csv(
    //     h_points,
    //     "../data/processed/contour_signal.csv"
    // );

    // std::cout << "Saved contour signal with "
    //         << h_points.size()
    //         << " boundary points.\n";

    cudaFree(out_d.boundaries);
    cudaFree(out_d.ranks);
    cudaFree(out_d.successors);

    //------------------------------------------------------
    //  SMOOTHING SIGNALS
    //------------------------------------------------------
    auto smoothing_time_start = std::chrono::high_resolution_clock::now();
    uint32_t smoothing_radius = 4; // magic value -- if increased too much corners are hard to detect
    DEBUG_TIMER_START(smoothing);
    thrust::device_vector<ContourSignalPointExt> d_smoothed = smooth_signal(h_points, d_points, smoothing_radius);
    DEBUG_TIMER_STOP(smoothing, "Smoothing Signals");

    DEBUG_TIMER_START(save_contour_signal);
    thrust::host_vector<ContourSignalPointExt> h_smoothed = d_smoothed; // this one is necessary for the steps after
    save_contour_signal_csv(h_smoothed, "../data/processed/contour_signal_smoothed.csv");
    DEBUG_TIMER_STOP(save_contour_signal, "Saving Smoothed Contour Signal");

    //------------------------------------------------------
    //  CORNER DETECTION
    //------------------------------------------------------
    DEBUG_TIMER_START(corner_detection);
    uint32_t peak_detection_radius = 100; // magic value -- lower makes it detect incorrect corners
    float prominece_threshold = 10.0f;    // magic value
    uint32_t distance_threshold = 0;

    thrust::host_vector<PieceCorners> h_corners = detect_corners(h_smoothed, d_smoothed, peak_detection_radius, prominece_threshold, distance_threshold);
    DEBUG_TIMER_STOP(corner_detection, "Corner Detection");

    DEBUG_TIMER_START(corner_validation);
    auto piece_id = 0;
    auto invalid_count = 0;
    // std::cout << "Invalid Pieces: \n";
    for (const auto &c : h_corners)
    {
        piece_id++;
        if (c.point_indices.size() != 4)
        {
            //     std::cout << " " << piece_id << " with " << c.point_indices.size() << " corners\n";
            invalid_count++;
        }
        for (uint8_t i = 0; i < c.point_indices.size(); ++i)
        {
            h_smoothed[c.point_indices.at(i)].distance = 0;
        }
    }
    // std::cout << "\n\n";
    // save_contour_signal_csv(h_smoothed, "../data/processed/contour_signal_smoothed_cornered.csv");

    std::cout << "Found " << h_corners.size() - invalid_count << " valid pieces \n\n";
    DEBUG_TIMER_STOP(corner_validation, "Corner Validation");

    // grayscale_image_t grayImage_cleaned = image;
    // grayscale_image_t corner_debug = visualize_corners(grayImage_cleaned, h_smoothed, h_corners);

    // save_gray_image(corner_debug, "../data/processed/detected_corners.png");

    //------------------------------------------------------
    //  CLASSIFICATION
    //------------------------------------------------------
    DEBUG_TIMER_START(classification);
    float edge_threshold = 4.0f;
    thrust::host_vector<ClassifiedPiece> classified = classify_edges(h_smoothed, h_corners, centroids, edge_threshold);
    DEBUG_TIMER_STOP(classification, "Classification");

    // for (const auto& p : classified) {
    //     if (!p.valid) {
    //         std::cout << "label=" << p.label
    //                 << " classification failed\n";
    //         continue;
    //     }

    //     std::cout << "label=" << p.label << " class=";

    //     for (char e : p.edges) {
    //         std::cout << e;
    //     }

    //     std::cout << " deviations=[";

    //     for (float d : p.edge_deviations) {
    //         std::cout << d << " ";
    //     }

    //     std::cout << "]\n";
    // }

    DEBUG_TIMER_START(visualization);
    visualizer(original_image, d_smoothed, h_smoothed, h_corners, classified);
    DEBUG_TIMER_STOP(visualization, "Visualization");

    DEBUG_TIMER_START(saving);
    image_path.erase(image_path.find_last_of('.'));        // remove extension
    image_path.erase(0, image_path.find_last_of('/') + 1); // remove directories
    std::string output_path = "../data/processed/" + image_path + "_classified.jpg";
    save_rgb_image(original_image, output_path);
    DEBUG_TIMER_STOP(saving, "Saving Output Image");

    GLOBAL_TIMER_STOP(full_execution, "Full Execution");

    return 0;
}
