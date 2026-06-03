#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <string>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>

#include "types.hpp"
#include "preprocessing.cuh"
#include "piece_segmentation.cuh"
#include "ccl.cuh"
#include "pointerjumping.cuh"
#include "sort_distances.cuh"
#include "signal_smoothing.cuh"
#include "corner_detection.cuh"
#include "classification.cuh"

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

void save_contour_signal_csv(
    const thrust::host_vector<ContourSignalPoint>& points,
    const std::string& filename
) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + filename);
    }

    file << "label,rank,x,y,distance\n";

    for (const auto& p : points) {
        file << p.label << ","
             << p.rank << ","
             << p.x << ","
             << p.y << ","
             << p.distance << "\n";
    }
}

void save_contour_signal_csv(
    const thrust::host_vector<ContourSignalPointExt>& points,
    const std::string& filename
) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + filename);
    }

    file << "label,rank,x,y,distance,piece_length\n";

    for (const auto& p : points) {
        file << p.label << ","
             << p.rank << ","
             << p.x << ","
             << p.y << ","
             << p.distance << ","
             << p.piece_length << "\n";
    }
}

grayscale_image_t visualize_corners(
    const grayscale_image_t& base,
    const thrust::host_vector<ContourSignalPointExt>& points,
    const thrust::host_vector<PieceCorners>& corners
) {
    grayscale_image_t out = base;

    // darken original image so corner markers are visible
    for (auto& px : out.data) {
        px = px / 3;
    }

    for (const auto& piece : corners) {
        if (!piece.valid) {
            printf("Piece not valid\n");
            continue;
        }

        for (uint32_t idx : piece.point_indices) {
            const auto& p = points[idx];

            int cx = static_cast<int>(p.x);
            int cy = static_cast<int>(p.y);

            // draw white square marker
            for (int dy = -10; dy <= 10; ++dy) {
                for (int dx = -10; dx <= 10; ++dx) {
                    int x = cx + dx;
                    int y = cy + dy;

                    if (x >= 0 && x < out.width &&
                        y >= 0 && y < out.height) {
                        out.data[y * out.width + x] = 255;
                    }
                }
            }
        }
    }

    return out;
}

int save_rgb_image(const rgb_image_t& image, const std::string& filename) {
    int channels = 3;
    int stride = image.width * channels;

    return stbi_write_png(
        filename.c_str(),
        image.width,
        image.height,
        channels,
        image.data.data(),
        stride
    );
}

static void set_rgb_pixel(
    rgb_image_t& img,
    int x,
    int y,
    uint8_t r,
    uint8_t g,
    uint8_t b
) {
    if (x < 0 || x >= img.width || y < 0 || y >= img.height) {
        return;
    }

    int idx = 3 * (y * img.width + x);
    img.data[idx + 0] = r;
    img.data[idx + 1] = g;
    img.data[idx + 2] = b;
}

static void draw_square(
    rgb_image_t& img,
    int cx,
    int cy,
    int radius,
    uint8_t r,
    uint8_t g,
    uint8_t b
) {
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            set_rgb_pixel(img, cx + dx, cy + dy, r, g, b);
        }
    }
}

static void color_for_edge_class(
    char edge_class,
    uint8_t& r,
    uint8_t& g,
    uint8_t& b
) {
    switch (edge_class) {
        case 'L': // straight
            r = 255; g = 255; b = 255;
            break;

        case 'V': // knob / outward
            r = 255; g = 0; b = 0;
            break;

        case 'C': // hole / inward
            r = 0; g = 120; b = 255;
            break;

        default:
            r = 255; g = 0; b = 255;
            break;
    }
}

rgb_image_t visualize_edge_classes(
    const grayscale_image_t& base,
    const thrust::host_vector<ContourSignalPointExt>& points,
    const thrust::host_vector<PieceCorners>& corners,
    const thrust::host_vector<ClassifiedPiece>& classified
) {
    rgb_image_t out;
    out.width = base.width;
    out.height = base.height;
    out.data.resize(base.width * base.height * 3);

    // dark grayscale background
    for (int y = 0; y < base.height; ++y) {
        for (int x = 0; x < base.width; ++x) {
            uint8_t v = base.data[y * base.width + x] / 4;
            int idx = 3 * (y * base.width + x);

            out.data[idx + 0] = v;
            out.data[idx + 1] = v;
            out.data[idx + 2] = v;
        }
    }

    for (const auto& piece : classified) {
        if (!piece.valid || piece.edges.size() != 4) {
            continue;
        }

        const PieceCorners* piece_corners = nullptr;

        for (const auto& c : corners) {
            if (c.label == piece.label) {
                piece_corners = &c;
                break;
            }
        }

        if (piece_corners == nullptr ||
            !piece_corners->valid ||
            piece_corners->point_indices.size() != 4) {
            continue;
        }

        uint32_t first_corner_idx = piece_corners->point_indices[0];

        if (first_corner_idx >= points.size()) {
            continue;
        }

        uint32_t piece_start =
            first_corner_idx - points[first_corner_idx].rank;

        uint32_t piece_length =
            points[first_corner_idx].piece_length;

        if (piece_length == 0 || piece_start + piece_length > points.size()) {
            continue;
        }

        for (uint32_t edge = 0; edge < 4; ++edge) {
            uint32_t a_idx = piece_corners->point_indices[edge];
            uint32_t b_idx = piece_corners->point_indices[(edge + 1) % 4];

            if (a_idx >= points.size() || b_idx >= points.size()) {
                continue;
            }

            uint32_t rank_a = points[a_idx].rank;
            uint32_t rank_b = points[b_idx].rank;

            uint32_t edge_len =
                (rank_b + piece_length - rank_a) % piece_length;

            uint8_t r, g, b;
            color_for_edge_class(piece.edges[edge], r, g, b);

            // draw all contour points belonging to this edge
            for (uint32_t d = 0; d <= edge_len; ++d) {
                uint32_t rank = (rank_a + d) % piece_length;
                uint32_t idx = piece_start + rank;

                if (idx >= points.size()) {
                    continue;
                }

                const auto& p = points[idx];

                // slightly thick contour line
                draw_square(
                    out,
                    static_cast<int>(p.x),
                    static_cast<int>(p.y),
                    1,
                    r, g, b
                );
            }

            // draw corners in green on top
            const auto& a = points[a_idx];
            const auto& bpt = points[b_idx];

            draw_square(out, a.x, a.y, 4, 0, 255, 0);
            draw_square(out, bpt.x, bpt.y, 4, 0, 255, 0);
        }
    }

    return out;
}

__global__ void helloFromDevice() {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    return;
}

int main()
{
    helloFromDevice<<<2, 4>>>();
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
    auto piece_segmentation_duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(piece_segmentation_time_end - piece_segmentation_time_start);
    std::cout << "Piece segmentation time: "
              << piece_segmentation_duration_ms.count()
              << " microseconds\n\n";

    save_gray_image(image,"../data/processed/after_denoising.png");

    std:: cout << "[CCL] starting... " << std::endl;
    grayscale_image_t& grayImage_cleaned = image;

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
    arrays.successors = (uint32_t*)malloc(grayImage_cleaned.width* grayImage_cleaned.height * sizeof(unsigned int));
    arrays.ranks = (uint32_t*)malloc(grayImage_cleaned.width* grayImage_cleaned.height * sizeof(unsigned int));

    pointer_jumping(global_labels_host,arrays , grayImage_cleaned.width,grayImage_cleaned.height);
    auto pointer_jumping_time_end = std::chrono::high_resolution_clock::now();

    std:: cout << "[Pointer jumping] done" << std::endl;
    auto pointer_jumping_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pointer_jumping_time_end - pointer_jumping_time_start);

    std::cout << "Pointer jumping time: "
              << pointer_jumping_duration_ms.count()
              << " milliseconds\n\n";
    grayscale_image_t boundary_image = visualize_cll(arrays.boundaries, grayImage_cleaned);
    save_gray_image(boundary_image, "../data/processed/extracted_boundaries.png");

    auto centroid_time_start = std::chrono::high_resolution_clock::now();

    thrust::host_vector<Centroid> centroids = compute_centroids(arrays.boundaries, grayImage_cleaned.width, grayImage_cleaned.height);

    auto centroid_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - centroid_time_start);

    std::cout << "Centroid time: "
              << centroid_duration_ms.count()
              << " milliseconds\n\n";

    auto sort_time_start = std::chrono::high_resolution_clock::now();

    thrust::host_vector<ContourSignalPoint> h_points = sort_distances(arrays.boundaries, arrays.ranks, centroids, grayImage_cleaned.width, grayImage_cleaned.height);

    auto sort_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - sort_time_start);

     std::cout << "Sort time: "
              << sort_duration_ms.count()
              << " milliseconds\n\n";

    for (size_t i = 0; i < std::min<size_t>(h_points.size(), 100); ++i) {
        const auto& p = h_points[i];
        printf("label=%u rank=%u point=(%u,%u) dist=%f\n",
            p.label, p.rank, p.x, p.y, p.distance);
    }

    save_contour_signal_csv(
        h_points,
        "../data/processed/contour_signal.csv"
    );

    std::cout << "Saved contour signal with "
            << h_points.size()
            << " boundary points.\n";

    auto smoothing_time_start = std::chrono::high_resolution_clock::now();
    uint32_t smoothing_radius = 4; // magic value -- if increased too much corners are hard to detect
    thrust::host_vector<ContourSignalPointExt> h_smoothed = smooth_signal(h_points, smoothing_radius);
    auto smoothing_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - smoothing_time_start);
    std::cout << "Smoothing time: "
              << smoothing_duration_ms.count()
              << " milliseconds\n\n";


    save_contour_signal_csv(h_smoothed, "../data/processed/contour_signal_smoothed.csv");

    auto corner_detection_start = std::chrono::high_resolution_clock::now();
    uint32_t peak_detection_radius = 100; // magic value -- lower makes it detect incorrect corners
    float prominece_threshold = 10.0f; // magic value
    uint32_t distance_threshold = 0;
    thrust::host_vector<PieceCorners> h_corners = detect_corners(h_smoothed, peak_detection_radius, prominece_threshold, distance_threshold);
    auto corner_det_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - corner_detection_start);
    std::cout << "Corner detection time: "
              << corner_det_time_ms.count()
              << " milliseconds\n\n";

    auto piece_id = 0; auto invalid_count = 0;
    std::cout << "Invalid Pieces: \n";
    for(const auto& c : h_corners){
        piece_id++;
        if(c.point_indices.size()!= 4){
            std::cout << " " << piece_id << " with " << c.point_indices.size() << " corners\n";
            invalid_count++;
         }
        for(uint8_t i= 0; i< c.point_indices.size(); ++i){
            h_smoothed[c.point_indices.at(i)].distance = 0;
        }
    }
    std::cout << "\n\n";
    save_contour_signal_csv(h_smoothed, "../data/processed/contour_signal_smoothed_cornered.csv");

    std::cout << "Found " <<h_corners.size() -  invalid_count << " valid pieces \n\n";

    grayscale_image_t corner_debug = visualize_corners(grayImage_cleaned, h_smoothed, h_corners);

    save_gray_image(corner_debug, "../data/processed/detected_corners.png");

    auto classification_start = std::chrono::high_resolution_clock::now();
    float edge_threshold = 3.0f;
    thrust::host_vector<ClassifiedPiece> classified = classify_edges(h_smoothed, h_corners, centroids, edge_threshold);
    auto classification_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - classification_start);
    std::cout << "Classification time: "
              << classification_time_ms.count()
              << " microseconds\n\n";

    for (const auto& p : classified) {
        if (!p.valid) {
            std::cout << "label=" << p.label
                    << " classification failed\n";
            continue;
        }

        std::cout << "label=" << p.label << " class=";

        for (char e : p.edges) {
            std::cout << e;
        }

        std::cout << " deviations=[";

        for (float d : p.edge_deviations) {
            std::cout << d << " ";
        }

        std::cout << "]\n";
    }

    rgb_image_t class_debug = visualize_edge_classes(grayImage_cleaned, h_smoothed, h_corners, classified);
    save_rgb_image(class_debug, "../data/processed/classified_edges.png");
    
    return 0;
}
