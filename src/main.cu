#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// define which formats to export
#define EXPORT_AS_PNG (0)
#define EXPORT_AS_JPG (0)
#define EXPORT_AS_BMP (1)

#include <string>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <filesystem>
#include <future>
#include <vector>
#include <cstdlib>

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
#include "visualizer_exp.cuh"
#include "pipeline_workspace.hpp"

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

static int requested_stream_count(int image_count)
{
    int default_streams = std::min(3, std::max(1, image_count));
    const char *env = std::getenv("PUZZLE_STREAMS");

    if (env == nullptr)
    {
        return default_streams;
    }

    try
    {
        int value = std::stoi(env);

        // 1..4 is the useful range for this pipeline.
        // 6 streams caused nvJPEG/resource pressure and worse timings.
        return std::clamp(value, 1, std::min(4, std::max(1, image_count)));
    }
    catch (...)
    {
        return default_streams;
    }
}

struct image_context_t
{
    cudaStream_t stream = nullptr;
    pipeline_workspace_t ws{};
    visualizer_workspace_t vis_ws{};
    jpeg_decoder_t jpeg_decoder{};
    jpeg_encoder_t jpeg_encoder{};
};

static void init_image_context(image_context_t &ctx)
{
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking));
    ctx.ws.stream = ctx.stream;
    ctx.vis_ws.stream = ctx.stream;

    init_jpeg_decoder(ctx.jpeg_decoder);
    init_jpeg_encoder(ctx.jpeg_encoder, 50);
}

static void destroy_image_context(image_context_t &ctx)
{
    if (ctx.stream != nullptr)
    {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    }

    free_visualizer_workspace(ctx.vis_ws);
    destroy_jpeg_decoder(ctx.jpeg_decoder);
    destroy_jpeg_encoder(ctx.jpeg_encoder);

    if (ctx.stream != nullptr)
    {
        cudaStreamDestroy(ctx.stream);
        ctx.stream = nullptr;
    }
}

static bool validate_image_path(const std::string& image_path)
{
    if (std::filesystem::exists(image_path) == false)
    {
        std::cerr << "File does not exist or is not a regular file: "
                  << image_path << std::endl;
        return false;
    }

    if (std::filesystem::is_directory(image_path))
    {
        std::cerr << "Provided path is a directory, not a file: "
                  << image_path << std::endl;
        return false;
    }

    return true;
}

__global__ void helloFromDevice()
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    return;
}

void process_image(const std::string &image_path, image_context_t &ctx)
{
    GLOBAL_TIMER_START(images_execution);

    uint8_t *d_binary_image = nullptr;
    DEBUG_TIMER_START(preprocessing);
    rgb_image_d_t original_image_d = preprocessing(image_path, d_binary_image, ctx.ws, ctx.jpeg_decoder);
    DEBUG_TIMER_STOP(preprocessing, "Preprocessing");

    std::cout << "Image " << image_path << " loaded and converted to grayscale with size " << original_image_d.width << "x" << original_image_d.height << "." << std::endl;

    uint32_t image_width = original_image_d.width;
    uint32_t image_height = original_image_d.height;
    uint32_t pixel_count = image_width * image_height;

    // grayscale_image_t testimage;
    // testimage.data.resize(pixel_count);
    // testimage.width = image_width;
    // testimage.height = image_height;
    // cudaMemcpy(testimage.data.data(), d_binary_image, pixel_count, cudaMemcpyDeviceToHost);
    // save_gray_image(testimage,"../data/processed/grayimagefromcolour.png");

    //------------------------------------------------------
    // PIECE SEGMENTATION
    //------------------------------------------------------
    DEBUG_TIMER_START(piece_segmentation);
    uint8_t *piece_seg_out = piece_segmentation(d_binary_image, image_width, image_height, 5 , ctx.ws); // black-white enlargement
    DEBUG_TIMER_STOP(piece_segmentation, "Piece Segmentation");

    //cudaFree(d_binary_image); // not needed anymore, allocated in preprocessing and used in piece segmentation

    //------------------------------------------------------
    // CONNECTED COMPONENT LABELING
    //------------------------------------------------------
    DEBUG_TIMER_START(ccl);
    uint32_t *global_labels_d = playne_ccl(piece_seg_out, image_width, image_height, ctx.ws);
    DEBUG_TIMER_STOP(ccl, "CCL");

    //cudaFree(piece_seg_out);

    //------------------------------------------------------
    // POINTER JUMPING
    //------------------------------------------------------
    DEBUG_TIMER_START(pointer_jumping);
    ptr_jmp_out_t out_d = pointer_jumping(global_labels_d, image_width, image_height, ctx.ws);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    DEBUG_TIMER_STOP(pointer_jumping, "Pointer Jumping");

    //cudaFree(global_labels_d);

    //------------------------------------------------------
    // CENTROID CALUCLATION
    //------------------------------------------------------
    DEBUG_TIMER_START(centroid);
    thrust::device_vector<Centroid> d_centroids = compute_centroids(out_d.boundaries, image_width, image_height, ctx.ws);
    DEBUG_TIMER_STOP(centroid, "Centroid Calculation");
    DEBUG_TIMER_START(transfercentroid);
    thrust::host_vector<Centroid> centroids = d_centroids; // we need this for classification
    DEBUG_TIMER_STOP(transfercentroid, "Transfer Centroid");
    //------------------------------------------------------
    // SORTING SIGNALS
    //------------------------------------------------------
    DEBUG_TIMER_START(sorting);
    thrust::device_vector<ContourSignalPoint> d_points = sort_distances(out_d.boundaries, out_d.ranks, d_centroids, image_width, image_height, ctx.ws);
    DEBUG_TIMER_STOP(sorting, "Sorting Signals");
    DEBUG_TIMER_START(transfersorting);
    //thrust::host_vector<ContourSignalPoint> h_points = d_points; // this one is necessary for the steps after
    DEBUG_TIMER_STOP(transfersorting, "Transfer after Sort");
    //cudaFree(out_d.boundaries);
    //cudaFree(out_d.ranks);
    //cudaFree(out_d.successors);

    //------------------------------------------------------
    //  SMOOTHING SIGNALS
    //------------------------------------------------------
    DEBUG_TIMER_START(smoothing);
    uint32_t smoothing_radius = 2; // 4 is magic value -- if increased too much corners are hard to detect
    //thrust::device_vector<ContourSignalPointExt> d_smoothed = smooth_signal(h_points, d_points, smoothing_radius, ws);
    thrust::device_vector<ContourSignalPointExt> d_smoothed = smooth_signal(d_points, smoothing_radius, ctx.ws);
    DEBUG_TIMER_STOP(smoothing, "Smoothing Signals");

    DEBUG_TIMER_START(save_contour_signal);
    thrust::host_vector<ContourSignalPointExt> h_smoothed = d_smoothed; // this one is necessary for the steps after
    //save_contour_signal_csv(h_smoothed, "../data/processed/contour_signal_smoothed.csv");
    DEBUG_TIMER_STOP(save_contour_signal, "Saving Smoothed Contour Signal");

    //------------------------------------------------------
    //  CORNER DETECTION
    //------------------------------------------------------
    DEBUG_TIMER_START(corner_detection);
    uint32_t peak_detection_radius = 85; // magic value -- lower makes it detect incorrect corners -- higher makes it sometimes detect knobs as corners
    float prominece_threshold = 8.0f;    // magic value
    uint32_t distance_threshold = 0;

    thrust::host_vector<PieceCorners> h_corners = detect_corners(h_smoothed, d_smoothed, peak_detection_radius, prominece_threshold, distance_threshold, ctx.ws);
    DEBUG_TIMER_STOP(corner_detection, "Corner Detection");

    CPU_TIMER_START(corner_validation);

    // auto piece_id = 0;
    // auto invalid_count = 0;
    // for (const auto &c : h_corners)
    // {
    //     piece_id++;
    //     if (c.point_indices.size() != 4)
    //     {
    //         std::cout << " " << piece_id << " with " << c.point_indices.size() << " corners\n";
    //         invalid_count++;
    //     }
    //     for (uint8_t i = 0; i < c.point_indices.size(); ++i)
    //     {
    //         h_smoothed[c.point_indices.at(i)].distance = 0;
    //     }
    // }

    // std::cout << "Found " << h_corners.size() - invalid_count << " valid pieces \n\n";

    CPU_TIMER_STOP(corner_validation, "Corner Validation");

    //------------------------------------------------------
    //  CLASSIFICATION
    //------------------------------------------------------
    CPU_TIMER_START(classification);
    float edge_threshold = 15.0f;
    thrust::host_vector<ClassifiedPiece> classified = classify_edges(h_smoothed, h_corners, centroids, edge_threshold);
    CPU_TIMER_STOP(classification, "Classification");

    CPU_TIMER_START(visualization);

    std::string image_name = image_path;
    image_name.erase(image_name.find_last_of('.'));
    image_name.erase(0, image_name.find_last_of('/') + 1);

    std::string output_path =
        "../data/processed/" + image_name + "_classified.jpg";

    visualizer_exp(
        &original_image_d,
        output_path,
        image_width,
        image_height,
        h_smoothed,
        h_corners,
        classified,
        ctx.vis_ws,
        ctx.jpeg_encoder);
    //cudaFree(original_image_d.data);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CPU_TIMER_STOP(visualization, "Visualization and saving output image");

    GLOBAL_TIMER_STOP(images_execution, "Image Execution");
}

int main(int argc, char *argv[])
{
    const auto total_start = std::chrono::high_resolution_clock::now();
    GLOBAL_TIMER_START(total_execution);
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    std::vector<std::string> image_paths;

    // read in all image paths
    if (argc == 1)
    {
        // no image provided, use default
        image_paths.push_back("../data/1_p2.jpg");
    }
    else
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string image_path = argv[i];

            if (validate_image_path(image_path))
            {
                image_paths.push_back(image_path);
            }
        }
    }

    if (image_paths.empty())
    {
        // this might happen if there are paths given, but all of them are invalid
        std::cerr << "No valid images to process." << std::endl;
        return 1;
    }

    const int num_streams = requested_stream_count(static_cast<int>(image_paths.size()));

    std::cout << "Using " << num_streams << " CUDA stream(s). Set PUZZLE_STREAMS=N to override." << std::endl;

    std::vector<image_context_t> contexts(num_streams);

    for (auto &ctx : contexts)
    {
        init_image_context(ctx);
    }

    std::vector<std::future<void>> jobs(num_streams);

    for (size_t i = 0; i < image_paths.size(); ++i)
    {
        int slot = static_cast<int>(i % num_streams);

        if (jobs[slot].valid())
        {
            jobs[slot].get();
        }

        std::string image_path = image_paths[i];

        std::cout << std::endl
                  << "Processing image: "
                  << image_path
                  << std::endl;

        jobs[slot] = std::async(
            std::launch::async,
            [&, slot, image_path]()
            {
                CUDA_CHECK(cudaSetDevice(0));
                process_image(image_path, contexts[slot]);
            });
    }

    for (auto &job : jobs)
    {
        if (job.valid())
        {
            job.get();
        }
    }

    for (auto &ctx : contexts)
    {
        destroy_image_context(ctx);
    }

    GLOBAL_TIMER_STOP(total_execution, "Total Execution Time");
        const auto total_end =
        std::chrono::high_resolution_clock::now();

    const double total_ms =
        std::chrono::duration<double, std::milli>(
            total_end - total_start
        ).count();

    std::printf(
        "[cpu] %-45s %10.3f ms\n",
        "Total Execution Time",
        total_ms
    );

    return 0;
}
