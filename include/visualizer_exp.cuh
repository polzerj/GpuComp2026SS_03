#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <thrust/host_vector.h>
#include "types.hpp"
#include "signal_smoothing.cuh"
#include "corner_detection.cuh"
#include "classification.cuh"

struct issued_pixel_t {
    uint32_t pixel_index;
    uchar3 rgb;
};

struct visualizer_workspace_t {
    issued_pixel_t* d_pixels = nullptr;
    size_t capacity = 0;

    uchar3* d_image_copy = nullptr;
    size_t image_capacity = 0;
};

struct jpeg_encoder_t {
    nvjpegHandle_t handle = nullptr;
    nvjpegEncoderState_t state = nullptr;
    nvjpegEncoderParams_t params = nullptr;

    std::vector<uint8_t> bitstream;
};

void init_jpeg_encoder(
    jpeg_encoder_t& enc,
    int quality = 50
);

void destroy_jpeg_encoder(
    jpeg_encoder_t& enc
);

void free_visualizer_workspace(
    visualizer_workspace_t& ws
);

void visualizer_exp(
    rgb_image_d_t* original_image_d,
    const std::string& filename,
    uint32_t image_width,
    uint32_t image_height,
    const thrust::host_vector<ContourSignalPointExt>& points,
    const thrust::host_vector<PieceCorners>& corners,
    const thrust::host_vector<ClassifiedPiece>& classified,
    visualizer_workspace_t& ws,
    jpeg_encoder_t& jpeg_encoder,
    cudaStream_t stream
);

void visualizer_exp(
    rgb_image_d_t* original_image_d,
    const std::string& filename,
    uint32_t image_width,
    uint32_t image_height,
    const thrust::host_vector<ContourSignalPointExt>& points,
    const thrust::host_vector<PieceCorners>& corners,
    const thrust::host_vector<ClassifiedPiece>& classified,
    visualizer_workspace_t& ws,
    jpeg_encoder_t& jpeg_encoder
);
