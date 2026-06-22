#pragma once

#include <string>
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include "types.hpp"
#include "pipeline_workspace.hpp"

struct jpeg_decoder_t {
    nvjpegHandle_t handle = nullptr;
    nvjpegJpegState_t state = nullptr;
};

void init_jpeg_decoder(jpeg_decoder_t& dec);

void destroy_jpeg_decoder(jpeg_decoder_t& dec);

rgb_image_d_t preprocessing(
    const std::string& image_path,
    uint8_t*& d_result,
    pipeline_workspace_t& ws,
    jpeg_decoder_t& jpeg_decoder,
    cudaStream_t stream
);

rgb_image_d_t preprocessing(
    const std::string& image_path,
    uint8_t*& d_result,
    pipeline_workspace_t& ws,
    jpeg_decoder_t& jpeg_decoder
);
