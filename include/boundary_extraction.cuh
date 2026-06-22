#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "types.hpp"

void extract_boundary(
    uint32_t* global_labels,
    uint32_t* extracted_boundary_image,
    uint32_t image_width,
    uint32_t image_height,
    cudaStream_t stream
);

void extract_boundary(
    uint32_t* global_labels,
    uint32_t* extracted_boundary_image,
    uint32_t image_width,
    uint32_t image_height
);
