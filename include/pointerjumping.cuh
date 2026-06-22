#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "types.hpp"
#include "pipeline_workspace.hpp"

typedef struct pointer_jumping_outputs {
    uint32_t* successors;
    uint32_t* ranks;
    uint32_t* boundaries;
} ptr_jmp_out_t;

ptr_jmp_out_t pointer_jumping(
    uint32_t* global_labels,
    uint32_t image_width,
    uint32_t image_height,
    pipeline_workspace_t& ws,
    cudaStream_t stream
);

ptr_jmp_out_t pointer_jumping(
    uint32_t* global_labels,
    uint32_t image_width,
    uint32_t image_height,
    pipeline_workspace_t& ws
);

ptr_jmp_out_t pointer_jumping_old(
    uint32_t* global_labels,
    uint32_t image_width,
    uint32_t image_height,
    pipeline_workspace_t& ws,
    cudaStream_t stream
);

ptr_jmp_out_t pointer_jumping_old(
    uint32_t* global_labels,
    uint32_t image_width,
    uint32_t image_height,
    pipeline_workspace_t& ws
);
