#pragma once
#include "types.hpp"
#include "pipeline_workspace.hpp"

uint8_t *piece_segmentation(const brightness_t *d_input, int width, int height, size_t kernel_size, pipeline_workspace_t& ws);
