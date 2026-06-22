#pragma once

#include <cstdint>
#include <thrust/host_vector.h>

#include "centroid.cuh"
#include "pipeline_workspace.hpp"

struct ContourSignalPoint {
    float distance;
    uint32_t x;
    uint32_t y;
    uint32_t label;
    uint32_t rank;
};

// struct ContourSignalPointExt : ContourSignalPoint {
//     float piece_length;
// };

struct ContourSignalBuildResult {
    thrust::device_vector<Centroid> centroids;
    thrust::device_vector<ContourSignalPoint> points;
};

/// @brief Optimized function to build a sorted contour signal from the boundary and Centroid data.
/// @param d_boundaries A device pointer to the boundary image (uint32_t per pixel).
/// @param d_ranks A device pointer to the rank image (uint32_t per pixel).
/// @param width The width of the image.
/// @param height The height of the image.
/// @param ws The pipeline workspace.
/// @param stream The CUDA stream.
/// @return The result containing the sorted contour signal.
ContourSignalBuildResult build_sorted_contour_signal(
    const uint32_t* d_boundaries,
    const uint32_t* d_ranks,
    uint32_t width,
    uint32_t height,
    pipeline_workspace_t& ws,
    cudaStream_t stream = 0
);
