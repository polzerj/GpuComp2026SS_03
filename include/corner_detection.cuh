#pragma once

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "sort_distances.cuh"
#include "signal_smoothing.cuh"
#include "pipeline_workspace.hpp"

struct PieceCorners {
    uint32_t label;
    std::vector<uint32_t> corner_ranks;
    std::vector<uint32_t> point_indices; // indices into the sorted/smoothed vector
    bool valid;
};

thrust::host_vector<PieceCorners> detect_corners(
    thrust::host_vector<ContourSignalPointExt>& h_smoothed,
    thrust::device_vector<ContourSignalPointExt>& d_smoothed,
    uint32_t peak_detection_radius,
    float prominence_threshold,
    uint32_t distance_threshold,
    pipeline_workspace_t& ws,
    cudaStream_t stream
);

thrust::host_vector<PieceCorners> detect_corners(
    thrust::host_vector<ContourSignalPointExt>& h_smoothed,
    thrust::device_vector<ContourSignalPointExt>& d_smoothed,
    uint32_t peak_detection_radius,
    float prominence_threshold,
    uint32_t distance_threshold,
    pipeline_workspace_t& ws
);
