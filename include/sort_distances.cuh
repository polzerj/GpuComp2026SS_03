#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "centroid.cuh"
#include "pipeline_workspace.hpp"

struct ContourSignalPoint {
    float distance;
    uint32_t x;
    uint32_t y;
    uint32_t label;
    uint32_t rank;
};

struct ContourSignalBuildResult {
    thrust::device_vector<Centroid> centroids;
    thrust::device_vector<ContourSignalPoint> points;
};

ContourSignalBuildResult build_sorted_contour_signal(
    const uint32_t* d_boundaries,
    const uint32_t* d_ranks,
    uint32_t width,
    uint32_t height,
    pipeline_workspace_t& ws,
    cudaStream_t stream
);

ContourSignalBuildResult build_sorted_contour_signal(
    const uint32_t* d_boundaries,
    const uint32_t* d_ranks,
    uint32_t width,
    uint32_t height,
    pipeline_workspace_t& ws
);

// Backward-compatible overloads.
// Prefer build_sorted_contour_signal(...), because it returns both centroids
// and sorted points from one boundary compaction and one sort pass.
thrust::device_vector<ContourSignalPoint> sort_distances(
    const uint32_t* d_boundaries,
    const uint32_t* d_ranks,
    const thrust::device_vector<Centroid>& d_centroids,
    uint32_t width,
    uint32_t height,
    pipeline_workspace_t& ws,
    cudaStream_t stream
);

thrust::device_vector<ContourSignalPoint> sort_distances(
    const uint32_t* d_boundaries,
    const uint32_t* d_ranks,
    const thrust::device_vector<Centroid>& d_centroids,
    uint32_t width,
    uint32_t height,
    pipeline_workspace_t& ws
);
