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

thrust::device_vector<ContourSignalPoint> sort_distances(
    const uint32_t* d_boundaries,
    const uint32_t* d_ranks,
    const thrust::device_vector<Centroid>& d_centroids,
    uint32_t width,
    uint32_t height,
    pipeline_workspace_t& ws
);
