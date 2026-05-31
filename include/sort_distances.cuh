#pragma once

#include <cstdint>
#include <thrust/host_vector.h>

#include "centroid.cuh"

struct ContourSignalPoint {
    float distance;
    uint32_t x;
    uint32_t y;
    uint32_t label;
    uint32_t rank;
};

thrust::host_vector<ContourSignalPoint> sort_distances(
    const uint32_t* h_boundaries,
    const uint32_t* h_ranks,
    const thrust::host_vector<Centroid>& h_centroids,
    uint32_t width,
    uint32_t height
);