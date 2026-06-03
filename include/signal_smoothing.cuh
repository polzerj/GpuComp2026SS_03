#pragma once
#include "sort_distances.cuh"

struct ContourSignalPointExt {
    float distance;
    uint32_t x;
    uint32_t y;
    uint32_t label;
    uint32_t rank;
    uint32_t piece_length;
};

thrust::device_vector<ContourSignalPointExt> smooth_signal(const thrust::host_vector<ContourSignalPoint>& h_points,  thrust::device_vector<ContourSignalPoint>& d_input,
    uint32_t smoothing_radius);