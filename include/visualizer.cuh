#pragma once
#include "sort_distances.cuh"
#include "signal_smoothing.cuh"
#include "types.hpp"
#include "corner_detection.cuh"
#include "classification.cuh"
#include "types.hpp"


void visualizer(
    rgb_image_t &original_image,
    const thrust::device_vector<ContourSignalPointExt>& d_points,
    const thrust::host_vector <ContourSignalPointExt>& points,
    const thrust::host_vector<PieceCorners>& corners,
    const thrust::host_vector<ClassifiedPiece>& classified
);

