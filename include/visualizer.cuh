#pragma once
#include "sort_distances.cuh"
#include "signal_smoothing.cuh"
#include "types.hpp"
#include "corner_detection.cuh"
#include "classification.cuh"


rgb_image_t visualizer(
    const uint8_t* image_in_d, // this assumes for now its a greyscale image
    uint32_t width, uint32_t height,
    const thrust::device_vector<ContourSignalPointExt>& d_points,
    const thrust::host_vector <ContourSignalPointExt>& points,
    const thrust::host_vector<PieceCorners>& corners,
    const thrust::host_vector<ClassifiedPiece>& classified
);

struct DrawTask{

} typedef DrawTask_t;
