#pragma once

#include <cstdint>
#include <vector>
#include "corner_detection.cuh"

struct ClassifiedPiece {
    uint32_t label;              // component label / piece id
    std::vector<char> edges;     // 4 entries: 'L', 'V', 'C'
    std::vector<float> edge_deviations;
    std::vector<uint32_t> corner_ranks;
    bool valid;
};

thrust::host_vector<ClassifiedPiece> classify_edges(
    const thrust::host_vector<ContourSignalPointExt>& h_smoothed,
    const thrust::host_vector<PieceCorners>& corners,
    const thrust::host_vector<Centroid>& centroids,
    float edge_threshold
);
