#include "corner_detection.cuh"
#include <iostream>

__global__ void find_corners(const ContourSignalPointExt* d_smoothed,
    uint8_t* corners,
    float* scores,
    uint32_t peak_detection_radius,
    float prominence_threshold,
    uint32_t num_boundary)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_boundary)
            return;

        ContourSignalPointExt self = d_smoothed[idx];
        uint32_t length = self.piece_length;
        uint32_t rank = self.rank;
        uint32_t start_idx = idx - rank;
        uint32_t radius_max = peak_detection_radius;

        bool is_peak = true;
        float neighbor_sum = 0.0f;
        uint32_t neighbor_count = 0;

        for (uint32_t radius = 1; radius <= radius_max; ++radius) {
            uint32_t right_rank = (rank + radius) % length;
            uint32_t left_rank  = (rank + length - radius) % length;

            float right_distance = d_smoothed[start_idx + right_rank].distance;
            float left_distance  = d_smoothed[start_idx + left_rank].distance;

            if (self.distance <= right_distance || self.distance <= left_distance) {
                is_peak = false;
                break;
            }

            neighbor_sum += right_distance + left_distance;
            neighbor_count += 2;
        }

        if (!is_peak || neighbor_count == 0) {
            corners[idx] = 0;
            scores[idx] = 0.0f;
            return;
        }

        float neighbor_avg = neighbor_sum / static_cast<float>(neighbor_count);
        float prominence = self.distance - neighbor_avg;

        bool accepted = prominence >= prominence_threshold;

        corners[idx] = accepted ? 1 : 0;
        scores[idx] = accepted ? prominence : 0.0f;
}

struct CornerCandidate {
    uint32_t index;
    uint32_t rank;
    float score;
    float distance;
};

static uint32_t cyclic_distance(uint32_t a, uint32_t b, uint32_t length) {
    uint32_t diff = (a > b) ? (a - b) : (b - a);
    return std::min(diff, length - diff);
}

static PieceCorners select_corners_for_piece(
    const thrust::host_vector<ContourSignalPointExt>& h_smoothed,
    const thrust::host_vector<uint8_t>& h_is_corner,
    const thrust::host_vector<float>& h_scores,
    uint32_t start,
    uint32_t end,
    uint32_t min_corner_distance
) {
    PieceCorners piece;
    piece.label = h_smoothed[start].label;
    piece.valid = false;

    uint32_t length = end - start;

    if (min_corner_distance == 0) {
        min_corner_distance = std::max<uint32_t>(1, length / 8);
    }

    std::vector<CornerCandidate> candidates;

    for (uint32_t i = start; i < end; ++i) {
        if (h_is_corner[i] == 0) {
            continue;
        }

        candidates.push_back({
            i,
            h_smoothed[i].rank,
            h_scores[i],
            h_smoothed[i].distance
        });
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const CornerCandidate& a, const CornerCandidate& b) {
            return a.score > b.score;
        }
    );

    std::vector<CornerCandidate> selected;

    for (const auto& cand : candidates) {
        bool far_enough = true;

        for (const auto& chosen : selected) {
            uint32_t dist = cyclic_distance(
                cand.rank,
                chosen.rank,
                length
            );

            if (dist < min_corner_distance) {
                far_enough = false;
                break;
            }
        }

        if (far_enough) {
            selected.push_back(cand);
        }

        if (selected.size() == 4) {
            break;
        }
    }

    if (selected.size() != 4) {
        return piece;
    }

    std::sort(
        selected.begin(),
        selected.end(),
        [](const CornerCandidate& a, const CornerCandidate& b) {
            return a.rank < b.rank;
        }
    );

    for (const auto& c : selected) {
        piece.corner_ranks.push_back(c.rank);
        piece.point_indices.push_back(c.index);
    }

    piece.valid = true;
    return piece;
}

thrust::host_vector<PieceCorners> detect_corners(
    thrust::host_vector<ContourSignalPointExt>& h_smoothed,
    uint32_t peak_detection_radius,
    float prominence_threshold,
    uint32_t distance_threshold
) {
    uint32_t num_points = static_cast<uint32_t>(h_smoothed.size());

    if (num_points == 0) {
        return thrust::host_vector<PieceCorners>();
    }

    thrust::device_vector<ContourSignalPointExt> d_smoothed = h_smoothed;
    thrust::device_vector<uint8_t> d_corners(num_points, 0);
    thrust::device_vector<float> d_scores(num_points, 0.0f);

    uint32_t threads = 256;
    uint32_t blocks = (num_points + threads - 1) / threads;

    find_corners<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_smoothed.data()),
        thrust::raw_pointer_cast(d_corners.data()),
        thrust::raw_pointer_cast(d_scores.data()),
        peak_detection_radius,
        prominence_threshold,
        num_points
    );

    thrust::host_vector<uint8_t> h_corners_bool = d_corners;
    thrust::host_vector<float> h_scores = d_scores;

    thrust::host_vector<PieceCorners> h_corners;

    uint32_t start = 0;

    while (start < num_points) {
        uint32_t label = h_smoothed[start].label;
        uint32_t length = h_smoothed[start].piece_length;
        uint32_t end = start + length;

        PieceCorners piece = select_corners_for_piece(
            h_smoothed,
            h_corners_bool,
            h_scores,
            start,
            end,
            distance_threshold
        );

        h_corners.push_back(piece);

        start = end;
    }

    return h_corners;
}
