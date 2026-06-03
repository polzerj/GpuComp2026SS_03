#include "classification.cuh"

static uint32_t edge_length_between(uint32_t rank_a, uint32_t rank_b, uint32_t piece_length
) {
    return (rank_b + piece_length - rank_a) % piece_length;
}

static char classify_single_edge(
    const thrust::host_vector<ContourSignalPointExt>& points,
    uint32_t piece_start,
    uint32_t piece_length,
    uint32_t corner_a_idx,
    uint32_t corner_b_idx,
    float edge_threshold,
    float& mean_deviation_out
) {
    uint32_t rank_a = points[corner_a_idx].rank;
    uint32_t rank_b = points[corner_b_idx].rank;

    uint32_t edge_len = edge_length_between(rank_a, rank_b, piece_length);

    mean_deviation_out = 0.0f;

    float dist_a = points[corner_a_idx].distance;
    float dist_b = points[corner_b_idx].distance;

    uint32_t from = edge_len / 4;
    uint32_t to   = (3 * edge_len) / 4;

    float deviation_sum = 0.0f;
    uint32_t count = 0;

    for (uint32_t d = from; d <= to; ++d) {
        uint32_t rank = (rank_a + d) % piece_length;
        uint32_t idx = piece_start + rank;

        float t = d * 1.0 / edge_len;

        float baseline = (1.0f - t) * dist_a + t * dist_b; // line interpolation

        float deviation = points[idx].distance - baseline;

        deviation_sum += deviation;
        count++;
    }

    if (count == 0) {
        return 'L';
    }

    float mean_deviation = deviation_sum / count;
    mean_deviation_out = mean_deviation;

    if (mean_deviation > edge_threshold) {
        return 'V'; // knob / outward
    }

    if (mean_deviation < -edge_threshold) {
        return 'C'; // hole / inward
    }

    return 'L'; // straight
}

thrust::host_vector<ClassifiedPiece> classify_edges(
    const thrust::host_vector<ContourSignalPointExt>& h_smoothed,
    const thrust::host_vector<PieceCorners>& corners,
    float edge_threshold
){
    thrust::host_vector<ClassifiedPiece> result;

    for (const auto& piece : corners) {
        ClassifiedPiece classified;
        classified.label = piece.label;
        classified.valid = false;
        classified.edges.resize(4);
        classified.edge_deviations.resize(4);
        classified.corner_ranks = piece.corner_ranks;

        uint32_t first_corner_idx = piece.point_indices[0];
        uint32_t piece_start = first_corner_idx - h_smoothed[first_corner_idx].rank;
        uint32_t piece_length = h_smoothed[first_corner_idx].piece_length;
        for (int edge = 0; edge < 4; edge++)
        {
            uint32_t corner_idx_a = piece.point_indices[edge];
            uint32_t corner_idx_b = piece.point_indices[(edge + 1) % 4];
            float mean_deviation = 0.0f;

            char edge_label = classify_single_edge(
                h_smoothed,
                piece_start,
                piece_length,
                corner_idx_a,
                corner_idx_b,
                edge_threshold,
                mean_deviation
            );
            classified.edges[edge] = edge_label;
            classified.edge_deviations[edge] = mean_deviation;

        }
        classified.valid = true;
        result.push_back(classified);
    }
    return result;
}
