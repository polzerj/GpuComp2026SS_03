#include "classification.cuh"
#include "timer.hpp"

#include <cmath>
#include <unordered_map>
#include <utility>

static uint32_t edge_length_between(
    uint32_t rank_a,
    uint32_t rank_b,
    uint32_t piece_length)
{
    return (rank_b + piece_length - rank_a) % piece_length;
}

static float cross2d(float ax, float ay, float bx, float by)
{
    return ax * by - ay * bx;
}

static char classify_single_edge(
    const thrust::host_vector<ContourSignalPointExt> &points,
    uint32_t piece_start,
    uint32_t piece_length,
    uint32_t corner_a_idx,
    uint32_t corner_b_idx,
    float center_x,
    float center_y,
    float edge_threshold,
    float &mean_deviation_out)
{
    mean_deviation_out = 0.0f;

    const auto &a = points[corner_a_idx];
    const auto &b = points[corner_b_idx];

    uint32_t rank_a = a.rank;
    uint32_t rank_b = b.rank;

    uint32_t edge_len = edge_length_between(rank_a, rank_b, piece_length);

    float ax = static_cast<float>(a.x);
    float ay = static_cast<float>(a.y);
    float bx = static_cast<float>(b.x);
    float by = static_cast<float>(b.y);

    float abx = bx - ax;
    float aby = by - ay;

    float ab_len = std::sqrt(abx * abx + aby * aby);

    float acx = center_x - ax;
    float acy = center_y - ay;

    float center_side = cross2d(abx, aby, acx, acy);
    float center_sign = center_side > 0.0f ? 1.0f : -1.0f;

    uint32_t from = edge_len / 4;
    uint32_t to = (3 * edge_len) / 4;

    float deviation_sum = 0.0f;
    uint32_t count = 0;

    for (uint32_t d = from; d <= to; ++d)
    {
        uint32_t rank = (rank_a + d) % piece_length;
        uint32_t idx = piece_start + rank;

        if (idx >= points.size())
        {
            continue;
        }

        const auto &p = points[idx];

        float px = static_cast<float>(p.x);
        float py = static_cast<float>(p.y);

        float apx = px - ax;
        float apy = py - ay;

        float point_side = cross2d(abx, aby, apx, apy);
        float signed_outward_distance = -(point_side / ab_len) * center_sign;

        deviation_sum += signed_outward_distance;
        count++;
    }

    if (count == 0)
    {
        return 'L';
    }

    float mean_deviation = deviation_sum / static_cast<float>(count);
    mean_deviation_out = mean_deviation;

    if (mean_deviation > edge_threshold)
    {
        return 'V'; // knob / outward
    }

    if (mean_deviation < -edge_threshold)
    {
        return 'C'; // hole / inward
    }

    return 'L';
}

thrust::host_vector<ClassifiedPiece> classify_edges(
    const thrust::host_vector<ContourSignalPointExt> &h_smoothed,
    const thrust::host_vector<PieceCorners> &corners,
    const thrust::host_vector<Centroid> &centroids,
    float edge_threshold)
{
    thrust::host_vector<ClassifiedPiece> result;

    std::unordered_map<uint32_t, std::pair<float, float>> centroid_by_label;

    for (const auto &c : centroids)
    {
        centroid_by_label[c.label] = {c.x, c.y};
    }

    for (const auto &piece : corners)
    {
        ClassifiedPiece classified;
        classified.label = piece.label;
        classified.valid = false;
        classified.corner_ranks = piece.corner_ranks;
        classified.edges.resize(4);
        classified.edge_deviations.resize(4);

        if (!piece.valid ||
            piece.point_indices.size() != 4 ||
            piece.corner_ranks.size() != 4)
        {
            result.push_back(classified);
            continue;
        }

        auto centroid_it = centroid_by_label.find(piece.label);

        if (centroid_it == centroid_by_label.end())
        {
            result.push_back(classified);
            continue;
        }

        float center_x = centroid_it->second.first;
        float center_y = centroid_it->second.second;

        uint32_t first_corner_idx = piece.point_indices[0];

        if (first_corner_idx >= h_smoothed.size())
        {
            result.push_back(classified);
            continue;
        }

        uint32_t piece_start = first_corner_idx - h_smoothed[first_corner_idx].rank;

        uint32_t piece_length = h_smoothed[first_corner_idx].piece_length;

        if (piece_length == 0 || piece_start + piece_length > h_smoothed.size())
        {
            result.push_back(classified);
            continue;
        }

        bool ok = true;

        for (uint32_t edge = 0; edge < 4; ++edge)
        {
            uint32_t corner_a_idx = piece.point_indices[edge];
            uint32_t corner_b_idx = piece.point_indices[(edge + 1) % 4];

            float mean_deviation = 0.0f;

            char edge_label = classify_single_edge(
                h_smoothed,
                piece_start,
                piece_length,
                corner_a_idx,
                corner_b_idx,
                center_x,
                center_y,
                edge_threshold,
                mean_deviation);

            classified.edges[edge] = edge_label;
            classified.edge_deviations[edge] = mean_deviation;

            if (corner_a_idx >= h_smoothed.size() ||
                corner_b_idx >= h_smoothed.size())
            {
                ok = false;
            }
        }

        classified.valid = ok;
        result.push_back(classified);
    }

    return result;
}
