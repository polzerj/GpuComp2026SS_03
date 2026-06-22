#include "signal_smoothing.cuh"

#include <cstdint>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include "timer.hpp"

__global__ void mark_piece_lengths_by_start_kernel(
    const ContourSignalPoint* points,
    uint32_t num_points,
    uint32_t* length_at_start
) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= num_points) {
        return;
    }

    uint32_t label = points[idx].label;

    bool is_last_point_of_piece =
        (idx + 1 == num_points) ||
        (points[idx + 1].label != label);

    if (is_last_point_of_piece) {
        uint32_t length = points[idx].rank + 1;
        uint32_t piece_start = idx + 1 - length;

        if (piece_start < num_points) {
            length_at_start[piece_start] = length;
        }
    }
}

__global__ void smooth_signal_kernel(
    const ContourSignalPoint* input,
    ContourSignalPointExt* output,
    const uint32_t* length_at_start,
    uint32_t num_points,
    uint32_t radius
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= num_points) {
        return;
    }

    const ContourSignalPoint self = input[tid];

    ContourSignalPointExt out;
    out.distance = self.distance;
    out.x = self.x;
    out.y = self.y;
    out.label = self.label;
    out.rank = self.rank;
    out.piece_length = 0;

    uint32_t rank = self.rank;

    if (tid < rank) {
        output[tid] = out;
        return;
    }

    uint32_t start = tid - rank;

    if (start >= num_points) {
        output[tid] = out;
        return;
    }

    uint32_t length = length_at_start[start];

    out.piece_length = length;

    if (length == 0 ||
        rank >= length ||
        start + length > num_points) {
        output[tid] = out;
        return;
    }

    float sum = 0.0f;

    int r = static_cast<int>(radius);
    int len = static_cast<int>(length);
    int center_rank = static_cast<int>(rank);

    for (int offset = -r; offset <= r; ++offset) {
        int local_idx = center_rank + offset;

        if (local_idx < 0) {
            local_idx += len;
        } else if (local_idx >= len) {
            local_idx -= len;
        }

        uint32_t sample_idx = start + static_cast<uint32_t>(local_idx);
        sum += input[sample_idx].distance;
    }

    float inv_count = 1.0f / static_cast<float>(2 * radius + 1);
    out.distance = sum * inv_count;

    output[tid] = out;
}

thrust::device_vector<ContourSignalPointExt> smooth_signal(
    const thrust::device_vector<ContourSignalPoint>& d_input,
    uint32_t smoothing_radius,
    pipeline_workspace_t& ws,
    cudaStream_t stream
) {
    uint32_t num_points = static_cast<uint32_t>(d_input.size());

    if (num_points == 0) {
        return thrust::device_vector<ContourSignalPointExt>();
    }

    ws.d_smooth_length_at_start.ensure(num_points);

    uint32_t* d_length_at_start = ws.d_smooth_length_at_start.data();

    CUDA_CHECK(cudaMemsetAsync(
        d_length_at_start,
        0,
        static_cast<size_t>(num_points) * sizeof(uint32_t),
        stream
    ));

    thrust::device_vector<ContourSignalPointExt> d_output(num_points);

    uint32_t threads = 256;
    uint32_t blocks = (num_points + threads - 1) / threads;

    DETAILED_TIMER_START(mark_piece_lengths);

    mark_piece_lengths_by_start_kernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_input.data()),
        num_points,
        d_length_at_start
    );

    CUDA_KERNEL_CHECK();

    DETAILED_TIMER_STOP(mark_piece_lengths, "Mark piece lengths");

    DETAILED_TIMER_START(smooth_signal_kernel);

    smooth_signal_kernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_output.data()),
        d_length_at_start,
        num_points,
        smoothing_radius
    );

    CUDA_KERNEL_CHECK();

    DETAILED_TIMER_STOP(smooth_signal_kernel, "Smooth signal kernel");

    return d_output;
}

thrust::device_vector<ContourSignalPointExt> smooth_signal(
    const thrust::device_vector<ContourSignalPoint>& d_input,
    uint32_t smoothing_radius,
    pipeline_workspace_t& ws
) {
    return smooth_signal(d_input, smoothing_radius, ws, 0);
}
