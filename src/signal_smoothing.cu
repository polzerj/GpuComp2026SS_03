#include "signal_smoothing.cuh"

__global__ void smooth_signal_kernel(
    const ContourSignalPoint* input,
    ContourSignalPointExt* output,
    const uint32_t* segment_length,
    uint32_t num_points,
    uint32_t radius
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid  >= num_points)
    {
        return;
    }
    uint32_t rank = input[tid].rank;
    uint32_t start = tid - rank;
    uint32_t length = segment_length[tid];

    float sum = 0.0f;
    uint32_t count = 0;

    int r = radius;

    for(int offset = -r; offset <= r; offset++){
        int local_idx = rank + offset;
        local_idx = ((local_idx % static_cast<int>(length)) + static_cast<int>(length)) % static_cast<int>(length);
        sum += input[start + local_idx].distance;
        count++;
    }
    output[tid].label = input[tid].label;
    output[tid].rank = input[tid].rank;
    output[tid].x = input[tid].x;
    output[tid].y = input[tid].y;
    output[tid].distance = sum * 1.0 / count;
    output[tid].piece_length = length;
}

thrust::device_vector<ContourSignalPointExt> smooth_signal(const thrust::host_vector<ContourSignalPoint>& h_points, thrust::device_vector<ContourSignalPoint>& d_input,   
    uint32_t smoothing_radius)
    {
        uint32_t num_points = static_cast<uint32_t>(h_points.size());

        if (num_points == 0) {
            return thrust::host_vector<ContourSignalPointExt>();;
        }

        thrust::host_vector<uint32_t> h_segment_length(num_points);

        uint32_t start = 0;

        while (start < num_points) {
            uint32_t label = h_points[start].label;

            uint32_t end = start + 1;
            while (end < num_points && h_points[end].label == label) {
                end++;
            }

            uint32_t length = end - start;

            for (uint32_t i = start; i < end; ++i) {
                h_segment_length[i] = length;
            }

            start = end;
        }
        
        //thrust::device_vector<ContourSignalPoint> d_input = h_points;
        thrust::device_vector<ContourSignalPointExt> d_output(num_points);

        thrust::device_vector<uint32_t> d_segment_length = h_segment_length;

        uint32_t threads = 256;
        uint32_t blocks = (num_points + threads - 1) / threads;

        smooth_signal_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_input.data()),
            thrust::raw_pointer_cast(d_output.data()),
            thrust::raw_pointer_cast(d_segment_length.data()),
            num_points,
            smoothing_radius
        );


        //thrust::host_vector<ContourSignalPointExt> h_smoothed = d_output;
        return d_output;
    }
