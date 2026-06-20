
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include "centroid.cuh"
#include "timer.hpp"

#define THRESHOLD 200

struct is_positive
{
    __device__ bool operator()(const uint32_t c) const
    {
        return c > 0;
    }
};

struct count_at_least {
    uint32_t threshold;

    __host__ __device__
    bool operator()(uint32_t c) const {
        return c >= threshold;
    }
};

__global__ void compute_sums_kernel(
    const uint32_t *image,
    uint32_t width,
    uint32_t height,
    unsigned long long *sum_x,
    unsigned long long *sum_y,
    uint32_t *count)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    uint32_t idx = y * width + x;
    uint32_t label = image[idx];

    if (label > 0)
    {
        atomicAdd(&sum_x[label], (unsigned long long)x);
        atomicAdd(&sum_y[label], (unsigned long long)y);
        atomicAdd(&count[label], 1);
    }
}

__global__ void compactCentroidsKernel(
    const unsigned long long *sum_x,
    const unsigned long long *sum_y,
    const uint32_t *count,
    const uint32_t *valid_labels,
    Centroid *centroids,
    uint32_t num_valid)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid)
        return;

    uint32_t label = valid_labels[idx];
    uint32_t count_label = count[label];
    centroids[idx].x = (count_label >= THRESHOLD) ? sum_x[label] * 1.0 / count_label : -1.0f;
    centroids[idx].y = (count_label >= THRESHOLD) ? sum_y[label] * 1.0 / count_label : -1.0f;
    centroids[idx].label = label;
}

thrust::device_vector<Centroid> compute_centroids(const uint32_t *d_labels_in, uint32_t width, uint32_t height, pipeline_workspace_t& ws)
{
    auto num_pixels = width * height;

    // thrust::device_vector<uint32_t> d_labels(h_labels, h_labels + num_pixels);
    thrust::device_ptr<const uint32_t> d_labels = thrust::device_pointer_cast(d_labels_in);

    DETAILED_TIMER_START(find_max_label);
    uint32_t max_label = thrust::reduce(d_labels, d_labels + num_pixels, 0u, thrust::maximum<uint32_t>());
    cudaDeviceSynchronize();
    DETAILED_TIMER_STOP(find_max_label, "Find max label");

    if (max_label == 0)
    {
        return thrust::device_vector<Centroid>();
    }

    DETAILED_TIMER_START(compute_sums);
    // thrust::device_vector<unsigned long long> d_sum_x(max_label + 1, 0ull);
    // thrust::device_vector<unsigned long long> d_sum_y(max_label + 1, 0ull);
    // thrust::device_vector<uint32_t> d_count(max_label + 1, 0u);
    uint32_t label_capacity = max_label + 1;

    ws.d_centroid_sum_x.ensure(label_capacity);
    ws.d_centroid_sum_y.ensure(label_capacity);
    ws.d_centroid_count.ensure(label_capacity);
    ws.d_centroid_indices.ensure(label_capacity);
    ws.d_centroid_valid_labels.ensure(label_capacity);

    unsigned long long* d_sum_x = ws.d_centroid_sum_x.data();
    unsigned long long* d_sum_y = ws.d_centroid_sum_y.data();
    uint32_t* d_count = ws.d_centroid_count.data();
    uint32_t* d_indices = ws.d_centroid_indices.data();
    uint32_t* d_valid_labels = ws.d_centroid_valid_labels.data();

    cudaMemset(d_sum_x, 0, label_capacity * sizeof(unsigned long long));
    cudaMemset(d_sum_y, 0, label_capacity * sizeof(unsigned long long));
    cudaMemset(d_count, 0, label_capacity * sizeof(uint32_t));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    compute_sums_kernel<<<grid, block>>>(
        d_labels_in,
        width,
        height,
        d_sum_x,
        d_sum_y,
        d_count
    );

    cudaDeviceSynchronize();
    DETAILED_TIMER_STOP(compute_sums, "Compute sums and counts");

    DETAILED_TIMER_START(count_valid);
    // thrust::device_vector<uint32_t> indices(max_label + 1);
    // thrust::sequence(indices.begin(), indices.end());

    // // Count directly without needing a mask vector
    // //uint32_t num_valid = thrust::count_if(d_count.begin(), d_count.end(), is_positive());
    // uint32_t num_valid = thrust::count_if(d_count.begin(), d_count.end(), count_at_least{THRESHOLD});
    thrust::device_ptr<uint32_t> count_begin = thrust::device_pointer_cast(d_count);

    thrust::device_ptr<uint32_t> indices_begin = thrust::device_pointer_cast(d_indices);

    thrust::device_ptr<uint32_t> valid_labels_begin = thrust::device_pointer_cast(d_valid_labels);

    thrust::sequence(indices_begin, indices_begin + label_capacity);

    uint32_t num_valid = thrust::count_if(
        count_begin,
        count_begin + label_capacity,
        count_at_least{THRESHOLD}
    );
    DETAILED_TIMER_STOP(count_valid, "Count valid labels");

    if (num_valid == 0)
    {
        return thrust::device_vector<Centroid>();
    }

    DETAILED_TIMER_START(copy_valid_labels);
    // thrust::device_vector<uint32_t> d_valid_labels(num_valid);

    // // copy_if can use d_count as the stencil directly
    // //thrust::copy_if(indices.begin(), indices.end(), d_count.begin(), d_valid_labels.begin(), is_positive());
    // thrust::copy_if(indices.begin(), indices.end(), d_count.begin(), d_valid_labels.begin(), count_at_least{THRESHOLD});
    thrust::copy_if(
        indices_begin,
        indices_begin + label_capacity,
        count_begin,
        valid_labels_begin,
        count_at_least{THRESHOLD}
    );
    DETAILED_TIMER_STOP(copy_valid_labels, "Copy valid labels");

    DETAILED_TIMER_START(compact_centroids);
    thrust::device_vector<Centroid> d_centroids(num_valid);
    dim3 block2(256);
    dim3 grid2((num_valid + block2.x - 1) / block2.x);

    compactCentroidsKernel<<<grid2, block2>>>(
        d_sum_x,
        d_sum_y,
        d_count,
        d_valid_labels,
        thrust::raw_pointer_cast(d_centroids.data()),
        num_valid
    );
    cudaDeviceSynchronize();
    DETAILED_TIMER_STOP(compact_centroids, "Compact centroids");

    // thrust::host_vector<Centroid> h_centroids(num_valid);
    // thrust::copy(d_centroids.begin(), d_centroids.end(), h_centroids.begin());

    return d_centroids;
}
