
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include "centroid.cuh"

struct is_positive {
    __device__ bool operator()(const uint32_t c) const {
        return c > 0;
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
        atomicAdd(&sum_x[label], (unsigned long long) x);
        atomicAdd(&sum_y[label], (unsigned long long) y);
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
    centroids[idx].x = sum_x[label] * 1.0 / count_label;
    centroids[idx].y = sum_y[label] * 1.0 / count_label;
    centroids[idx].label = label;
}

thrust::host_vector<Centroid> compute_centroids(const uint32_t *h_labels, uint32_t width, uint32_t height)
{
    auto num_pixels = width * height;

    thrust::device_vector<uint32_t> d_labels(h_labels, h_labels + num_pixels);
    
    uint32_t max_label = thrust::reduce(d_labels.begin(), d_labels.end(), 0u, thrust::maximum<uint32_t>());

    if (max_label == 0)
    {
        return thrust::host_vector<Centroid>(); 
    }

    thrust::device_vector<unsigned long long> d_sum_x(max_label + 1, 0ull);
    thrust::device_vector<unsigned long long> d_sum_y(max_label + 1, 0ull);
    thrust::device_vector<uint32_t> d_count(max_label + 1, 0u);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    compute_sums_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_labels.data()), width, height,
        thrust::raw_pointer_cast(d_sum_x.data()),
        thrust::raw_pointer_cast(d_sum_y.data()),
        thrust::raw_pointer_cast(d_count.data())
    );

    thrust::device_vector<uint32_t> indices(max_label + 1);
    thrust::sequence(indices.begin(), indices.end());

    // Count directly without needing a mask vector
    uint32_t num_valid = thrust::count_if(d_count.begin(), d_count.end(), is_positive());
    
    if (num_valid == 0)
    {
        return thrust::host_vector<Centroid>();
    }

    thrust::device_vector<uint32_t> d_valid_labels(num_valid);
    
    // copy_if can use d_count as the stencil directly
    thrust::copy_if(indices.begin(), indices.end(), d_count.begin(), d_valid_labels.begin(), is_positive());

    thrust::device_vector<Centroid> d_centroids(num_valid);
    dim3 block2(256);
    dim3 grid2((num_valid + block2.x - 1) / block2.x);
    
    compactCentroidsKernel<<<grid2, block2>>>(
        thrust::raw_pointer_cast(d_sum_x.data()),
        thrust::raw_pointer_cast(d_sum_y.data()),
        thrust::raw_pointer_cast(d_count.data()),
        thrust::raw_pointer_cast(d_valid_labels.data()),
        thrust::raw_pointer_cast(d_centroids.data()),
        num_valid
    );

    thrust::host_vector<Centroid> h_centroids(num_valid);
    thrust::copy(d_centroids.begin(), d_centroids.end(), h_centroids.begin());

    return h_centroids;
}