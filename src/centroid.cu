
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "centroid.cuh"

__global__ void compute_sums_kernel(
    const uint32_t *image,
    uint32_t width,
    uint32_t height,
    uint32_t *sum_x,
    uint32_t *sum_y,
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
        atomicAdd(&sum_x[label], x);
        atomicAdd(&sum_y[label], y);
        atomicAdd(&count[label], 1);
    }
}

__global__ void compactCentroidsKernel(
    const uint32_t *sum_x,
    const uint32_t *sum_y,
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
    centroids[idx].x = sum_x[label] / count_label;
    centroids[idx].y = sum_y[label] / count_label;
    centroids[idx].label = label;
}

thrust::host_vector<Centroid> compute_centroids(const uint32_t *image, uint32_t width, uint32_t height)
{
    auto num_pixels = width * height;

    thrust::device_ptr<const uint32_t> d_image(image);
    uint32_t max_label = thrust::reduce(d_image, d_image + num_pixels, 0u, thrust::maximum<uint32_t>());

    if (max_label == 0)
    {
        return thrust::host_vector<Centroid>(); // No non-zero labels
    }

    thrust::device_vector<uint32_t> d_sum_x(max_label + 1, 0.0f);
    thrust::device_vector<uint32_t> d_sum_y(max_label + 1, 0.0f);
    thrust::device_vector<uint32_t> d_count(max_label + 1, 0u);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    compute_sums_kernel<<<grid, block>>>(
        image, width, height,
        thrust::raw_pointer_cast(d_sum_x.data()),
        thrust::raw_pointer_cast(d_sum_y.data()),
        thrust::raw_pointer_cast(d_count.data()));


    thrust::device_vector<uint32_t> indices(max_label + 1);
    thrust::sequence(indices.begin(), indices.end());

    thrust::device_vector<bool> mask(max_label + 1);
    thrust::transform(d_count.begin(), d_count.end(), mask.begin(),
                      [] __device__(uint32_t c)
                      { return c > 0; });

    uint32_t num_valid = thrust::reduce(mask.begin(), mask.end(), 0u, thrust::plus<uint32_t>());
    if (num_valid == 0)
    {
        return thrust::host_vector<Centroid>();
    }

    thrust::device_vector<uint32_t> d_valid_labels(num_valid);
    thrust::copy_if(indices.begin(), indices.end(), mask.begin(), d_valid_labels.begin(), thrust::identity<bool>());

    thrust::device_vector<Centroid> d_centroids(num_valid);dim3 block2(256);
    dim3 grid2((num_valid + block2.x - 1) / block2.x);
    compactCentroidsKernel<<<grid2, block2>>>(
        thrust::raw_pointer_cast(d_sum_x.data()),
        thrust::raw_pointer_cast(d_sum_y.data()),
        thrust::raw_pointer_cast(d_count.data()),
        thrust::raw_pointer_cast(d_valid_labels.data()),
        thrust::raw_pointer_cast(d_centroids.data()),
        num_valid);

    thrust::host_vector<Centroid> h_centroids(num_valid);
    thrust::copy(d_centroids.begin(), d_centroids.end(), h_centroids.begin());

    return h_centroids;
}
