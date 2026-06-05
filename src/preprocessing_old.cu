#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/scan.h>

#include "types.hpp"

#include "stb_image.h"
#include "preprocessing_old.cuh"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#define GAUSSIAN_KERNEL_SIZE 5
#define GAUSSIAN_SIGMA 1.0
#define OTSU_NUM_BINS 256

__device__ double *get_gaussian_kernel(const size_t kernel_size, const double sigma)
{
    assert(kernel_size % 2 == 1);

    double *kernel = new double[kernel_size * kernel_size];
    double sum = 0.0;

    for (size_t y = 0; y < kernel_size; ++y)
    {
        for (size_t x = 0; x < kernel_size; ++x)
        {
            const double element = exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[y * kernel_size + x] = element;
            sum += element;
        }
    }

    for (size_t i = 0; i < kernel_size * kernel_size; ++i)
    {
        kernel[i] /= sum;
    }

    return kernel;
}

__global__ void gaussian_blur_kernel(const brightness_t *input, brightness_t *output, const int width, const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    double *kernel = get_gaussian_kernel(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA);

    double pixel_value = 0.0;
    for (int x_offset = -GAUSSIAN_KERNEL_SIZE / 2; x_offset < GAUSSIAN_KERNEL_SIZE / 2; ++x_offset)
    {
        for (int y_offset = -GAUSSIAN_KERNEL_SIZE / 2; y_offset < GAUSSIAN_KERNEL_SIZE / 2; ++y_offset)
        {
            if (x + x_offset >= 0 && x + x_offset < width && y + y_offset >= 0 && y + y_offset < height)
            {
                // Perform Gaussian blur operation
                const size_t kernel_index = (y_offset + GAUSSIAN_KERNEL_SIZE / 2) * GAUSSIAN_KERNEL_SIZE + (x_offset + GAUSSIAN_KERNEL_SIZE / 2);
                const size_t input_index = (y + y_offset) * width + (x + x_offset);

                pixel_value += kernel[kernel_index] * input[input_index];
            }
        }
    }

    delete (kernel);
    output[y * width + x] = (brightness_t)pixel_value;
}

__global__ void calculate_histogram_kernel(const brightness_t *input, unsigned int *histogram, const int width, const int height, const int num_bins)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const size_t index = y * width + x;
    assert(index < width * height);

    const brightness_t pixel_value = input[index];

    assert(pixel_value < num_bins);

    atomicAdd(&histogram[pixel_value], 1);
}

__global__ void calc_variance_between_kernel(const unsigned int *histogram, const unsigned int *cumulative_histogram, unsigned int *variance_between, const int width, const int height, const int num_bins)
{
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_bins)
        return;

    if (histogram[t] == 0)
    {
        variance_between[t] = 0;
        return;
    }

    const unsigned sum_total = cumulative_histogram[num_bins - 1];
    const unsigned weight_background = cumulative_histogram[t];
    const unsigned weight_foreground = sum_total - weight_background;

    if (weight_background == 0 || weight_foreground == 0)
    {
        variance_between[t] = 0;
        return;
    }

    const unsigned sum_background = t * histogram[t];
    const double mean_background = (double)sum_background / weight_background;
    const double mean_foreground = (double)(sum_total - sum_background) / weight_foreground;
    variance_between[t] = (unsigned int)(weight_background * weight_foreground * (mean_background - mean_foreground) * (mean_background - mean_foreground));
}

grayscale_image_t
preprocessing(std::string image_path, uint8_t *d_result)
{
    // return grayscale_image_t(); // TODO: Remove this and continue debugging

    if (image_path.empty())
    {
        fprintf(stderr, "Error: Image path is empty.\n");
        exit(EXIT_FAILURE);
    }

    int channels;

    grayscale_image_t image;
    image.data = std::vector<uint8_t>();

    unsigned char *img_data = stbi_load(image_path.c_str(), &image.width, &image.height, &channels, 1);
    if (!img_data)
    {
        fprintf(stderr, "Error: Failed to load image '%s'.\n", image_path.c_str());
        exit(EXIT_FAILURE);
    }

    image.data.assign(img_data, img_data + image.width * image.height);

    stbi_image_free(img_data);

    std::cout << "Image loaded: " << image_path << " (width: " << image.width << ", height: " << image.height << ", channels: " << channels << ")\n";

    size_t free, total;
    cudaError_t error = cudaMemGetInfo(&free, &total);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: Failed to get GPU memory info.\n");
        exit(EXIT_FAILURE);
    }
    std::cout << "Free GPU memory: " << free / (1024 * 1024) << " MB (" << free << " bytes)\n";
    std::cout << "Total GPU memory: " << total / (1024 * 1024) << " MB (" << total << " bytes)\n";
    std::cout << "Need " << (image.width * image.height * sizeof(brightness_t)) / (1024 * 1024) << " MB for input and output images\n";

    thrust::host_vector<brightness_t> h_input(image.data.begin(), image.data.end());
    thrust::device_vector<brightness_t> d_input(h_input);
    std::cout << "Image data copied to device.\n";

    thrust::device_vector<brightness_t> d_output(image.width * image.height);

    dim3 block(16, 16);
    dim3 grid((image.width + block.x - 1) / block.x, (image.height + block.y - 1) / block.y);

    gaussian_blur_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_output.data()),
        image.width,
        image.height);

    std::cout << "Gaussian blur applied.\n";

    error = cudaMemGetInfo(&free, &total);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: Failed to get GPU memory info.\n");
        exit(EXIT_FAILURE);
    }
    std::cout << "Free GPU memory: " << free / (1024 * 1024) << " MB (" << free << " bytes)\n";

    thrust::device_vector<unsigned int> d_histogram(OTSU_NUM_BINS, 0u);

    std::cout << "Calculating histogram 1...\n";

    thrust::device_vector<unsigned int> variance_between(OTSU_NUM_BINS, 0u);

    std::cout << "Calculating histogram 2...\n";

    calculate_histogram_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_histogram.data()),
        image.width,
        image.height,
        OTSU_NUM_BINS);

    std::cout << "Histogram calculated.\n";

    thrust::device_vector<unsigned int> cumulative_histogram(OTSU_NUM_BINS, 0u);
    thrust::inclusive_scan(d_histogram.begin(), d_histogram.end(), cumulative_histogram.begin());

    calc_variance_between_kernel<<<1, OTSU_NUM_BINS>>>(
        thrust::raw_pointer_cast(d_histogram.data()),
        thrust::raw_pointer_cast(cumulative_histogram.data()),
        thrust::raw_pointer_cast(variance_between.data()),
        image.width, image.height,
        OTSU_NUM_BINS);

    std::cout << "Variance between calculated.\n";

    brightness_t optimal_threshold = thrust::reduce(variance_between.begin(), variance_between.end(), 0, thrust::maximum<unsigned int>());

    thrust::transform(d_output.begin(), d_output.end(), d_output.begin(), [optimal_threshold] __device__(brightness_t pixel)
                      { return pixel >= optimal_threshold ? 255 : 0; });

    thrust::host_vector<brightness_t> h_image = d_output;
    image.data.assign(h_image.begin(), h_image.end());

    return image;
}
