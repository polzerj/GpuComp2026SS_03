#include <cstdio>
#include <cuda_runtime.h>
#include "types.hpp"

__global__ void erode(const brightness_t* input, brightness_t* output, int width, int height, size_t kernel_size){
    extern __shared__ brightness_t tile[];
    int radius = kernel_size / 2;
    int tile_width = blockDim.x + 2 * radius;
    int tile_height = blockDim.y + 2 * radius;
    int tile_size = tile_width * tile_height; // 16 x 16 block plus radius/2 extra on the sides

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int tid_in_block = threadIdx.x + blockDim.x * threadIdx.y;
    int num_threads_per_block = blockDim.y * blockDim.x;

    // copy values into shared array for every block
    for (int idx = tid_in_block; idx < tile_size; idx += num_threads_per_block)
    {
        int local_y = idx / tile_width;
        int local_x = idx % tile_width;

        int global_y = blockIdx.y * blockDim.y + local_y - radius;
        int global_x = blockIdx.x * blockDim.x + local_x - radius;

        brightness_t value = 0;

        if(global_x >= 0 && global_x < width && global_y >= 0 && global_y < height){
            value = input[global_y * width + global_x];
        }

        tile[local_y * tile_width + local_x] = value;
    }
    
    __syncthreads();
    if(x >= width || y >= height){ // rectangle of threads started is slightly bigger than image
        return;
    }


    bool keep_white = true;

    for (int dy = -radius; dy <= radius; dy++)
    {
        for (int dx = -radius; dx <= radius; dx++)
        {
           brightness_t value = tile[(threadIdx.y + radius + dy)*tile_width + (threadIdx.x + radius + dx)];
           if (value != 255)
           {
            keep_white = false;
           }
        }  
    }
    
    output[x + width * y] = keep_white ? 255 : 0;
}

__global__ void dilate(const brightness_t* input, brightness_t* output, int width, int height, size_t kernel_size){
    extern __shared__ brightness_t tile[];
    int radius = kernel_size / 2;
    int tile_width = blockDim.x + 2 * radius;
    int tile_height = blockDim.y + 2 * radius;
    int tile_size = tile_height * tile_width;

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int tid_in_block = threadIdx.x + threadIdx.y * blockDim.x;
    int num_threads_per_block = blockDim.y * blockDim.x;

    for (int idx = tid_in_block; idx < tile_size; idx += num_threads_per_block)
    {
        int local_y = idx / tile_width;
        int local_x = idx % tile_width;

        int global_y = blockIdx.y * blockDim.y + local_y - radius;
        int global_x = blockIdx.x * blockDim.x + local_x - radius;

        brightness_t value = 0;
        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height)
        {
            value = input[global_y * width + global_x];
        }
        tile[local_y * tile_width + local_x] = value;
    }
    
    __syncthreads();
    if(x >= width || y >= height){ // rectangle of threads started is slightly bigger than image
        return;
    }

    bool make_white = false;

    for (int dy = -radius; dy <= radius; dy++)
    {
        for (int dx = -radius; dx <= radius; dx++)
        {
            brightness_t value = tile[(threadIdx.y + radius + dy) * tile_width + (threadIdx.x + radius + dx)];
            if (value == 255)
            {
                make_white = true;
            }
                         
        }
    }
    output[x + width * y] = make_white ? 255 : 0;
}


void piece_segmentation(grayscale_image_t& image){
    // allocate memory
    int num_pixels = image.height * image.width;
    size_t size = num_pixels * sizeof(brightness_t);
    size_t kernel_size = 5;

    brightness_t* h_image = image.data.data();
    brightness_t* d_input;
    brightness_t* d_tmp;
    brightness_t* d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_tmp, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_image, size, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((image.width + block.x - 1) / block.x , (image.height + block.y - 1) / block.y);
    size_t radius = kernel_size / 2;
    size_t shared_mem_bytes = (block.x + 2 * radius)*(block.y + 2 * radius)*sizeof(brightness_t);

    // Morphological open
    erode<<<grid,block, shared_mem_bytes>>>(d_input, d_tmp, image.width, image.height, kernel_size);
    // cudaError_t err = cudaGetLastError();
    // if(err != cudaSuccess){
    //     printf("Erode kernel launch failed: %s\n", cudaGetErrorString(err));
    // }
    // err = cudaDeviceSynchronize();
    // if(err != cudaSuccess){
    //     printf("Runtime error: %s\n", cudaGetErrorString(err));
    // }
    //erode<<<grid,block>>>(d_tmp, d_output, image.width, image.height, kernel_size);


    //dilate<<<grid, block>>>(d_output, d_tmp, image.width, image.height, kernel_size);
    // err = cudaGetLastError();
    // if(err != cudaSuccess){
    //     printf("Dilate kernel launch failed: %s\n", cudaGetErrorString(err));
    // }
    // err = cudaDeviceSynchronize();
    // if(err != cudaSuccess){
    //     printf("Runtime error: %s\n", cudaGetErrorString(err));
    // }
    dilate<<<grid, block, shared_mem_bytes>>>(d_tmp, d_output, image.width, image.height, kernel_size);

    //Copy results back from GPU to CPU
    cudaMemcpy(h_image, d_output, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_tmp);
    cudaFree(d_output);
}
