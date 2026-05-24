#include <cstdio>
#include <cuda_runtime.h>
#include "types.hpp"

__global__ void erode(brightness_t* input, brightness_t* output, int width, int height, size_t kernel_size){
    //extern __shared__ brightness_t tile[];
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x >= width || y >= height){ // rectangle of threads started is slightly bigger than image
        return;
    }

    bool keep_white = true;
    size_t kernel_size_half = kernel_size / 2;

    for (int dy = -kernel_size_half; dy <= kernel_size_half; dy++)
    {
        int ny = y + dy;    
        for (int dx = -kernel_size_half; dx <= kernel_size_half; dx++)
        {
            int nx = x + dx;
            brightness_t b = 0;
            if (!(nx < 0 || nx >= width || ny < 0 || ny >= height)){
                b = input[nx + ny * width];
            } 
            if (b != 255){
                keep_white = false;
            }  
        }
    }
    
    
    
    output[x + width * y] = keep_white ? 255 : 0;
}

__global__ void dilate(brightness_t* input, brightness_t* output, int width, int height, size_t kernel_size){
    //extern __shared__ brightness_t tile[];
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x >= width || y >= height){ // rectangle of threads started is slightly bigger than image
        return;
    }

    bool keep_black = true;
    size_t kernel_size_half = kernel_size / 2;

    for (int dy = -kernel_size_half; dy <= kernel_size_half; dy++)
    {
        int ny = y + dy;    
        for (int dx = -kernel_size_half; dx <= kernel_size_half; dx++)
        {
            int nx = x + dx;
            if (!(nx < 0 || nx >= width || ny < 0 || ny >= height)){
                if (input[nx + width * ny] == 255){
                    keep_black = false;
                }  
            } 
        }
    }
    output[x + width * y] = keep_black ? 0 : 255;
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

    // Morphological open
    erode<<<grid,block>>>(d_input, d_tmp, image.width, image.height, kernel_size);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Erode kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("Runtime error: %s\n", cudaGetErrorString(err));
    }

    dilate<<<grid, block>>>(d_tmp, d_output, image.width, image.height, kernel_size);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Dilate kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("Runtime error: %s\n", cudaGetErrorString(err));
    }
    //Copy results back from GPU to CPU
    cudaMemcpy(h_image, d_output, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_tmp);
    cudaFree(d_output);
}
