#include <cstdio>
#include <cuda_runtime.h>
#include "types.hpp"

__global__ void segment(brightness_t* image, int width, int height){

    int num_pixels = width * height;
    // instantiate shared memories
    extern __shared__ brightness_t tile1[];
    extern __shared__ brightness_t tile2[];





}

void piece_segmentation(grayscale_image_t image){
    (void) image;
    printf("Piece segmentation is not implemented yet.\n");
    // allocate memory
    int num_pixels = image.height * image.width;
    size_t size = num_pixels * sizeof(brightness_t);
    brightness_t* h_image = image.data.data();
    brightness_t* d_image;
    cudaMalloc(&d_image, size);
    cudaMemcpy(d_image, h_image, size, cudaMemcpyHostToDevice);
    // call to the kernel
    segment<<<1,num_pixels>>>(d_image, image.width, image.height);
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    // Copy result back from GPU to CPU
    cudaMemcpy(h_image, d_image, size, cudaMemcpyDeviceToHost);
    // Cleanup
    cudaFree(d_image);
}
