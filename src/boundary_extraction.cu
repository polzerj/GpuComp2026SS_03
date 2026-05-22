#ifndef BOUNDARY
#define BOUNDARY
#include <cstdint>
#include "types.hpp"

#define BLOCK_WIDTH_X  128
#define BLOCK_WIDTH_Y  4


__global__ void boundary_extraction(uint32_t * global_labels, uint32_t * boundary, uint32_t image_width, uint32_t image_height) {

    uint32_t pixel_id_x = ((blockIdx.x * blockDim.x) + threadIdx.x);
    uint32_t pixel_id_y= ((blockIdx.y * blockDim.y) + threadIdx.y);
    
    uint32_t unique_thread_id = pixel_id_x + pixel_id_y * image_width;

    uint32_t block_local_x = threadIdx.x;
    uint32_t block_local_y = threadIdx.y;
    uint32_t block_local_unique_id = block_local_x + BLOCK_WIDTH_X * block_local_y;

    // load data into shared memory 
    __shared__ uint32_t shared_block[BLOCK_WIDTH_Y + 2][BLOCK_WIDTH_X + 2]; // we need to load more than just the tile to be able to process all neighbours
       
    shared_block[block_local_y + 1][block_local_x + 1] = 0;
    if (block_local_x == 0) {
        shared_block[block_local_y + 1][0] = 0;
        shared_block[block_local_y + 1][BLOCK_WIDTH_X + 1] = 0;
    }
    if (block_local_y == 0) {
        shared_block[0][block_local_x + 1] = 0;
        shared_block[BLOCK_WIDTH_Y + 1][block_local_x + 1] = 0;
    }

    uint32_t pixel_value = 0; 
    if(pixel_id_x < image_width && pixel_id_y < image_height)
    {
        // fill shared memory
        pixel_value = global_labels[unique_thread_id];
        shared_block[block_local_y + 1][block_local_x + 1] = pixel_value; // load TILE

        if(block_local_x == 0) shared_block[block_local_y + 1][0] = (pixel_id_x > 0 ) ? global_labels[pixel_id_x - 1 + pixel_id_y * image_width] : 0;// load halo on the left
        if(block_local_x == BLOCK_WIDTH_X - 1) shared_block[block_local_y + 1][BLOCK_WIDTH_X + 1] = (pixel_id_x < image_width - 1  ) ? global_labels[pixel_id_x + 1 + pixel_id_y * image_width] : 0;// load halo on the right
        if(block_local_y == 0) shared_block[0][block_local_x + 1 ] = (pixel_id_y > 0 ) ? global_labels[pixel_id_x + (pixel_id_y - 1) * image_width] : 0;// load halo on the top
        if(block_local_y == BLOCK_WIDTH_Y - 1) shared_block[BLOCK_WIDTH_Y + 1][block_local_x + 1] = (pixel_id_y < image_height - 1 ) ? global_labels[pixel_id_x + (pixel_id_y + 1) * image_width] : 0;// load halo on the bottom  
       
        
    }    
    __syncthreads();

    uint32_t final_value = 0;
    if(pixel_id_x < image_width && pixel_id_y < image_height ){  

        if(pixel_value != 0) {
            //get neighbourhood values
            uint32_t neighbour_top_value = shared_block[block_local_y - 1 + 1][block_local_x + 1];
            uint32_t neighbour_left_value = shared_block[block_local_y + 1][block_local_x - 1 + 1];
            uint32_t neighbour_right_value = shared_block[block_local_y + 1][block_local_x + 1 + 1];
            uint32_t neighbour_bottom_value = shared_block[block_local_y + 1 + 1][block_local_x + 1];

            final_value= (neighbour_bottom_value == 0 || neighbour_left_value == 0 || neighbour_right_value == 0 || neighbour_top_value == 0) ? pixel_value : 0;
        }
     boundary[unique_thread_id] =final_value;
    }
}

void extract_boundary( uint32_t * global_labels, uint32_t * extracted_boundary_image, uint32_t image_width, uint32_t image_height) {
    //setup all the necessary memory on device and host
    uint32_t pixel_count = image_height * image_width;
	
    // Create Grid/Block
    dim3 block(BLOCK_WIDTH_X, BLOCK_WIDTH_Y);
    dim3 grid(ceil(image_width/(float)block.x), ceil(image_height/(float)block.y)); 

    uint32_t  *d_output;
    uint32_t   *d_input;

    // Allocate device memory
    cudaMalloc((void**) &d_output, pixel_count* sizeof(uint32_t));
    cudaMalloc((void**) &d_input,  pixel_count * sizeof(uint32_t));

    cudaMemcpy(d_input, global_labels, pixel_count*sizeof(uint32_t), cudaMemcpyHostToDevice);

    boundary_extraction <<<grid,block>>>(d_input,d_output,image_width,image_height);

    cudaMemcpy(extracted_boundary_image, d_output, pixel_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

}


#endif