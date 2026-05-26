#ifndef PTRJUMP
#define PTRJUMP


#include <cstdint>
#include "types.hpp"
#include "pointerjumping.cuh"

#define BLOCK_WIDTH_X 32
#define BLOCK_WIDTH_Y 32
#define SHARED_B_X (BLOCK_WIDTH_X + 2)
#define SHARED_B_Y (BLOCK_WIDTH_Y + 2)
#define SHARED_BLOCK_SIZE (SHARED_B_X * SHARED_B_Y)
#define SHARED_ELEM_CNT (BLOCK_WIDTH_X * BLOCK_WIDTH_Y)

__global__ void init_pointer_list_and_extract_boundaries(uint32_t *image_data, uint32_t image_width, uint32_t image_height, ptr_jmp_out_t data){
    //we need to make a list where within each piece boundary each node points to its successor 
    uint32_t pixel_id_x = ((blockIdx.x * blockDim.x) + threadIdx.x);
    uint32_t pixel_id_y= ((blockIdx.y * blockDim.y) + threadIdx.y);
    
    uint32_t unique_thread_id = pixel_id_x + pixel_id_y * image_width;
    
    uint32_t block_local_x = threadIdx.x + 1;
    uint32_t block_local_y = threadIdx.y + 1;

    // load data into shared memory 
    __shared__ uint32_t shared_block[BLOCK_WIDTH_Y + 2][BLOCK_WIDTH_X + 2]; // we need to load more than just the tile to be able to process all neighbours

    for (int i = (threadIdx.x + threadIdx.y * blockDim.x); i < SHARED_BLOCK_SIZE; i += SHARED_ELEM_CNT) {
        int s_y = i / SHARED_B_X;
        int s_x = i % SHARED_B_X;

        // Offset by -1 because shared (1,1) is global (block_start_x, block_start_y)
        int g_x = (int)(blockIdx.x * blockDim.x) + s_x - 1;
        int g_y = (int)(blockIdx.y * blockDim.y) + s_y - 1;

        if (g_x >= 0 && g_x < image_width && g_y >= 0 && g_y < image_height) {
            shared_block[s_y][s_x] = image_data[g_y * image_width + g_x];
        } else {
            shared_block[s_y][s_x] = 0; // Boundary condition: black/empty
        }
    }
    __syncthreads();


    uint32_t successor = unique_thread_id;
    uint32_t final_value = 0;
    uint32_t pixel_value = 0;
    uint32_t rank = 0;
    if(pixel_id_x < image_width && pixel_id_y < image_height ){  

        pixel_value = shared_block[block_local_y][block_local_x];

        if(pixel_value != 0) {
            //get all 8 neighbourhood values
            uint32_t neighbour_top_value = shared_block[block_local_y - 1 ][block_local_x];
            uint32_t neighbour_left_value = shared_block[block_local_y][block_local_x - 1 ];
            uint32_t neighbour_right_value = shared_block[block_local_y][block_local_x + 1];
            uint32_t neighbour_bottom_value = shared_block[block_local_y + 1][block_local_x];

            final_value= (neighbour_bottom_value == 0 || neighbour_left_value == 0 || neighbour_right_value == 0 || neighbour_top_value == 0) ? pixel_value : 0;

            if(final_value > 0){
                const int8_t dx[8] = { 0,  1, 1, 1, 0, -1, -1, -1}; // top,top right, right, bottom right, bottom, left, top left
                const int8_t dy[8] = {-1, -1, 0, 1, 1, 1,  0, -1};
                int8_t start_rotation = 127;
                if (neighbour_top_value == 0) start_rotation = 0;
                else if (neighbour_right_value == 0) start_rotation = 2;
                else if (neighbour_bottom_value == 0) start_rotation = 4; 
                else if (neighbour_left_value == 0) start_rotation = 6;
                
                if(start_rotation != 127){
                    for(int8_t clockwise_steps = 0; clockwise_steps < 8; ++clockwise_steps){
                        int8_t rotation = (clockwise_steps + start_rotation) % 8; 
                        int32_t x_value = dx[rotation] + block_local_x;
                        int32_t y_value = dy[rotation] + block_local_y;

                        if(shared_block[y_value][x_value] == pixel_value) {
                            successor = uint32_t((int32_t) pixel_id_x + dx[rotation] + ((int32_t) pixel_id_y + dy[rotation]) * (int32_t) image_width); 
                            rank = 1;
                            break;
                        }
                    }


                }
                
            }
        }
    }
    //set boundaries list
    data.boundaries[unique_thread_id] = final_value;
    data.successors[unique_thread_id] = successor;
    data.ranks[unique_thread_id] = rank;

}

__global__ void list_ranking(ptr_jmp_out_t in, uint32_t image_width, uint32_t image_height){


}
__device__ uint32_t get_next(uint32_t* successors, uint32_t current_pixel){
    
}

void pointer_jumping(uint32_t * global_labels, ptr_jmp_out_t out, uint32_t image_width, uint32_t image_height) {
    
    //setup all the necessary memory on device and host
    uint32_t pixel_count = image_height * image_width;
	
    // Create Grid/Block
    dim3 block(BLOCK_WIDTH_X, BLOCK_WIDTH_Y);
    dim3 grid(ceil(image_width/(float)block.x), ceil(image_height/(float)block.y)); 

    uint32_t array_size =  pixel_count* sizeof(uint32_t);
    uint32_t *d_image_data_in;
    ptr_jmp_out_t device_data;

    cudaMalloc((void**) &d_image_data_in, array_size);
    cudaMalloc((void**) &device_data.boundaries, array_size );
    cudaMalloc((void**) &device_data.ranks, array_size );
    cudaMalloc((void**) &device_data.successors, array_size );

    cudaMemcpy(d_image_data_in, global_labels, pixel_count*sizeof(uint32_t), cudaMemcpyHostToDevice);

    init_pointer_list_and_extract_boundaries <<<grid,block>>>(d_image_data_in,image_width,image_height, device_data);

    cudaMemcpy(out.boundaries, device_data.boundaries, pixel_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);



    cudaFree(device_data.boundaries);
    cudaFree(d_image_data_in);
    cudaFree(device_data.ranks);
    cudaFree(device_data.successors);
}


#endif