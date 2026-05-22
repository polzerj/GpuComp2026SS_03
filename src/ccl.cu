// this implements connected component labeling as is presented in https://ieeexplore.ieee.org/document/8274991
// it is called the playne-equivalence algorithm
// it is pretty cool

#ifndef CCL
#define CCL
#include <cstdint>
#include "types.hpp"
// keep X at 32/4 unless you want to change the launch bounds for the reduction kernel (see original implementation)
#define BLOCK_WIDTH_X  32
#define BLOCK_WIDTH_Y  4

__device__ __inline__ uint32_t find_root(uint32_t *labels_array, uint32_t label) {
	
    //labels_array[labels] points to the specific pixel label is from --> this pixel might be connected
    //to a pixel with lower starting index and might therefore be labled with a lower number
    //if so we follow the chain until we cant go further meaning a label that points to itself
	uint32_t resolved = labels_array[label];

	// Follow chain
	while(label != resolved) {
		// Move to next
		label = resolved;
		resolved = labels_array[label];
	}

	// Return label
	return label;
}

__device__ __inline__ uint32_t reduction(uint32_t *labels_array, uint32_t label1, uint32_t label2){

    uint32_t resolved1 = (label1 != label2) ? labels_array[label1] : 0;
    uint32_t resolved2 = (label1 != label2) ? labels_array[label2] : 0;


    // find the roots for both labels if they are not the same already
	while(label1 != resolved1 && label1 != label2) {
		// Move to next
		label1 = resolved1;
		resolved1 = labels_array[label1];
	}

    while(label2 != resolved2 && label1 != label2) {
		// Move to next
		label2 = resolved2;
		resolved2 = labels_array[label2];
	}
    
    uint32_t merge_label;

    //the reduction step merges clusters of labels that are connected.
    //there might be multiple clusters that are connected together. 
    // for each iteration we look at the two labels(roots) of our connection node and its neighbour cluster
    // and set both to the lower of the two atomically. We do this until we 
    // reach a steady state and they are equal. (their root nodes point to the same place)
    // if a third cluster has a lower label it will update the corresponding root node to point at the correct lower root
    while(label1 != label2) {
		// Label 2 should be smallest
		if(label1 < label2) {
			// Swap Labels
			label1 = label1 ^ label2;
			label2 = label1 ^ label2;
			label1 = label1 ^ label2;
		}

		// AtomicMin label1 to label2
		merge_label = atomicMin(&labels_array[label1], label2);
		label1 = (label1 == merge_label) ? label2 : merge_label;
	}

    return label1;

}


__global__ void propagate_labels(uint32_t *global_labels, uint32_t image_width, uint32_t image_height){
    // per pixel kernel that finds the root label for every pixel
    // we are using a 2D kernel with a tupel (x,y) uniquely identifying a pixel in the image
    uint32_t thread_id = ((blockIdx.y * blockDim.y) + threadIdx.y) * image_width +
							((blockIdx.x * blockDim.x) + threadIdx.x);

    if(thread_id < image_width * image_height){
        global_labels[thread_id] = find_root(global_labels,global_labels[thread_id]);
    }

}

__global__ void local_blocked_ccl(uint32_t *global_labels, uint8_t* global_image_bw, uint32_t image_width, uint32_t image_height){

    extern __shared__ uint32_t local_label_cache[]; // we can dynamically allocate this by passing its size through a implicit parameter in the launch

    uint32_t pixel_id_x = ((blockIdx.x * blockDim.x) + threadIdx.x);
    uint32_t pixel_id_y= ((blockIdx.y * blockDim.y) + threadIdx.y);
    
    uint32_t unique_thread_id = pixel_id_x + pixel_id_y * image_width;

    if(pixel_id_x < image_width && pixel_id_y < image_height){

        uint32_t local_pixel_x = threadIdx.x;
        uint32_t local_pixel_y = threadIdx.y;
        uint32_t unique_local_id = local_pixel_x + local_pixel_y * blockDim.x;

        //-------------------
        // GET IMAGE VALUES
        //------------------
        // we copy a trick from Playne, where if you start the blocks with width in the X-dim = 32 we know all threads in a warp are
        // in the same y dimension and can therefore share information without loads but instead using warp level communication

        uint8_t image_value_self = global_image_bw[unique_thread_id];
        uint8_t image_value_top = (pixel_id_y > 0) ? global_image_bw[pixel_id_x+ (pixel_id_y - 1)* image_width]: 0; // we have to make a mem access here unfortunatelly
        uint8_t image_value_left= __shfl_up_sync(__activemask(), image_value_self, 1);// we shuffle up the image value of the pixel in the warp that is 1 to our left
        uint8_t image_value_top_left = __shfl_up_sync(__activemask(), image_value_top, 1); // we need this for later to check if pixel is a border pixel

        //-----------------------
        // INITIALISATION -----> we initialize each pixel with the lowest neighbours value
        //-----------------------
        
        //CHECK CONNECTIONS
        bool connected_to_top = (local_pixel_y > 0) ? image_value_top == image_value_self : false;
        bool connected_to_left = (local_pixel_x > 0) ? image_value_left == image_value_self : false;
        bool connected_to_top_left = (local_pixel_x > 0 && local_pixel_y > 0) ? image_value_top_left == image_value_self: false;
        
        //INITIALISE BASED ON CONNECTIONS
        uint32_t label_pixel = (connected_to_left) ? (local_pixel_x - 1)+ (local_pixel_y) * blockDim.x : unique_local_id; //connected to left? --> if yes adopt its value
        label_pixel = (connected_to_top) ? (local_pixel_x) + (local_pixel_y- 1) * blockDim.x : label_pixel; //connected to top? --> if yes adopt its value (top will be lower than left so it takes precedence)
        
        //write to local cache
        local_label_cache[unique_local_id] = label_pixel;

        __syncthreads();

        //---------------------------
        // ANALYSIS --> Analysis is label propagation
        //--------------------------
        
        local_label_cache[unique_local_id] = find_root(local_label_cache, label_pixel);

        __syncthreads();

        //-------------------------
        // REDUCTION --> in the reduction step we need to merge connected clusters of labels since the top always takes precedence and labels might not have been updated accordingly yet
        // playne introduces some checks to reduce the amount of reductions we actually have to do --> there is some great descriptions for this in the paper
        //-------------------------
        
        if(connected_to_left && connected_to_top && !(connected_to_top_left))
        {
            uint32_t neighbour_label = local_label_cache[ (local_pixel_x- 1)+ (local_pixel_y) * blockDim.x ] ; // if we are a reduction node we need the label 
            //of the neighbour on our left since the top label would have propagated during init
            
            label_pixel = reduction(local_label_cache, label_pixel, neighbour_label);
        }

         __syncthreads();
        
        //---------------------------
        // ANALYSIS --> Analysis is label propagation
        //--------------------------
        
        label_pixel = find_root(local_label_cache, label_pixel);
        
        //we need to convert these local labels to global ones ---> this can be spead up still
        const unsigned int local_x =  label_pixel  % blockDim.x;;
		const unsigned int local_y = (label_pixel / blockDim.x) % blockDim.y; 
     
        global_labels[unique_thread_id] =  ((blockIdx.y * blockDim.y) + local_y) * image_width + ((blockIdx.x * blockDim.x) + local_x);


    }
}

__global__ void block_border_reduction_x(uint32_t *global_labels, uint8_t* global_image_bw, uint32_t image_width, uint32_t image_height){
        //this reduces along the vertical seams that means the kernel will launch a thread for each horizontal line (span the entire height)
        // but only one thread per block along the x dimension so we get the seams
        // we also have to "flip" the image so the threads along the x dimension are above each other on the same seam so we can get the right
        // structure for shuffeling up the values

        uint32_t pixel_seam_x = BLOCK_WIDTH_X + BLOCK_WIDTH_X * ((blockIdx.y * blockDim.y) + threadIdx.y); // this represents one vertical line on the seam
        uint32_t pixel_y = ((blockIdx.x * blockDim.x) + threadIdx.x); // this iterates between the pixels on the seam
        uint32_t unique_thread_id = pixel_seam_x + pixel_y * image_width;

        if(pixel_seam_x < image_width && pixel_y < image_height){
            // Get image and label values
            const uint8_t pixel_value = global_image_bw[unique_thread_id];

            // Neighbour Values
            const uint8_t pixel_value_neighbour_left = global_image_bw[pixel_y*image_width + pixel_seam_x - 1];

            const bool top_pixel_in_block = (pixel_y % BLOCK_WIDTH_Y) == 0; // is the ,current pixel the top most of the block


            // Fetch Neighbours
            const uint8_t pixel_value_neighbour_top= __shfl_up_sync(__activemask(), pixel_value,   1);
            const uint8_t pixel_value_neighbour_top_left = __shfl_up_sync(__activemask(), pixel_value_neighbour_left, 1);

            // If connected to left neighbour
            if((pixel_value == pixel_value_neighbour_left) && (top_pixel_in_block || (pixel_value != pixel_value_neighbour_top) || (pixel_value != pixel_value_neighbour_top_left))) {
                // Get Labels
                uint32_t pixel_label = global_labels[unique_thread_id];
                uint32_t neighbour_label = global_labels[pixel_y * image_width + pixel_seam_x - 1];

                // Reduction
                reduction(global_labels, pixel_label, neighbour_label);
            }
        }

}

__global__ void block_border_reduction_y(uint32_t *global_labels, uint8_t* global_image_bw, uint32_t image_width, uint32_t image_height){
    //same logic as above applies just now with the seams along the y dimension

    	// Calculate Index
	uint32_t pixel_x = ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32_t pixel_seam_y = ((blockIdx.y * blockDim.y) + threadIdx.y) * BLOCK_WIDTH_Y + BLOCK_WIDTH_Y;
    uint32_t unique_thread_id = pixel_seam_y * image_width + pixel_x;

	// Check Range
	if(pixel_x < image_width && pixel_seam_y < image_height) {
		// Get image and label values
		const uint8_t pixel_value   = global_image_bw[unique_thread_id];
		const uint8_t pixel_value_neighbour_top = global_image_bw[(pixel_seam_y-1)*image_width + pixel_x];


        //neighbours
		const uint8_t pixel_value_neighbour_left   = __shfl_up_sync(__activemask(), pixel_value,   1);
		const uint8_t pixel_value_neighbour_top_left = __shfl_up_sync(__activemask(), pixel_value_neighbour_top, 1);

		// If connected to neighbour
		if (pixel_value == pixel_value_neighbour_top && ((threadIdx.x == 0) || (pixel_value != pixel_value_neighbour_left) || (pixel_value != pixel_value_neighbour_top_left))) {
			// Get labels
			unsigned int pixel_label = global_labels[unique_thread_id];
			unsigned int neighbour_label = global_labels[(pixel_seam_y-1)*image_width + pixel_x];

			// Reduction
			reduction(global_labels, pixel_label, neighbour_label);
		}
	}
    
}

void playne_ccl (uint32_t *global_labels, const grayscale_image_t &grayImage) {

    //setup all the necessary memory on device and host
    uint32_t pixel_count = grayImage.width * grayImage.height;
	
    // Create Grid/Block
    dim3 block(BLOCK_WIDTH_X, BLOCK_WIDTH_Y ); // keep at 32/4
    dim3 grid(ceil(grayImage.width/(float)block.x), ceil(grayImage.height/(float)block.y)); 

    // Create Border X/Y Grid
    dim3 border_grid_x(ceil(grayImage.height/(float)block.x), ceil(grid.x/(float)block.y));
    dim3 border_grid_y(ceil(grayImage.width/(float)block.x), ceil(grid.y/(float)block.y));

    uint32_t  *d_labels;
    uint8_t   *d_image;

    // Allocate device memory
    cudaMalloc((void**) &d_labels, pixel_count* sizeof(uint32_t));
    cudaMalloc((void**) &d_image,  pixel_count * sizeof(uint8_t));

    cudaMemcpy(d_image, grayImage.data.data(), pixel_count*sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

    local_blocked_ccl<<< grid, block, block.x*block.y*sizeof(uint32_t) >>>(d_labels, d_image, grayImage.width, grayImage.height);
    cudaDeviceSynchronize();

    // Label Reduction
    block_border_reduction_y <<< border_grid_y, block, 0, stream1 >>> (d_labels, d_image, grayImage.width, grayImage.height );
    block_border_reduction_x <<< border_grid_x, block, 0, stream2 >>> (d_labels, d_image, grayImage.width, grayImage.height);
    cudaDeviceSynchronize();

    // Analysis
    propagate_labels <<< grid, block >>>(d_labels, grayImage.width, grayImage.height);

    // 9. Copy results back to host
    cudaMemcpy(global_labels, d_labels, pixel_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 10. Cleanup
    cudaFree(d_labels);
    cudaFree(d_image);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    //count the components and normalize labels (using prefix scan probably) 


}

grayscale_image_t visualize_cll(uint32_t *global_labels, const grayscale_image_t &grayImage){
    //some kind of visual representation of the labeling. 
    // Create a map of Root ID -> Random Color

    grayscale_image_t grayImage_o;
    grayImage_o.width = grayImage.width;
    grayImage_o.height = grayImage.height;
    grayImage_o.data.resize(grayImage.height * grayImage.width);

    for(uint32_t i = 0; i < grayImage.height * grayImage.width; i++) {
        uint32_t value = global_labels[i];
        
        if(value == 0){//if(grayImage.data[i] == 0) { //is background
            grayImage_o.data[i] = 0;
        } 
        else {
            // 2. It's an object. Get the root ID.
            uint32_t root = global_labels[i];

            grayImage_o.data[i] = (uint8_t)((root * 2654435761u % 254) + 1);
        }
    }

    return grayImage_o;


}


#endif


