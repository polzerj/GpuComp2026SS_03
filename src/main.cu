#include "types.hpp"
#include "preprocessing.cuh"
#include "piece_segmentation.cuh"


int main(){
    grayscale_image_t image = preprocessing("image.jpg");
    piece_segmentation(image);
    return 0;
}
