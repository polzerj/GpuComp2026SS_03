#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include "types.hpp"

grayscale_image_t preprocessing(std::string image_path){
    (void) image_path;

    grayscale_image_t image;
    image.width = 0;
    image.height = 0;
    image.data = nullptr;

    printf("Preprocessing is not implemented yet.\n");

    return image;
}
