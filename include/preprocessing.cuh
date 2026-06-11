#pragma once
#include "types.hpp"
#include <string>

rgb_image_d_t preprocessing(const std::string &image_path, uint8_t *&d_result);
