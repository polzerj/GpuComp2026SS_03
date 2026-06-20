#include <cstdint>
#include "types.hpp"
#include "pipeline_workspace.hpp"


uint32_t* playne_ccl (uint8_t*d_image, uint32_t width, uint32_t height, pipeline_workspace_t& ws);
grayscale_image_t visualize_cll(uint32_t *global_labels, const grayscale_image_t &grayImage);
