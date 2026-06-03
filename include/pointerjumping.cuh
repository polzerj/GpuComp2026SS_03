#include <cstdint>
#include "types.hpp"

typedef struct pointer_jumping_outputs {
    uint32_t* successors;
    uint32_t* ranks;
    uint32_t* boundaries;
}ptr_jmp_out_t;

ptr_jmp_out_t pointer_jumping(uint32_t * global_labels, uint32_t image_width, uint32_t image_height);
