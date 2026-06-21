#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "workspace.hpp"
#include "types.hpp"

struct pipeline_workspace_t {
    size_t pixel_capacity = 0;

    // device_buffer_t<uint8_t> d_u8_a;
    // device_buffer_t<uint8_t> d_u8_b;
    // device_buffer_t<uint8_t> d_u8_c;

    // device_buffer_t<float> d_float_a;
    // device_buffer_t<float> d_float_b;

    // device_buffer_t<uint32_t> d_u32_a;
    // device_buffer_t<uint32_t> d_u32_b;
    // device_buffer_t<uint32_t> d_u32_c;
    // device_buffer_t<uint32_t> d_u32_d;
    
    device_buffer_t<uchar3> d_rgb;
    device_buffer_t<uint8_t> d_gray;
    device_buffer_t<uint8_t> d_blur;
    device_buffer_t<uint8_t> d_binary;

    device_buffer_t<int> d_hist;
    device_buffer_t<float> d_threshold;

    device_buffer_t<brightness_t> d_seg_tmp;
    device_buffer_t<brightness_t> d_seg_out;

    device_buffer_t<uint32_t> d_ccl_labels;

    device_buffer_t<uint32_t> d_ptr_boundaries;
    device_buffer_t<uint32_t> d_ptr_successors;
    device_buffer_t<uint32_t> d_ptr_ranks;
    device_buffer_t<uint32_t> d_ptr_predecessors;

    device_buffer_t<uint2> d_ptr_work_a;
    device_buffer_t<uint2> d_ptr_work_b;

    device_buffer_t<uint32_t> d_ptr_done;

    device_buffer_t<unsigned long long> d_centroid_sum_x;
    device_buffer_t<unsigned long long> d_centroid_sum_y;
    device_buffer_t<uint32_t> d_centroid_count;
    device_buffer_t<uint32_t> d_centroid_indices;
    device_buffer_t<uint32_t> d_centroid_valid_labels;

    device_buffer_t<float> d_sort_center_x;
    device_buffer_t<float> d_sort_center_y;
    device_buffer_t<uint32_t> d_sort_boundary_ids;
    device_buffer_t<uint64_t> d_sort_keys;

    device_buffer_t<uint32_t> d_smooth_length_at_start;

    device_buffer_t<uint8_t> d_corner_flags;
    device_buffer_t<float> d_corner_scores;

    device_buffer_t<uint2> d_work_a;
    device_buffer_t<uint2> d_work_b;


    void ensure_pixel_capacity(size_t pixel_count) {
        if (pixel_capacity >= pixel_count) {
            return;
        }

        d_rgb.ensure(pixel_count);
        d_gray.ensure(pixel_count);
        d_blur.ensure(pixel_count);
        d_binary.ensure(pixel_count);

        // d_u8_a.ensure(pixel_count);
        // d_u8_b.ensure(pixel_count);
        // d_u8_c.ensure(pixel_count);

        // d_float_a.ensure(pixel_count);
        // d_float_b.ensure(pixel_count);

        // d_u32_a.ensure(pixel_count);
        // d_u32_b.ensure(pixel_count);
        // d_u32_c.ensure(pixel_count);
        // d_u32_d.ensure(pixel_count);

        d_work_a.ensure(pixel_count);
        d_work_b.ensure(pixel_count);

        pixel_capacity = pixel_count;
    }
    void ensure_preprocessing_small_buffers() {
        d_hist.ensure(256);
        d_threshold.ensure(1);
    }
};
