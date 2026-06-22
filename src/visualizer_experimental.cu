#include "visualizer_exp.cuh"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "timer.hpp"

//#define DRAW_COLORED_EDGES
//#define DRAW_CORNERS

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
        throw std::runtime_error(msg);
    }
}

static void check_nvjpeg(nvjpegStatus_t status, const char* msg) {
    if (status != NVJPEG_STATUS_SUCCESS) {
        std::cerr << msg << ": nvJPEG error code "
                  << static_cast<int>(status) << "\n";
        throw std::runtime_error(msg);
    }
}

__global__ void draw_visuals_kernel(
    const issued_pixel_t* pixels,
    uchar3* image,
    uint32_t num_pixels
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_pixels) {
        return;
    }

    issued_pixel_t p = pixels[idx];
    image[p.pixel_index] = p.rgb;
}

// void free_visualizer_workspace(
//     visualizer_workspace_t& ws
// ) {
//     if (ws.d_pixels != nullptr) {
//         cudaFree(ws.d_pixels);
//         ws.d_pixels = nullptr;
//     }

//     ws.capacity = 0;
// }

void free_visualizer_workspace(
    visualizer_workspace_t& ws
) {
    if (ws.d_pixels != nullptr) {
        cudaFree(ws.d_pixels);
        ws.d_pixels = nullptr;
    }

    ws.capacity = 0;

    if (ws.d_image_copy != nullptr) {
        cudaFree(ws.d_image_copy);
        ws.d_image_copy = nullptr;
    }

    ws.image_capacity = 0;
}

static void ensure_image_copy_capacity(
    visualizer_workspace_t& ws,
    size_t required_pixels
) {
    if (required_pixels == 0) {
        return;
    }

    if (ws.image_capacity >= required_pixels) {
        return;
    }

    if (ws.d_image_copy != nullptr) {
        check_cuda(cudaFree(ws.d_image_copy), "cudaFree visualizer image copy");
        ws.d_image_copy = nullptr;
        ws.image_capacity = 0;
    }

    check_cuda(
        cudaMalloc(&ws.d_image_copy, required_pixels * sizeof(uchar3)),
        "cudaMalloc visualizer image copy"
    );

    ws.image_capacity = required_pixels;
}

static void ensure_pixel_capacity(
    visualizer_workspace_t& ws,
    size_t required
) {
    if (required == 0) {
        return;
    }

    if (ws.capacity >= required) {
        return;
    }

    if (ws.d_pixels != nullptr) {
        check_cuda(cudaFree(ws.d_pixels), "cudaFree visualizer pixel buffer");
        ws.d_pixels = nullptr;
        ws.capacity = 0;
    }

    check_cuda(
        cudaMalloc(&ws.d_pixels, required * sizeof(issued_pixel_t)),
        "cudaMalloc visualizer pixel buffer"
    );

    ws.capacity = required;
}

static void upload_and_draw_pixels(
    visualizer_workspace_t& ws,
    const std::vector<issued_pixel_t>& pixels,
    uchar3* d_image,
    cudaStream_t stream
) {
    if (pixels.empty()) {
        return;
    }

    ensure_pixel_capacity(ws, pixels.size());

    check_cuda(
        cudaMemcpyAsync(
            ws.d_pixels,
            pixels.data(),
            pixels.size() * sizeof(issued_pixel_t),
            cudaMemcpyHostToDevice,
            stream
        ),
        "cudaMemcpy issued pixels"
    );

    uint32_t threads = 256;
    uint32_t blocks =
        static_cast<uint32_t>((pixels.size() + threads - 1) / threads);

    draw_visuals_kernel<<<blocks, threads, 0, stream>>>(
        ws.d_pixels,
        d_image,
        static_cast<uint32_t>(pixels.size())
    );

    //CUDA_KERNEL_CHECK();
}

void init_jpeg_encoder(
    jpeg_encoder_t& enc,
    int quality
) {
    check_nvjpeg(
        nvjpegCreate(NVJPEG_BACKEND_DEFAULT, nullptr, &enc.handle),
        "nvjpegCreate"
    );

    check_nvjpeg(
        nvjpegEncoderStateCreate(enc.handle, &enc.state, nullptr),
        "nvjpegEncoderStateCreate"
    );

    check_nvjpeg(
        nvjpegEncoderParamsCreate(enc.handle, &enc.params, nullptr),
        "nvjpegEncoderParamsCreate"
    );

    check_nvjpeg(
        nvjpegEncoderParamsSetQuality(enc.params, quality, nullptr),
        "nvjpegEncoderParamsSetQuality"
    );

    check_nvjpeg(
        nvjpegEncoderParamsSetSamplingFactors(
            enc.params,
            NVJPEG_CSS_420,
            nullptr
        ),
        "nvjpegEncoderParamsSetSamplingFactors"
    );
}

void destroy_jpeg_encoder(
    jpeg_encoder_t& enc
) {
    if (enc.params != nullptr) {
        nvjpegEncoderParamsDestroy(enc.params);
        enc.params = nullptr;
    }

    if (enc.state != nullptr) {
        nvjpegEncoderStateDestroy(enc.state);
        enc.state = nullptr;
    }

    if (enc.handle != nullptr) {
        nvjpegDestroy(enc.handle);
        enc.handle = nullptr;
    }

    enc.bitstream.clear();
}

static void save_image_nvjpeg_reuse(
    jpeg_encoder_t& enc,
    const std::string& filename,
    uchar3* d_rgb,
    uint32_t width,
    uint32_t height,
    cudaStream_t stream
) {
    if (enc.handle == nullptr ||
        enc.state == nullptr ||
        enc.params == nullptr) {
        throw std::runtime_error("jpeg_encoder_t was not initialized");
    }

    static_assert(
        sizeof(uchar3) == 3,
        "uchar3 must be 3 bytes for NVJPEG_INPUT_RGBI"
    );

    nvjpegImage_t nv_image{};
    nv_image.channel[0] = reinterpret_cast<unsigned char*>(d_rgb);
    nv_image.pitch[0] = width * sizeof(uchar3);

    check_nvjpeg(
        nvjpegEncodeImage(
            enc.handle,
            enc.state,
            enc.params,
            &nv_image,
            NVJPEG_INPUT_RGBI,
            width,
            height,
            stream
        ),
        "nvjpegEncodeImage"
    );

    size_t length = 0;

    check_nvjpeg(
        nvjpegEncodeRetrieveBitstream(
            enc.handle,
            enc.state,
            nullptr,
            &length,
            stream
        ),
        "nvjpegEncodeRetrieveBitstream size"
    );

    enc.bitstream.resize(length);

    check_nvjpeg(
        nvjpegEncodeRetrieveBitstream(
            enc.handle,
            enc.state,
            enc.bitstream.data(),
            &length,
            stream
        ),
        "nvjpegEncodeRetrieveBitstream data"
    );

    check_cuda(cudaStreamSynchronize(stream), "visualizer encode stream synchronize");

    std::ofstream file(filename, std::ios::out | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + filename);
    }

    file.write(
        reinterpret_cast<const char*>(enc.bitstream.data()),
        static_cast<std::streamsize>(length)
    );
}

static void issue_rgb_pixel(
    std::vector<issued_pixel_t>& pixel_buffer,
    uint32_t image_width,
    uint32_t image_height,
    int x,
    int y,
    uint8_t r,
    uint8_t g,
    uint8_t b
) {
    if (x < 0 || y < 0) {
        return;
    }

    if (x >= static_cast<int>(image_width) ||
        y >= static_cast<int>(image_height)) {
        return;
    }

    issued_pixel_t pixel;
    pixel.pixel_index =
        static_cast<uint32_t>(y) * image_width + static_cast<uint32_t>(x);
    pixel.rgb = make_uchar3(r, g, b);

    pixel_buffer.push_back(pixel);
}

static void draw_square(
    std::vector<issued_pixel_t>& pixel_buffer,
    uint32_t image_width,
    uint32_t image_height,
    int cx,
    int cy,
    int radius,
    uint8_t r,
    uint8_t g,
    uint8_t b
) {
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            issue_rgb_pixel(
                pixel_buffer,
                image_width,
                image_height,
                cx + dx,
                cy + dy,
                r, g, b
            );
        }
    }
}

static void draw_rect_outline(
    std::vector<issued_pixel_t>& pixel_buffer,
    uint32_t image_width,
    uint32_t image_height,
    int x0,
    int y0,
    int x1,
    int y1,
    uint8_t r,
    uint8_t g,
    uint8_t b
) {
    if (x0 > x1) {
        std::swap(x0, x1);
    }

    if (y0 > y1) {
        std::swap(y0, y1);
    }

    for (int x = x0; x <= x1; ++x) {
        issue_rgb_pixel(pixel_buffer, image_width, image_height, x, y0, r, g, b);
        issue_rgb_pixel(pixel_buffer, image_width, image_height, x, y1, r, g, b);
    }

    for (int y = y0; y <= y1; ++y) {
        issue_rgb_pixel(pixel_buffer, image_width, image_height, x0, y, r, g, b);
        issue_rgb_pixel(pixel_buffer, image_width, image_height, x1, y, r, g, b);
    }
}

static void color_for_edge_class(
    char edge_class,
    uint8_t& r,
    uint8_t& g,
    uint8_t& b
) {
    switch (edge_class) {
        case 'L':
            r = 255; g = 255; b = 100;
            break;

        case 'V':
            r = 255; g = 0; b = 0;
            break;

        case 'C':
            r = 0; g = 120; b = 255;
            break;

        default:
            r = 255; g = 0; b = 255;
            break;
    }
}

/*
 * Tiny block font for L, V, C, and ?.
 * Each glyph is 3x5.
 */
static const uint8_t* glyph_for_char(char c) {
    static const uint8_t glyph_L[5] = {
        0b100,
        0b100,
        0b100,
        0b100,
        0b111
    };

    static const uint8_t glyph_V[5] = {
        0b101,
        0b101,
        0b101,
        0b101,
        0b010
    };

    static const uint8_t glyph_C[5] = {
        0b111,
        0b100,
        0b100,
        0b100,
        0b111
    };

    static const uint8_t glyph_Q[5] = {
        0b111,
        0b001,
        0b011,
        0b000,
        0b010
    };

    switch (c) {
        case 'L':
            return glyph_L;
        case 'V':
            return glyph_V;
        case 'C':
            return glyph_C;
        default:
            return glyph_Q;
    }
}

static void draw_char_3x5(
    std::vector<issued_pixel_t>& pixel_buffer,
    uint32_t image_width,
    uint32_t image_height,
    int x,
    int y,
    char c,
    uint8_t r,
    uint8_t g,
    uint8_t b,
    int scale = 2
) {
    const uint8_t* glyph = glyph_for_char(c);

    for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 3; ++col) {
            bool on = (glyph[row] >> (2 - col)) & 1;

            if (!on) {
                continue;
            }

            for (int sy = 0; sy < scale; ++sy) {
                for (int sx = 0; sx < scale; ++sx) {
                    issue_rgb_pixel(
                        pixel_buffer,
                        image_width,
                        image_height,
                        x + col * scale + sx,
                        y + row * scale + sy,
                        r, g, b
                    );
                }
            }
        }
    }
}

static void draw_edge_label_string(
    std::vector<issued_pixel_t>& pixel_buffer,
    uint32_t image_width,
    uint32_t image_height,
    int x,
    int y,
    const std::vector<char>& edges,
    uint32_t top_edge_index
) {
    if (edges.size() != 4) {
        return;
    }

    int cursor_x = x;

    for (uint32_t i = 0; i < 4; ++i) {
        uint32_t edge_idx = (top_edge_index + i) % 4;
        char c = edges[edge_idx];

        uint8_t r = 255;
        uint8_t g = 255;
        uint8_t b = 255;
        int scale = 5;

        draw_char_3x5(
            pixel_buffer,
            image_width,
            image_height,
            cursor_x,
            y,
            c,
            r, g, b,
            scale
        );

        cursor_x += 4 * scale;
    }
}

void visualizer_exp(
    rgb_image_d_t* original_image_d,
    const std::string& filename,
    uint32_t image_width,
    uint32_t image_height,
    const thrust::host_vector<ContourSignalPointExt>& points,
    const thrust::host_vector<PieceCorners>& corners,
    const thrust::host_vector<ClassifiedPiece>& classified,
    visualizer_workspace_t& ws,
    jpeg_encoder_t& jpeg_encoder,
    cudaStream_t stream
) {
    CPU_TIMER_START(visualizer_total);

    std::vector<issued_pixel_t> pixel_buffer;
    pixel_buffer.reserve(points.size() * 12 + classified.size() * 2500);
    // pixel_buffer.reserve(classified.size() * 4000);

    std::unordered_map<uint32_t, const PieceCorners*> corner_lookup;
    corner_lookup.reserve(corners.size());

    for (const auto& c : corners) {
        corner_lookup[c.label] = &c;
    }

    CPU_TIMER_START(build_visualizer_pixels);

    for (const auto& piece : classified) {
        if (!piece.valid || piece.edges.size() != 4) {
            continue;
        }

        auto it = corner_lookup.find(piece.label);

        if (it == corner_lookup.end()) {
            continue;
        }

        const PieceCorners* piece_corners = it->second;

        if (piece_corners == nullptr ||
            !piece_corners->valid ||
            piece_corners->point_indices.size() != 4) {
            continue;
        }

        uint32_t first_corner_idx = piece_corners->point_indices[0];

        if (first_corner_idx >= points.size()) {
            continue;
        }

        uint32_t piece_start =
            first_corner_idx - points[first_corner_idx].rank;

        uint32_t piece_length =
            points[first_corner_idx].piece_length;

        if (piece_length == 0 ||
            piece_start + piece_length > points.size()) {
            continue;
        }

        uint32_t min_x = image_width;
        uint32_t min_y = image_height;
        uint32_t max_x = 0;
        uint32_t max_y = 0;

        for (uint32_t i = 0; i < piece_length; ++i) {
            const auto& p = points[piece_start + i];

            min_x = std::min(min_x, p.x);
            min_y = std::min(min_y, p.y);
            max_x = std::max(max_x, p.x);
            max_y = std::max(max_y, p.y);
        }

        draw_rect_outline(
            pixel_buffer,
            image_width,
            image_height,
            static_cast<int>(min_x) - 2,
            static_cast<int>(min_y) - 2,
            static_cast<int>(max_x) + 2,
            static_cast<int>(max_y) + 2,
            0, 255, 0
        );

        uint32_t c0 = first_corner_idx;
        uint32_t c1 = piece_corners->point_indices[1];
        uint32_t c3 = piece_corners->point_indices[3];
        uint32_t top_edge_index = 3;

        if (c1 < points.size()) {
            if (points[c1].y < points[c3].y) {
                top_edge_index = 0;
            }
        }

        draw_edge_label_string(
            pixel_buffer,
            image_width,
            image_height,
            static_cast<int>(min_x),
            static_cast<int>(min_y) - 35,
            piece.edges,
            top_edge_index
        );
#ifdef DRAW_COLORED_EDGES

        for (uint32_t edge = 0; edge < 4; ++edge) {
            uint32_t a_idx = piece_corners->point_indices[edge];
            uint32_t b_idx = piece_corners->point_indices[(edge + 1) % 4];

            if (a_idx >= points.size() || b_idx >= points.size()) {
                continue;
            }

            uint32_t rank_a = points[a_idx].rank;
            uint32_t rank_b = points[b_idx].rank;

            uint32_t edge_len =
                (rank_b + piece_length - rank_a) % piece_length;

            uint8_t r, g, b;
            color_for_edge_class(piece.edges[edge], r, g, b);

            for (uint32_t d = 0; d <= edge_len; ++d) {
                uint32_t rank = (rank_a + d) % piece_length;
                uint32_t idx = piece_start + rank;

                if (idx >= points.size()) {
                    continue;
                }

                const auto& p = points[idx];

                draw_square(
                    pixel_buffer,
                    image_width,
                    image_height,
                    static_cast<int>(p.x),
                    static_cast<int>(p.y),
                    1,
                    r, g, b
                );
            }
        }
#endif
#ifdef DRAW_CORNERS

        for (uint32_t corner_idx : piece_corners->point_indices) {
            if (corner_idx >= points.size()) {
                continue;
            }

            const auto& p = points[corner_idx];

            draw_square(
                pixel_buffer,
                image_width,
                image_height,
                static_cast<int>(p.x),
                static_cast<int>(p.y),
                4,
                0, 255, 0
            );
        }
#endif
    }

    CPU_TIMER_STOP(build_visualizer_pixels, "Build visualizer pixel buffer");

    // uchar3* d_image = original_image_d->data;

    // GPU_TIMER_START(draw_visual_pixels);
    // upload_and_draw_pixels(ws, pixel_buffer, d_image, stream);
    // GPU_TIMER_STOP(draw_visual_pixels, "Draw visual pixels");

    // CPU_TIMER_START(save_visualized_image);
    // save_image_nvjpeg_reuse(
    //     jpeg_encoder,
    //     filename,
    //     d_image,
    //     image_width,
    //     image_height
    // );
    // CPU_TIMER_STOP(save_visualized_image, "Save visualized image");

    // CPU_TIMER_STOP(visualizer_total, "Visualizer total");

    size_t num_pixels = static_cast<size_t>(image_width) * static_cast<size_t>(image_height);

    ensure_image_copy_capacity(ws, num_pixels);

    check_cuda(
        cudaMemcpyAsync(
            ws.d_image_copy,
            original_image_d->data,
            num_pixels * sizeof(uchar3),
            cudaMemcpyDeviceToDevice,
            stream
        ),
        "cudaMemcpy visualizer image copy"
    );

    uchar3* d_image = ws.d_image_copy;

    GPU_TIMER_START(draw_visual_pixels);
    upload_and_draw_pixels(ws, pixel_buffer, d_image, stream);
    GPU_TIMER_STOP(draw_visual_pixels, "Draw visual pixels");

    CPU_TIMER_START(save_visualized_image);
    save_image_nvjpeg_reuse(
        jpeg_encoder,
        filename,
        d_image,
        image_width,
        image_height,
        stream
    );
    CPU_TIMER_STOP(save_visualized_image, "Save visualized image");
}

void visualizer_exp(
    rgb_image_d_t* original_image_d,
    const std::string& filename,
    uint32_t image_width,
    uint32_t image_height,
    const thrust::host_vector<ContourSignalPointExt>& points,
    const thrust::host_vector<PieceCorners>& corners,
    const thrust::host_vector<ClassifiedPiece>& classified,
    visualizer_workspace_t& ws,
    jpeg_encoder_t& jpeg_encoder
) {
    visualizer_exp(original_image_d, filename, image_width, image_height, points, corners, classified, ws, jpeg_encoder, 0);
}
