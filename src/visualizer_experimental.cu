#include "visualizer_exp.cuh"
#include <chrono>
#include <nvjpeg.h>
#include <fstream>
#include <vector>
#include <iostream>


static void issue_rgb_pixel(
    thrust::host_vector<issued_pixel_t> &pixel_buffer,
    uint32_t image_width,
    uint32_t image_height,
    int x,
    int y,
    uint8_t r,
    uint8_t g,
    uint8_t b
) {

    if (x < 0 || x >= image_width || y < 0 || y >= image_height) {
        return;
    }
    issued_pixel_t pixel;
    pixel.pixel_index = (y * image_width + x); // no 3 since we are dealing with uchar3
    pixel.rgb.x = r;
    pixel.rgb.y = g;
    pixel.rgb.z = b;
    pixel_buffer.push_back(pixel);
}


void save_image_nvjpeg(const std::string& filename, uchar3* d_rgb, uint32_t width, uint32_t height) {
    nvjpegHandle_t nv_handle;
    nvjpegEncoderState_t nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;
    
    // 1. Initialize nvJPEG
    nvjpegCreate(NVJPEG_BACKEND_DEFAULT,NULL, &nv_handle);
    nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, NULL);
    nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, NULL);

    // 2. Set encoding parameters
    nvjpegEncoderParamsSetQuality(nv_enc_params, 50, NULL); // Quality 0-100
    nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_420, NULL);

    // 3. Prepare the image struct
    // nvJPEG expects an array of pointers to planes. 
    // For interleaved (RGBI), only channel[0] is used.
    nvjpegImage_t nv_image;
    nv_image.channel[0] = (unsigned char*)d_rgb;
    nv_image.pitch[0] = width * 3; // Number of bytes per row

    // 4. Encode the image
    nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
        &nv_image, NVJPEG_INPUT_RGBI, width, height, NULL);

    // 5. Retrieve the compressed bitstream size
    size_t length;
    nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, NULL);

    // 6. Download bitstream from GPU to Host
    std::vector<uint8_t> jpeg_data(length);
    nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, jpeg_data.data(), &length, NULL);

    // 7. Write to file
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (file.is_open()) {
        file.write((char*)jpeg_data.data(), length);
        file.close();
        std::cout << "Saved image to " << filename << std::endl;
    }

    // 8. Cleanup
    nvjpegEncoderParamsDestroy(nv_enc_params);
    nvjpegEncoderStateDestroy(nv_enc_state);
    nvjpegDestroy(nv_handle);
}

static void draw_rect_outline(
    thrust::host_vector<issued_pixel_t>& pixel_buffer, uint32_t image_width, uint32_t image_height,
    int min_x, int min_y, int max_x, int max_y,
    uint8_t r, uint8_t g, uint8_t b
) {
    // Top and Bottom lines
    for (int x = min_x; x <= max_x; ++x) {
        issue_rgb_pixel(pixel_buffer, image_width, image_height,x, min_y, r, g, b);
        issue_rgb_pixel(pixel_buffer, image_width, image_height, x, max_y, r, g, b);
    }
    // Left and Right lines
    for (int y = min_y; y <= max_y; ++y) {
        issue_rgb_pixel(pixel_buffer, image_width, image_height, min_x, y, r, g, b);
        issue_rgb_pixel(pixel_buffer, image_width, image_height, max_x, y, r, g, b);
    }
}
static void draw_square(
    thrust::host_vector<issued_pixel_t>& pixel_buffer, uint32_t image_width, uint32_t image_height,
    int cx,
    int cy,
    int radius,
    uint8_t r,
    uint8_t g,
    uint8_t b
) {
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            issue_rgb_pixel(pixel_buffer, image_width, image_height, cx + dx, cy + dy, r, g, b);
        }
    }
}

static void color_for_edge_class(
    char edge_class,
    uint8_t& r,
    uint8_t& g,
    uint8_t& b
) {
    switch (edge_class) {
        case 'L': // straight
            r = 255; g = 255; b = 100;
            break;

        case 'V': // knob / outward
            r = 255; g = 0; b = 0;
            break;

        case 'C': // hole / inward
            r = 0; g = 120; b = 255;
            break;

        default:
            r = 255; g = 0; b = 255;
            break;
    }
}


static void draw_char(thrust::host_vector<issued_pixel_t>& pixel_buffer, uint32_t image_width, uint32_t image_height,
     int x, int y, char c, int scale, uint8_t r, uint8_t g, uint8_t b) {
    static const uint8_t font_L[7] = {0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111};
    static const uint8_t font_V[7] = {0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100};
    static const uint8_t font_C[7] = {0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110};

    const uint8_t* bitmap = nullptr;
    if (c == 'L') bitmap = font_L;
    else if (c == 'V') bitmap = font_V;
    else if (c == 'C') bitmap = font_C;
    else return;

    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 5; ++j) {
            if ((bitmap[i] >> (4 - j)) & 1) {
                // Scale logic: draw a block of pixels for every 1 in the bitmap
                for (int sy = 0; sy < scale; ++sy) {
                    for (int sx = 0; sx < scale; ++sx) {
                        issue_rgb_pixel(pixel_buffer, image_width, image_height, x + (j * scale) + sx, y + (i * scale) + sy, r, g, b);
                    }
                }
            }
        }
    }
}
static void draw_edge_label_string(thrust::host_vector<issued_pixel_t>& pixel_buffer, uint32_t image_width, uint32_t image_height,
     int x, int y, const std::vector<char>& edges, uint32_t top_edge_index) {
    int scale = 5;
    int edge = top_edge_index;
    // Draw each character
    for (size_t i = 0; i < edges.size(); ++i) {

        draw_char(pixel_buffer, image_width, image_height, x + (i * 7*scale), y - scale*8, edges[(edge + i) % 4],scale, 255, 255, 255);
    }
}

__global__ void draw_visuals(issued_pixel_t* pixel_buffer, uchar3* image, uint32_t count){

    uint32_t idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx >=count)
        return;

    issued_pixel_t pixel = pixel_buffer[idx];
    image[pixel.pixel_index].x = pixel.rgb.x;
    image[pixel.pixel_index].y = pixel.rgb.y;
    image[pixel.pixel_index].z = pixel.rgb.z;
    
}




void visualizer_exp(
    rgb_image_d_t* original_image_d,
    const std::string& filename,
    uint32_t image_width, uint32_t image_height,
    const thrust::device_vector<ContourSignalPointExt>& d_points,
    const thrust::host_vector <ContourSignalPointExt>& points,
    const thrust::host_vector<PieceCorners>& corners,
    const thrust::host_vector<ClassifiedPiece>& classified

) {
    // rgb_image_t out;
    // out.width =width;
    // out.height =height;
    // out.data.resize(width * height * 3); // this takes forevecer (around 50 ms)
    // uint32_t width= original_image.width;
    // uint32_t height = original_image.height;
    // uint32_t pixel_count= original_image.width * original_image.height;

    
    // std::unordered_map<uint32_t, const PieceCorners*> corner_lookup;
    // for (const auto& c : corners) {
    //     corner_lookup[c.label] = &c;
    // }

    
    // //dark grayscale background --- in a kernel
    // uint32_t threads = 256;
    // uint32_t blocks = (pixel_count + threads - 1) / threads;

    // uint8_t* image_background_d; 
    // cudaMalloc(&image_background_d, 3 * pixel_count * sizeof(uint8_t));
    // set_background<<<blocks,threads>>>(image_in_d, image_background_d, pixel_count);
    // cudaMemcpy(out.data.data(),image_background_d, 3 * pixel_count * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    thrust::host_vector<issued_pixel_t> pixel_buffer;
    
    for (const auto& piece : classified) {
        if (!piece.valid || piece.edges.size() != 4) {
            continue;
        }

        const PieceCorners* piece_corners = nullptr;

        //find correct corner for the current piece
        for (const auto& c : corners) {
            if (c.label == piece.label) {
                piece_corners = &c;
                break;
            }
        }

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

        if (piece_length == 0 || piece_start + piece_length > points.size()) {
            continue;
        }



        float min_x = 1e10f, max_x = -1e10f;
        float min_y = 1e10f, max_y = -1e10f;

        for (uint32_t i = 0; i < piece_length; ++i) {
            const auto& p = points[piece_start + i];
            if (p.x < min_x) min_x = p.x;
            if (p.x > max_x) max_x = p.x;
            if (p.y < min_y) min_y = p.y;
            if (p.y > max_y) max_y = p.y;
        }

        // Draw the bounding box (e.g., White or Gray)
        // We add a small 2-pixel padding so it doesn't overlap the contour exactly
        draw_rect_outline(pixel_buffer, image_width, image_height,
            (int)min_x - 2, (int)min_y - 2, 
            (int)max_x + 2, (int)max_y + 2, 
            0, 255, 0); 

        uint8_t last_corner_index= piece_corners->point_indices[3];
        uint8_t top_edge_index = 3;

        //wenn der 1. corner in dem vektor die root ist und der x index vom 1. corner nicht größer als der vom letzten corner ist dann 
        if(points[first_corner_idx].rank == 0 && points[first_corner_idx].x < points[last_corner_index].x)
            top_edge_index = 0;
            
        draw_edge_label_string(pixel_buffer, image_width, image_height, (int)min_x, (int)min_y - 10, piece.edges, top_edge_index);

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

            // draw all contour points belonging to this edge
            for (uint32_t d = 0; d <= edge_len; ++d) {
                uint32_t rank = (rank_a + d) % piece_length;
                uint32_t idx = piece_start + rank;

                if (idx >= points.size()) {
                    continue;
                }

                const auto& p = points[idx];

                // slightly thick contour line
                draw_square(
                    pixel_buffer, image_width, image_height,
                    static_cast<int>(p.x),
                    static_cast<int>(p.y),
                    1,
                    r, g, b
                );
            }

            // draw corners in green on top
            const auto& a = points[a_idx];
            const auto& bpt = points[b_idx];

            draw_square(pixel_buffer, image_width, image_height, a.x, a.y, 4, 0, 255, 0);
            draw_square(pixel_buffer, image_width, image_height, bpt.x, bpt.y, 4, 0, 255, 0);
        }
    }

    thrust::device_vector<issued_pixel_t> pixel_buffer_d = pixel_buffer;

    uint32_t threads= 256;
    uint32_t blocks = (pixel_buffer.size()+ threads - 1)/threads;

    draw_visuals<<<blocks,threads>>>(thrust::raw_pointer_cast(pixel_buffer_d.data()), thrust::raw_pointer_cast(original_image_d->data.data()), pixel_buffer.size());

    cudaDeviceSynchronize();

    save_image_nvjpeg(filename, 
                      thrust::raw_pointer_cast(original_image_d->data.data()), 
                      image_width, 
                      image_height);


    return;
}
