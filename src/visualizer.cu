#include "visualizer.cuh"
#include <chrono>



static void set_rgb_pixel(
    rgb_image_t& img,
    int x,
    int y,
    uint8_t r,
    uint8_t g,
    uint8_t b
) {
    if (x < 0 || x >= img.width || y < 0 || y >= img.height) {
        return;
    }

    int idx = 3 * (y * img.width + x);
    img.data[idx + 0] = r;
    img.data[idx + 1] = g;
    img.data[idx + 2] = b;
}

static void draw_square(
    rgb_image_t& img,
    int cx,
    int cy,
    int radius,
    uint8_t r,
    uint8_t g,
    uint8_t b
) {
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            set_rgb_pixel(img, cx + dx, cy + dy, r, g, b);
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

__global__ void set_background(const uint8_t* base_image_data, uint8_t* image_out, uint32_t pixel_count){

    uint32_t idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx >= pixel_count)
        return;

    uint8_t v = base_image_data[idx] >> 2; // bitshift is faster than division by 4
    image_out[idx * 3 + 0] = v;
    image_out[idx * 3 + 1] = v;
    image_out[idx * 3 + 2] = v;
    
}

void visualizer(
    rgb_image_t &original_image,
    const thrust::device_vector<ContourSignalPointExt>& d_points,
    const thrust::host_vector <ContourSignalPointExt>& points,
    const thrust::host_vector<PieceCorners>& corners,
    const thrust::host_vector<ClassifiedPiece>& classified
) {
    // rgb_image_t out;
    // out.width =width;
    // out.height =height;
    // out.data.resize(width * height * 3); // this takes forevecer (around 50 ms)
    uint32_t width= original_image.width;
    uint32_t height = original_image.height;
    uint32_t pixel_count= original_image.width * original_image.height;

    
    std::unordered_map<uint32_t, const PieceCorners*> corner_lookup;
    for (const auto& c : corners) {
        corner_lookup[c.label] = &c;
    }

    
    // //dark grayscale background --- in a kernel
    // uint32_t threads = 256;
    // uint32_t blocks = (pixel_count + threads - 1) / threads;

    // uint8_t* image_background_d; 
    // cudaMalloc(&image_background_d, 3 * pixel_count * sizeof(uint8_t));
    // set_background<<<blocks,threads>>>(image_in_d, image_background_d, pixel_count);
    // cudaMemcpy(out.data.data(),image_background_d, 3 * pixel_count * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    
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
                    original_image,
                    static_cast<int>(p.x),
                    static_cast<int>(p.y),
                    1,
                    r, g, b
                );
            }

            // draw corners in green on top
            const auto& a = points[a_idx];
            const auto& bpt = points[b_idx];

            draw_square(original_image, a.x, a.y, 4, 0, 255, 0);
            draw_square(original_image, bpt.x, bpt.y, 4, 0, 255, 0);
        }
    }
    return;
}
