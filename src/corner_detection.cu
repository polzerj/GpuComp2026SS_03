#include "corner_detection.cuh"
#include <iostream>

__global__ void find_corners(const ContourSignalPointExt* d_smoothed, bool* corners,
    uint32_t peak_detection_radius, float prominence_threshold, uint32_t num_boundary)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_boundary)
            return;
        
        ContourSignalPointExt self = d_smoothed[idx];
        ContourSignalPointExt neighbour_next, neighbour_prev;

        uint32_t start_idx = idx - self.rank;
        uint32_t neighbour_offset_right = (self.rank + 1 ) % self.piece_length;
        uint32_t neighbour_offset_left = 
            (((static_cast<int>(self.rank) - 1) % static_cast<int>(self.piece_length)) + static_cast<int>(self.piece_length)) % static_cast<int>(self.piece_length);
        neighbour_next = d_smoothed[start_idx + neighbour_offset_right];
        neighbour_prev = d_smoothed[start_idx + neighbour_offset_left];

        bool is_peak = false;
        // we are at least some kind of local maxima
        if(self.distance > neighbour_next.distance && self.distance > neighbour_prev.distance)
        {

            is_peak= true;
            float distance_avg = 0;
            for(uint8_t radius = 1; radius< peak_detection_radius; ++radius){

                neighbour_offset_right = (self.rank + radius )% self.piece_length;
                neighbour_offset_left = ((self.rank - radius + self.piece_length) % self.piece_length);

                
                neighbour_next = d_smoothed[start_idx + neighbour_offset_right];
                neighbour_prev = d_smoothed[start_idx + neighbour_offset_left];

                if(self.distance < neighbour_next.distance || self.distance < neighbour_prev.distance) 
                {
                    is_peak = false;
                    break;
                }
                
                distance_avg = (distance_avg + neighbour_next.distance + neighbour_prev.distance);


            }
            distance_avg = distance_avg / (2 * peak_detection_radius - 1);
            is_peak = (is_peak && prominence_threshold < (self.distance - distance_avg) );

        

        }

        corners[idx] = is_peak;


}


thrust::host_vector<PieceCorners> detect_corners(thrust::host_vector<ContourSignalPointExt>& h_smoothed, 
    uint32_t peak_detection_radius, float prominence_threshold, uint32_t distance_threshold){

        uint32_t num_points = static_cast<uint32_t>(h_smoothed.size());

        if (num_points == 0) {
            return thrust::host_vector<PieceCorners>();
        }
        
        thrust::device_vector<ContourSignalPointExt> d_smoothed = h_smoothed;
        thrust::device_vector<bool> d_corners(num_points, false);

        uint32_t threads = 256;
        uint32_t blocks = (num_points + threads - 1) / threads;

        find_corners <<<blocks, threads>>> (
            thrust::raw_pointer_cast(d_smoothed.data()),
            thrust::raw_pointer_cast(d_corners.data()),
            peak_detection_radius,
            prominence_threshold,
            num_points
        );

        thrust::host_vector<bool> h_corners_bool = d_corners;
        thrust::host_vector<PieceCorners> h_corners;
        if (num_points == 0) return h_corners;

        uint32_t current_label = h_smoothed[0].label;
        PieceCorners current_piece;
        current_piece.label = current_label;
        current_piece.valid = false; 
        int corners_found = 0;

        for (uint32_t i = 0; i < num_points; ++i) {

            if (h_smoothed[i].label != current_label) {
                current_piece.valid = (corners_found == 4); 
                h_corners.push_back(current_piece);

                current_label = h_smoothed[i].label;
                corners_found = 0;
                current_piece = PieceCorners();
                current_piece.label = current_label;
                current_piece.valid = false;
            }

            if (h_corners_bool[i]) {
                current_piece.corner_ranks.push_back(h_smoothed[i].rank);
                current_piece.point_indices.push_back(i);
                corners_found++; 
            }
        }

        current_piece.valid = (corners_found == 4);
        h_corners.push_back(current_piece);

        return h_corners;
    }