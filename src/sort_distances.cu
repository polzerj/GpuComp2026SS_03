#include "sort_distances.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

struct IsBoundary {
    __device__ bool operator()(uint32_t label) const {
        return label != 0;
    }
};

__global__ void scatter_centroids_by_label(
    const Centroid* centroids,
    uint32_t num_centroids,
    float* center_x,
    float* center_y
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_centroids) {
        return;
    }

    uint32_t label = centroids[i].label;
    center_x[label] = centroids[i].x;
    center_y[label] = centroids[i].y;
}

__global__ void make_signal_keys_and_values(
    const uint32_t* boundary_ids,
    const uint32_t* boundaries,
    const uint32_t* ranks,
    const float* center_x,
    const float* center_y,
    uint64_t* keys,
    ContourSignalPoint* points,
    uint32_t width,
    uint32_t num_boundary
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_boundary) {
        return;
    }

    uint32_t pixel_id = boundary_ids[i];

    uint32_t label = boundaries[pixel_id];
    uint32_t rank = ranks[pixel_id];

    uint32_t x = pixel_id % width;
    uint32_t y = pixel_id / width;

    float dx = (x*1.0) - center_x[label];
    float dy = (y*1.0) - center_y[label];

    float dist = sqrtf(dx * dx + dy * dy);

    keys[i] = (static_cast<uint64_t>(label) << 32) | static_cast<uint64_t>(rank);

    ContourSignalPoint p;
    p.distance = dist;
    p.x = x;
    p.y = y;
    p.label = label;
    p.rank = rank;

    points[i] = p;
}

__global__ void filter_boundaries( const uint32_t* boundaries_in, uint32_t* boundaries_out, uint32_t pixel_count,
    const float* center_x, const float* center_y){

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= pixel_count) {
        return;
    }
    uint32_t label = boundaries_in[i];

    boundaries_out[i] = (center_x[label]== -1.0f && center_y[label] == -1.0f) ? 0 : label;
    
}


thrust::device_vector<ContourSignalPoint> sort_distances(
    const uint32_t* d_boundaries,
    const uint32_t* d_ranks,
    const thrust::device_vector<Centroid>& d_centroids,
    uint32_t width,
    uint32_t height
) {
    uint32_t pixel_count = width * height;
    
    if (pixel_count == 0 || d_centroids.empty()) {
        return thrust::host_vector<ContourSignalPoint>();
    }
    
    // thrust::device_vector<uint32_t> d_boundaries(h_boundaries, h_boundaries + pixel_count);
    // thrust::device_vector<uint32_t> d_ranks(h_ranks, h_ranks + pixel_count);

    //thrust::device_ptr<uint32_t> d_labels = thrust::device_pointer_cast(d_boundaries);
    ///////thrust::device_ptr<uint32_t> d_labels = thrust::device_pointer_cast(d_ranks);
    
    
    // ---------------------------------------------------------------------
    //    Build sparse centroid lookup arrays:
    //    center_x[label], center_y[label]
    // ---------------------------------------------------------------------

    //thrust::device_vector<Centroid> d_centroids = h_centroids;
    uint32_t max_label = pixel_count;
    //uint32_t max_label = 0;
    // this can be made parallel or left out but that increases memory usage
    // for(const auto& c : d_centroids) {
    //     if(c.label > max_label) max_label = c.label;
    // }
    // if left out this needs to take pixel_count instead of max_label + 1
    thrust::device_vector<float> d_center_x(pixel_count, -1.0f);
    thrust::device_vector<float> d_center_y(pixel_count, -1.0f);

    uint32_t threads = 256;
    uint32_t centroid_blocks = (static_cast<uint32_t>(d_centroids.size()) + threads - 1) / threads;

    scatter_centroids_by_label<<<centroid_blocks, threads>>>(
        thrust::raw_pointer_cast(d_centroids.data()),
        static_cast<uint32_t>(d_centroids.size()),
        thrust::raw_pointer_cast(d_center_x.data()),
        thrust::raw_pointer_cast(d_center_y.data())
    );

    
    // ---------------------------------------------------------------------
    // Filter out small blobs and noise
    // ---------------------------------------------------------------------
    uint32_t filter_blocks = (pixel_count + threads - 1) / threads;
    uint32_t * d_boundaries_out;
    cudaMalloc(&d_boundaries_out, pixel_count * sizeof(uint32_t));

    filter_boundaries<<<filter_blocks, threads>>> (d_boundaries, 
        d_boundaries_out, pixel_count,
        thrust::raw_pointer_cast(d_center_x.data()),
        thrust::raw_pointer_cast(d_center_y.data()));
    

    // ---------------------------------------------------------------------
    // Compact boundary pixel IDs
    // ---------------------------------------------------------------------

    thrust::device_vector<uint32_t> d_all_ids(pixel_count);
    thrust::sequence(d_all_ids.begin(), d_all_ids.end());

    thrust::device_vector<uint32_t> d_boundary_ids(pixel_count);

    thrust::device_ptr<uint32_t> wrapped_out = thrust::device_pointer_cast(d_boundaries_out);
    auto boundary_end = thrust::copy_if(
        d_all_ids.begin(),
        d_all_ids.end(),
        wrapped_out,
        d_boundary_ids.begin(),
        IsBoundary()
    );  // copy_if returns an iterator pointing to the end of the copied sequence

    uint32_t num_boundary = static_cast<uint32_t>(boundary_end - d_boundary_ids.begin()); 

    d_boundary_ids.resize(num_boundary);

    if (num_boundary == 0) {
        return thrust::host_vector<ContourSignalPoint>();
    }


    // ---------------------------------------------------------------------
    //    Create packed sort keys and signal values
    //    key = (label << 32) | rank
    // ---------------------------------------------------------------------

    thrust::device_vector<uint64_t> d_keys(num_boundary);
    thrust::device_vector<ContourSignalPoint> d_points(num_boundary);

    uint32_t signal_blocks = (num_boundary + threads - 1) / threads;

    make_signal_keys_and_values<<<signal_blocks, threads>>>(
        thrust::raw_pointer_cast(d_boundary_ids.data()),
        d_boundaries_out,
        d_ranks,
        thrust::raw_pointer_cast(d_center_x.data()),
        thrust::raw_pointer_cast(d_center_y.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        thrust::raw_pointer_cast(d_points.data()),
        width,
        num_boundary
    );

    cudaFree(d_boundaries_out);

    thrust::sort_by_key(
        d_keys.begin(),
        d_keys.end(),
        d_points.begin()
    );

    //thrust::host_vector<ContourSignalPoint> h_points = d_points;
    
    return d_points;
}