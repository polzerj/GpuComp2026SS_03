#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/system/cuda/execution_policy.h>

#include "centroid.cuh"
#include "timer.hpp"

#define THRESHOLD 200

struct count_at_least {
    uint32_t threshold;

    __host__ __device__
    bool operator()(uint32_t c) const {
        return c >= threshold;
    }
};

struct IsBoundaryPixelId {
    const uint32_t* labels;

    __host__ __device__
    bool operator()(uint32_t pixel_id) const {
        return labels[pixel_id] != 0u;
    }
};

__global__ void init_boundary_accumulators_kernel(
    const uint32_t* boundary_ids,
    const uint32_t* labels_in,
    uint32_t* labels_out,
    unsigned long long* sum_x,
    unsigned long long* sum_y,
    uint32_t* count,
    uint32_t width,
    uint32_t num_boundary)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boundary) {
        return;
    }

    uint32_t pixel_id = boundary_ids[i];
    labels_out[i] = labels_in[pixel_id];
    sum_x[i] = static_cast<unsigned long long>(pixel_id % width);
    sum_y[i] = static_cast<unsigned long long>(pixel_id / width);
    count[i] = 1u;
}

struct accum_tuple_sum {
    template <typename TupleA, typename TupleB>
    __host__ __device__
    thrust::tuple<unsigned long long, unsigned long long, uint32_t>
    operator()(const TupleA& a, const TupleB& b) const {
        return thrust::make_tuple(
            static_cast<unsigned long long>(thrust::get<0>(a)) +
                static_cast<unsigned long long>(thrust::get<0>(b)),
            static_cast<unsigned long long>(thrust::get<1>(a)) +
                static_cast<unsigned long long>(thrust::get<1>(b)),
            static_cast<uint32_t>(thrust::get<2>(a)) +
                static_cast<uint32_t>(thrust::get<2>(b))
        );
    }
};

__global__ void compact_reduced_centroids_kernel(
    const uint32_t* reduced_labels,
    const unsigned long long* reduced_sum_x,
    const unsigned long long* reduced_sum_y,
    const uint32_t* reduced_count,
    const uint32_t* valid_component_indices,
    Centroid* centroids,
    uint32_t num_valid)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_valid) {
        return;
    }

    uint32_t component_index = valid_component_indices[i];
    uint32_t c = reduced_count[component_index];

    Centroid out;
    out.label = reduced_labels[component_index];
    out.x = static_cast<float>(
        static_cast<double>(reduced_sum_x[component_index]) /
        static_cast<double>(c)
    );
    out.y = static_cast<float>(
        static_cast<double>(reduced_sum_y[component_index]) /
        static_cast<double>(c)
    );

    centroids[i] = out;
}

thrust::device_vector<Centroid> compute_centroids(
    const uint32_t* d_labels_in,
    uint32_t width,
    uint32_t height,
    pipeline_workspace_t& ws,
    cudaStream_t stream)
{
    const uint32_t num_pixels = width * height;
    auto policy = thrust::cuda::par.on(stream);

    if (num_pixels == 0) {
        return thrust::device_vector<Centroid>();
    }

    // ---------------------------------------------------------------------
    // Compact only the actually-used boundary pixels.
    //
    // The old implementation allocated arrays indexed by sparse pixel-id
    // labels and then memset/scanned almost the whole image-sized label range.
    // For 5100x7016 images that is tens of millions of entries per stream.
    // This version pays one full scan to compact boundary pixel IDs, then all
    // centroid work scales with num_boundary instead of width*height.
    // ---------------------------------------------------------------------
    DETAILED_TIMER_START_STREAM(compact_centroid_boundaries, stream);

    ws.d_sort_boundary_ids.ensure(num_pixels);
    uint32_t* d_boundary_ids = ws.d_sort_boundary_ids.data();
    thrust::device_ptr<uint32_t> boundary_ids_begin =
        thrust::device_pointer_cast(d_boundary_ids);

    auto ids_begin = thrust::make_counting_iterator<uint32_t>(0);
    auto ids_end = ids_begin + num_pixels;

    auto boundary_end = thrust::copy_if(
        policy,
        ids_begin,
        ids_end,
        boundary_ids_begin,
        IsBoundaryPixelId{d_labels_in}
    );

    const uint32_t num_boundary =
        static_cast<uint32_t>(boundary_end - boundary_ids_begin);

    DETAILED_TIMER_STOP(
        compact_centroid_boundaries,
        "Compact centroid boundary pixels"
    );

    if (num_boundary == 0) {
        return thrust::device_vector<Centroid>();
    }

    // ---------------------------------------------------------------------
    // Build one accumulator tuple per boundary pixel:
    // key   = label
    // value = (x, y, 1)
    // ---------------------------------------------------------------------
    DETAILED_TIMER_START_STREAM(init_centroid_accumulators, stream);

    ws.d_centroid_labels.ensure(num_boundary);
    ws.d_centroid_sum_x.ensure(num_boundary);
    ws.d_centroid_sum_y.ensure(num_boundary);
    ws.d_centroid_count.ensure(num_boundary);

    uint32_t* d_labels = ws.d_centroid_labels.data();
    unsigned long long* d_sum_x = ws.d_centroid_sum_x.data();
    unsigned long long* d_sum_y = ws.d_centroid_sum_y.data();
    uint32_t* d_count = ws.d_centroid_count.data();

    uint32_t threads = 256;
    uint32_t blocks = (num_boundary + threads - 1) / threads;

    init_boundary_accumulators_kernel<<<blocks, threads, 0, stream>>>(
        d_boundary_ids,
        d_labels_in,
        d_labels,
        d_sum_x,
        d_sum_y,
        d_count,
        width,
        num_boundary
    );

    DETAILED_TIMER_STOP(init_centroid_accumulators, "Init centroid accumulators");

    // ---------------------------------------------------------------------
    // Sort by label and reduce equal labels. This avoids atomicAdd hot-spots
    // and avoids huge sparse sum/count arrays indexed by pixel-id labels.
    // ---------------------------------------------------------------------
    DETAILED_TIMER_START_STREAM(sort_centroid_labels, stream);

    thrust::device_ptr<uint32_t> labels_begin =
        thrust::device_pointer_cast(d_labels);
    thrust::device_ptr<unsigned long long> sum_x_begin =
        thrust::device_pointer_cast(d_sum_x);
    thrust::device_ptr<unsigned long long> sum_y_begin =
        thrust::device_pointer_cast(d_sum_y);
    thrust::device_ptr<uint32_t> count_begin =
        thrust::device_pointer_cast(d_count);

    auto values_begin = thrust::make_zip_iterator(
        thrust::make_tuple(sum_x_begin, sum_y_begin, count_begin)
    );

    thrust::sort_by_key(
        policy,
        labels_begin,
        labels_begin + num_boundary,
        values_begin
    );

    DETAILED_TIMER_STOP(sort_centroid_labels, "Sort centroid labels");

    DETAILED_TIMER_START_STREAM(reduce_centroid_labels, stream);

    ws.d_centroid_reduced_labels.ensure(num_boundary);
    ws.d_centroid_reduced_sum_x.ensure(num_boundary);
    ws.d_centroid_reduced_sum_y.ensure(num_boundary);
    ws.d_centroid_reduced_count.ensure(num_boundary);

    uint32_t* d_reduced_labels = ws.d_centroid_reduced_labels.data();
    unsigned long long* d_reduced_sum_x = ws.d_centroid_reduced_sum_x.data();
    unsigned long long* d_reduced_sum_y = ws.d_centroid_reduced_sum_y.data();
    uint32_t* d_reduced_count = ws.d_centroid_reduced_count.data();

    thrust::device_ptr<uint32_t> reduced_labels_begin =
        thrust::device_pointer_cast(d_reduced_labels);
    thrust::device_ptr<unsigned long long> reduced_sum_x_begin =
        thrust::device_pointer_cast(d_reduced_sum_x);
    thrust::device_ptr<unsigned long long> reduced_sum_y_begin =
        thrust::device_pointer_cast(d_reduced_sum_y);
    thrust::device_ptr<uint32_t> reduced_count_begin =
        thrust::device_pointer_cast(d_reduced_count);

    auto reduced_values_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
            reduced_sum_x_begin,
            reduced_sum_y_begin,
            reduced_count_begin
        )
    );

    auto reduced_end = thrust::reduce_by_key(
        policy,
        labels_begin,
        labels_begin + num_boundary,
        values_begin,
        reduced_labels_begin,
        reduced_values_begin,
        thrust::equal_to<uint32_t>(),
        accum_tuple_sum{}
    );

    const uint32_t num_components =
        static_cast<uint32_t>(reduced_end.first - reduced_labels_begin);

    DETAILED_TIMER_STOP(reduce_centroid_labels, "Reduce centroid labels");

    if (num_components == 0) {
        return thrust::device_vector<Centroid>();
    }

    // ---------------------------------------------------------------------
    // Keep only real components and drop tiny noise blobs.
    // valid_component_indices stores indices into the reduced arrays, not labels.
    // ---------------------------------------------------------------------
    DETAILED_TIMER_START_STREAM(filter_valid_centroids, stream);

    ws.d_centroid_valid_labels.ensure(num_components);
    uint32_t* d_valid_component_indices = ws.d_centroid_valid_labels.data();
    thrust::device_ptr<uint32_t> valid_begin =
        thrust::device_pointer_cast(d_valid_component_indices);

    auto component_ids_begin = thrust::make_counting_iterator<uint32_t>(0);
    auto component_ids_end = component_ids_begin + num_components;

    auto valid_end = thrust::copy_if(
        policy,
        component_ids_begin,
        component_ids_end,
        reduced_count_begin,
        valid_begin,
        count_at_least{THRESHOLD}
    );

    const uint32_t num_valid =
        static_cast<uint32_t>(valid_end - valid_begin);

    DETAILED_TIMER_STOP(filter_valid_centroids, "Filter valid centroids");

    if (num_valid == 0) {
        return thrust::device_vector<Centroid>();
    }

    DETAILED_TIMER_START_STREAM(compact_centroids, stream);

    thrust::device_vector<Centroid> d_centroids(num_valid);
    uint32_t compact_blocks = (num_valid + threads - 1) / threads;

    compact_reduced_centroids_kernel<<<compact_blocks, threads, 0, stream>>>(
        d_reduced_labels,
        d_reduced_sum_x,
        d_reduced_sum_y,
        d_reduced_count,
        d_valid_component_indices,
        thrust::raw_pointer_cast(d_centroids.data()),
        num_valid
    );

    DETAILED_TIMER_STOP(compact_centroids, "Compact centroids");

    return d_centroids;
}

thrust::device_vector<Centroid> compute_centroids(
    const uint32_t* d_labels_in,
    uint32_t width,
    uint32_t height,
    pipeline_workspace_t& ws)
{
    return compute_centroids(d_labels_in, width, height, ws, 0);
}
