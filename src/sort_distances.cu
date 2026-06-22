#include "sort_distances.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/tuple.h>

#include "timer.hpp"

// ---------------------------------
#define CENTROID_THRESHOLD 200u

struct IsBoundaryPixelId
{
    const uint32_t *boundaries;
    __host__ __device__ bool operator()(uint32_t pixel_id) const
    {
        return boundaries[pixel_id] != 0u;
    }
};

struct PointToLabel
{
    __host__ __device__
        uint32_t
        operator()(const ContourSignalPoint &p) const
    {
        return p.label;
    }
};

struct PointToAccumTuple
{
    __host__ __device__
        thrust::tuple<unsigned long long, unsigned long long, uint32_t>
        operator()(const ContourSignalPoint &p) const
    {
        return thrust::make_tuple(
            static_cast<unsigned long long>(p.x),
            static_cast<unsigned long long>(p.y),
            1u);
    }
};

struct AccumTupleSum
{
    template <typename TupleA, typename TupleB>
    __host__ __device__
        thrust::tuple<unsigned long long, unsigned long long, uint32_t>
        operator()(const TupleA &a, const TupleB &b) const
    {
        return thrust::make_tuple(
            static_cast<unsigned long long>(thrust::get<0>(a)) +
                static_cast<unsigned long long>(thrust::get<0>(b)),
            static_cast<unsigned long long>(thrust::get<1>(a)) +
                static_cast<unsigned long long>(thrust::get<1>(b)),
            static_cast<uint32_t>(thrust::get<2>(a)) +
                static_cast<uint32_t>(thrust::get<2>(b)));
    }
};

struct CountAtLeast
{
    uint32_t threshold;
    __host__ __device__ bool operator()(uint32_t c) const
    {
        return c >= threshold;
    }
};

struct PointLabelIsZero {
    __host__ __device__
    bool operator()(const ContourSignalPoint& p) const {
        return p.label == 0u;
    }
};

__global__ void make_signal_keys_and_points_kernel(
    const uint32_t *boundary_ids,
    const uint32_t *boundaries,
    const uint32_t *ranks,
    uint64_t *keys,
    ContourSignalPoint *points,
    uint32_t width,
    uint32_t num_boundary)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_boundary)
        return;

    uint32_t pixel_id = boundary_ids[i];
    uint32_t label = boundaries[pixel_id];
    uint32_t rank = ranks[pixel_id];
    uint32_t x = pixel_id % width;
    uint32_t y = pixel_id / width;

    keys[i] = (static_cast<uint64_t>(label) << 32) | static_cast<uint64_t>(rank);

    ContourSignalPoint p;
    p.distance = 0.0f; // Will be filled later
    p.x = x;
    p.y = y;
    p.label = label;
    p.rank = rank;
    points[i] = p;
}

__global__ void compact_reduced_centroids_kernel(
    const uint32_t *reduced_labels,
    const unsigned long long *reduced_sum_x,
    const unsigned long long *reduced_sum_y,
    const uint32_t *reduced_count,
    const uint32_t *valid_component_indices,
    Centroid *centroids,
    uint32_t num_valid)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_valid)
        return;

    uint32_t component_index = valid_component_indices[i];
    uint32_t c = reduced_count[component_index];

    Centroid out;
    out.label = reduced_labels[component_index];
    out.x = static_cast<float>(static_cast<double>(reduced_sum_x[component_index]) / static_cast<double>(c));
    out.y = static_cast<float>(static_cast<double>(reduced_sum_y[component_index]) / static_cast<double>(c));
    centroids[i] = out;
}

__device__ int find_centroid_by_label(
    const Centroid *centroids,
    uint32_t num_centroids,
    uint32_t label)
{
    uint32_t lo = 0;
    uint32_t hi = num_centroids;

    while (lo < hi)
    {
        uint32_t mid = lo + ((hi - lo) >> 1);
        uint32_t mid_label = centroids[mid].label;
        if (mid_label < label)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }

    if (lo < num_centroids && centroids[lo].label == label)
    {
        return static_cast<int>(lo);
    }
    return -1;
}

__global__ void apply_centroids_to_points_kernel(
    ContourSignalPoint *points,
    uint32_t num_points,
    const Centroid *centroids,
    uint32_t num_centroids)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;

    ContourSignalPoint p = points[i];
    int centroid_index = find_centroid_by_label(centroids, num_centroids, p.label);

    if (centroid_index < 0)
    {
        p.label = 0u; // Mark for removal
        p.distance = 0.0f;
        points[i] = p;
        return;
    }

    Centroid c = centroids[centroid_index];
    float dx = static_cast<float>(p.x) - c.x;
    float dy = static_cast<float>(p.y) - c.y;
    p.distance = sqrtf(dx * dx + dy * dy);
    points[i] = p;
}

// ---------------------------------

struct IsBoundary
{
    __device__ bool operator()(uint32_t label) const
    {
        return label != 0;
    }
};

struct IsValidBoundaryId {
    const uint32_t* boundaries;
    const float* center_x;
    const float* center_y;
    uint32_t center_capacity;

    __host__ __device__
    bool operator()(uint32_t pixel_id) const {
        uint32_t label = boundaries[pixel_id];

        if (label == 0) {
            return false;
        }

        if (label >= center_capacity) {
            return false;
        }

        // Only keep labels for which a valid centroid exists.
        return center_x[label] >= 0.0f && center_y[label] >= 0.0f;
    }
};

__global__ void scatter_centroids_by_label(
    const Centroid *centroids,
    uint32_t num_centroids,
    float *center_x,
    float *center_y)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_centroids)
    {
        return;
    }

    uint32_t label = centroids[i].label;
    center_x[label] = centroids[i].x;
    center_y[label] = centroids[i].y;
}

__global__ void make_signal_keys_and_values(
    const uint32_t *boundary_ids,
    const uint32_t *boundaries,
    const uint32_t *ranks,
    const float *center_x,
    const float *center_y,
    uint64_t *keys,
    ContourSignalPoint *points,
    uint32_t width,
    uint32_t num_boundary)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_boundary)
    {
        return;
    }

    uint32_t pixel_id = boundary_ids[i];

    uint32_t label = boundaries[pixel_id];
    uint32_t rank = ranks[pixel_id];

    uint32_t x = pixel_id % width;
    uint32_t y = pixel_id / width;

    float dx = (x * 1.0) - center_x[label];
    float dy = (y * 1.0) - center_y[label];

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

__global__ void filter_boundaries(const uint32_t *boundaries_in, uint32_t *boundaries_out, uint32_t pixel_count,
                                  const float *center_x, const float *center_y)
{

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= pixel_count)
    {
        return;
    }
    uint32_t label = boundaries_in[i];

    boundaries_out[i] = (center_x[label] == -1.0f && center_y[label] == -1.0f) ? 0 : label;
}



ContourSignalBuildResult build_sorted_contour_signal(
    const uint32_t* d_boundaries,
    const uint32_t* d_ranks,
    uint32_t width,
    uint32_t height,
    pipeline_workspace_t& ws,
    cudaStream_t stream)
{
    ContourSignalBuildResult result;
    const uint32_t pixel_count = width * height;
    auto policy = thrust::cuda::par.on(stream);

    if (pixel_count == 0) {
        return result;
    }

    const uint32_t threads = 256;

    // --- Compact boundary pixel IDs ---
    DETAILED_TIMER_START_STREAM(compact_boundaries, stream);
    ws.d_sort_boundary_ids.ensure(pixel_count);
    uint32_t* d_boundary_ids = ws.d_sort_boundary_ids.data();

    thrust::device_ptr<uint32_t> boundary_ids_begin = thrust::device_pointer_cast(d_boundary_ids);
    auto ids_begin = thrust::make_counting_iterator<uint32_t>(0);
    auto ids_end = ids_begin + pixel_count;

    auto boundary_end = thrust::copy_if(
        policy,
        ids_begin,
        ids_end,
        boundary_ids_begin,
        IsBoundaryPixelId{d_boundaries}
    );

    const uint32_t num_boundary = static_cast<uint32_t>(boundary_end - boundary_ids_begin);
    DETAILED_TIMER_STOP(compact_boundaries, "Compact boundary pixel IDs");

    if (num_boundary == 0) {
        return result;
    }

    // --- Create and sort signal points ---
    DETAILED_TIMER_START_STREAM(make_and_sort_signal_points, stream);
    result.points.resize(num_boundary);
    ws.d_sort_keys.ensure(num_boundary);
    uint64_t* d_keys = ws.d_sort_keys.data();

    uint32_t signal_blocks = (num_boundary + threads - 1) / threads;
    make_signal_keys_and_points_kernel<<<signal_blocks, threads, 0, stream>>>(
        d_boundary_ids,
        d_boundaries,
        d_ranks,
        d_keys,
        thrust::raw_pointer_cast(result.points.data()),
        width,
        num_boundary
    );
    CUDA_KERNEL_CHECK();

    thrust::device_ptr<uint64_t> keys_begin = thrust::device_pointer_cast(d_keys);
    thrust::sort_by_key(
        policy,
        keys_begin,
        keys_begin + num_boundary,
        result.points.begin()
    );
    DETAILED_TIMER_STOP(make_and_sort_signal_points, "Make and sort signal points");

    // --- Compute centroids from sorted points ---
    DETAILED_TIMER_START_STREAM(reduce_centroids_from_points, stream);
    ws.d_centroid_reduced_labels.ensure(num_boundary);
    ws.d_centroid_reduced_sum_x.ensure(num_boundary);
    ws.d_centroid_reduced_sum_y.ensure(num_boundary);
    ws.d_centroid_reduced_count.ensure(num_boundary);

    uint32_t* d_reduced_labels = ws.d_centroid_reduced_labels.data();
    unsigned long long* d_reduced_sum_x = ws.d_centroid_reduced_sum_x.data();
    unsigned long long* d_reduced_sum_y = ws.d_centroid_reduced_sum_y.data();
    uint32_t* d_reduced_count = ws.d_centroid_reduced_count.data();

    thrust::device_ptr<uint32_t> reduced_labels_begin = thrust::device_pointer_cast(d_reduced_labels);
    thrust::device_ptr<unsigned long long> reduced_sum_x_begin = thrust::device_pointer_cast(d_reduced_sum_x);
    thrust::device_ptr<unsigned long long> reduced_sum_y_begin = thrust::device_pointer_cast(d_reduced_sum_y);
    thrust::device_ptr<uint32_t> reduced_count_begin = thrust::device_pointer_cast(d_reduced_count);

    auto point_labels_begin = thrust::make_transform_iterator(result.points.begin(), PointToLabel{});
    auto point_values_begin = thrust::make_transform_iterator(result.points.begin(), PointToAccumTuple{});
    auto reduced_values_begin = thrust::make_zip_iterator(
        thrust::make_tuple(reduced_sum_x_begin, reduced_sum_y_begin, reduced_count_begin)
    );

    auto reduced_end = thrust::reduce_by_key(
        policy,
        point_labels_begin,
        point_labels_begin + num_boundary,
        point_values_begin,
        reduced_labels_begin,
        reduced_values_begin,
        thrust::equal_to<uint32_t>(),
        AccumTupleSum{}
    );

    const uint32_t num_components = static_cast<uint32_t>(reduced_end.first - reduced_labels_begin);
    DETAILED_TIMER_STOP(reduce_centroids_from_points, "Reduce centroids from sorted points");

    if (num_components == 0) {
        result.points.clear();
        return result;
    }

    // --- Filter out small components (noise) ---
    DETAILED_TIMER_START_STREAM(filter_valid_centroids, stream);
    ws.d_centroid_valid_labels.ensure(num_components);
    uint32_t* d_valid_component_indices = ws.d_centroid_valid_labels.data();
    thrust::device_ptr<uint32_t> valid_begin = thrust::device_pointer_cast(d_valid_component_indices);

    auto component_ids_begin = thrust::make_counting_iterator<uint32_t>(0);
    auto component_ids_end = component_ids_begin + num_components;

    auto valid_end = thrust::copy_if(
        policy,
        component_ids_begin,
        component_ids_end,
        reduced_count_begin,
        valid_begin,
        CountAtLeast{CENTROID_THRESHOLD}
    );

    const uint32_t num_valid = static_cast<uint32_t>(valid_end - valid_begin);
    DETAILED_TIMER_STOP(filter_valid_centroids, "Filter valid centroids");

    if (num_valid == 0) {
        result.points.clear();
        return result;
    }

    // --- Compact valid centroids ---
    DETAILED_TIMER_START_STREAM(compact_centroids, stream);
    result.centroids.resize(num_valid);
    uint32_t centroid_blocks = (num_valid + threads - 1) / threads;

    compact_reduced_centroids_kernel<<<centroid_blocks, threads, 0, stream>>>(
        d_reduced_labels,
        d_reduced_sum_x,
        d_reduced_sum_y,
        d_reduced_count,
        d_valid_component_indices,
        thrust::raw_pointer_cast(result.centroids.data()),
        num_valid
    );
    CUDA_KERNEL_CHECK();
    DETAILED_TIMER_STOP(compact_centroids, "Compact centroids");

    // --- Apply centroids to points and remove noise ---
    DETAILED_TIMER_START_STREAM(apply_centroids_to_points, stream);
    uint32_t blocks = (num_boundary + threads - 1) / threads;
    apply_centroids_to_points_kernel<<<blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(result.points.data()),
        num_boundary,
        thrust::raw_pointer_cast(result.centroids.data()),
        num_valid
    );
    CUDA_KERNEL_CHECK();

    auto filtered_end = thrust::remove_if(
        policy,
        result.points.begin(),
        result.points.end(),
        PointLabelIsZero{}
    );
    result.points.erase(filtered_end, result.points.end());
    DETAILED_TIMER_STOP(apply_centroids_to_points, "Apply centroids to points");

    return result;
}

thrust::device_vector<ContourSignalPoint> sort_distances(
    const uint32_t *d_boundaries,
    const uint32_t *d_ranks,
    const thrust::device_vector<Centroid> &d_centroids,
    uint32_t width,
    uint32_t height,
    pipeline_workspace_t& ws)
{
    cudaStream_t stream = ws.stream;
    auto policy = thrust::cuda::par.on(stream);

    uint32_t pixel_count = width * height;

    if (pixel_count == 0 || d_centroids.empty())
    {
        return thrust::device_vector<ContourSignalPoint>();
    }

    // thrust::device_vector<uint32_t> d_boundaries(h_boundaries, h_boundaries + pixel_count);
    // thrust::device_vector<uint32_t> d_ranks(h_ranks, h_ranks + pixel_count);

    // thrust::device_ptr<uint32_t> d_labels = thrust::device_pointer_cast(d_boundaries);
    ///////thrust::device_ptr<uint32_t> d_labels = thrust::device_pointer_cast(d_ranks);

    // ---------------------------------------------------------------------
    //    Build sparse centroid lookup arrays:
    //    center_x[label], center_y[label]
    // ---------------------------------------------------------------------

    // thrust::device_vector<Centroid> d_centroids = h_centroids;

    DETAILED_TIMER_START(scatter_centroids);
    uint32_t max_label = pixel_count;
    // uint32_t max_label = 0;
    //  this can be made parallel or left out but that increases memory usage
    //  for(const auto& c : d_centroids) {
    //      if(c.label > max_label) max_label = c.label;
    //  }
    //  if left out this needs to take pixel_count instead of max_label + 1
    // thrust::device_vector<float> d_center_x(pixel_count, -1.0f);
    // thrust::device_vector<float> d_center_y(pixel_count, -1.0f);
    ws.d_sort_center_x.ensure(pixel_count);
    ws.d_sort_center_y.ensure(pixel_count);

    float* d_center_x = ws.d_sort_center_x.data();
    float* d_center_y = ws.d_sort_center_y.data();

    thrust::device_ptr<float> center_x_begin =
        thrust::device_pointer_cast(d_center_x);

    thrust::device_ptr<float> center_y_begin =
        thrust::device_pointer_cast(d_center_y);

    thrust::fill(policy, center_x_begin, center_x_begin + pixel_count, -1.0f);
    thrust::fill(policy, center_y_begin, center_y_begin + pixel_count, -1.0f);

    uint32_t threads = 256;
    uint32_t centroid_blocks = (static_cast<uint32_t>(d_centroids.size()) + threads - 1) / threads;

    scatter_centroids_by_label<<<centroid_blocks, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_centroids.data()),
        static_cast<uint32_t>(d_centroids.size()),
        d_center_x,
        d_center_y
    );

    DETAILED_TIMER_STOP(scatter_centroids, "Scatter centroids by label");

    // ---------------------------------------------------------------------
    // Filter out small blobs and noise
    // ---------------------------------------------------------------------
    // DETAILED_TIMER_START(filter_boundaries);
    // uint32_t filter_blocks = (pixel_count + threads - 1) / threads;
    // uint32_t *d_boundaries_out;
    // cudaMalloc(&d_boundaries_out, pixel_count * sizeof(uint32_t));

    // filter_boundaries<<<filter_blocks, threads>>>(d_boundaries,
    //                                               d_boundaries_out, pixel_count,
    //                                               thrust::raw_pointer_cast(d_center_x.data()),
    //                                               thrust::raw_pointer_cast(d_center_y.data()));

    // cudaDeviceSynchronize();
    // DETAILED_TIMER_STOP(filter_boundaries, "Filter boundaries");

    // // ---------------------------------------------------------------------
    // // Compact boundary pixel IDs
    // // ---------------------------------------------------------------------

    DETAILED_TIMER_START(compact_boundaries);
    // thrust::device_vector<uint32_t> d_all_ids(pixel_count);
    // thrust::sequence(d_all_ids.begin(), d_all_ids.end());

    // thrust::device_vector<uint32_t> d_boundary_ids(pixel_count);

    // thrust::device_ptr<uint32_t> wrapped_out = thrust::device_pointer_cast(d_boundaries_out);
    // auto boundary_end = thrust::copy_if(
    //     d_all_ids.begin(),
    //     d_all_ids.end(),
    //     wrapped_out,
    //     d_boundary_ids.begin(),
    //     IsBoundary()); // copy_if returns an iterator pointing to the end of the copied sequence

    auto ids_begin = thrust::make_counting_iterator<uint32_t>(0);
    auto ids_end   = ids_begin + pixel_count;

    // thrust::device_vector<uint32_t> d_boundary_ids(pixel_count);

    // auto boundary_end = thrust::copy_if(
    //     ids_begin,
    //     ids_end,
    //     d_boundary_ids.begin(),
    //     IsValidBoundaryId{
    //         d_boundaries,
    //         thrust::raw_pointer_cast(d_center_x.data()),
    //         thrust::raw_pointer_cast(d_center_y.data()),
    //         static_cast<uint32_t>(d_center_x.size())
    //     }
    // );

    // uint32_t num_boundary = static_cast<uint32_t>(boundary_end - d_boundary_ids.begin());

    // d_boundary_ids.resize(num_boundary);
    ws.d_sort_boundary_ids.ensure(pixel_count);

    uint32_t* d_boundary_ids = ws.d_sort_boundary_ids.data();

    thrust::device_ptr<uint32_t> boundary_ids_begin =
        thrust::device_pointer_cast(d_boundary_ids);

    auto boundary_end = thrust::copy_if(
        policy,
        ids_begin,
        ids_end,
        boundary_ids_begin,
        IsValidBoundaryId{
            d_boundaries,
            d_center_x,
            d_center_y,
            pixel_count
        }
    );

    uint32_t num_boundary = static_cast<uint32_t>(boundary_end - boundary_ids_begin);

    DETAILED_TIMER_STOP(compact_boundaries, "Compact boundary pixel IDs");

    if (num_boundary == 0)
    {
        return thrust::device_vector<ContourSignalPoint>();
    }

    // ---------------------------------------------------------------------
    //    Create packed sort keys and signal values
    //    key = (label << 32) | rank
    // ---------------------------------------------------------------------

    DETAILED_TIMER_START(sort_keys_and_values);
    // thrust::device_vector<uint64_t> d_keys(num_boundary);
    // thrust::device_vector<ContourSignalPoint> d_points(num_boundary);
    ws.d_sort_keys.ensure(num_boundary);

    uint64_t* d_keys = ws.d_sort_keys.data();

    thrust::device_ptr<uint64_t> keys_begin =
        thrust::device_pointer_cast(d_keys);

    thrust::device_vector<ContourSignalPoint> d_points(num_boundary);

    uint32_t signal_blocks = (num_boundary + threads - 1) / threads;

    // make_signal_keys_and_values<<<signal_blocks, threads>>>(
    //     thrust::raw_pointer_cast(d_boundary_ids.data()),
    //     d_boundaries,
    //     d_ranks,
    //     thrust::raw_pointer_cast(d_center_x.data()),
    //     thrust::raw_pointer_cast(d_center_y.data()),
    //     thrust::raw_pointer_cast(d_keys.data()),
    //     thrust::raw_pointer_cast(d_points.data()),
    //     width,
    //     num_boundary);
    make_signal_keys_and_values<<<signal_blocks, threads, 0, stream>>>(
        d_boundary_ids,
        d_boundaries,
        d_ranks,
        d_center_x,
        d_center_y,
        d_keys,
        thrust::raw_pointer_cast(d_points.data()),
        width,
        num_boundary
    );


    // thrust::sort_by_key(
    //     d_keys.begin(),
    //     d_keys.end(),
    //     d_points.begin());
    thrust::sort_by_key(
        policy,
        keys_begin,
        keys_begin + num_boundary,
        d_points.begin()
    );

    // thrust::host_vector<ContourSignalPoint> h_points = d_points;

    DETAILED_TIMER_STOP(sort_keys_and_values, "Sort keys and values");
    return d_points;
}
