
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include "centroid.cuh"


thrust::host_vector<Centroid> computeCentroids(const uint32_t *image, uint32_t width, uint32_t height){
    auto numPixels = width * height;

    thrust::host_vector<uint32_t> h_image(image, image + numPixels);
    thrust::device_vector<uint32_t> d_image = h_image;

    thrust::device_vector<uint32_t> d_x(numPixels);
    thrust::device_vector<uint32_t> d_y(numPixels);


    thrust::tabulate(d_x.begin(), d_x.end(), [width] __device__ (size_t i) {
        return i % width;
    });
    thrust::tabulate(d_y.begin(), d_y.end(), [width] __device__ (size_t i) {
        return i / width;
    });

    thrust::device_vector<CentroidSum> d_centroidSums(numPixels);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_image.begin(), d_x.begin(), d_y.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_image.end(), d_x.end(), d_y.end())),
        d_centroidSums.begin(),
        [] __device__ (thrust::tuple<uint32_t, uint32_t, uint32_t> t) {
            uint32_t label = thrust::get<0>(t);
            uint32_t x = thrust::get<1>(t);
            uint32_t y = thrust::get<2>(t);
            return CentroidSum(label == 0 ? 0 : x, label == 0 ? 0 : y, label == 0 ? 0 : 1);
        }
    );

    // sort by label

    thrust::sort_by_key(d_image.begin(), d_image.end(), d_centroidSums.begin());

    // reduce by label
    thrust::device_vector<CentroidSum> d_reducedCentroidSums(numPixels);
    thrust::device_vector<uint32_t> d_labels(numPixels);
    auto new_end = thrust::reduce_by_key(
        d_image.begin(), d_image.end(),
        d_centroidSums.begin(),
        d_labels.begin(),
        d_reducedCentroidSums.begin()
    );

    size_t numCentroids = new_end.first - d_labels.begin();
    thrust::device_vector<Centroid> d_centroids(numCentroids);

    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_labels.begin(), d_reducedCentroidSums.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_labels.begin() + numCentroids, d_reducedCentroidSums.begin() + numCentroids)),
        d_centroids.begin(),
        [] __device__ (thrust::tuple<uint32_t, CentroidSum> t) {
            uint32_t label = thrust::get<0>(t);
            CentroidSum sum = thrust::get<1>(t);
            return Centroid{sum.x_sum / sum.count, sum.y_sum / sum.count, label};
        }
    );

    thrust::host_vector<Centroid> h_centroids = d_centroids;
    return h_centroids;



}
