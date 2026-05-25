#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

struct Centroid {
    float x;
    float y;
    uint32_t label;
};

struct CentroidSum {
    float x_sum;
    float y_sum;
    uint32_t count;

    __host__ __device__
    CentroidSum(float x, float y, uint32_t c) : x_sum(x), y_sum(y), count(c) {}

    __host__ __device__
    CentroidSum() : x_sum(0), y_sum(0), count(0) {}

    __host__ __device__
    CentroidSum operator+( const CentroidSum& other ) const {
        return CentroidSum(x_sum + other.x_sum, y_sum + other.y_sum, count + other.count);
    }

};

thrust::host_vector<Centroid> computeCentroids(const uint32_t *image, uint32_t width, uint32_t height);
