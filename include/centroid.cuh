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

thrust::device_vector<Centroid> compute_centroids(const uint32_t *d_labels_in, uint32_t width, uint32_t height);
