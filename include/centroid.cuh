#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include "pipeline_workspace.hpp"

struct Centroid {
    float x;
    float y;
    uint32_t label;
};
