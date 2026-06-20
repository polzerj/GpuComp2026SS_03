#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <iostream>

#include "timer.hpp"

template <typename T>
struct device_buffer_t {
    T* ptr = nullptr;
    size_t capacity = 0;

    device_buffer_t() = default;

    device_buffer_t(const device_buffer_t&) = delete;
    device_buffer_t& operator=(const device_buffer_t&) = delete;

    ~device_buffer_t() {
        release();
    }

    void ensure(size_t required_count) {
        if (capacity >= required_count) {
            return;
        }

        release();

        cudaMalloc(&ptr, required_count * sizeof(T));
        capacity = required_count;
    }

    void release() {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }

        capacity = 0;
    }

    T* data() {
        return ptr;
    }

    const T* data() const {
        return ptr;
    }
};
