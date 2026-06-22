#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>

// ------------------------------------------------------------
// Enable timing from compiler flags, for example:
// nvcc ... -DENABLE_TIMING
// nvcc ... -DENABLE_DETAILED_TIMING
// ------------------------------------------------------------

#ifdef ENABLE_DETAILED_TIMING
#ifndef ENABLE_TIMING
#define ENABLE_TIMING
#endif
#endif

// ------------------------------------------------------------
// CUDA error checking
// ------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr,                                                \
                         "CUDA error at %s:%d: %s\n",                           \
                         __FILE__,                                              \
                         __LINE__,                                              \
                         cudaGetErrorString(err__));                            \
        }                                                                       \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                     \
    do {                                                                        \
        cudaError_t err__ = cudaGetLastError();                                 \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr,                                                \
                         "CUDA kernel launch error at %s:%d: %s\n",             \
                         __FILE__,                                              \
                         __LINE__,                                              \
                         cudaGetErrorString(err__));                            \
        }                                                                       \
    } while (0)


// ------------------------------------------------------------
// CPU wall-clock timer
// Use this for:
// total runtime, image runtime, classification, validation,
// visualization, file loading/saving, CPU code.
// ------------------------------------------------------------

struct cpu_timer_t {
    std::chrono::high_resolution_clock::time_point start;

    cpu_timer_t() {
        reset();
    }

    void reset() {
        start = std::chrono::high_resolution_clock::now();
    }

    float elapsed_ms() const {
        auto stop = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::milli>(stop - start).count();
    }
};


// ------------------------------------------------------------
// GPU CUDA-event timer
// Use this for:
// kernel-heavy CUDA stages.
// This measures GPU timeline time, not CPU code time.
// ------------------------------------------------------------

struct gpu_timer_t {
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    cudaStream_t stream = 0;

    explicit gpu_timer_t(cudaStream_t stream_ = 0)
        : stream(stream_)
    {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        CUDA_CHECK(cudaEventRecord(start_event, stream));
    }

    gpu_timer_t(const gpu_timer_t&) = delete;
    gpu_timer_t& operator=(const gpu_timer_t&) = delete;

    ~gpu_timer_t() {
        if (start_event) {
            cudaEventDestroy(start_event);
        }

        if (stop_event) {
            cudaEventDestroy(stop_event);
        }
    }

    float elapsed_ms() {
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));

        return ms;
    }
};


// ------------------------------------------------------------
// Timing macros
// ------------------------------------------------------------

#ifdef ENABLE_TIMING

#define CPU_TIMER_START(name)                                                   \
    cpu_timer_t cpu_timer_##name

#define CPU_TIMER_STOP(name, label)                                             \
    do {                                                                        \
        float ms__ = cpu_timer_##name.elapsed_ms();                             \
        std::printf("[cpu] %-40s %10.3f ms\n", label, ms__);                    \
    } while (0)

#define GPU_TIMER_START(name)                                                   \
    gpu_timer_t gpu_timer_##name

#define GPU_TIMER_START_STREAM(name, stream)                                    \
    gpu_timer_t gpu_timer_##name(stream)

#define GPU_TIMER_STOP(name, label)                                             \
    do {                                                                        \
        float ms__ = gpu_timer_##name.elapsed_ms();                             \
        std::printf("[gpu] %-40s %10.3f ms\n", label, ms__);                    \
    } while (0)

#else

#define CPU_TIMER_START(name)
#define CPU_TIMER_STOP(name, label)

#define GPU_TIMER_START(name)
#define GPU_TIMER_START_STREAM(name, stream)
#define GPU_TIMER_STOP(name, label)

#endif


// ------------------------------------------------------------
// Detailed timing macros
// Only active with -DENABLE_DETAILED_TIMING
// ------------------------------------------------------------

#ifdef ENABLE_DETAILED_TIMING

#define DETAILED_GPU_TIMER_START(name)                                          \
    gpu_timer_t detailed_gpu_timer_##name

#define DETAILED_GPU_TIMER_START_STREAM(name, stream)                           \
    gpu_timer_t detailed_gpu_timer_##name(stream)

#define DETAILED_GPU_TIMER_STOP(name, label)                                    \
    do {                                                                        \
        float ms__ = detailed_gpu_timer_##name.elapsed_ms();                    \
        std::printf("    [gpu detail] %-32s %10.3f ms\n", label, ms__);         \
    } while (0)

#define DETAILED_CPU_TIMER_START(name)                                          \
    cpu_timer_t detailed_cpu_timer_##name

#define DETAILED_CPU_TIMER_STOP(name, label)                                    \
    do {                                                                        \
        float ms__ = detailed_cpu_timer_##name.elapsed_ms();                    \
        std::printf("    [cpu detail] %-32s %10.3f ms\n", label, ms__);         \
    } while (0)

#else

#define DETAILED_GPU_TIMER_START(name)
#define DETAILED_GPU_TIMER_START_STREAM(name, stream)
#define DETAILED_GPU_TIMER_STOP(name, label)

#define DETAILED_CPU_TIMER_START(name)
#define DETAILED_CPU_TIMER_STOP(name, label)

#endif


// ------------------------------------------------------------
// Compatibility aliases for older names
// ------------------------------------------------------------

// Use GLOBAL_TIMER for total/image runtime.
// This is CPU wall-clock time.
#define GLOBAL_TIMER_START(name)                                                \
    CPU_TIMER_START(name)

#define GLOBAL_TIMER_STOP(name, label)                                          \
    CPU_TIMER_STOP(name, label)


// Use DEBUG_TIMER for GPU stages.
// For CPU stages, use CPU_TIMER directly.
#define DEBUG_TIMER_START(name)                                                 \
    GPU_TIMER_START(name)

#define DEBUG_TIMER_START_STREAM(name, stream)                                  \
    GPU_TIMER_START_STREAM(name, stream)

#define DEBUG_TIMER_STOP(name, label)                                           \
    GPU_TIMER_STOP(name, label)


// Detailed GPU timing aliases.
#define DEBUG_DETAILED_TIMER_START(name)                                        \
    DETAILED_GPU_TIMER_START(name)

#define DEBUG_DETAILED_TIMER_START_STREAM(name, stream)                         \
    DETAILED_GPU_TIMER_START_STREAM(name, stream)

#define DEBUG_DETAILED_TIMER_STOP(name, label)                                  \
    DETAILED_GPU_TIMER_STOP(name, label)

#define DETAILED_TIMER_START(name)                                              \
    DETAILED_GPU_TIMER_START(name)

#define DETAILED_TIMER_START_STREAM(name, stream)                               \
    DETAILED_GPU_TIMER_START_STREAM(name, stream)

#define DETAILED_TIMER_STOP(name, label)                                        \
    DETAILED_GPU_TIMER_STOP(name, label)

