#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define DETAILED_TIMING
// #define TIMING

#ifdef DETAILED_TIMING
#define DEBUG_TIMING
#endif

#ifdef DEBUG_TIMING
#define TIMING
#endif

#if defined(TIMING) || defined(DEBUG_TIMING)
struct stage_timer_t
{
    cudaEvent_t start, stop;
    stage_timer_t()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start); // insert start marker in default stream
    }
    float elapsed_ms()
    {
        cudaEventRecord(stop);      // insert stop marker
        cudaEventSynchronize(stop); // wait for GPU to reach stop
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return ms;
    }
};
#endif

#ifdef TIMING
#define GLOBAL_TIMER_START(name) stage_timer_t _global_timer_##name
#define GLOBAL_TIMER_STOP(name, label) printf("[timing] %-42s %8.3f ms\n", label, _global_timer_##name.elapsed_ms())
#else
#define GLOBAL_TIMER_START(name)
#define GLOBAL_TIMER_STOP(name, label)
#endif // TIMING

#ifdef DEBUG_TIMING
#define DEBUG_TIMER_START(name) stage_timer_t _debug_timer_##name
#define DEBUG_TIMER_STOP(name, label) printf("[timing] %-42s %8.3f ms\n\n", label, _debug_timer_##name.elapsed_ms())
#else
#define DEBUG_TIMER_START(name)
#define DEBUG_TIMER_STOP(name, label)
#endif // DEBUG_TIMING

#ifdef DETAILED_TIMING
#define DETAILED_TIMER_START(name) stage_timer_t _detailed_timer_##name
#define DETAILED_TIMER_STOP(name, label) printf("[timing]   %-40s %8.3f ms\n", label, _detailed_timer_##name.elapsed_ms())
#else
#define DETAILED_TIMER_START(name)
#define DETAILED_TIMER_STOP(name, label)
#endif // DETAILED_TIMING
