#ifndef PTRJUMP
#define PTRJUMP

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <cuda/std/utility>
#include "types.hpp"
#include "timer.hpp"
#include "pointerjumping.cuh"
#include <cuda_device_runtime_api.h>

#define BLOCK_WIDTH_X 32
#define BLOCK_WIDTH_Y 32
#define SHARED_B_X (BLOCK_WIDTH_X + 2)
#define SHARED_B_Y (BLOCK_WIDTH_Y + 2)
#define SHARED_BLOCK_SIZE (SHARED_B_X * SHARED_B_Y)
#define SHARED_ELEM_CNT (BLOCK_WIDTH_X * BLOCK_WIDTH_Y)
#define NIL UINT32_MAX

__global__ void init_pointer_list_and_extract_boundaries(uint32_t *image_data, uint32_t image_width, uint32_t image_height, ptr_jmp_out_t data)
{
    // we need to make a list where within each piece boundary each node points to its successor
    uint32_t pixel_id_x = ((blockIdx.x * blockDim.x) + threadIdx.x);
    uint32_t pixel_id_y = ((blockIdx.y * blockDim.y) + threadIdx.y);

    uint32_t unique_thread_id = pixel_id_x + pixel_id_y * image_width;

    uint32_t block_local_x = threadIdx.x + 1;
    uint32_t block_local_y = threadIdx.y + 1;

    // load data into shared memory
    __shared__ uint32_t shared_block[BLOCK_WIDTH_Y + 2][BLOCK_WIDTH_X + 2]; // we need to load more than just the tile to be able to process all neighbours

    for (int i = (threadIdx.x + threadIdx.y * blockDim.x); i < SHARED_BLOCK_SIZE; i += SHARED_ELEM_CNT)
    {
        int s_y = i / SHARED_B_X;
        int s_x = i % SHARED_B_X;

        // Offset by -1 because shared (1,1) is global (block_start_x, block_start_y)
        int g_x = (int)(blockIdx.x * blockDim.x) + s_x - 1;
        int g_y = (int)(blockIdx.y * blockDim.y) + s_y - 1;

        if (g_x >= 0 && g_x < image_width && g_y >= 0 && g_y < image_height)
        {
            shared_block[s_y][s_x] = image_data[g_y * image_width + g_x];
        }
        else
        {
            shared_block[s_y][s_x] = 0; // Boundary condition: black/empty
        }
    }
    __syncthreads();

    if (pixel_id_x >= image_width || pixel_id_y >= image_height)
    {
        return;
    }

    uint32_t successor = unique_thread_id;
    uint32_t final_value = 0;
    uint32_t pixel_value = 0;
    uint32_t rank = 0;

    pixel_value = shared_block[block_local_y][block_local_x];

    // if(pixel_value != 0) { // unnecessary condition, avoid warp divergence
    // get all 8 neighbourhood values
    uint32_t neighbour_top_value = shared_block[block_local_y - 1][block_local_x];
    uint32_t neighbour_left_value = shared_block[block_local_y][block_local_x - 1];
    uint32_t neighbour_right_value = shared_block[block_local_y][block_local_x + 1];
    uint32_t neighbour_bottom_value = shared_block[block_local_y + 1][block_local_x];

    final_value = (neighbour_bottom_value == 0 || neighbour_left_value == 0 || neighbour_right_value == 0 || neighbour_top_value == 0) ? pixel_value : 0;

    if (final_value > 0)
    {
        const int8_t dx[8] = {0, 1, 1, 1, 0, -1, -1, -1}; // top,top right, right, bottom right, bottom, bottom left, left, top left
        const int8_t dy[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
        int8_t start_rotation = 127;
        if (neighbour_top_value == 0)
            start_rotation = 0;
        else if (neighbour_right_value == 0)
            start_rotation = 2;
        else if (neighbour_bottom_value == 0)
            start_rotation = 4;
        else if (neighbour_left_value == 0)
            start_rotation = 6;

        if (start_rotation != 127)
        {
            for (int8_t clockwise_steps = 0; clockwise_steps < 8; ++clockwise_steps)
            {
                int8_t rotation = (clockwise_steps + start_rotation) % 8;
                int32_t x_value = dx[rotation] + block_local_x;
                int32_t y_value = dy[rotation] + block_local_y;

                if (shared_block[y_value][x_value] == pixel_value)
                {
                    successor = uint32_t((int32_t)pixel_id_x + dx[rotation] + ((int32_t)pixel_id_y + dy[rotation]) * (int32_t)image_width);
                    rank = 1;
                    break;
                }
            }
        }
    }
    //}

    // set boundaries list
    data.boundaries[unique_thread_id] = final_value;
    data.successors[unique_thread_id] = successor;
    data.ranks[unique_thread_id] = rank;
}

// Unneeded kernels under the assumption that label is the pixel id of the most topleft pixel

// __global__ void find_roots(const uint32_t* boundaries, uint32_t* roots, uint32_t pixel_count)
// {
//     uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

//     if (id >= pixel_count) {
//         return;
//     }

//     uint32_t label = boundaries[id];

//     if (label != 0) {
//         atomicMin(&roots[label], id);
//     }
// }

__global__ void build_predecessors(
    const uint32_t *boundaries,
    const uint32_t *successors,
    uint32_t *predecessors,
    uint32_t pixel_count)
{
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= pixel_count)
        return;

    if (boundaries[id] == 0)
        return;

    uint32_t succ = successors[id];

    if (succ < pixel_count)
    {
        predecessors[succ] = id;
    }
}

// __global__ void break_at_roots_for_ranks(
//     const uint32_t* roots,
//     uint32_t* predecessors,
//     uint32_t* ranks,
//     uint32_t pixel_count
// ) {
//     uint32_t label = blockIdx.x * blockDim.x + threadIdx.x;
//     if (label >= pixel_count) return;

//     uint32_t root = roots[label];

//     if (root == NIL) return;

//     predecessors[root] = NIL;
//     ranks[root] = 0;
// }

__global__ void break_roots_by_label(
    const uint32_t *boundaries,
    uint32_t *predecessors,
    uint32_t *ranks,
    uint32_t pixel_count)
{
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= pixel_count)
    {
        return;
    }

    uint32_t label = boundaries[id];

    if (label == 0)
    {
        return;
    }

    // If label is smallest pixel index of the component,
    // then the root pixel has id == label.
    if (id == label)
    {
        predecessors[id] = NIL;
        ranks[id] = 0;
    }
}

__global__ void list_rank_step(
    const uint32_t *boundaries,
    const uint32_t *ptr_in,
    const uint32_t *rank_in,
    uint32_t *ptr_out,
    uint32_t *rank_out,
    uint32_t pixel_count,
    uint32_t *done)
{
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= pixel_count)
        return;

    if (boundaries[id] == 0)
    {
        ptr_out[id] = NIL;
        rank_out[id] = 0;
        return;
    }
    // i dont particularly like the naming p --> maybe change it to succ_ptr or smth
    uint32_t p = ptr_in[id];

    if (p == NIL)
    {
        ptr_out[id] = NIL;
        rank_out[id] = rank_in[id];
    }
    else
    {
        ptr_out[id] = ptr_in[p];                 // Set n.next ← n.next.next
        rank_out[id] = rank_in[id] + rank_in[p]; // Set d[n] ← d[n] + d[n.next]
        atomicExch(done, 0);
    }
}



__global__ void list_ranking_chained(
    int blocks, int threads,
    const uint32_t *boundaries,
    uint32_t *ptr_in, uint32_t *rank_in,
    uint32_t *ptr_out, uint32_t *rank_out,
    uint32_t pixel_count,
    uint32_t *done,
    uint32_t **final_winner_out,
    uint32_t current_round,
    uint32_t max_rounds)
{
    // Ensure only thread 0 manages the logic
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // Reset the done flag for this round
    *done = 1;

    // Launch the processing for THIS round
    list_rank_step<<<blocks, threads>>>(
        boundaries, ptr_in, rank_in, ptr_out, rank_out, pixel_count, done);

    // Instead of Synchronizing, we use cudaStreamTailLaunch.
    // This tells CUDA: "Launch the next step once THIS kernel and its children are finished."

    if (current_round < max_rounds) {
        // We launch the management for the NEXT round into the Tail Stream
        // Note: we SWAP the pointers in the arguments for the next launch
        list_ranking_chained<<<1, 1, 0, cudaStreamTailLaunch>>>(
            blocks, threads,
            boundaries,
            ptr_out, rank_out, // Swapped: Out becomes In
            ptr_in, rank_in,   // Swapped: In becomes Out
            pixel_count,
            done,
            final_winner_out,
            current_round + 1,
            max_rounds
        );
    } else {
        // Final round: set the winner
        *final_winner_out = rank_in;
    }
}
// Optimized ranking step using packed uint2
__global__ void list_rank_step_packed(
    const uint32_t* boundaries,
    const uint2*    data_in,    // .x = predecessor, .y = rank
    uint2*          data_out,
    uint32_t        pixel_count,
    uint32_t*       done)
{
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= pixel_count || boundaries[id] == 0) return;

    uint2 current = data_in[id];
    uint32_t p = current.x;

    if (p == NIL) {
        data_out[id] = current;
    } else {
        // Fetch neighbor's pointer and rank in a single 64-bit transaction
        uint2 neighbor = __ldg(&data_in[p]);

        uint2 updated;
        updated.x = neighbor.x;             // jump
        updated.y = current.y + neighbor.y; // accumulate rank
        data_out[id] = updated;

        // Warp-level check: if any thread in this warp isn't done, mark global 'done' as 0
        if (updated.x != NIL) {
            if (__any_sync(0xFFFFFFFF, true)) {
                if ((threadIdx.x & 31) == 0) atomicExch(done, 0);
            }
        }
    }
}

// Helper to pack/unpack
__global__ void pack_data(const uint32_t* p, const uint32_t* r, uint2* dst, uint32_t n) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) dst[id] = make_uint2(p[id], r[id]);
}

__global__ void unpack_data(const uint2* src, uint32_t* p, uint32_t* r, uint32_t n) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) { p[id] = src[id].x; r[id] = src[id].y; }
}


// ptr_jmp_out_t pointer_jumping(uint32_t *global_labels, uint32_t image_width, uint32_t image_height)
// {

//     ptr_jmp_out_t out;

//     // setup all the necessary memory on device and host
//     uint32_t pixel_count = image_height * image_width;
//     size_t bytes = pixel_count * sizeof(uint32_t);

//     // Create Grid/Block
//     dim3 block(BLOCK_WIDTH_X, BLOCK_WIDTH_Y);
//     dim3 grid(ceil(image_width / (float)block.x), ceil(image_height / (float)block.y));

//     uint32_t array_size = pixel_count * sizeof(uint32_t);
//     // uint32_t *d_image_data_in;
//     ptr_jmp_out_t device_data;

//     // cudaMalloc((void**) &d_image_data_in, array_size);
//     cudaMalloc((void **)&device_data.boundaries, array_size);
//     cudaMalloc((void **)&device_data.ranks, array_size);
//     cudaMalloc((void **)&device_data.successors, array_size);

//     // cudaMemcpy(d_image_data_in, global_labels, pixel_count*sizeof(uint32_t), cudaMemcpyHostToDevice);
//     DETAILED_TIMER_START(init_list_and_extract_boundaries);
//     init_pointer_list_and_extract_boundaries<<<grid, block>>>(global_labels, image_width, image_height, device_data);
//     cudaDeviceSynchronize();
//     DETAILED_TIMER_STOP(init_list_and_extract_boundaries, "Init pointer list and extract boundaries");

//     // uint32_t* d_roots;            // root per component label
//     uint32_t *d_predecessors_a; // pointer-jumping input
//     uint32_t *d_predecessors_b; // pointer-jumping output
//     uint32_t *d_ranks_a;        // rank input
//     uint32_t *d_ranks_b;        // rank output

//     // cudaMalloc(&d_roots, bytes);
//     DETAILED_TIMER_START(malloc);
//     cudaMalloc(&d_predecessors_a, bytes);
//     cudaMalloc(&d_predecessors_b, bytes);
//     cudaMalloc(&d_ranks_a, bytes);
//     cudaMalloc(&d_ranks_b, bytes);
//     // cudaMemset(d_roots, 0xFF, bytes);
//     cudaMemset(d_predecessors_a, 0xFF, bytes);
//     cudaMemcpy(d_ranks_a, device_data.ranks, bytes, cudaMemcpyDeviceToDevice);
//     DETAILED_TIMER_STOP(malloc, "Malloc for pointer jumping arrays");

//     int threads = BLOCK_WIDTH_X * BLOCK_WIDTH_Y; // make 1D
//     int blocks = (pixel_count + threads - 1) / threads;

//     // find_roots<<<blocks, threads>>>(device_data.boundaries, d_roots, pixel_count);

//     DETAILED_TIMER_START(build_predecessors);
//     build_predecessors<<<blocks, threads>>>(device_data.boundaries, device_data.successors, d_predecessors_a, pixel_count);
//     cudaDeviceSynchronize();
//     DETAILED_TIMER_STOP(build_predecessors, "Build predecessors list");

//     DETAILED_TIMER_START(break_roots);
//     break_roots_by_label<<<blocks, threads>>>(device_data.boundaries, d_predecessors_a, d_ranks_a, pixel_count);
//     cudaDeviceSynchronize();
//     DETAILED_TIMER_STOP(break_roots, "Break roots by label");

//     uint32_t rounds = 0;
//     while ((1u << rounds) < pixel_count)
//     { // log2 loop
//         rounds++;
//     }
//     uint32_t host_done = 0;
//     uint32_t *d_done;
//     cudaMalloc(&d_done, sizeof(int));

//     DETAILED_TIMER_START(pointer_jumping);
//     for (uint32_t r = 0; r < rounds; ++r)
//     {
//         int initial_val = 1;
//         cudaMemcpy(d_done, &initial_val, sizeof(uint32_t), cudaMemcpyHostToDevice);

//         list_rank_step<<<blocks, threads>>>(
//             device_data.boundaries,
//             d_predecessors_a,
//             d_ranks_a,
//             d_predecessors_b,
//             d_ranks_b,
//             pixel_count,
//             d_done);

//         cudaMemcpy(&host_done, d_done, sizeof(uint32_t), cudaMemcpyDeviceToHost);

//         if (host_done == 1)
//         {
//             break;
//         }

//         std::swap(d_predecessors_a, d_predecessors_b);
//         std::swap(d_ranks_a, d_ranks_b);
//     }

//     //this is unfortunatelly slower
//     // uint32_t**final_pointer;
//     // cudaMalloc(&final_pointer, sizeof(uint32_t*));

//     // // Start the chain
//     // list_ranking_chained<<<1, 1>>>(
//     //     blocks, threads,
//     //     device_data.boundaries,
//     //     d_predecessors_a, d_ranks_a,
//     //     d_predecessors_b, d_ranks_b,
//     //     pixel_count,
//     //     d_done,
//     //     final_pointer,
//     //     0,      // current_round
//     //     rounds  // max_rounds
//     // );
//     // cudaDeviceSynchronize();

//     // uint32_t *h_winning_buffer_addr;
//     // cudaMemcpy(&h_winning_buffer_addr, final_pointer, sizeof(uint32_t*), cudaMemcpyDeviceToHost);

//     DETAILED_TIMER_STOP(pointer_jumping, "Pointer Jumping");

//     // cudaMemcpy(out.boundaries, device_data.boundaries, pixel_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
//     // cudaMemcpy(out.successors, device_data.successors, pixel_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
//     // cudaMemcpy(out.ranks, d_ranks_a, pixel_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

//     cudaMemcpy(device_data.ranks, d_ranks_a, bytes, cudaMemcpyDeviceToDevice);
//     out.boundaries = device_data.boundaries;
//     out.successors = device_data.successors;
//     out.ranks = device_data.ranks;

//     // printf("Printing boundaries\n");
//     // for (uint32_t y = 0; y < image_height; ++y) {
//     //     for (uint32_t x = 0; x < image_width; ++x) {
//     //         uint32_t id = x + y * image_width;

//     //         if (out.boundaries[id] != 0) {
//     //             uint32_t succ = out.successors[id];

//     //             uint32_t succ_x = succ % image_width;
//     //             uint32_t succ_y = succ / image_width;

//     //             printf(
//     //                 "pixel (%u,%u) id=%u label=%u -> successor id=%u (%u,%u), rank=%u\n",
//     //                 x,
//     //                 y,
//     //                 id,
//     //                 out.boundaries[id],
//     //                 succ,
//     //                 succ_x,
//     //                 succ_y,
//     //                 out.ranks[id]
//     //             );
//     //         }
//     //     }
//     // }

//     // uint32_t bad = 0;

//     // for (uint32_t id = 0; id < pixel_count; ++id) {
//     //     uint32_t label = out.boundaries[id];

//     //     if (label == 0) {
//     //         continue;
//     //     }

//     //     uint32_t succ = out.successors[id];

//     //     if (succ >= pixel_count) {
//     //         printf("BAD: successor out of range: id=%u succ=%u\n", id, succ);
//     //         bad++;
//     //         continue;
//     //     }

//     //     if (out.boundaries[succ] != label) {
//     //         printf("BAD: successor has different label: id=%u label=%u succ=%u succ_label=%u\n",
//     //             id, label, succ, out.boundaries[succ]);
//     //         bad++;
//     //         continue;
//     //     }

//     //     if (succ == label) {
//     //         // id is the last pixel before root
//     //         // rank[id] should be max rank of that component
//     //         continue;
//     //     }

//     //     if (out.ranks[succ] != out.ranks[id] + 1) {
//     //         printf("BAD rank continuity: id=%u rank=%u succ=%u succ_rank=%u label=%u\n",
//     //             id, out.ranks[id], succ, out.ranks[succ], label);
//     //         bad++;
//     //     }
//     // }

//     // printf("Rank continuity check: bad=%u\n", bad);

//     // cudaFree(d_image_data_in);
//     // cudaFree(device_data.boundaries);
//     //  cudaFree(device_data.ranks);
//     //  cudaFree(device_data.successors);
//     cudaFree(d_predecessors_a);
//     cudaFree(d_predecessors_b);
//     cudaFree(d_ranks_a);
//     cudaFree(d_ranks_b);

//     return out;
// }

__global__ void walk_boundary_pieces_kernel(
    const uint32_t* __restrict__ boundaries,
    const uint32_t* __restrict__ successors,
    uint32_t* __restrict__ ranks,
    uint32_t pixel_count)
{
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= pixel_count) return;

    uint32_t label = boundaries[id];

    // Only the 'Root' (id == label) of a component starts the walk.
    // In your system, labels are the ID of the top-left pixel of the component.
    if (label == 0 || id != label) return;

    uint32_t curr = successors[id];
    uint32_t dist = 1;

    // The root itself is rank 0
    ranks[id] = 0;

    // Traverse the chain
    // The init kernel creates a loop for closed boundaries.
    // We walk until we return to the start ID or hit NIL.
    while (curr != id && curr != NIL) {
        ranks[curr] = dist++;

        // Use __ldg to exploit the Read-Only/Texture cache for pointer chasing
        uint32_t next = __ldg(&successors[curr]);

        // Safety: break if we hit a self-loop or dead end
        if (next == curr || next == NIL) break;

        curr = next;

        // Massive safety break (prevents infinite GPU kernels on data errors)
        if (dist > 65535) break;
    }
}


ptr_jmp_out_t pointer_jumping_old(uint32_t *global_labels, uint32_t image_width, uint32_t image_height, pipeline_workspace_t& ws)
{
    uint32_t pixel_count = image_height * image_width;
    size_t bytes = pixel_count * sizeof(uint32_t);
    dim3 block(BLOCK_WIDTH_X, BLOCK_WIDTH_Y);
    dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);
    int threads_1d = 256;
    int blocks_1d = (pixel_count + threads_1d - 1) / threads_1d;

    ws.d_ptr_boundaries.ensure(pixel_count);
    ws.d_ptr_ranks.ensure(pixel_count);
    ws.d_ptr_successors.ensure(pixel_count);
    ws.d_ptr_predecessors.ensure(pixel_count);
    ws.d_ptr_work_a.ensure(pixel_count);
    ws.d_ptr_work_b.ensure(pixel_count);
    ws.d_ptr_done.ensure(1);

    ptr_jmp_out_t out;

    DETAILED_TIMER_START(malloc);
    // cudaMalloc(&out.boundaries, bytes);
    // cudaMalloc(&out.ranks,      bytes);
    // cudaMalloc(&out.successors, bytes);
    // uint2 *d_work_a, *d_work_b;
    // cudaMalloc(&d_work_a, pixel_count * sizeof(uint2));
    // cudaMalloc(&d_work_b, pixel_count * sizeof(uint2));
    // uint32_t *d_predecessors;
    // cudaMalloc(&d_predecessors, bytes);
    // cudaMemset(d_predecessors, 0xFF, bytes); // NIL
    out.boundaries = ws.d_ptr_boundaries.data();
    out.ranks = ws.d_ptr_ranks.data();
    out.successors = ws.d_ptr_successors.data();

    uint32_t* d_predecessors = ws.d_ptr_predecessors.data();
    uint2* d_work_a = ws.d_ptr_work_a.data();
    uint2* d_work_b = ws.d_ptr_work_b.data();
    uint32_t* d_done = ws.d_ptr_done.data();
    cudaMemset(out.boundaries, 0, bytes);
    cudaMemset(out.ranks, 0, bytes);
    cudaMemset(out.successors, 0xFF, bytes);
    cudaMemset(d_predecessors, 0xFF, bytes);
    DETAILED_TIMER_STOP(malloc, "Allocate arrays");


    // 1. Init and Build Predecessors
    DETAILED_TIMER_START(init_list_and_extract_boundaries);
    init_pointer_list_and_extract_boundaries<<<grid, block>>>(global_labels, image_width, image_height, out);
    DETAILED_TIMER_STOP(init_list_and_extract_boundaries, "Init pointer list and extract boundaries");

    DETAILED_TIMER_START(buildandbreak);
    build_predecessors<<<blocks_1d, threads_1d>>>(out.boundaries, out.successors, d_predecessors, pixel_count);
    break_roots_by_label<<<blocks_1d, threads_1d>>>(out.boundaries, d_predecessors, out.ranks, pixel_count);
    DETAILED_TIMER_STOP(buildandbreak, "Build Predecessors| Break roots by label");



    DETAILED_TIMER_START(pointer_jumping);
    pack_data<<<blocks_1d, threads_1d>>>(d_predecessors, out.ranks, d_work_a, pixel_count);

    //uint32_t *d_done;
    //cudaMalloc(&d_done, sizeof(uint32_t));

    uint32_t rounds = 32 - __builtin_clz(pixel_count);
    for (uint32_t r = 0; r < rounds; ++r) {
        uint32_t h_done = 1;
        cudaMemcpy(d_done, &h_done, sizeof(uint32_t), cudaMemcpyHostToDevice);

        list_rank_step_packed<<<blocks_1d, threads_1d>>>(
            out.boundaries, d_work_a, d_work_b, pixel_count, d_done
        );

        cudaMemcpy(&h_done, d_done, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        std::swap(d_work_a, d_work_b);
        if (h_done == 1) break;
    }

    // 3. Final Result Unpack
    // Results are in d_work_a because of the swap at the end of the loop
    unpack_data<<<blocks_1d, threads_1d>>>(d_work_a, out.successors, out.ranks, pixel_count);
    DETAILED_TIMER_STOP(pointer_jumping, "Pointerjumping");
    // Cleanup
    // cudaFree(d_predecessors);
    // cudaFree(d_work_a);
    // cudaFree(d_work_b);
    // cudaFree(d_done);

    return out;
}


//sequential list ranking
ptr_jmp_out_t pointer_jumping(uint32_t *global_labels, uint32_t image_width, uint32_t image_height, pipeline_workspace_t& ws)
{
    cudaStream_t stream = ws.stream;

    uint32_t pixel_count = image_height * image_width;
    size_t bytes = pixel_count * sizeof(uint32_t);

    dim3 block2d(BLOCK_WIDTH_X, BLOCK_WIDTH_Y);
    dim3 grid2d((image_width + block2d.x - 1) / block2d.x, (image_height + block2d.y - 1) / block2d.y);

    int threads1d = 256;
    int blocks1d = (pixel_count + threads1d - 1) / threads1d;

    // Workspace allocation
    ws.d_ptr_boundaries.ensure(pixel_count);
    ws.d_ptr_ranks.ensure(pixel_count);
    ws.d_ptr_successors.ensure(pixel_count);

    ptr_jmp_out_t out;
    out.boundaries = ws.d_ptr_boundaries.data();
    out.ranks = ws.d_ptr_ranks.data();
    out.successors = ws.d_ptr_successors.data();

    // Reset ranks to 0
    cudaMemsetAsync(out.ranks, 0, bytes, stream);

    // Step 1: Extract boundaries and identify immediate successors (neighbors)
    DETAILED_TIMER_START(init_list);
    init_pointer_list_and_extract_boundaries<<<grid2d, block2d, 0, stream>>>(global_labels, image_width, image_height, out);
    DETAILED_TIMER_STOP(init_list, "Extract boundaries and successors");

    // Step 2: One thread per piece walk
    // This replaces the pointer-jumping loop entirely.
    DETAILED_TIMER_START(piece_walk);
    walk_boundary_pieces_kernel<<<blocks1d, threads1d, 0, stream>>>(
        out.boundaries, out.successors, out.ranks, pixel_count
    );
    DETAILED_TIMER_STOP(piece_walk, "Piece-wise Boundary Walk");

    return out;
}
#endif
