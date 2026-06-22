#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/system/cuda/execution_policy.h>

#include "types.hpp"
#include "timer.hpp"
#include "pipeline_workspace.hpp"

#include "stb_image.h"
#include "preprocessing.cuh"

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>

// ── Tunable parameters ─────────────────────────────────────────────────────────
static constexpr int BLOCK_X = 32;
static constexpr int BLOCK_Y = 16;
static constexpr int HIST_BINS = 256;
static constexpr int HIST_BLOCK = 1024;
static constexpr float GAUSSIAN_SIGMA = 2.0f;
static constexpr int RADIUS = 6;
#define MAX_KERNEL_HALF 15
//#define PREPROC_TIMING

// for timing
__global__ void warmup_kernel() {}

// get  errors
static void cudaCheck(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("CUDA error: ") + cudaGetErrorString(err) +
            " at " + file + ":" + std::to_string(line));
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

// ── timer ───
// Enable with make debug for kernel timings

struct pipeline_t
{
    uchar3 *d_rgb = nullptr;     //   RGB input
    uint8_t *d_gray = nullptr;     //  luma , Gaussian output
    uint8_t *d_binary = nullptr; //  binary after threshold
    int width = 0;
    int height = 0;
};

static pipeline_t alloc_pipeline(int width, int height)
{
    pipeline_t p;
    p.width = width;
    p.height = height;
    size_t N = (size_t)width * height;

    CUDA_CHECK(cudaMalloc(&p.d_rgb, N * sizeof(uchar3)));
    CUDA_CHECK(cudaMalloc(&p.d_gray, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p.d_binary, N * sizeof(uint8_t)));

    return p;
}
// cudafrees
static void free_pipeline(pipeline_t &p, bool free_rgb = true, bool free_gray = true, bool free_binary = true)
{
    if (free_rgb)
    {
        cudaFree(p.d_rgb);
        p.d_rgb = nullptr;
    }
    if (free_gray)
    {
        cudaFree(p.d_gray);
        p.d_gray = nullptr;
    }
    if (free_binary)
    {
        cudaFree(p.d_binary);
        p.d_binary = nullptr;
    }
}

// RGB → luma
__global__ void rgba_to_luma_kernel(const uchar3 *src, uint8_t *dst, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uchar3 p = src[i];
    dst[i] = (uint8_t)(0.299f * p.x + 0.587f * p.y + 0.114f * p.z + 0.5f);
}
// ── Gaussian kernel

#define MAX_KERNEL_HALF 15 // sigma is at 2.0
// kernel coefficients in constant memory
__constant__ float c_gaussian[2 * MAX_KERNEL_HALF + 1]; // 31 elements...

// compute gaussian in h_k
// Normalize
// Fill c_gaussian in constant mem with kernel elements.
static void build_gaussian_kernel(float sigma)
{
    //int radius = (int)ceilf(3.0f * sigma);

    int ksize = 2 * RADIUS + 1;
    float h_k[2 * RADIUS + 1];
    float sum = 0.0f;

    for (int i = -RADIUS; i <= RADIUS; ++i)
    {
        h_k[i + RADIUS] = expf(-(float)(i * i) / (2.0f * sigma * sigma));
        sum += h_k[i + RADIUS];
    }
    for (int i = 0; i < ksize; ++i)
        h_k[i] /= sum;

    CUDA_CHECK(cudaMemcpyToSymbol(c_gaussian, h_k, ksize * sizeof(float)));

}

// Handle edge pixels
// Mirror coordinate to [0, limit-1].
__device__ static int clamp_coord(int v, int limit)
{
    return max(0, min(v, limit - 1));
}

// ─── separable Gaussian horizontal & vertical ─────
template <typename T, int RADIUS>
__global__ void gaussian_fused(const T *__restrict__ src,
                               uint8_t *__restrict__ dst,
                               int W, int H)
{
    constexpr int ROWS = BLOCK_Y + 2 * RADIUS;
    constexpr int INW  = BLOCK_X + 2 * RADIUS;

    __shared__ float smem_in[ROWS][INW];
    __shared__ float smem_hc[ROWS][BLOCK_X];

    const int x0  = blockIdx.x * BLOCK_X;
    const int y0  = blockIdx.y * BLOCK_Y;
    const int tid = threadIdx.y * BLOCK_X + threadIdx.x;

    // Phase 1: load input tile + halos into smem_in
    for (int i = tid; i < ROWS * INW; i += BLOCK_X * BLOCK_Y) {
        int r = i / INW, c = i % INW;
        smem_in[r][c] = (float)src[clamp_coord(y0 + r - RADIUS, H) * W +
                                   clamp_coord(x0 + c - RADIUS, W)];
    }
    __syncthreads();

    // horizontal convolution for every row
    // Result stored in smem_hc
    for (int i = tid; i < ROWS * BLOCK_X; i += BLOCK_X * BLOCK_Y) {
        int r = i / BLOCK_X, c = i % BLOCK_X;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k <= 2 * RADIUS; ++k)
            sum += c_gaussian[k] * smem_in[r][c + k];
        smem_hc[r][c] = sum;
    }
    __syncthreads();

    // Phase 3: vertical convolution → global output
    const int x = x0 + threadIdx.x;
    const int y = y0 + threadIdx.y;
    if (x >= W || y >= H) return;

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k <= 2 * RADIUS; ++k)
        sum += c_gaussian[k] * smem_hc[threadIdx.y + k][threadIdx.x];
    dst[y * W + x] = (uint8_t)(sum + 0.5f);
}

// ─── Histogram for Otsu ───────
__global__ void histogram_kernel(const uint8_t *__restrict__ luma,
                                 int *hist, int N)
{
    extern __shared__ int s_hist[];   // [nwarps × HIST_BINS]

    const int tid    = threadIdx.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int nwarps = blockDim.x / 32;
    int *my_hist = s_hist + warpId * HIST_BINS;

    for (int i = laneId; i < HIST_BINS; i += 32)
        my_hist[i] = 0;
    __syncthreads();

    int idx4 = blockIdx.x * blockDim.x + tid;
    if (idx4 * 4 < N) {
        uchar4 v = reinterpret_cast<const uchar4 *>(luma)[idx4];
        atomicAdd(&my_hist[v.x], 1);
        atomicAdd(&my_hist[v.y], 1);
        atomicAdd(&my_hist[v.z], 1);
        atomicAdd(&my_hist[v.w], 1);
    }
    __syncthreads();

    // Reduce all warp slices into global histogram
    for (int i = tid; i < HIST_BINS; i += blockDim.x) {
        int sum = 0;
        for (int w = 0; w < nwarps; ++w)
            sum += s_hist[w * HIST_BINS + i];
        atomicAdd(&hist[i], sum);
    }
}

// ─── Otsu threshold  ──────
// one thread per bin
// Hillis–Steele inclusive prefix scan over hist[]

__global__ void otsu_kernel(const int *__restrict__ hist, int N, float *out)
{
    __shared__ int s_count[HIST_BINS];  // cumulative pixel count  for threshold t
    __shared__ float s_wsum[HIST_BINS]; // cumulative weighted sum for threshold t
    __shared__ float s_var[HIST_BINS];
    __shared__ int s_idx[HIST_BINS];

    int t = threadIdx.x;

    s_count[t] = hist[t];
    s_wsum[t] = (float)t * hist[t];
    __syncthreads();

    // inter-class variance for its threshold
    for (int stride = 1; stride < HIST_BINS; stride <<= 1)
    {
        auto delta_count = (t >= stride) ? s_count[t - stride] : 0;
        auto delta_wsum = (t >= stride) ? s_wsum[t - stride] : 0.0f;
        __syncthreads();
        s_count[t] += delta_count;
        s_wsum[t] += delta_wsum;
        __syncthreads();
    }
    int total_pixels = s_count[HIST_BINS - 1];
    float total_wsum = s_wsum[HIST_BINS - 1];
    int bg_count = s_count[t];
    int fg_count = total_pixels - bg_count;

    float var = 0.0f;
    if (bg_count > 0 && fg_count > 0)
    {
        float mean_bg = s_wsum[t] / (float)bg_count;
        float mean_fg = (total_wsum - s_wsum[t]) / (float)fg_count;
        float diff = mean_bg - mean_fg;
        var = (float)bg_count * (float)fg_count * diff * diff;
    }

    s_var[t] = var;
    s_idx[t] = t;
    __syncthreads();

    // argmax reduction
    for (int stride = 128; stride >= 1; stride >>= 1)
    {
        if (t < stride && s_var[t + stride] > s_var[t])
        {
            s_var[t] = s_var[t + stride];
            s_idx[t] = s_idx[t + stride];
        }
        __syncthreads();
    }

    if (t == 0)
        *out = (float)s_idx[0];
}

// ─── Otsu threshold application ──────
// All threads read the same value
// Processes 4 pixels per thread via float4 load + packed uint32 write.

__global__ void apply_threshold_kernel(const uint8_t *__restrict__ luma,
                                       uint8_t *__restrict__ binary,
                                       int W, int H, const float *threshold)
{
    const uint8_t thresh = (uint8_t)*threshold;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y   = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx * 4 >= W || y >= H) return;

    uchar4 v = reinterpret_cast<const uchar4 *>(luma)[y * (W / 4) + idx];

    uchar4 result;
    result.x = v.x > thresh ? 255 : 0;
    result.y = v.y > thresh ? 255 : 0;
    result.z = v.z > thresh ? 255 : 0;
    result.w = v.w > thresh ? 255 : 0;

    reinterpret_cast<uchar4 *>(binary)[y * (W / 4) + idx] = result;
}

// ── Debug saves ────────────────────────────────────────────────────────────────
// Compiled in only when -DPREPROC_DEBUG is passed.  Each helper downloads one
// device buffer, converts it if needed, and writes a PNG to ../data/processed/.
/*
#ifdef PREPROC_DEBUG
static void save_float_stage(const float *d_buf, int W, int H, const char *path)
{
    std::vector<float>   h_f(W * H);
    std::vector<uint8_t> h_u8(W * H);
    cudaMemcpy(h_f.data(), d_buf, (size_t)W * H * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < W * H; ++i)
        h_u8[i] = (uint8_t)fminf(255.0f, fmaxf(0.0f, h_f[i]));
    stbi_write_png(path, W, H, 1, h_u8.data(), W);
}

static void save_uint8_stage(const uint8_t *d_buf, int W, int H, const char *path)
{
    std::vector<uint8_t> h_buf(W * H);
    cudaMemcpy(h_buf.data(), d_buf, (size_t)W * H, cudaMemcpyDeviceToHost);
    stbi_write_png(path, W, H, 1, h_buf.data(), W);
}
#endif  // PREPROC_DEBUG
*/

// AI Generated image loading instead of stb
//  seems faster

// Reads a JPEG file into a host byte buffer.
static std::vector<uint8_t> read_file(const std::string &path)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("Failed to open image: " + path);
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    fread(buf.data(), 1, sz, f);
    fclose(f);
    return buf;
}

static pipeline_t make_pipeline_from_workspace(
    int width,
    int height,
    pipeline_workspace_t& ws
) {
    size_t N = static_cast<size_t>(width) * static_cast<size_t>(height);

    ws.d_rgb.ensure(N);
    ws.d_gray.ensure(N);
    ws.d_blur.ensure(N);
    ws.d_binary.ensure(N);

    ws.d_hist.ensure(HIST_BINS);
    ws.d_threshold.ensure(1);

    pipeline_t p;
    p.width = width;
    p.height = height;
    p.d_rgb = ws.d_rgb.data();
    p.d_gray = ws.d_gray.data();
    p.d_binary = ws.d_binary.data();

    return p;
}

void init_jpeg_decoder(jpeg_decoder_t& dec)
{
    nvjpegCreate(
        NVJPEG_BACKEND_DEFAULT,
        nullptr,
        &dec.handle
    );

    nvjpegJpegStateCreate(
        dec.handle,
        &dec.state
    );
}

void destroy_jpeg_decoder(jpeg_decoder_t& dec)
{
    if (dec.state != nullptr) {
        nvjpegJpegStateDestroy(dec.state);
        dec.state = nullptr;
    }

    if (dec.handle != nullptr) {
        nvjpegDestroy(dec.handle);
        dec.handle = nullptr;
    }
}

// Decodes a JPEG directly to device memory (d_rgb) using nvJPEG.
// Returns image dimensions via W, H.
static void nvjpeg_decode(const std::vector<uint8_t>& jpeg, pipeline_t& pipe, int& W, int& H, pipeline_workspace_t& ws, jpeg_decoder_t& dec, cudaStream_t stream)
{
    // nvjpegHandle_t handle;
    // nvjpegJpegState_t state;
    // CUDA_CHECK((cudaError_t)nvjpegCreate(NVJPEG_BACKEND_DEFAULT, nullptr, &handle));
    // CUDA_CHECK((cudaError_t)nvjpegJpegStateCreate(handle, &state));

    // Query image dimensions before decoding
    int ncomp;
    nvjpegChromaSubsampling_t css;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    //nvjpegGetImageInfo(handle, jpeg.data(), jpeg.size(), &ncomp, &css, widths, heights);
    nvjpegGetImageInfo( dec.handle, jpeg.data(), jpeg.size(), &ncomp, &css, widths, heights);
    W = widths[0];
    H = heights[0];

    //pipe = alloc_pipeline(W, H);
    pipe = make_pipeline_from_workspace(W, H, ws);

    // Point nvJPEG output at d_rgb — interleaved RGB matches uchar3 layout
    nvjpegImage_t img{};
    img.channel[0] = reinterpret_cast<uint8_t *>(pipe.d_rgb);
    img.pitch[0] = W * 3;

    DETAILED_TIMER_START(nvjpegdecode);
    //CUDA_CHECK((cudaError_t)nvjpegDecode(handle, state, jpeg.data(), jpeg.size(), NVJPEG_OUTPUT_RGBI, &img, 0));
    nvjpegStatus_t decode_status = nvjpegDecode(dec.handle, dec.state, jpeg.data(), jpeg.size(), NVJPEG_OUTPUT_RGBI, &img, stream);

    if (decode_status != NVJPEG_STATUS_SUCCESS)
    {
        throw std::runtime_error("nvjpegDecode failed with status " + std::to_string(static_cast<int>(decode_status)));
    }

    DETAILED_TIMER_STOP(nvjpegdecode, "nvJPEG decode");

    // nvjpegJpegStateDestroy(state);
    // nvjpegDestroy(handle);
}

static void jpeg_load(const std::vector<uint8_t> &jpeg, pipeline_t &pipe, int &W, int &H, cudaStream_t stream)
{
    int n;
    unsigned char *data = stbi_load_from_memory(jpeg.data(), jpeg.size(), &W, &H, &n, 3);
    if (!data)
        throw std::runtime_error("Failed to decode image with stb_image");

    pipe = alloc_pipeline(W, H);
    CUDA_CHECK(cudaMemcpyAsync(pipe.d_rgb, data, (size_t)W * H * 3, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    stbi_image_free(data);
}

// MEMCPY
static void download_result(const pipeline_t &p, grayscale_image_t &out, cudaStream_t stream)
{
    size_t N = (size_t)p.width * p.height;
    out.width = p.width;
    out.height = p.height;
    out.data.resize(N);
    CUDA_CHECK(cudaMemcpyAsync(out.data.data(), p.d_binary, N, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

static void download_rgb_result(const pipeline_t &p, rgb_image_t &out, cudaStream_t stream)
{
    size_t N = (size_t)p.width * p.height;
    out.width = p.width;
    out.height = p.height;
    out.data.resize(N * 3);
    CUDA_CHECK(cudaMemcpyAsync(out.data.data(), p.d_rgb, N * 3, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

rgb_image_d_t preprocessing(const std::string &imagePath, uint8_t *&d_result,  pipeline_workspace_t& ws, jpeg_decoder_t& jpeg_decoder)
{
    DETAILED_TIMER_START(imageread);
    cudaStream_t stream = ws.stream;
    auto policy = thrust::cuda::par.on(stream);

    int W, H;
    auto jpeg = read_file(imagePath);
    pipeline_t pipe{};
    nvjpeg_decode(jpeg, pipe, W, H, ws, jpeg_decoder, stream);
    DETAILED_TIMER_STOP(imageread, "image read + decode");
    printf("[image] %s - %d x %d\n", imagePath.c_str(), W, H);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    //  RGB → luma
    DETAILED_TIMER_START(total);
    DETAILED_TIMER_START(luma);
    dim3 luma_block(256);
    dim3 luma_grid((W * H + 255) / 256);
    rgba_to_luma_kernel<<<luma_grid, luma_block, 0, stream>>>(pipe.d_rgb, pipe.d_gray, W*H);
    DETAILED_TIMER_STOP(luma, "rgba_to_luma");

    DETAILED_TIMER_START(download_rgb);
    //rgb_image_t input_rgb;
    //download_rgb_result(pipe, input_rgb);
    rgb_image_d_t input_rgb;
    input_rgb.width = W;
    input_rgb.height = H;
    input_rgb.data = pipe.d_rgb;
    //pipe.d_rgb = nullptr; // ownership moved
    DETAILED_TIMER_STOP(download_rgb, "download RGB result");

    //cudaFree(pipe.d_rgb);
    //pipe.d_rgb = nullptr;

    // Gaussian blur, constant memory is shared by all streams, so initialize it once.
    static std::once_flag gaussian_once;
    std::call_once(gaussian_once, []() { build_gaussian_kernel(GAUSSIAN_SIGMA); });

    //const int rows_gauss = block.y + 2 * radius;
    //size_t smem_gauss = (size_t)rows_gauss * (block.x + 2 * radius + block.x) * sizeof(float);

    DETAILED_TIMER_START(gauss);
    // float* d_blur = nullptr;
    //cudaMalloc(&d_blur, sizeof(float) * W * H);
    uint8_t* d_blur = ws.d_blur.data();

    gaussian_fused<uint8_t, RADIUS><<<grid, block, 0, stream>>>(pipe.d_gray, d_blur, W, H);
    //std::swap(pipe.d_gray, d_blur);
    //cudaFree(d_blur);
    pipe.d_gray = d_blur;
    DETAILED_TIMER_STOP(gauss, "gaussian_fused");

    // Otsu threshold
    const int num_pixels = W * H;
    // int *d_hist;
    // float *d_threshold;
    // // Allocate
    // CUDA_CHECK(cudaMalloc(&d_hist, HIST_BINS * sizeof(int)));
    // CUDA_CHECK(cudaMalloc(&d_threshold, sizeof(float)));
    // CUDA_CHECK(cudaMemsetAsync(d_hist, 0, HIST_BINS * sizeof(int), stream));

    int* d_hist = ws.d_hist.data();
    float* d_threshold = ws.d_threshold.data();

    CUDA_CHECK(cudaMemsetAsync(d_hist, 0, HIST_BINS * sizeof(int), stream));
    DETAILED_TIMER_START(hist);
    const size_t smem_hist = (HIST_BLOCK / 32) * HIST_BINS * sizeof(int);
    histogram_kernel<<<(num_pixels / 4 + HIST_BLOCK - 1) / HIST_BLOCK, HIST_BLOCK, smem_hist, stream>>>(pipe.d_gray, d_hist, num_pixels);
    DETAILED_TIMER_STOP(hist, "histogram");

    DETAILED_TIMER_START(otsu);
    otsu_kernel<<<1, HIST_BINS, 0, stream>>>(d_hist, num_pixels, d_threshold);
    DETAILED_TIMER_STOP(otsu, "otsu_gpu");
    //cudaFree(d_hist);


#ifdef PREPROC_TIMING
    float h_threshold;
    CUDA_CHECK(cudaMemcpyAsync(&h_threshold, d_threshold, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("[preprocessing] Otsu threshold = %.1f\n", h_threshold);
#endif

    dim3 thresh_grid((W / 4 + block.x - 1) / block.x, grid.y);
    DETAILED_TIMER_START(thresh);
    apply_threshold_kernel<<<thresh_grid, block,0,stream>>>(pipe.d_gray, pipe.d_binary, W, H, d_threshold);
    DETAILED_TIMER_STOP(thresh, "apply_threshold");

    DETAILED_TIMER_START(invert);
    // count white pixels in binary result
    thrust::device_ptr<uint8_t> d_binary_ptr(pipe.d_binary);
    size_t white_pixels = thrust::count(policy, d_binary_ptr, d_binary_ptr + num_pixels, 255);
    if (white_pixels > (num_pixels - white_pixels))
    {
        // Invert binary if foreground is majority
        printf("[preprocessing] Inverting binary result (foreground is majority: %zu white pixels to %zu black pixels)\n", white_pixels, num_pixels - white_pixels);
        thrust::transform(policy, d_binary_ptr, d_binary_ptr + num_pixels, d_binary_ptr, [] __device__(uint8_t v)
                          { return 255 - v; });
    }
    DETAILED_TIMER_STOP(invert, "invert binary if needed");

    DETAILED_TIMER_STOP(total, "preprocessing without transfers");
    // save_uint8_stage(pipe.d_binary, W, H, "data/processed//main/debug_3_binary.png"); // final stage 1 output

    // DETAILED_TIMER_START(download);
    // grayscale_image_t result;
    // download_result(pipe, result);
    // DETAILED_TIMER_STOP(download, "download result");
    // free_pipeline(pipe, true, true, false); // keep binary for further processing in main
    // d_result = pipe.d_binary;
    d_result = pipe.d_binary;
    return input_rgb;
}
