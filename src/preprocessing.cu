#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/scan.h>

#include "types.hpp"

#include "stb_image.h"
#include "preprocessing.cuh"

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// ── Tunable parameters ─────────────────────────────────────────────────────────
static constexpr int BLOCK_X = 32;
static constexpr int BLOCK_Y = 8;
static constexpr int HIST_BINS = 256;
static constexpr int HIST_BLOCK = 1024;
static constexpr float GAUSSIAN_SIGMA = 2.0f;
#define MAX_KERNEL_HALF 15
#define PREPROC_TIMING

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

#ifdef PREPROC_TIMING
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
#define TIMER_START(name) stage_timer_t _timer_##name
#define TIMER_STOP(name, label) printf("[timing] %-30s %7.3f ms\n", label, _timer_##name.elapsed_ms())
#else
#define TIMER_START(name)
#define TIMER_STOP(name, label)
#endif // PREPROC_TIMING

struct pipeline_t
{
    uchar3 *d_rgb = nullptr;     //   RGB input
    float *d_gray = nullptr;     //  luma , Gaussian output
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
__global__ void rgba_to_luma_kernel(const uchar3 *__restrict__ src,
                                    float *__restrict__ dst,
                                    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    uchar3 p = src[y * W + x];
    dst[y * W + x] = 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
}

// ── Gaussian kernel

#define MAX_KERNEL_HALF 15 // sigma is at 2.0
// kernel coefficients in constant memory
__constant__ float c_gaussian[2 * MAX_KERNEL_HALF + 1]; // 31 elements...

// compute gaussian in h_k
// Normalize
// Fill c_gaussian in constant mem with kernel elements.
static int build_gaussian_kernel(float sigma)
{
    int radius = (int)ceilf(3.0f * sigma);

    int ksize = 2 * radius + 1;
    float h_k[2 * radius + 1];
    float sum = 0.0f;

    for (int i = -radius; i <= radius; ++i)
    {
        h_k[i + radius] = expf(-(float)(i * i) / (2.0f * sigma * sigma));
        sum += h_k[i + radius];
    }
    for (int i = 0; i < ksize; ++i)
        h_k[i] /= sum;

    CUDA_CHECK(cudaMemcpyToSymbol(c_gaussian, h_k, ksize * sizeof(float)));
    return radius;
}

// Handle edge pixels
// Mirror coordinate to [0, limit-1].
__device__ static int clamp_coord(int v, int limit)
{
    return max(0, min(v, limit - 1));
}

// ─── separable Gaussian horizontal & vertical ─────
__global__ void gaussian_fused(const float *__restrict__ src,
                               float *__restrict__ dst,
                               int W, int H, int radius)
{
    extern __shared__ float smem[];
    const int rows = blockDim.y + 2 * radius; // smem rows
    const int in_w = blockDim.x + 2 * radius;
    float *smem_in = smem;
    float *smem_hc = smem + rows * in_w; // horizontal-conv output

    const int x0 = blockIdx.x * blockDim.x;
    const int y0 = blockIdx.y * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int bsz = blockDim.x * blockDim.y;

    // load full input tile and halos
    // Clamp edges
    for (int i = tid; i < rows * in_w; i += bsz)
    {
        int r = i / in_w, c = i % in_w;
        int gx = clamp_coord(x0 + c - radius, W);
        int gy = clamp_coord(y0 + r - radius, H);
        smem_in[i] = src[gy * W + gx];
    }
    __syncthreads();

    // horizontal convolution for every row
    // Result stored in smem_hc
    for (int i = tid; i < rows * (int)blockDim.x; i += bsz)
    {
        int r = i / blockDim.x, c = i % blockDim.x;
        float sum = 0.0f;
        for (int k = 0; k <= 2 * radius; ++k)
            sum += c_gaussian[k] * smem_in[r * in_w + c + k];

        smem_hc[r * blockDim.x + c] = sum;
    }
    __syncthreads();

    // vertical convolution, one output pixel per thread.
    int x = x0 + threadIdx.x;
    int y = y0 + threadIdx.y;
    if (x >= W || y >= H)
        return;

    float sum = 0.0f;
    for (int k = 0; k <= 2 * radius; ++k)
        sum += c_gaussian[k] * smem_hc[(threadIdx.y + k) * blockDim.x + threadIdx.x];

    dst[y * W + x] = sum;
}

// ─── Histogram for Otsu ───────
__global__ void histogram_kernel(const float *__restrict__ luma,
                                 int *hist, int N)
{
    __shared__ int s_hist[HIST_BINS];

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    for (int i = tid; i < HIST_BINS; i += block_size)
        s_hist[i] = 0;
    __syncthreads();

    // Each thread processes 4 pixels
    int idx4 = blockIdx.x * block_size + tid; // group-of-4 index
    if (idx4 * 4 < N)
    {
        float4 v = reinterpret_cast<const float4 *>(luma)[idx4];
        atomicAdd(&s_hist[min(255, max(0, (int)v.x))], 1);
        atomicAdd(&s_hist[min(255, max(0, (int)v.y))], 1);
        atomicAdd(&s_hist[min(255, max(0, (int)v.z))], 1);
        atomicAdd(&s_hist[min(255, max(0, (int)v.w))], 1);
    }
    __syncthreads();

    for (int i = tid; i < HIST_BINS; i += block_size)
        atomicAdd(&hist[i], s_hist[i]);
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

__global__ void apply_threshold_kernel(const float *__restrict__ luma,
                                       uint8_t *__restrict__ binary,
                                       int W, int H, float threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // group-of-4 index
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx * 4 >= W || y >= H)
        return;

    float4 v = reinterpret_cast<const float4 *>(luma)[y * (W / 4) + idx];

    uchar4 result;
    result.x = v.x > threshold ? 255 : 0;
    result.y = v.y > threshold ? 255 : 0;
    result.z = v.z > threshold ? 255 : 0;
    result.w = v.w > threshold ? 255 : 0;

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

// Decodes a JPEG directly to device memory (d_rgb) using nvJPEG.
// Returns image dimensions via W, H.
static void nvjpeg_decode(const std::vector<uint8_t> &jpeg, pipeline_t &pipe, int &W, int &H)
{
    nvjpegHandle_t handle;
    nvjpegJpegState_t state;
    CUDA_CHECK((cudaError_t)nvjpegCreate(NVJPEG_BACKEND_DEFAULT, nullptr, &handle));
    CUDA_CHECK((cudaError_t)nvjpegJpegStateCreate(handle, &state));

    // Query image dimensions before decoding
    int ncomp;
    nvjpegChromaSubsampling_t css;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    nvjpegGetImageInfo(handle, jpeg.data(), jpeg.size(), &ncomp, &css, widths, heights);
    W = widths[0];
    H = heights[0];

    pipe = alloc_pipeline(W, H);

    // Point nvJPEG output at d_rgb — interleaved RGB matches uchar3 layout
    nvjpegImage_t img{};
    img.channel[0] = reinterpret_cast<uint8_t *>(pipe.d_rgb);
    img.pitch[0] = W * 3;

    TIMER_START(nvjpegdecode);
    CUDA_CHECK((cudaError_t)nvjpegDecode(handle, state, jpeg.data(), jpeg.size(),
                                         NVJPEG_OUTPUT_RGBI, &img, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    TIMER_STOP(nvjpegdecode, "nvJPEG decode");

    nvjpegJpegStateDestroy(state);
    nvjpegDestroy(handle);
}

static void jpeg_load(const std::vector<uint8_t> &jpeg, pipeline_t &pipe, int &W, int &H)
{
    int n;
    unsigned char *data = stbi_load_from_memory(jpeg.data(), jpeg.size(), &W, &H, &n, 3);
    if (!data)
        throw std::runtime_error("Failed to decode image with stb_image");

    pipe = alloc_pipeline(W, H);
    CUDA_CHECK(cudaMemcpy(pipe.d_rgb, data, (size_t)W * H * 3, cudaMemcpyHostToDevice));
    stbi_image_free(data);
}

// MEMCPY
static void download_result(const pipeline_t &p, grayscale_image_t &out)
{
    size_t N = (size_t)p.width * p.height;
    out.width = p.width;
    out.height = p.height;
    out.data.resize(N);
    CUDA_CHECK(cudaMemcpy(out.data.data(), p.d_binary, N, cudaMemcpyDeviceToHost));
}

static void download_rgb_result(const pipeline_t &p, rgb_image_t &out)
{
    size_t N = (size_t)p.width * p.height;
    out.width = p.width;
    out.height = p.height;
    out.data.resize(N * 3);
    CUDA_CHECK(cudaMemcpy(out.data.data(), p.d_rgb, N * 3, cudaMemcpyDeviceToHost));
}

rgb_image_t preprocessing(const std::string &imagePath, uint8_t *&d_result)
{
    TIMER_START(imageread);
    int W, H;
    auto jpeg = read_file(imagePath);
    pipeline_t pipe{};
    nvjpeg_decode(jpeg, pipe, W, H);
    TIMER_STOP(imageread, "image read + decode");
    printf("[image] %s - %d x %d\n", imagePath.c_str(), W, H);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    //  RGB → luma
    TIMER_START(total);
    TIMER_START(luma);
    rgba_to_luma_kernel<<<grid, block>>>(pipe.d_rgb, pipe.d_gray, W, H);
    TIMER_STOP(luma, "rgba_to_luma");

    TIMER_START(download_rgb);
    rgb_image_t input_rgb;
    download_rgb_result(pipe, input_rgb);
    TIMER_STOP(download_rgb, "download RGB result");

    cudaFree(pipe.d_rgb);
    pipe.d_rgb = nullptr;

    // Gaussian blur
    const int radius = build_gaussian_kernel(GAUSSIAN_SIGMA);

    const int rows_gauss = block.y + 2 * radius;
    size_t smem_gauss = (size_t)rows_gauss * (block.x + 2 * radius + block.x) * sizeof(float);

    TIMER_START(gauss);
    gaussian_fused<<<grid, block, smem_gauss>>>(pipe.d_gray, pipe.d_gray, W, H, radius);
    TIMER_STOP(gauss, "gaussian_fused");

    // Otsu threshold
    const int num_pixels = W * H;
    int *d_hist;
    float *d_threshold;
    // Allocate
    CUDA_CHECK(cudaMalloc(&d_hist, HIST_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_threshold, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_hist, 0, HIST_BINS * sizeof(int)));

    TIMER_START(hist);
    histogram_kernel<<<(num_pixels / 4 + HIST_BLOCK - 1) / HIST_BLOCK, HIST_BLOCK>>>(pipe.d_gray, d_hist, num_pixels);
    TIMER_STOP(hist, "histogram");

    TIMER_START(otsu);
    otsu_kernel<<<1, HIST_BINS>>>(d_hist, num_pixels, d_threshold);
    TIMER_STOP(otsu, "otsu_gpu");
    cudaFree(d_hist);

    float h_threshold;
    CUDA_CHECK(cudaMemcpy(&h_threshold, d_threshold, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_threshold);

#ifdef PREPROC_TIMING
    printf("[timing] Otsu threshold = %.1f\n", h_threshold);
#endif

    dim3 thresh_grid((W / 4 + block.x - 1) / block.x, grid.y);
    TIMER_START(thresh);
    apply_threshold_kernel<<<thresh_grid, block>>>(pipe.d_gray, pipe.d_binary, W, H, h_threshold);
    TIMER_STOP(thresh, "apply_threshold");

    TIMER_STOP(total, "── total without transfers");
    // save_uint8_stage(pipe.d_binary, W, H, "data/processed//main/debug_3_binary.png"); // final stage 1 output

    // TIMER_START(download);
    // grayscale_image_t result;
    // download_result(pipe, result);
    // TIMER_STOP(download, "download result");
    free_pipeline(pipe, true, true, false); // keep binary for further processing in main
    d_result = pipe.d_binary;

    return input_rgb;
}
