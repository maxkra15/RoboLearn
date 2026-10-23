// SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
// SPDX-License-Identifier: MIT
// Original FP32 FlashSAC normalization, parameter projection and target EMA.
// Include at the end of kernels.cu; use its common error/device helpers.
//
// Persistent-plan C ABI (pointer tables/lengths/kinds are host arrays copied at creation):
// void* rl_flash_projection_create(size_t count, const void* const* first,
//     const void* const* second, const size_t* lengths, const int* kinds);
// int rl_flash_projection_step(void* plan, void* stream);
// int rl_flash_projection_destroy(void* plan);
// void* rl_flash_projection_create_with_shadows(size_t count, const void* const* first,
//     const void* const* second, const size_t* lengths, const int* kinds,
//     const void* const* shadows);  // optional FP16 unit-linear row destinations
// kinds: 0=unit Linear row, 1=joint BN affine vector, 2=RMS scale vector.
// void* rl_flash_ema_create(size_t count, const void* const* source,
//     const void* const* target, const size_t* lengths, float tau);
// int rl_flash_ema_step(void* plan, void* stream);
// int rl_flash_ema_destroy(void* plan);
// void* rl_flash_ema_create_with_shadows(size_t count, const void* const* source,
//     const void* const* target, const size_t* lengths, float tau,
//     const void* const* shadows);  // optional FP16 target destinations
//
// Allocation-free normalization ABI (row-major B x C, all arrays FP32):
// int rl_flash_bn_forward(x,w,b,running_mean,running_var,mean,var,partial,y,
//     int B,int C,float eps,float momentum,int training,int relu,void* stream);
// int rl_flash_bn_backward(x,dy,y,w,mean,var,partial,sums,dx,dw,db,
//     int B,int C,float eps,int training,int relu,void* stream);
// int rl_flash_rms_forward(x,w,var,y,int B,int C,float eps,void* stream);
// int rl_flash_rms_backward(x,dy,w,var,partial,dx,dw,int B,int C,float eps,void* stream);
// Forward partial/scratch: BN 2*ceil(B/256)*C; RMS var B.
// Backward scratch: BN partial 2*ceil(B/256)*C and sums 2*C;
// RMS partial ceil(B/256)*C. Unused gradient pointers may be null.
// BN biased variance is used for normalization; running variance uses B/(B-1).
// Gradients accumulate into dx/dw/db, never overwrite other tape contributions.
// All run calls enqueue on the supplied stream without allocation or sync.
// Plan create/destroy must be outside capture. Preserve buffers/plan lifetime.

#pragma once

#include <cuda_fp16.h>

namespace {
namespace flash_native {

constexpr int Threads = 256;
constexpr int Channels = 32;
constexpr int Groups = Threads / Channels;
constexpr int Chunk = 256;
constexpr int ApplyRows = 64;

struct Projection {
    float* first;
    float* second;
    std::size_t length;
    int kind;
    __half* shadow;
};

struct Pair {
    const float* source;
    float* target;
    std::size_t length;
    __half* shadow;
};

template<class Descriptor>
struct Plan {
    Descriptor* descriptors = nullptr;
    unsigned count = 0;
    float tau = 0.0f;

    int close(bool report = false) noexcept {
        const cudaError_t status = descriptors ? cudaFree(descriptors) : cudaSuccess;
        descriptors = nullptr;
        if (status != cudaSuccess && report) set_error("Destroy FlashSAC plan: %s", cudaGetErrorString(status));
        return status == cudaSuccess ? 0 : -1;
    }

    ~Plan() { close(); }
};

template<class Descriptor>
std::unique_ptr<Plan<Descriptor>> create_plan(const std::vector<Descriptor>& descriptors) {
    require(!descriptors.empty() && descriptors.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
            "FlashSAC descriptor count must fit a nonempty CUDA grid.");
    auto plan = std::make_unique<Plan<Descriptor>>();
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&plan->descriptors), descriptors.size() * sizeof(Descriptor)),
               "Allocate FlashSAC descriptors");
    check_cuda(cudaMemcpy(plan->descriptors, descriptors.data(), descriptors.size() * sizeof(Descriptor), cudaMemcpyHostToDevice),
               "Copy FlashSAC descriptors");
    plan->count = static_cast<unsigned>(descriptors.size());
    return plan;
}

__device__ float block_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = Threads / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    return scratch[0];
}

__global__ void project(const Projection* descriptors) {
    const Projection row = descriptors[blockIdx.x];
    float value = 0.0f;
    for (std::size_t i = threadIdx.x; i < row.length; i += Threads) {
        value += row.first[i] * row.first[i];
        if (row.kind == 1) value += row.second[i] * row.second[i];
    }
    __shared__ float scratch[Threads];
    const float total = block_sum(value, scratch);
    __shared__ float factor;
    if (threadIdx.x == 0) factor = row.kind == 0 ? fmaxf(sqrtf(total), 1.0e-8f)
        : sqrtf(static_cast<float>(row.length)) / sqrtf(total + 1.0e-8f);
    __syncthreads();
    for (std::size_t i = threadIdx.x; i < row.length; i += Threads) {
        const float normalized = row.kind == 0 ? row.first[i] / factor : row.first[i] * factor;
        row.first[i] = normalized;
        if (row.shadow) row.shadow[i] = __float2half_rn(normalized);
        if (row.kind == 1) row.second[i] *= factor;
    }
}

__global__ void ema(const Pair* descriptors, float tau) {
    const Pair values = descriptors[blockIdx.x];
    for (std::size_t i = threadIdx.x; i < values.length; i += Threads) {
        const float updated = (1.0f - tau) * values.target[i] + tau * values.source[i];
        values.target[i] = updated;
        if (values.shadow) values.shadow[i] = __float2half_rn(updated);
    }
}

// The channel tile gives coalesced loads. Eight lane groups cooperate over
// batch rows, then combine Welford moments in a fixed order without atomics.
__device__ void combine(float incoming_mean, float incoming_m2, int incoming_count,
                        float& mean, float& m2, int& count) {
    if (incoming_count == 0) return;
    if (count == 0) {
        mean = incoming_mean;
        m2 = incoming_m2;
        count = incoming_count;
        return;
    }
    const int total = count + incoming_count;
    const float delta = incoming_mean - mean;
    mean += delta * (static_cast<float>(incoming_count) / static_cast<float>(total));
    m2 += incoming_m2 + delta * delta * (static_cast<float>(count) * static_cast<float>(incoming_count) / static_cast<float>(total));
    count = total;
}

__global__ void bn_moments(const float* x, float* partial, int batch, int channels, int chunks) {
    const int lane = threadIdx.x % Channels;
    const int group = threadIdx.x / Channels;
    const int channel = blockIdx.x * Channels + lane;
    const int start = blockIdx.y * Chunk;
    const int end = start + Chunk < batch ? start + Chunk : batch;
    float mean = 0.0f, m2 = 0.0f;
    int count = 0;
    for (int row = start + group; row < end; row += Groups) {
        const float value = channel < channels ? x[static_cast<std::size_t>(row) * channels + channel] : 0.0f;
        ++count;
        const float delta = value - mean;
        mean += delta / static_cast<float>(count);
        m2 += delta * (value - mean);
    }
    __shared__ float means[Threads], squares[Threads];
    __shared__ int counts[Threads];
    means[threadIdx.x] = mean;
    squares[threadIdx.x] = m2;
    counts[threadIdx.x] = count;
    __syncthreads();
    if (group == 0 && channel < channels) {
        mean = 0.0f; m2 = 0.0f; count = 0;
        for (int other = 0; other < Groups; ++other) {
            const int index = other * Channels + lane;
            combine(means[index], squares[index], counts[index], mean, m2, count);
        }
        partial[blockIdx.y * channels + channel] = mean;
        partial[(chunks + blockIdx.y) * channels + channel] = m2;
    }
}

__global__ void bn_statistics(const float* partial, float* mean, float* variance,
                              float* running_mean, float* running_var, int batch, int channels, int chunks, float momentum) {
    const int channel = blockIdx.x * Threads + threadIdx.x;
    if (channel >= channels) return;
    float average = 0.0f, m2 = 0.0f;
    int count = 0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int length = batch - chunk * Chunk < Chunk ? batch - chunk * Chunk : Chunk;
        combine(partial[chunk * channels + channel], partial[(chunks + chunk) * channels + channel], length,
                average, m2, count);
    }
    const float var = m2 / static_cast<float>(batch);
    mean[channel] = average;
    variance[channel] = var;
    if (running_mean && running_var) {
        const float unbiased = var * static_cast<float>(batch) / static_cast<float>(batch - 1);
        running_mean[channel] = (1.0f - momentum) * running_mean[channel] + momentum * average;
        running_var[channel] = (1.0f - momentum) * running_var[channel] + momentum * unbiased;
    }
}

__global__ void bn_apply(const float* x, const float* weight, const float* bias, const float* mean,
                         const float* variance, float* output, int batch, int channels, float eps, bool relu) {
    const int lane = threadIdx.x % Channels, group = threadIdx.x / Channels;
    const int channel = blockIdx.x * Channels + lane;
    __shared__ float averages[Channels], deviations[Channels], weights[Channels], biases[Channels];
    if (group == 0 && channel < channels) {
        averages[lane] = mean[channel];
        deviations[lane] = sqrtf(variance[channel] + eps);
        weights[lane] = weight ? weight[channel] : 1.0f;
        biases[lane] = bias ? bias[channel] : 0.0f;
    }
    __syncthreads();
    if (channel >= channels) return;
    const int end = (blockIdx.y + 1) * ApplyRows < batch ? (blockIdx.y + 1) * ApplyRows : batch;
    for (int row = blockIdx.y * ApplyRows + group; row < end; row += Groups) {
        const std::size_t index = static_cast<std::size_t>(row) * channels + channel;
        float value = (x[index] - averages[lane]) / deviations[lane];
        if (weight) value *= weights[lane];
        if (bias) value += biases[lane];
        output[index] = relu && value <= 0.0f ? 0.0f : value;
    }
}

__global__ void bn_gradient_partials(const float* x, const float* dy, const float* output, const float* mean,
                                     const float* variance, float* partial, int batch, int channels, int chunks,
                                     float eps, bool relu) {
    const int lane = threadIdx.x % Channels, group = threadIdx.x / Channels;
    const int channel = blockIdx.x * Channels + lane;
    const int end = (blockIdx.y + 1) * Chunk < batch ? (blockIdx.y + 1) * Chunk : batch;
    float sum = 0.0f, sum_x = 0.0f;
    if (channel < channels) {
        const float inverse = 1.0f / sqrtf(variance[channel] + eps);
        for (int row = blockIdx.y * Chunk + group; row < end; row += Groups) {
            const std::size_t index = static_cast<std::size_t>(row) * channels + channel;
            const float gradient = relu && output[index] <= 0.0f ? 0.0f : dy[index];
            const float normalized = (x[index] - mean[channel]) * inverse;
            sum += gradient;
            sum_x += gradient * normalized;
        }
    }
    __shared__ float sums[Threads], xs[Threads];
    sums[threadIdx.x] = sum;
    xs[threadIdx.x] = sum_x;
    __syncthreads();
    if (group == 0 && channel < channels) {
        sum = 0.0f; sum_x = 0.0f;
        for (int other = 0; other < Groups; ++other) {
            sum += sums[other * Channels + lane];
            sum_x += xs[other * Channels + lane];
        }
        partial[blockIdx.y * channels + channel] = sum;
        partial[(chunks + blockIdx.y) * channels + channel] = sum_x;
    }
}

// Reduce the existing chunk partials in their original order, then reuse the
// per-channel statistics for a coalesced batch tile. Only tile zero writes the
// affine adjoints, so the operation needs no atomics or separate reduce launch.
__global__ void bn_backward_apply(const float* x, const float* dy, const float* output, const float* weight,
                                  const float* mean, const float* variance, const float* partial, float* sums,
                                  float* dx, float* dw, float* db, int batch, int channels, int chunks,
                                  float eps, bool training, bool relu) {
    const int lane = threadIdx.x % Channels, group = threadIdx.x / Channels;
    const int channel = blockIdx.x * Channels + lane;
    __shared__ float averages[Channels], inverses[Channels], weights[Channels];
    __shared__ float gradient_means[Channels], normalized_means[Channels];
    if (group == 0 && channel < channels) {
        averages[lane] = mean[channel];
        inverses[lane] = 1.0f / sqrtf(variance[channel] + eps);
        weights[lane] = weight ? weight[channel] : 1.0f;
        float sum = 0.0f, sum_x = 0.0f;
        if (training || dw || db) {
            for (int chunk = 0; chunk < chunks; ++chunk) {
                sum += partial[chunk * channels + channel];
                sum_x += partial[(chunks + chunk) * channels + channel];
            }
        }
        gradient_means[lane] = sum / static_cast<float>(batch);
        normalized_means[lane] = sum_x / static_cast<float>(batch);
        if (blockIdx.y == 0) {
            sums[channel] = gradient_means[lane];
            sums[channels + channel] = normalized_means[lane];
            if (dw) dw[channel] += sum_x;
            if (db) db[channel] += sum;
        }
    }
    __syncthreads();
    if (!dx || channel >= channels) return;
    const int end = (blockIdx.y + 1) * ApplyRows < batch ? (blockIdx.y + 1) * ApplyRows : batch;
    for (int row = blockIdx.y * ApplyRows + group; row < end; row += Groups) {
        const std::size_t index = static_cast<std::size_t>(row) * channels + channel;
        float gradient = relu && output[index] <= 0.0f ? 0.0f : dy[index];
        if (training) gradient = gradient - gradient_means[lane]
            - (x[index] - averages[lane]) * inverses[lane] * normalized_means[lane];
        dx[index] += weights[lane] * inverses[lane] * gradient;
    }
}

__global__ void rms_forward(const float* x, const float* weight, float* variance, float* output, int channels, float eps) {
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * channels;
    float squares = 0.0f;
    for (int channel = threadIdx.x; channel < channels; channel += Threads) squares += x[offset + channel] * x[offset + channel];
    __shared__ float scratch[Threads];
    const float var = block_sum(squares, scratch) / static_cast<float>(channels);
    if (threadIdx.x == 0) variance[blockIdx.x] = var;
    for (int channel = threadIdx.x; channel < channels; channel += Threads) {
        output[offset + channel] = (x[offset + channel] / sqrtf(var + eps)) * (weight ? weight[channel] : 1.0f);
    }
}

__global__ void rms_gradient_partials(const float* x, const float* dy, const float* variance,
                                      float* partial, int batch, int channels, float eps) {
    const int lane = threadIdx.x % Channels, group = threadIdx.x / Channels;
    const int channel = blockIdx.x * Channels + lane;
    const int end = (blockIdx.y + 1) * Chunk < batch ? (blockIdx.y + 1) * Chunk : batch;
    float sum = 0.0f;
    if (channel < channels) {
        for (int row = blockIdx.y * Chunk + group; row < end; row += Groups) {
            const std::size_t index = static_cast<std::size_t>(row) * channels + channel;
            sum += dy[index] * (x[index] / sqrtf(variance[row] + eps));
        }
    }
    __shared__ float scratch[Threads];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    if (group == 0 && channel < channels) {
        sum = 0.0f;
        for (int other = 0; other < Groups; ++other) sum += scratch[other * Channels + lane];
        partial[blockIdx.y * channels + channel] = sum;
    }
}

__global__ void rms_dw(const float* partial, float* dw, int channels, int chunks) {
    const int channel = blockIdx.x * Threads + threadIdx.x;
    if (channel >= channels) return;
    float sum = 0.0f;
    for (int chunk = 0; chunk < chunks; ++chunk) sum += partial[chunk * channels + channel];
    dw[channel] += sum;
}

__global__ void rms_dx(const float* x, const float* dy, const float* weight, const float* variance,
                       float* dx, int channels, float eps) {
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * channels;
    float sum = 0.0f;
    for (int channel = threadIdx.x; channel < channels; channel += Threads) {
        sum += dy[offset + channel] * (weight ? weight[channel] : 1.0f) * x[offset + channel];
    }
    __shared__ float scratch[Threads];
    const float mean = block_sum(sum, scratch) / static_cast<float>(channels);
    const float inverse = 1.0f / sqrtf(variance[blockIdx.x] + eps);
    for (int channel = threadIdx.x; channel < channels; channel += Threads) {
        float gradient = dy[offset + channel] * (weight ? weight[channel] : 1.0f);
        gradient -= x[offset + channel] * inverse * inverse * mean;
        dx[offset + channel] += inverse * gradient;
    }
}

int launched(const char* operation) noexcept {
    const cudaError_t status = cudaPeekAtLastError();
    if (status == cudaSuccess) return 0;
    set_error("%s: %s", operation, cudaGetErrorString(status));
    return -1;
}

bool dimensions(int batch, int channels, float eps) noexcept {
    if (batch > 0 && channels > 0 && std::isfinite(eps) && eps > 0.0f &&
        static_cast<std::size_t>(batch) * channels <= static_cast<std::size_t>(std::numeric_limits<int>::max())) return true;
    set_error("FlashSAC normalization requires positive finite epsilon and B*C fitting signed int.");
    return false;
}

}  // namespace flash_native
}  // namespace

static void* create_flash_projection(std::size_t count, const void* const* first, const void* const* second,
                                     const std::size_t* lengths, const int* kinds, const void* const* shadows) noexcept {
    last_error[0] = '\0';
    try {
        require(count && first && second && lengths && kinds, "Invalid projection metadata.");
        std::vector<flash_native::Projection> descriptors;
        descriptors.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            require(first[i] && lengths[i] && kinds[i] >= 0 && kinds[i] <= 2 && (kinds[i] != 1 || second[i]),
                    "Invalid projection row/vector.");
            require(!shadows || !shadows[i] || (kinds[i] == 0 && aligned(shadows[i], alignof(__half))),
                    "Projection shadows require aligned FP16 unit-linear row destinations.");
            descriptors.push_back({static_cast<float*>(const_cast<void*>(first[i])),
                                   static_cast<float*>(const_cast<void*>(second[i])), lengths[i], kinds[i],
                                   shadows ? static_cast<__half*>(const_cast<void*>(shadows[i])) : nullptr});
        }
        return flash_native::create_plan(descriptors).release();
    } catch (const Failure&) { return nullptr; }
    catch (...) { set_error("Unexpected host error creating projection plan."); return nullptr; }
}

extern "C" void* rl_flash_projection_create(std::size_t count, const void* const* first, const void* const* second,
                                            const std::size_t* lengths, const int* kinds) noexcept {
    return create_flash_projection(count, first, second, lengths, kinds, nullptr);
}

extern "C" void* rl_flash_projection_create_with_shadows(std::size_t count, const void* const* first,
                                                         const void* const* second, const std::size_t* lengths,
                                                         const int* kinds, const void* const* shadows) noexcept {
    return create_flash_projection(count, first, second, lengths, kinds, shadows);
}

extern "C" int rl_flash_projection_step(void* opaque, void* stream) noexcept {
    last_error[0] = '\0';
    if (!opaque) { set_error("Projection plan is null."); return -1; }
    auto* plan = static_cast<flash_native::Plan<flash_native::Projection>*>(opaque);
    flash_native::project<<<plan->count, flash_native::Threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(plan->descriptors);
    return flash_native::launched("Run FlashSAC projections");
}

extern "C" int rl_flash_projection_destroy(void* opaque) noexcept {
    last_error[0] = '\0';
    auto* plan = static_cast<flash_native::Plan<flash_native::Projection>*>(opaque);
    if (!plan) return 0;
    const int result = plan->close(true);
    delete plan;
    return result;
}

static void* create_flash_ema(std::size_t count, const void* const* source, const void* const* target,
                              const std::size_t* lengths, float tau, const void* const* shadows) noexcept {
    last_error[0] = '\0';
    try {
        require(count && source && target && lengths && std::isfinite(tau) && tau >= 0.0f && tau <= 1.0f,
                "Invalid EMA metadata/tau.");
        std::vector<flash_native::Pair> descriptors;
        for (std::size_t i = 0; i < count; ++i) {
            require(source[i] && target[i] && lengths[i], "Invalid EMA tensor.");
            require(!shadows || !shadows[i] || aligned(shadows[i], alignof(__half)),
                    "EMA shadows require aligned FP16 destinations.");
            for (std::size_t offset = 0; offset < lengths[i]; offset += 1024) {
                const std::size_t length = lengths[i] - offset < 1024 ? lengths[i] - offset : 1024;
                descriptors.push_back({static_cast<const float*>(source[i]) + offset,
                                       static_cast<float*>(const_cast<void*>(target[i])) + offset, length,
                                       shadows && shadows[i] ? static_cast<__half*>(const_cast<void*>(shadows[i])) + offset : nullptr});
            }
        }
        auto plan = flash_native::create_plan(descriptors);
        plan->tau = tau;
        return plan.release();
    } catch (const Failure&) { return nullptr; }
    catch (...) { set_error("Unexpected host error creating EMA plan."); return nullptr; }
}

extern "C" void* rl_flash_ema_create(std::size_t count, const void* const* source, const void* const* target,
                                     const std::size_t* lengths, float tau) noexcept {
    return create_flash_ema(count, source, target, lengths, tau, nullptr);
}

extern "C" void* rl_flash_ema_create_with_shadows(std::size_t count, const void* const* source,
                                                  const void* const* target, const std::size_t* lengths,
                                                  float tau, const void* const* shadows) noexcept {
    return create_flash_ema(count, source, target, lengths, tau, shadows);
}

extern "C" int rl_flash_ema_step(void* opaque, void* stream) noexcept {
    last_error[0] = '\0';
    if (!opaque) { set_error("EMA plan is null."); return -1; }
    auto* plan = static_cast<flash_native::Plan<flash_native::Pair>*>(opaque);
    flash_native::ema<<<plan->count, flash_native::Threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(plan->descriptors, plan->tau);
    return flash_native::launched("Run FlashSAC target EMA");
}

extern "C" int rl_flash_ema_destroy(void* opaque) noexcept {
    last_error[0] = '\0';
    auto* plan = static_cast<flash_native::Plan<flash_native::Pair>*>(opaque);
    if (!plan) return 0;
    const int result = plan->close(true);
    delete plan;
    return result;
}

extern "C" int rl_flash_bn_forward(const float* x, const float* w, const float* b, float* running_mean, float* running_var,
                                    float* mean, float* var, float* partial, float* y, int batch, int channels,
                                    float eps, float momentum, int training, int relu, void* stream) noexcept {
    last_error[0] = '\0';
    if (!flash_native::dimensions(batch, channels, eps)) return -1;
    if (!x || !y || !mean || !var || !partial || (!training && (!running_mean || !running_var)) ||
        (training && running_mean && batch < 2)) { set_error("Invalid BN forward arrays/count."); return -1; }
    const cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    const int chunks = (batch + flash_native::Chunk - 1) / flash_native::Chunk;
    if (training) {
        const dim3 grid((channels + 31) / 32, chunks);
        flash_native::bn_moments<<<grid, 256, 0, st>>>(x, partial, batch, channels, chunks);
        flash_native::bn_statistics<<<(channels + 255) / 256, 256, 0, st>>>(partial, mean, var, running_mean, running_var,
                                                                        batch, channels, chunks, momentum);
    }
    const dim3 apply_grid((channels + 31) / 32, (batch + flash_native::ApplyRows - 1) / flash_native::ApplyRows);
    flash_native::bn_apply<<<apply_grid, 256, 0, st>>>(x, w, b, training ? mean : running_mean,
                                                     training ? var : running_var, y, batch, channels, eps, relu != 0);
    return flash_native::launched("Run FlashSAC BatchNorm forward");
}

extern "C" int rl_flash_bn_backward(const float* x, const float* dy, const float* y, const float* w,
                                     const float* mean, const float* var, float* partial, float* sums,
                                     float* dx, float* dw, float* db, int batch, int channels, float eps,
                                     int training, int relu, void* stream) noexcept {
    last_error[0] = '\0';
    if (!flash_native::dimensions(batch, channels, eps)) return -1;
    if (!x || !dy || !y || !mean || !var || !partial || !sums) { set_error("Invalid BN backward arrays."); return -1; }
    const cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    const int chunks = (batch + 255) / 256;
    if (training || dw || db) {
        const dim3 grid((channels + 31) / 32, chunks);
        flash_native::bn_gradient_partials<<<grid, 256, 0, st>>>(x, dy, y, mean, var, partial, batch, channels, chunks, eps, relu != 0);
    }
    if (dx || dw || db) {
        const dim3 grid((channels + 31) / 32, dx ? (batch + flash_native::ApplyRows - 1) / flash_native::ApplyRows : 1);
        flash_native::bn_backward_apply<<<grid, 256, 0, st>>>(x, dy, y, w, mean, var, partial, sums, dx, dw, db,
                                                            batch, channels, chunks, eps, training != 0, relu != 0);
    }
    return flash_native::launched("Run FlashSAC BatchNorm backward");
}

extern "C" int rl_flash_rms_forward(const float* x, const float* w, float* var, float* y,
                                     int batch, int channels, float eps, void* stream) noexcept {
    last_error[0] = '\0';
    if (!flash_native::dimensions(batch, channels, eps)) return -1;
    if (!x || !var || !y) { set_error("Invalid RMSNorm forward arrays."); return -1; }
    flash_native::rms_forward<<<batch, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(x, w, var, y, channels, eps);
    return flash_native::launched("Run FlashSAC RMSNorm forward");
}

extern "C" int rl_flash_rms_backward(const float* x, const float* dy, const float* w, const float* var,
                                      float* partial, float* dx, float* dw, int batch, int channels, float eps, void* stream) noexcept {
    last_error[0] = '\0';
    if (!flash_native::dimensions(batch, channels, eps)) return -1;
    if (!x || !dy || !var || !partial) { set_error("Invalid RMSNorm backward arrays."); return -1; }
    const cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    const int chunks = (batch + 255) / 256;
    if (dw) {
        const dim3 grid((channels + 31) / 32, chunks);
        flash_native::rms_gradient_partials<<<grid, 256, 0, st>>>(x, dy, var, partial, batch, channels, eps);
        flash_native::rms_dw<<<(channels + 255) / 256, 256, 0, st>>>(partial, dw, channels, chunks);
    }
    if (dx) flash_native::rms_dx<<<batch, 256, 0, st>>>(x, dy, w, var, dx, channels, eps);
    return flash_native::launched("Run FlashSAC RMSNorm backward");
}
