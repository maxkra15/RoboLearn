// SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
// SPDX-License-Identifier: MIT
// Cooperative FlashSAC categorical heads and distribution projection.
#pragma once

#include <math_constants.h>

namespace {
namespace categorical_native {

template<bool Maximum>
__device__ float reduce(float value, float* shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset; offset >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, value, offset);
        value = Maximum ? fmaxf(value, other) : value + other;
    }
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = threadIdx.x < blockDim.x / 32 ? shared[lane] : (Maximum ? -CUDART_INF_F : 0.0f);
    if (warp == 0) {
        for (int offset = 16; offset; offset >>= 1) {
            const float other = __shfl_down_sync(0xffffffff, value, offset);
            value = Maximum ? fmaxf(value, other) : value + other;
        }
        if (lane == 0) shared[0] = value;
    }
    __syncthreads();
    const float result = shared[0];
    __syncthreads();
    return result;
}

__global__ void forward(
    const float* raw, const float* bias, const float* support,
    float* log_probs, float* values, int bins) {
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * bins;
    __shared__ float shared[8];
    float maximum = -CUDART_INF_F;
    for (int j = threadIdx.x; j < bins; j += blockDim.x) maximum = fmaxf(maximum, raw[base + j] + bias[j]);
    maximum = reduce<true>(maximum, shared);
    float total = 0.0f;
    for (int j = threadIdx.x; j < bins; j += blockDim.x) total += expf(raw[base + j] + bias[j] - maximum);
    total = logf(reduce<false>(total, shared));
    float expectation = 0.0f;
    for (int j = threadIdx.x; j < bins; j += blockDim.x) {
        const float lp = raw[base + j] + bias[j] - maximum - total;
        log_probs[base + j] = lp;
        expectation += expf(lp) * support[j];
    }
    expectation = reduce<false>(expectation, shared);
    if (threadIdx.x == 0) values[blockIdx.x] = expectation;
}

__global__ void backward(
    const float* log_probs, const float* support, const float* dlog_probs,
    const float* dvalues, float* scratch, float* draw, int bins) {
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * bins;
    __shared__ float shared[8];
    const float dq = dvalues ? dvalues[blockIdx.x] : 0.0f;
    float total = 0.0f;
    for (int j = threadIdx.x; j < bins; j += blockDim.x) {
        const float probability = expf(log_probs[base + j]);
        total += (dlog_probs ? dlog_probs[base + j] : 0.0f) + dq * probability * support[j];
    }
    total = reduce<false>(total, shared);
    for (int j = threadIdx.x; j < bins; j += blockDim.x) {
        const float probability = expf(log_probs[base + j]);
        const float gradient = (dlog_probs ? dlog_probs[base + j] : 0.0f)
            + probability * (dq * support[j] - total);
        if (draw) draw[base + j] += gradient;
        if (scratch) scratch[base + j] = gradient;
    }
}

__device__ void maximum(float* address, float value) {
    auto* bits = reinterpret_cast<unsigned int*>(address);
    unsigned int old = *bits;
    while (__uint_as_float(old) < value) {
        const unsigned int expected = old;
        old = atomicCAS(bits, expected, __float_as_uint(value));
        if (old == expected) break;
    }
}

__global__ void project_target(
    const float* q, const float* log_probs, const float* reward, const float* terminated,
    const float* discount, const float* next_log_probs, const float* alpha,
    float* target, float* maximum_entropy, int batch, int bins, float min_v, float max_v) {
    extern __shared__ float projected[];
    const int row = blockIdx.x;
    const float entropy = alpha[0] * next_log_probs[row];
    const int selected = q[3 * batch + row] < q[batch + row] ? 1 : 0;
    const std::size_t source = (static_cast<std::size_t>(selected) * (2 * batch) + batch + row) * bins;
    const float width = (max_v - min_v) / static_cast<float>(bins - 1);
    for (int j = threadIdx.x; j < bins; j += blockDim.x) projected[j] = 0.0f;
    __syncthreads();
    for (int j = threadIdx.x; j < bins; j += blockDim.x) {
        const float support = min_v + static_cast<float>(j) * width;
        const float value = reward[row] + discount[row] * (support - entropy) * (1.0f - terminated[row]);
        const float location = (fminf(fmaxf(value, min_v), max_v) - min_v) / width;
        const int lower = min(max(static_cast<int>(floorf(location)), 0), bins - 1);
        const int upper = min(lower + 1, bins - 1);
        const float fraction = location - static_cast<float>(lower);
        const float probability = expf(log_probs[source + j]);
        atomicAdd(projected + lower, probability * (1.0f - fraction));
        atomicAdd(projected + upper, probability * fraction);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < bins; j += blockDim.x)
        target[static_cast<std::size_t>(row) * bins + j] = projected[j];
    if (threadIdx.x == 0) maximum(maximum_entropy, entropy);
}

inline int status(const char* operation) {
    const cudaError_t error = cudaPeekAtLastError();
    if (error == cudaSuccess) return 0;
    set_error("%s: %s", operation, cudaGetErrorString(error));
    return -1;
}

}  // namespace categorical_native
}  // namespace

extern "C" int rl_flash_categorical_forward(
    const void* raw, const void* bias, const void* support, void* log_probs,
    void* values, int batch, int bins, void* stream) noexcept {
    last_error[0] = '\0';
    if (!raw || !bias || !support || !log_probs || !values || batch < 1 || bins < 2) {
        set_error("Categorical forward requires nonempty FP32 buffers and at least two bins.");
        return -1;
    }
    categorical_native::forward<<<batch, bins <= 128 ? 128 : 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(raw), static_cast<const float*>(bias), static_cast<const float*>(support),
        static_cast<float*>(log_probs), static_cast<float*>(values), bins);
    return categorical_native::status("Categorical forward");
}

extern "C" int rl_flash_categorical_backward(
    const void* log_probs, const void* support, const void* dlog_probs, const void* dvalues,
    void* scratch, void* draw, void* dbias, int batch, int bins, void* stream) noexcept {
    last_error[0] = '\0';
    if (!log_probs || !support || (!draw && !dbias) || (dbias && !scratch) || batch < 1 || bins < 2) {
        set_error("Categorical backward requires FP32 buffers, gradients and valid dimensions.");
        return -1;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    categorical_native::backward<<<batch, bins <= 128 ? 128 : 256, 0, cuda_stream>>>(
        static_cast<const float*>(log_probs), static_cast<const float*>(support),
        static_cast<const float*>(dlog_probs), static_cast<const float*>(dvalues),
        dbias ? static_cast<float*>(scratch) : nullptr, static_cast<float*>(draw), bins);
    if (categorical_native::status("Categorical backward") != 0) return -1;
    if (dbias) bias_gradient<<<bins, kThreads, 0, cuda_stream>>>(
        static_cast<const float*>(scratch), static_cast<float*>(dbias), batch, bins);
    return categorical_native::status("Categorical bias backward");
}

extern "C" int rl_flash_categorical_target(
    const void* q, const void* log_probs, const void* reward, const void* terminated,
    const void* discount, const void* next_log_probs, const void* alpha,
    void* target, void* maximum_entropy, int batch, int bins, float min_v, float max_v, void* stream) noexcept {
    last_error[0] = '\0';
    if (!q || !log_probs || !reward || !terminated || !discount || !next_log_probs || !alpha ||
        !target || !maximum_entropy || batch < 1 || bins < 2 || bins > 4096 || !(min_v < max_v)) {
        set_error("Categorical target requires FP32 buffers, valid support and 2..4096 bins.");
        return -1;
    }
    categorical_native::project_target<<<batch, bins <= 128 ? 128 : 256,
        static_cast<std::size_t>(bins) * sizeof(float), reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(q), static_cast<const float*>(log_probs), static_cast<const float*>(reward),
        static_cast<const float*>(terminated), static_cast<const float*>(discount),
        static_cast<const float*>(next_log_probs), static_cast<const float*>(alpha),
        static_cast<float*>(target), static_cast<float*>(maximum_entropy), batch, bins, min_v, max_v);
    return categorical_native::status("Categorical target");
}
