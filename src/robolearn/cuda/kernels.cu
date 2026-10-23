// SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
// SPDX-License-Identifier: MIT
//
// C ABI (all matrices/device state FP32; transpose flags are 0=N, 1=T):
// const char* rl_last_error();
// void* rl_plan_create(int m, int n, int k, int transa, int transb,
//                      int lda, int ldb, int ldd, int use_bias, int allow_tf32,
//                      size_t workspace_limit);
// size_t rl_plan_workspace(void* plan);  // SIZE_MAX on error
// int rl_plan_run(void* plan, const void* A, const void* B, const void* C,
//                 void* D, const void* bias, void* workspace,
//                 size_t workspace_bytes, float beta, void* stream);
// int rl_plan_destroy(void* plan);
// int rl_bias_grad(const void* dy, void* db, int batch, int channels, void* stream);
// void* rl_adam_create(size_t count, const void* const* parameters,
//                      const void* const* gradients, const void* const* m1,
//                      const void* const* m2, const size_t* sizes,
//                      void* timestep, void* learning_rate,
//                      double beta1, double beta2, double eps);
// int rl_adam_step(void* plan, void* stream);
// int rl_adam_destroy(void* plan);
//
// Matmul is column-major D=op(A)*op(B)+beta*C, alpha=1, with optional
// length-m bias. C may be null only for beta=0. In-place C=D is supported.
// Strict default uses CUBLAS_COMPUTE_32F_PEDANTIC, never implicit TF32.
// Run/step enqueue work on the caller's stream; this code does not allocate or
// sync there. Warm each plan outside capture before capturing its first call,
// including any cuBLAS/kernel module preparation performed by CUDA libraries.
// Create/destroy may allocate/synchronize and must run outside CUDA capture.
// Plans are tied to the current CUDA device and must not be used concurrently.
// Device buffers/workspace remain caller-owned and alive through graph replay.
// Adam pointer tables/sizes are host arrays copied at creation. Parameters,
// gradients, moments, float32 timestep and learning rate stay caller-owned.
// Adam increments timestep once per step, applies existing clipped gradients,
// and preserves Warp-NN's bias-corrected formula (no weight decay/AMSGrad).
// Status is 0 on success, -1 on synchronous error. Asynchronous device errors
// are observed at the caller's synchronization boundary, not hidden here.

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace {

thread_local char last_error[512] = "";

struct Failure {};

void set_error(const char* format, ...) noexcept {
    va_list args;
    va_start(args, format);
    std::vsnprintf(last_error, sizeof(last_error), format, args);
    va_end(args);
}

void require(bool condition, const char* message) {
    if (!condition) {
        set_error("%s", message);
        throw Failure{};
    }
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        set_error("%s: %s", operation, cudaGetErrorString(status));
        throw Failure{};
    }
}

void check_blas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        set_error("%s: cuBLAS status %d", operation, static_cast<int>(status));
        throw Failure{};
    }
}

bool aligned(const void* pointer, std::size_t bytes) noexcept {
    return reinterpret_cast<std::uintptr_t>(pointer) % bytes == 0;
}

struct DensePlan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t output = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    std::size_t workspace_bytes = 0;
    bool bias = false;

    int close(bool report = false) noexcept {
        int result = 0;
        auto release = [&](cublasStatus_t status) {
            if (status != CUBLAS_STATUS_SUCCESS && result == 0) {
                result = -1;
                if (report) {
                    set_error("Destroy dense plan: cuBLAS status %d", static_cast<int>(status));
                }
            }
        };
        if (preference) release(cublasLtMatmulPreferenceDestroy(preference));
        if (output) release(cublasLtMatrixLayoutDestroy(output));
        if (b) release(cublasLtMatrixLayoutDestroy(b));
        if (a) release(cublasLtMatrixLayoutDestroy(a));
        if (operation) release(cublasLtMatmulDescDestroy(operation));
        if (handle) release(cublasLtDestroy(handle));
        preference = nullptr;
        output = b = a = nullptr;
        operation = nullptr;
        handle = nullptr;
        return result;
    }

    ~DensePlan() noexcept { close(); }
};

constexpr unsigned kThreads = 256;
constexpr std::size_t kAdamChunk = 1024;

__global__ void bias_gradient(const float* dy, float* db, int batch, int channels) {
    const int channel = static_cast<int>(blockIdx.x);
    float value = 0.0f;
    for (int row = static_cast<int>(threadIdx.x); row < batch; row += blockDim.x) {
        value += dy[static_cast<std::size_t>(row) * channels + channel];
    }
    __shared__ float partial[kThreads];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (unsigned offset = kThreads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) db[channel] += partial[0];
}

struct AdamTensor {
    float* parameter;
    const float* gradient;
    float* m1;
    float* m2;
    std::size_t size;
};

struct AdamBlock {
    std::size_t tensor;
    std::size_t offset;
};

struct AdamPlan {
    AdamTensor* tensors = nullptr;
    AdamBlock* blocks = nullptr;
    float* correction = nullptr;
    float* timestep = nullptr;
    const float* learning_rate = nullptr;
    unsigned block_count = 0;
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float one_minus_beta1 = 0.0f;
    float one_minus_beta2 = 0.0f;
    float epsilon = 0.0f;

    int close(bool report = false) noexcept {
        int result = 0;
        auto release = [&](void* pointer) {
            if (!pointer) return;
            cudaError_t status = cudaFree(pointer);
            if (status != cudaSuccess && result == 0) {
                result = -1;
                if (report) set_error("Destroy Adam plan: %s", cudaGetErrorString(status));
            }
        };
        release(correction);
        release(blocks);
        release(tensors);
        correction = nullptr;
        blocks = nullptr;
        tensors = nullptr;
        return result;
    }

    ~AdamPlan() noexcept { close(); }
};

__global__ void adam_prepare(float* timestep, float beta1, float beta2, float* correction) {
    const float t = timestep[0] + 1.0f;
    timestep[0] = t;
    correction[0] = 1.0f / (1.0f - powf(beta1, t));
    correction[1] = 1.0f / (1.0f - powf(beta2, t));
}

__global__ void adam_update(
    const AdamTensor* tensors, const AdamBlock* blocks, const float* correction,
    const float* learning_rate, float beta1, float beta2,
    float one_minus_beta1, float one_minus_beta2, float epsilon) {
    const AdamBlock block = blocks[blockIdx.x];
    const AdamTensor tensor = tensors[block.tensor];
    const std::size_t remaining = tensor.size - block.offset;
    const std::size_t length = remaining < kAdamChunk ? remaining : kAdamChunk;
    for (std::size_t j = threadIdx.x; j < length; j += blockDim.x) {
        const std::size_t i = block.offset + j;
        const float gradient = tensor.gradient[i];
        const float m1 = beta1 * tensor.m1[i] + one_minus_beta1 * gradient;
        const float m2 = beta2 * tensor.m2[i] + one_minus_beta2 * (gradient * gradient);
        const float m1_hat = m1 * correction[0];
        const float m2_hat = m2 * correction[1];
        tensor.m1[i] = m1;
        tensor.m2[i] = m2;
        tensor.parameter[i] -= learning_rate[0] * (m1_hat / (sqrtf(m2_hat) + epsilon));
    }
}

}  // namespace

extern "C" const char* rl_last_error() noexcept { return last_error; }

extern "C" void* rl_plan_create(
    int m, int n, int k, int transa, int transb, int lda, int ldb, int ldd,
    int use_bias, int allow_tf32, std::size_t workspace_limit) noexcept {
    last_error[0] = '\0';
    try {
        require(m > 0 && n > 0 && k > 0, "Dense dimensions must be positive.");
        require((transa == 0 || transa == 1) && (transb == 0 || transb == 1),
                "Transpose flags must be 0 or 1.");
        require((use_bias == 0 || use_bias == 1) && (allow_tf32 == 0 || allow_tf32 == 1),
                "Bias and TF32 flags must be 0 or 1.");
        const int a_rows = transa ? k : m;
        const int a_columns = transa ? m : k;
        const int b_rows = transb ? n : k;
        const int b_columns = transb ? k : n;
        require(lda >= a_rows && ldb >= b_rows && ldd >= m, "Invalid column-major leading dimension.");
        auto plan = std::make_unique<DensePlan>();
        plan->bias = use_bias != 0;
        check_blas(cublasLtCreate(&plan->handle), "Create cuBLASLt handle");
        const cublasComputeType_t compute = allow_tf32
            ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F_PEDANTIC;
        check_blas(cublasLtMatmulDescCreate(&plan->operation, compute, CUDA_R_32F), "Create matmul descriptor");
        const cublasOperation_t op_a = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t op_b = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
        check_blas(cublasLtMatmulDescSetAttribute(
            plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)), "Set op(A)");
        check_blas(cublasLtMatmulDescSetAttribute(
            plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)), "Set op(B)");
        if (plan->bias) {
            const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            check_blas(cublasLtMatmulDescSetAttribute(
                plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)), "Set bias epilogue");
            // Heuristics inspect alignment but do not dereference this hint.
            const void* bias_alignment_hint = reinterpret_cast<const void*>(alignof(float));
            check_blas(cublasLtMatmulDescSetAttribute(
                plan->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias_alignment_hint, sizeof(bias_alignment_hint)), "Set bias alignment hint");
        }
        check_blas(cublasLtMatrixLayoutCreate(
            &plan->a, CUDA_R_32F, a_rows, a_columns, lda), "Create A layout");
        check_blas(cublasLtMatrixLayoutCreate(
            &plan->b, CUDA_R_32F, b_rows, b_columns, ldb), "Create B layout");
        check_blas(cublasLtMatrixLayoutCreate(
            &plan->output, CUDA_R_32F, m, n, ldd), "Create C/D layout");
        check_blas(cublasLtMatmulPreferenceCreate(&plan->preference), "Create matmul preference");
        check_blas(cublasLtMatmulPreferenceSetAttribute(
            plan->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_limit, sizeof(workspace_limit)), "Set workspace limit");
        const std::uint32_t alignment = alignof(float);
        for (const auto attribute : {
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
            check_blas(cublasLtMatmulPreferenceSetAttribute(
                plan->preference, attribute, &alignment, sizeof(alignment)), "Set matrix alignment");
        }
        cublasLtMatmulHeuristicResult_t candidates[8]{};
        int returned = 0;
        check_blas(cublasLtMatmulAlgoGetHeuristic(
            plan->handle, plan->operation, plan->a, plan->b, plan->output, plan->output,
            plan->preference, 8, candidates, &returned), "Select matmul algorithm");
        bool found = false;
        for (int i = 0; i < returned; ++i) {
            if (candidates[i].state == CUBLAS_STATUS_SUCCESS && candidates[i].workspaceSize <= workspace_limit) {
                plan->algorithm = candidates[i].algo;
                plan->workspace_bytes = candidates[i].workspaceSize;
                found = true;
                break;
            }
        }
        require(found, "No cuBLASLt algorithm supports this shape, precision and workspace limit.");
        return plan.release();
    } catch (const Failure&) {
        return nullptr;
    } catch (...) {
        set_error("Unexpected host error while creating dense plan.");
        return nullptr;
    }
}

extern "C" std::size_t rl_plan_workspace(void* opaque) noexcept {
    last_error[0] = '\0';
    if (!opaque) {
        set_error("Dense plan is null.");
        return std::numeric_limits<std::size_t>::max();
    }
    return static_cast<DensePlan*>(opaque)->workspace_bytes;
}

extern "C" int rl_plan_run(
    void* opaque, const void* a, const void* b, const void* c, void* d,
    const void* bias, void* workspace, std::size_t workspace_bytes,
    float beta, void* stream) noexcept {
    last_error[0] = '\0';
    try {
        require(opaque && a && b && d, "Dense plan, A, B and D must be nonnull.");
        auto* plan = static_cast<DensePlan*>(opaque);
        require(std::isfinite(beta) && (beta == 0.0f || c), "Nonzero beta requires C; beta must be finite.");
        require(aligned(a, alignof(float)) && aligned(b, alignof(float)) && aligned(d, alignof(float)) &&
                (!c || aligned(c, alignof(float))), "Dense buffers must be FP32-aligned.");
        require(workspace_bytes >= plan->workspace_bytes &&
                (plan->workspace_bytes == 0 || (workspace && aligned(workspace, 256))),
                "Dense workspace is too small, null, or not 256-byte aligned.");
        if (plan->bias) {
            require(bias && aligned(bias, alignof(float)), "Dense bias must be a nonnull FP32-aligned device buffer.");
            check_blas(cublasLtMatmulDescSetAttribute(
                plan->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)), "Set runtime bias");
        }
        const float alpha = 1.0f;
        check_blas(cublasLtMatmul(
            plan->handle, plan->operation, &alpha, a, plan->a, b, plan->b,
            &beta, c ? c : d, plan->output, d, plan->output, &plan->algorithm,
            workspace, plan->workspace_bytes, reinterpret_cast<cudaStream_t>(stream)), "Run matmul");
        return 0;
    } catch (const Failure&) {
        return -1;
    } catch (...) {
        set_error("Unexpected host error while running dense plan.");
        return -1;
    }
}

extern "C" int rl_plan_destroy(void* opaque) noexcept {
    last_error[0] = '\0';
    if (!opaque) return 0;
    auto* plan = static_cast<DensePlan*>(opaque);
    const int result = plan->close(true);
    delete plan;
    return result;
}

extern "C" int rl_bias_grad(
    const void* dy, void* db, int batch, int channels, void* stream) noexcept {
    last_error[0] = '\0';
    if (!dy || !db || batch <= 0 || channels <= 0 ||
        !aligned(dy, alignof(float)) || !aligned(db, alignof(float))) {
        set_error("Bias gradient requires FP32 buffers and positive batch/channels.");
        return -1;
    }
    bias_gradient<<<static_cast<unsigned>(channels), kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(dy), static_cast<float*>(db), batch, channels);
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        set_error("Run bias gradient: %s", cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

extern "C" void* rl_adam_create(
    std::size_t count, const void* const* parameters, const void* const* gradients,
    const void* const* m1, const void* const* m2, const std::size_t* sizes,
    void* timestep, void* learning_rate, double beta1, double beta2, double eps) noexcept {
    last_error[0] = '\0';
    try {
        require(count && parameters && gradients && m1 && m2 && sizes && timestep && learning_rate,
                "Adam requires nonempty host pointer tables and device step/lr buffers.");
        require(std::isfinite(beta1) && std::isfinite(beta2) && std::isfinite(eps) &&
                beta1 >= 0.0 && beta1 < 1.0 && beta2 >= 0.0 && beta2 < 1.0 && eps > 0.0,
                "Adam betas must lie in [0,1) and epsilon must be positive.");
        require(aligned(timestep, alignof(float)) && aligned(learning_rate, alignof(float)),
                "Adam step/lr buffers must be FP32-aligned.");
        auto plan = std::make_unique<AdamPlan>();
        std::vector<AdamTensor> tensors;
        std::vector<AdamBlock> blocks;
        tensors.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            require(parameters[i] && gradients[i] && m1[i] && m2[i] && sizes[i] > 0,
                    "Adam tensors must be nonnull and nonempty.");
            require(aligned(parameters[i], alignof(float)) && aligned(gradients[i], alignof(float)) &&
                    aligned(m1[i], alignof(float)) && aligned(m2[i], alignof(float)),
                    "Adam tensors must be FP32-aligned.");
            const std::size_t chunks = (sizes[i] - 1) / kAdamChunk + 1;
            require(chunks <= static_cast<std::size_t>(std::numeric_limits<int>::max()) - blocks.size(),
                    "Adam tensor block count exceeds CUDA grid limits.");
            tensors.push_back({
                static_cast<float*>(const_cast<void*>(parameters[i])),
                static_cast<const float*>(gradients[i]),
                static_cast<float*>(const_cast<void*>(m1[i])),
                static_cast<float*>(const_cast<void*>(m2[i])), sizes[i]});
            for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
                blocks.push_back({i, chunk * kAdamChunk});
            }
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&plan->tensors), tensors.size() * sizeof(AdamTensor)),
                   "Allocate Adam tensor metadata");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&plan->blocks), blocks.size() * sizeof(AdamBlock)),
                   "Allocate Adam block metadata");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&plan->correction), 2 * sizeof(float)),
                   "Allocate Adam correction scratch");
        check_cuda(cudaMemcpy(plan->tensors, tensors.data(), tensors.size() * sizeof(AdamTensor), cudaMemcpyHostToDevice),
                   "Copy Adam tensor metadata");
        check_cuda(cudaMemcpy(plan->blocks, blocks.data(), blocks.size() * sizeof(AdamBlock), cudaMemcpyHostToDevice),
                   "Copy Adam block metadata");
        plan->block_count = static_cast<unsigned>(blocks.size());
        plan->timestep = static_cast<float*>(timestep);
        plan->learning_rate = static_cast<const float*>(learning_rate);
        plan->beta1 = static_cast<float>(beta1);
        plan->beta2 = static_cast<float>(beta2);
        plan->one_minus_beta1 = static_cast<float>(1.0 - beta1);
        plan->one_minus_beta2 = static_cast<float>(1.0 - beta2);
        plan->epsilon = static_cast<float>(eps);
        return plan.release();
    } catch (const Failure&) {
        return nullptr;
    } catch (...) {
        set_error("Unexpected host error while creating Adam plan.");
        return nullptr;
    }
}

extern "C" int rl_adam_step(void* opaque, void* stream) noexcept {
    last_error[0] = '\0';
    if (!opaque) {
        set_error("Adam plan is null.");
        return -1;
    }
    auto* plan = static_cast<AdamPlan*>(opaque);
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    adam_prepare<<<1, 1, 0, cuda_stream>>>(
        plan->timestep, plan->beta1, plan->beta2, plan->correction);
    cudaError_t status = cudaPeekAtLastError();
    if (status == cudaSuccess) {
        adam_update<<<plan->block_count, kThreads, 0, cuda_stream>>>(
            plan->tensors, plan->blocks, plan->correction, plan->learning_rate,
            plan->beta1, plan->beta2, plan->one_minus_beta1, plan->one_minus_beta2, plan->epsilon);
        status = cudaPeekAtLastError();
    }
    if (status != cudaSuccess) {
        set_error("Run Adam: %s", cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

extern "C" int rl_adam_destroy(void* opaque) noexcept {
    last_error[0] = '\0';
    if (!opaque) return 0;
    auto* plan = static_cast<AdamPlan*>(opaque);
    const int result = plan->close(true);
    delete plan;
    return result;
}
