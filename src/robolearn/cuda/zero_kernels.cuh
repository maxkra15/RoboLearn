// SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
// SPDX-License-Identifier: MIT
// Immutable, disjoint FP32 zero spans; create/destroy only outside capture.
// void* rl_zero_create(size_t count, const void* const* pointers, const size_t* lengths);
// int rl_zero_step(void* plan, void* stream);
// int rl_zero_destroy(void* plan);
#pragma once

namespace {
namespace zero_native {

constexpr std::size_t Chunk = 4096;

struct Block {
    float* pointer;
    unsigned length;
};

struct Plan {
    Block* blocks = nullptr;
    unsigned count = 0;

    int close(bool report = false) noexcept {
        if (!blocks) return 0;
        const cudaError_t result = cudaFree(blocks);
        blocks = nullptr;
        count = 0;
        if (result != cudaSuccess) {
            if (report) set_error("Destroy zero plan: %s", cudaGetErrorString(result));
            return -1;
        }
        return 0;
    }

    ~Plan() noexcept { close(); }
};

__global__ void clear(const Block* blocks) {
    const Block block = blocks[blockIdx.x];
    if (reinterpret_cast<std::uintptr_t>(block.pointer) % alignof(uint4) == 0) {
        auto* vectors = reinterpret_cast<uint4*>(block.pointer);
        const unsigned count = block.length / 4;
        for (unsigned i = threadIdx.x; i < count; i += blockDim.x) vectors[i] = make_uint4(0, 0, 0, 0);
        for (unsigned i = count * 4 + threadIdx.x; i < block.length; i += blockDim.x) block.pointer[i] = 0.0f;
    } else {
        for (unsigned i = threadIdx.x; i < block.length; i += blockDim.x) block.pointer[i] = 0.0f;
    }
}

}  // namespace zero_native
}  // namespace

extern "C" void* rl_zero_create(
    std::size_t count, const void* const* pointers, const std::size_t* lengths) noexcept {
    last_error[0] = '\0';
    try {
        require(count && pointers && lengths, "Zero plan requires nonempty host pointer/length tables.");
        std::vector<zero_native::Block> blocks;
        std::uintptr_t previous_end = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const auto start = reinterpret_cast<std::uintptr_t>(pointers[i]);
            require(start && aligned(pointers[i], alignof(float)) && lengths[i],
                    "Zero plan requires aligned, nonempty FP32 spans.");
            require(lengths[i] <= (std::numeric_limits<std::uintptr_t>::max() - start) / sizeof(float),
                    "Zero span pointer arithmetic overflows.");
            require(start >= previous_end, "Zero spans must be sorted and nonoverlapping.");
            previous_end = start + lengths[i] * sizeof(float);
            const std::size_t chunks = (lengths[i] - 1) / zero_native::Chunk + 1;
            require(chunks <= static_cast<std::size_t>(std::numeric_limits<int>::max()) - blocks.size(),
                    "Zero plan block count exceeds CUDA grid limits.");
            auto* pointer = static_cast<float*>(const_cast<void*>(pointers[i]));
            for (std::size_t offset = 0; offset < lengths[i]; offset += zero_native::Chunk) {
                const std::size_t remaining = lengths[i] - offset;
                blocks.push_back({pointer + offset, static_cast<unsigned>(
                    remaining < zero_native::Chunk ? remaining : zero_native::Chunk)});
            }
        }
        auto plan = std::make_unique<zero_native::Plan>();
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&plan->blocks), blocks.size() * sizeof(zero_native::Block)),
                   "Allocate grouped-zero descriptors");
        check_cuda(cudaMemcpy(plan->blocks, blocks.data(), blocks.size() * sizeof(zero_native::Block), cudaMemcpyHostToDevice),
                   "Copy grouped-zero descriptors");
        plan->count = static_cast<unsigned>(blocks.size());
        return plan.release();
    } catch (const Failure&) {
        return nullptr;
    } catch (...) {
        set_error("Unexpected host error while creating zero plan.");
        return nullptr;
    }
}

extern "C" int rl_zero_step(void* opaque, void* stream) noexcept {
    last_error[0] = '\0';
    if (!opaque) {
        set_error("Zero plan is null.");
        return -1;
    }
    const auto* plan = static_cast<const zero_native::Plan*>(opaque);
    zero_native::clear<<<plan->count, kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(plan->blocks);
    const cudaError_t result = cudaPeekAtLastError();
    if (result != cudaSuccess) {
        set_error("Grouped zero launch: %s", cudaGetErrorString(result));
        return -1;
    }
    return 0;
}

extern "C" int rl_zero_destroy(void* opaque) noexcept {
    last_error[0] = '\0';
    if (!opaque) return 0;
    auto* plan = static_cast<zero_native::Plan*>(opaque);
    const int result = plan->close(true);
    delete plan;
    return result;
}
