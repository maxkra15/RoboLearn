# Experimental native CUDA learning kernels

This optional backend targets the dense arithmetic shared by Warp PPO and Warp
FlashSAC. It preserves their network architectures, loss kernels, normalization,
replay/GAE, update order, clipping and checkpoint arrays. Defaults remain Warp.

## Scope

- Planned [cuBLASLt](https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublaslt-api)
  matrix products for forward, input gradients and weight gradients. PPO bias
  addition uses the forward epilogue; FlashSAC dense layers remain bias-free.
- An original CUDA bias-gradient reduction and fused multi-tensor Adam update.
  Adam retains independent actor/critic/temperature counters where required,
  FP32 moments, device learning rates, and caller-owned gradient clipping.
- Persistent zero-copy Warp arrays, cached per-shape workspace, and the current
  Warp CUDA stream. No Torch tensor conversion, simulator, logger, or per-update
  device allocation is part of this backend.

The first implementation explicitly requests pedantic FP32 GEMMs. It does not
enable TF32 or mixed precision silently. `CudaLinear(allow_tf32=True)` is an
explicit low-level experiment, outside the default learner recipes. Different
floating-point reduction orders can still change updates and learning outcomes.

This is a short sequence of GPU kernels captured as a graph. FlashSAC's batch
normalization, parameter-gradient reductions and optimizer dependencies limit
whole-network fusion. Its existing residual blocks and categorical critics are
retained. The algorithm-specific kernels remain authored in Warp and compiled to
CUDA; this backend replaces dense arithmetic and Adam, rather than reimplementing
every loss in CUDA C++.

## Use

Install RoboLearn's `warp` extra in the supported environment, plus an NVCC CUDA
development toolkit with cuBLASLt headers/libraries and a compatible C++ compiler.
There is no additional Torch dependency or vendored CUDA library.

```python
from robolearn.warp import PPOConfig, WarpPPO

ppo = WarpPPO(
    123,
    37,
    1024,
    24,
    config=PPOConfig(
        hidden_dims=(256, 128, 128),
        dense_backend="cuda",
    ),
)
```

```python
from robolearn.flashsac import FlashSACConfig
from robolearn.warp import WarpFlashSAC

sac = WarpFlashSAC(
    123,
    37,
    1024,
    cfg=FlashSACConfig(
        device="cuda:0",
        use_amp=False,
        use_compile=False,
    ),
    dense_backend="cuda",
)
```

These snippets select the backend, not the complete matched G1 hyperparameters;
see the existing [PPO](warp.md) and [FlashSAC](warp_flashsac.md) guides.
Use `ppo.warmup()` or `sac.prepare(capture=True)` before timing or outer capture.
Every inference/training shape must be exercised before capturing it. Keep the
learners and their native plans alive for the lifetime of dependent graphs.
Checkpoints restore arrays in place, retaining graph pointer addresses.

Compilation is lazy on first native preparation. To compile without using a GPU:

```python
from robolearn.cuda import build_library

library = build_library(architecture=89)  # L40/RTX 4090; explicit target, no GPU query.
```

`ROBOLEARN_CUDA_NVCC` overrides the compiler; `ROBOLEARN_CUDA_CACHE_DIR` overrides
the build cache. The cache records source/compiler/architecture/flags, build log
and shared library. No native binary is built during ordinary package import or
installation. Build or plan failures are explicit; there is no backend fallback.

## Measurements

The standalone examples run independently of Isaac Lab:

```bash
uv run python examples/benchmark_cuda_learning.py --output logs/cuda/primitives-ppo.json
uv run python examples/diagnose_ppo_update.py --dense-backend cuda --output logs/cuda/ppo-reference.json
uv run python examples/benchmark_warp_flashsac.py --dense-backend cuda --disable-tf32 \
    --output logs/cuda/flashsac-fp32.json
```

The first script compares native dense forward/gradients and Adam parameters and
moments against FP64 formulas, including irregular shapes, frozen parameters,
gradient accumulation, eager/captured PPO, and checkpoint restoration. It times
complete captured PPO updates at the recorded G1 shapes. The small PPO reference
also compares a complete update with Torch formulas. FlashSAC uses shared initial
weights for descriptive component measurements; its stochastic trajectories and
optimizer state diverge across recipes.

Report compilation/preparation separately from warmed update time. Compare a
full optimizer step, not only inference or one GEMM, and report precision, shapes,
attempted/completed updates and numerical health. Synthetic timings establish
neither walking quality nor time to a trained policy. No speedup is promised.

The native CUDA code is original RoboLearn code. Algorithm authors and upstream
FlashSAC attribution remain in [ATTRIBUTION.md](../ATTRIBUTION.md) and
[CITATIONS.bib](../CITATIONS.bib). Warp-NN and NVIDIA CUDA libraries retain their
respective licenses.
