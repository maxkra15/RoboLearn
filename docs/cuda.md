# Experimental native CUDA learning kernels

The optional CUDA backend accelerates dense arithmetic in Warp PPO and Warp
FlashSAC. It retains their networks, loss equations, replay/GAE, clipping and
update order. The default backend remains Warp. Native CUDA is experimental;
synthetic update timings do not establish task quality or time to a trained policy.

## Scope and precision

- [cuBLASLt](https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublaslt-api)
  plans implement dense forward, input gradients and weight gradients.
- Native bias reductions and multi-tensor Adam retain FP32 parameters, gradients,
  moments, device learning rates and the existing checkpoint arrays.
- Persistent Warp arrays and the current Warp stream avoid Torch conversions and
  per-update allocation. The caller owns environments, rollouts, logging sinks,
  run directories and checkpoint schedules; see the [Isaac Lab guide](isaaclab.md).

The default native dense mode requests pedantic FP32 GEMMs. It does not enable
TF32 silently. Low-level `CudaLinear(allow_tf32=True)` is a separate experiment.
Reduction order can still change floating-point trajectories and learning outcomes.

Native FlashSAC with `cfg.use_amp=True` uses FP16 dense operands with FP32
accumulation, outputs, activations, master parameters, adjoints and Adam moments.
This differs from Torch autocast's intermediate FP16 rounding. Actor and critic
have independent dynamic loss scales; temperature learning and action inference
remain FP32. Nonfinite attempts skip parameter, moment and completed-step updates.
The scheduler, projections and target EMA follow the authors' attempted-update
cadence, including overflow skips. Scales and attempt/skip counters are saved.

`fused_updates=True` selects native BatchNorm/ReLU, RMSNorm, categorical heads and
targets, network parameter projections, target EMA and grouped adjoint clearing.
Replay and remaining loss kernels use Warp. This is an ordered sequence of GPU
kernels captured as a graph; it retains FlashSAC's residual and categorical networks.

The fused AMP path reuses FP16 weight copies refreshed by projection/EMA and
state restoration. At proven private BatchNorm/dense edges, normalization also
writes the consuming layer's existing FP16 activation or adjoint buffer. FP32
outputs and accumulated gradients remain available, with the same FP16 rounding
as an explicit conversion. Other edges retain their ordinary conversions.

## Install and select the backend

Install the `warp` extra plus an NVCC development toolkit with cuBLASLt
headers/libraries and a compatible C++ compiler. Linux and one CUDA device per
learner are supported. The native library has no Torch dependency; Torch examples
and upstream comparisons additionally need the `flashsac` extra.

```bash
uv sync --extra warp --extra flashsac
```

```python
from robolearn.warp import PPOConfig, WarpPPO

ppo = WarpPPO(
    123,
    37,
    1024,
    24,
    config=PPOConfig(hidden_dims=(256, 128, 128), dense_backend="cuda"),
)
ppo.warmup()
```

```python
from robolearn.flashsac import FlashSACConfig
from robolearn.warp import WarpFlashSAC

sac = WarpFlashSAC(
    123,
    37,
    num_envs=1024,
    cfg=FlashSACConfig(device="cuda:0", use_amp=True, use_compile=False),
    dense_backend="cuda",
    fused_updates=True,
    fresh_training_gradients=False,
)
sac.prepare(capture=True)
```

These snippets select backend options; they are not complete matched G1 training
recipes. See the [PPO](warp.md) and [FlashSAC](warp_flashsac.md) guides for learner
inputs, transitions and update usage. Leave `use_amp=False` for FP32, or
`fused_updates=False` for the original Warp normalization path. Fusion and AMP
are disabled by default.

Compilation is lazy at first preparation. An explicit architecture permits a
build without querying or using a GPU:

```python
from robolearn.cuda import build_library

library = build_library(architecture=89)  # L40/RTX 4090.
```

`ROBOLEARN_CUDA_NVCC` selects the compiler; `ROBOLEARN_CUDA_CACHE_DIR` selects the
cache. The cache retains source/compiler/architecture/flags, build log and library.
Ordinary package import/install does not build it. Build or plan failures raise;
there is no automatic backend fallback.

## Capture, storage and checkpoints

Warm every inference/training shape and precision mode before outer capture.
Keep the learner and plans alive while dependent graphs exist. Construct the
learner on its intended device: native plans and graphs require fixed device and
owned buffer storage; migration with `.to()` and private pointer replacement
after capture are unsupported. Preparation and diagnostics belong outside timing.

Checkpoint loading copies into existing arrays and refreshes derived FP16 copies,
retaining graph addresses. Resume optimizer/scaler state with the same precision
recipe. For policy transfer to another precision, set `cfg.load_optimizer=False`.
Native checkpoints preserve FP32 masters, Adam state, counters and loss scales;
FP16 conversion buffers are derived storage rather than extra checkpoint arrays.

`cuda_autotune=True` optionally probes at most eight cuBLASLt candidates per dense
operation before capture. Private scratch preserves model/gradient/RNG state.
A replacement needs a 3% median improvement and two paired wins over the default
heuristic. Plans/workspace stay fixed after preparation; `cuda_tuning_metadata()`
returns host records. Include tuning in startup costs and measure complete updates:
scratch GEMM improvements need not improve the learner. This option defaults False.

## Optional fresh training gradients

`fresh_training_gradients=True` requires `dense_backend="cuda"`, `cfg.use_amp=True`
and `fused_updates=True`. It defaults False and affects only owned private residual
`w1`/`w2` output adjoints during optimizer training phases. Their first backward
writer assigns the complete FP32 gradient and its rounded FP16 copy, allowing
those verified spans to leave the post-update clear. All remaining adjoints clear
normally; omitted private spans are finite but need not be zero after an update.

Ordinary taped calls, retained/repeated backward and `apply_optimizer=False`
diagnostics keep accumulation semantics. Checkpoint/warmup restoration clears
omitted spans. Ownership preflight precedes eager backward and capture recording:
supported storage changes select ordinary accumulation/full clearing; invalid
aliasing or extents raise before backward/optimizer. Captured graphs retain the
fixed-storage contract above, rather than checking every pointer on each replay.

Inspect actual activation outside timers:

```python
# Construct with fresh_training_gradients=True, then prepare, before inspecting it.
status = sac.fresh_gradient_metadata()
for phase in ("actor", "critic"):
    assert status[phase]["enabled"]
    assert status[phase]["fallback_reason"] is None
    print(phase, status[phase]["owned_raw_bytes"], status[phase]["remaining_clear_bytes"])
```

Use this after constructing with the option and preparing the learner. The method
reads host metadata without synchronizing device values; it returns `{}` when
disabled. Report fallback instead of attributing timings to an inactive mode.
Buffer-byte counts describe storage/clear scope, not measured bandwidth or speed.

## Validate and measure

Run focused numerical/capture/checkpoint diagnostics before timing on a GPU:

```bash
uv run python examples/diagnose_cuda_flashsac.py --output logs/cuda/flash-diagnostics.json
uv run python examples/benchmark_cuda_learning.py --output logs/cuda/primitives-ppo.json
uv run python examples/diagnose_ppo_update.py --dense-backend cuda --output logs/cuda/ppo-reference.json
```

The FlashSAC diagnostics cover dense/scaler/normalization/categorical math,
conversion buffers, ordinary/fresh clearing, overflow recovery, eager/captured
updates and checkpoint continuation. PPO primitives use FP64 references; the
small PPO reference checks a complete update against Torch formulas. Longer PPO
FP32 trajectory differences are recorded separately from mandatory finite/control
checks; `--strict-ppo-parity` makes their numerical budgets mandatory too.

The balanced entrypoint compares four recipes: official compiled Torch AMP,
RoboLearn Torch AMP, native fused AMP, and native fused AMP with plan tuning.
Use the pristine [official source revision](https://github.com/Holiday-Robot/FlashSAC/tree/87edc9061150ae9e962dd84e6544e27a1554b3ab).
Its import also needs benchmark-only OmegaConf and CPU JAX; the recorded setup
used `omegaconf==2.4.0`, `jax[cpu]==0.11.2` and `JAX_PLATFORMS=cpu`. Python 3.12
comparisons extend beyond the authors' documented Python 3.10/3.11 matrix.

```bash
JAX_PLATFORMS=cpu uv run python examples/benchmark_flashsac_balanced.py \
    --official-root /path/to/FlashSAC --rounds 8 --schedule-seed 17 \
    --warmup 100 --updates 500 --actor-calls 100 --output logs/cuda/balanced.json
JAX_PLATFORMS=cpu uv run python examples/benchmark_flashsac_balanced.py \
    --official-root /path/to/FlashSAC --rounds 8 --schedule-seed 17 \
    --warmup 100 --updates 500 --actor-calls 100 --fresh-training-gradients \
    --output logs/cuda/balanced-fresh.json
```

The flag affects both native rows, with their labels unchanged; official/port
workers receive no fresh-gradient flag. Results record requested/actual modes
and the worker's untimed activation evidence. Without the flag both native rows
use complete clearing. Each worker has fresh optimizer/scaler history, identical
canonical weights and replay fixtures. Pass `--initial-weights` to reuse the first
run's exported canonical NPZ across separate runs. Compare same-round ratios;
separate invocations are not paired merely because schedule seeds match.

Two Williams blocks balance positions and immediate predecessors within rounds;
cross-round boundaries are not separately balanced. Preconditioning warms shared
disk caches, so startup is not an independent cold-start comparison. Tuned plans
are chosen separately in each worker. Keep all rounds, observed ranges, raw JSON,
source hashes, checkpoints and engineering-target misses. Repeat on separate
allocations; timing rounds are not independent training seeds or confidence bounds.

Report startup separately from warmed complete-update time, with precision,
shapes, attempted/completed/skipped updates and finite updated weights. Every
timed update returns matched CPU scalar metrics; those synchronization costs are
included. Saved worker checkpoints are evidence of state, while focused diagnostics
check reload/continuation. Profiling runs outside timing and restores state; kernel
categories/counts are not conserved counts of algorithmic work or causal speedups.

## Dated results

<a id="comparison-with-upstream-learners-23-october-2026"></a>

- [24 October CUDA FlashSAC report](https://maxkra15.github.io/reports/2026-10-24-flashsac-cuda/)
  records frozen implementations, numerical recipes and single-allocation timings.
- [20 October G1 report](https://maxkra15.github.io/reports/2026-10-20-g1-main-baselines/)
  measures task learning with its recorded backends; it does not validate later
  native CUDA changes. See [attribution](../ATTRIBUTION.md) for authors and licenses.

The [23 October PPO comparison](https://github.com/maxkra15/RoboLearn/blob/8611ba548644a4ee55482551f6fe4f327cf5d618/docs/cuda.md)
recorded about 2.31× faster synthetic updates than stock RSL-RL for its pinned recipe.

Historical timings belong to their pinned sources and protocols. No backend is
promised to win across devices, shapes or workloads; measure learning quality
separately from synthetic learner throughput.
