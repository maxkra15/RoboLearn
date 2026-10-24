# Experimental native CUDA learning kernels

This optional backend targets the dense arithmetic shared by Warp PPO and Warp
FlashSAC. It preserves their network architectures, loss and normalization
equations, replay/GAE, update order, clipping and checkpoint arrays. Defaults
remain Warp.

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

### Native FlashSAC mixed precision and fusion

FlashSAC can opt into FP16 dense operands with FP32 accumulation, outputs,
activations, master parameters, gradients and Adam moments. Actor and critic
optimizers use independent device-side dynamic loss scales. Nonfinite gradient
attempts skip parameter, moment and completed-step updates; attempt/skip counters
and scales are checkpointed. Temperature learning and environment action
inference remain FP32. The scheduler, parameter projections and target EMA follow
the authors' attempted-update cadence, including overflow skips.

```python
agent = WarpFlashSAC(
    123,
    37,
    num_envs=1024,
    cfg=FlashSACConfig(device="cuda:0", use_amp=True, use_compile=False),
    dense_backend="cuda",
    fused_updates=True,
)
agent.prepare(capture=True)
```

`fused_updates=True` selects cooperative native normalization with fused ReLU in
residual blocks, one whole-network parameter-projection launch, and one
multi-tensor target EMA launch. Losses and the actor/temperature/critic update
order are preserved. Reduction order and mixed-precision rounding differ from
Torch autocast. Batch-statistic and optimizer dependencies retain multiple
ordered kernels inside each captured update graph.

The fused path reuses FP16 weight copies across batch shapes. Projection and
target EMA refresh them alongside their FP32 masters; author-state loading,
checkpoint restoration and warmup restoration refresh them before reuse.
Dense activations and adjoints use vectorized conversions. Cooperative
categorical heads combine bias, log-softmax and expected value, with a matching
combined backward; target distributions use shared-memory projection. These
changes preserve the network and loss equations while changing reduction order.

Gradient clearing groups the complete recorded adjoint collection into one native
launch per optimizer phase. Persistent pointer inventories handle overlapping
views and preserve Warp's initialization bookkeeping; unsupported or changed
inventories retain Warp's ordinary clear. BatchNorm reduces backward coefficients
once per channel before coalesced gradient application, using its existing scratch.

The options are independent: leave `use_amp=False` for a fusion-only experiment,
or `fused_updates=False` to measure mixed precision with the original Warp
normalization. The default remains FP32 with fusion disabled. Restore the same
precision to resume optimizer/scaler state; use `load_optimizer=False` when
transferring a policy checkpoint to a different precision recipe.

Standalone diagnostics exercise rounded-input GEMM references, overflow skips,
capture, normalization gradients and checkpoint continuation. Synthetic update
benchmarks support `--compile-fp32`, `--native-variants fp32 fp32-fused amp amp-fused`
and `--profile-native`; profiler work is outside measured update time and restores
the learner afterward. Performance and task-learning quality require measurements.

This is a sequence of GPU kernels captured as a graph. FlashSAC's batch
normalization, parameter-gradient reductions and optimizer dependencies limit
whole-network fusion. Its existing residual blocks and categorical critics are
retained. Replay and remaining loss kernels use Warp. Dense products,
normalization, categorical heads and targets, parameter projection, target EMA,
gradient clearing and Adam use native CUDA in the fused path.

## Use

Install RoboLearn's `warp` extra in the supported environment, plus an NVCC CUDA
development toolkit with cuBLASLt headers/libraries and a compatible C++ compiler.
The initial native backend supports Linux and a single CUDA device per learner.
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
Construct each native learner on its intended CUDA device. Native layers,
prepared plans and captured graphs have fixed device and buffer ownership;
device migration with `.to()` is unsupported. Checkpoints restore arrays in
place, retaining graph pointer addresses.

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

Long PPO update differences are recorded separately from the mandatory primitive,
control-state, finite-weight and checkpoint checks. `--strict-ppo-parity` also
requires every recorded FP32 trajectory budget to pass. A timing result with
numerical differences is labeled explicitly and does not establish equivalence.

Report compilation/preparation separately from warmed update time. Compare a
full optimizer step, not only inference or one GEMM, and report precision, shapes,
attempted/completed updates and numerical health. Synthetic timings establish
neither walking quality nor time to a trained policy. No speedup is promised.

### Matched FlashSAC follow-up, 24 October 2026

Source: [`69941d3`](https://github.com/maxkra15/RoboLearn/commit/69941d39ef987f43a9052a576b1f679ce4db2a05),
compared with the preceding native implementation
[`33c5ba9`](https://github.com/maxkra15/RoboLearn/commit/33c5ba95cb02ae1c74e56b20b3489cacc5be8d10)
and the unmodified official FlashSAC commit pinned below. One fresh L40 allocation
uses the same G1-shaped networks, batch size 2,048, and runtime versions below.
No compiler-cache archive is imported. These are synthetic learner measurements.

Each row contains five continuous-state blocks of 20 warm and 200 measured calls.
Times are medians of synchronized block means, including the same host scalar
metric cadence; compilation, checkpointing and profiling are excluded.

| Implementation | First pass, ms/call | Repeat pass, ms/call |
| --- | ---: | ---: |
| Official FlashSAC, compiled FP16 AMP | 4.045 | 3.858 |
| RoboLearn Torch port, compiled FP16 AMP | 4.206 | 4.007 |
| Native CUDA, captured FP16 dense / FP32 state | 3.794 | 3.798 |

Native used **1.6--6.2% less median update time** than official Torch AMP across
the two passes. Their observed block ranges overlap: this is a small, conditional
median advantage, not evidence of a consistent winner. The preceding native
implementation measured 3.825 ms before and 3.838 ms after the first comparison;
the additional grouped clears and BatchNorm changes improved native time by
about 1%. A native strict-FP32 control measured 6.129 ms. The earlier
weight-copy and categorical optimizations reduced native AMP time from
4.464--4.467 ms to 3.828 ms in a separate matched allocation.

All nine timing rows completed 1,000 critic, 500 actor and 500 temperature measured
updates, with zero skips. All four benchmark commands loaded the same 89 canonical
initial arrays exactly and used bit-identical transition fixtures. Twelve focused
diagnostic stages passed, including numerical primitives, overflow handling,
aliased-gradient clearing, eager/captured execution, exact checkpoint restoration
and continued graph replay. Complete-learner floating drift remains descriptive
under the recorded sanity gates, rather than strict Torch/native equivalence.
All optimized native clear plans were active; the focused continuation checks
confirmed their adjoints were zero.

The native recipe retains FP32 intermediates and separate actor/critic scales;
Torch autocast rounds more intermediates and uses a shared scaler. Torch permits
TF32 in this AMP recipe; native does not. Rows and processes run in fixed order
with shared caches, so the blocks are not independent runs or confidence
intervals. Official first-pass startup took 444 seconds including compilation
and autotuning; native preparation followed the diagnostic and preceding-native
processes and is not an independently cold startup comparison. No environment
learning, walking quality or end-to-end simulator speed was measured.

Reproduce with [the focused diagnostic](../examples/diagnose_cuda_flashsac.py)
and [the official comparison](../examples/benchmark_official_flashsac.py). Pass
`--official-root <pristine-87edc90-checkout> --output <fresh-result>.json`, using
`--recipe amp --allow-tf32 --blocks 5 --warmup 20 --updates 200 --actor-calls 100`,
`--native-variants amp-fused fp32-fused --profile-native` for the first pass,
and `--native-variants amp-fused` for the repeat. The first invocation saves
`common-initial-weights.npz` in its artifact directory; pass that same file with
`--initial-weights` to subsequent commands. To bracket the first comparison,
run the current driver with `PYTHONPATH` pointing to the pinned `33c5ba9/src`
and `--only-native --native-variants amp-fused` before and after it. The backend
remains experimental pending policy-learning checks.

### L40 measurements, 23 October 2026

Source: [`c4dd800`](https://github.com/maxkra15/RoboLearn/commit/c4dd800c5d4ffd7dd2c2629862e4ee02eab04e48).
One NVIDIA L40, CUDA 13.0, Torch 2.11.0, Warp 1.17.0 and Warp-NN 0.4.1.
These are synchronized mean update times from fixed synthetic data. The native
backend requests pedantic FP32; Torch TF32 was disabled. Compilation and
preparation are excluded from the table.

| Learner | Implementation | Milliseconds per update |
| --- | --- | ---: |
| PPO | Warp, captured FP32 | 57.171 |
| PPO | Native dense/Adam, captured FP32 | 31.174 |
| FlashSAC | Warp, captured FP32 | 29.864 |
| FlashSAC | Native dense/Adam, captured FP32 | 12.979 |
| FlashSAC | Torch, eager FP32 | 11.487 |
| FlashSAC | Torch, compiled FP16 AMP | 4.027 |

PPO uses 123 observations, 37 actions, hidden widths 256/128/128, 1,024 worlds,
a 24-step rollout and five epochs with four minibatches. Each table update
contains 20 optimizer steps; 50 measured calls completed all 1,000 steps per
backend after five warmup calls. The native backend achieved a **1.83x speedup**
over Warp.

FlashSAC retains the existing residual actor and twin categorical critics,
with batch size 2,048. Each update is one critic/replay call; actor and
temperature updates occur every second call. Each recipe completed 100 critic,
50 actor and 50 temperature steps, with zero skipped optimizer steps. Warmup was
10 calls for FP32 and 20 for the AMP comparison. Native FP32 achieved a **2.30x
speedup** over Warp FP32, but Torch FP32 and compiled AMP remained faster than
native FP32. PPO and FlashSAC update units differ.

Native dense forward/backward and Adam passed their FP64 reference checks.
Both PPO backends restored checkpoints exactly and successfully replayed an
update after loading. Native PPO passed the recorded long-update numerical
budgets, with maximum eager/captured state difference `1.54e-5`. The small native
PPO/Torch reference passed, with parameter and Adam-moment errors below `1.1e-6`.
The stock Warp PPO first-moment array for the distribution parameter exceeded
its float-array budget; its exact counters and all adaptive learning-rate
decisions matched. This difference remains explicitly recorded in the raw data.

These measurements cover dense operations and Adam together. They use sequential
processes on one GPU, with cache reuse and order effects. Torch AMP warmup took
480 seconds, including compilation and autotuning; startup measurements are
reported by the scripts but do not constitute an independently cold comparison.
No environment was trained or evaluated in this experiment. The backend remains
experimental; full policy learning checks and profiling of the remaining
normalization/reduction work are needed before recommending it for training.

### Comparison with upstream learners, 23 October 2026

The following experiment executes **RSL-RL 5.5.1's actual PyTorch PPO**, the
authors' unmodified FlashSAC agent, RoboLearn's Torch FlashSAC port, and the native
CUDA backend. Source: [`8611ba5`](https://github.com/maxkra15/RoboLearn/commit/8611ba548644a4ee55482551f6fe4f327cf5d618).
It uses one L40, with the runtime versions and G1-shaped networks described above.
Values are medians of three synchronized block means; setup and compilation are
excluded. Each recipe runs its rows and blocks in one ordered process, with
shared compiler caches, rather than independent runs or confidence intervals.

| PPO implementation | Precision | Milliseconds per full update |
| --- | --- | ---: |
| RSL-RL, Isaac Lab default recipe | FP32, TF32 allowed | 73.636 |
| RSL-RL, eager control | Strict FP32 | 73.221 |
| RSL-RL, official actor/critic compile | Strict FP32 | 81.920 |
| RSL-RL, official compile and autocast | BF16, FP32 parameters/Adam | 89.826 |
| RoboLearn, captured native CUDA | Strict FP32 | 31.925 |

Each PPO call includes final critic bootstrap, GAE, advantage normalization,
shuffle, and all 20 minibatch optimizer steps. Each block restores common initial
weights and empty Adam state, then runs five warm and 50 timed calls: **3,000
measured optimizer steps per parameter per backend** across three blocks. The
native path took 0.434 times the default RSL time, a **2.31x speedup** in this
synthetic learner experiment.

This comparison includes the implementations' control and metric costs. Stock
RSL retains Python adaptive-KL decisions and per-minibatch scalar reads; the
native path captures the full update and reads health metrics once per call.
RSL reports mean minibatch losses, while native health describes the last
minibatch. These costs and reporting semantics differ. The result cannot be
attributed solely to dense matrix arithmetic.

| FlashSAC implementation | FP32 control, ms/call | Compiled AMP recipe, ms/call |
| --- | ---: | ---: |
| Official FlashSAC | 11.751 | 3.892 |
| RoboLearn Torch port | 11.578 | 4.144 |
| RoboLearn native CUDA, strict FP32 in both processes | 12.976 | 12.986 |

One FlashSAC call samples replay and updates the critic; actor and temperature
update every second call. These units differ from PPO. Each row completed **300
critic, 150 actor and 150 temperature updates**, with zero skipped steps, across
three continuous-state blocks. All implementations used common initial weights,
matching replay data and the same host scalar metric cadence. Their random
streams and updated trajectories differ.

Torch TF32 is disabled in the FP32 control. The AMP recipe allows TF32 and uses
compiled FP16 autocast for the official and Torch-port learners; native CUDA
continues to use pedantic FP32. Native FP32 was slower than official FP32, and
the official AMP recipe was fastest. The Torch port took about **6.5% longer**
than official AMP. The official agent unconditionally compiles categorical-target
and reward-normalization helpers that the port currently leaves uncompiled; this
experiment does not isolate the contribution of that difference. Official AMP's
first warmup took 416 seconds including compilation/autotuning. Subsequent rows
reuse caches, so startup costs are not independently cold measurements.

All rows passed finite-metric and updated-weight checks. PPO passed exact
checkpoint restoration and a post-load update; FlashSAC checkpoints were saved
in each backend's native format, with no reload check in this experiment.
Initial native/RSL bootstrap and GAE alignment was checked before optimization.
No task learning, policy quality, or end-to-end simulator speed was measured.

Reproduce using [the RSL comparison](../examples/benchmark_rsl_ppo_cuda.py) and
[the official FlashSAC comparison](../examples/benchmark_official_flashsac.py).
FlashSAC source is pinned to
[`87edc90`](https://github.com/Holiday-Robot/FlashSAC/tree/87edc9061150ae9e962dd84e6544e27a1554b3ab)
and verified before import, without source patches or runtime monkeypatches.
Its declared Python range excludes Python 3.12: execution here on Python 3.12.14
and Torch 2.11 is a current-runtime experiment, outside that declared matrix.

The native CUDA code is original RoboLearn code. Algorithm authors and upstream
FlashSAC attribution remain in [ATTRIBUTION.md](../ATTRIBUTION.md) and
[CITATIONS.bib](../CITATIONS.bib). Warp-NN and NVIDIA CUDA libraries retain their
respective licenses.

### FlashSAC fusion and mixed precision, 23 October 2026

Source: [`35f9bf1`](https://github.com/maxkra15/RoboLearn/commit/35f9bf14ff8065e3befe85640ce6f7c8313de699).
This follow-up keeps the G1-shaped architecture and batch size above, on one L40
with the runtime versions above. Each row has three continuous-state blocks of
20 warm calls and 100 measured calls. Values are medians of synchronized block
means, including the default host scalar metrics. Compilation, checkpointing
and profiling are outside measured update time.

Initial weights match the pristine author weights exactly within each command.
Across the three command processes, 19 normalized dense arrays differ by at
most `5.96e-8`; the transition fixture is bit-identical. Comparisons within each
command retain matched initialization.

| Implementation | Precision | Fusions | Milliseconds per call |
| --- | --- | --- | ---: |
| Official FlashSAC, networks eager | Strict FP32 | Upstream helpers | 12.048 |
| RoboLearn Torch port, eager | Strict FP32 | Disabled | 11.878 |
| Native CUDA | Strict FP32 | Disabled | 13.032 |
| Native CUDA | Strict FP32 | Enabled | 6.477 |
| Native CUDA | FP16 dense operands, FP32 state | Disabled | 11.026 |
| Native CUDA | FP16 dense operands, FP32 state | Enabled | 4.557 |

Fusion reduces native FP32 call time by about half. Fusion plus mixed precision
achieves a **2.86x speedup over the previous native FP32 path**. The fusion bundle
includes native normalization, BN+ReLU, parameter projections and target EMA;
this experiment does not isolate each component's contribution. All rows
completed 300 critic, 150 actor and 150 temperature updates, with zero skipped
optimizer attempts.

Separate compiled controls use the same budgets and metric cadence:

| Implementation | Compiled FP32 control, ms/call | Authors' AMP recipe, ms/call |
| --- | ---: | ---: |
| Official FlashSAC | 5.198 | 3.917 |
| RoboLearn Torch port | 5.199 | 4.091 |
| Native CUDA, fused | 6.500, strict FP32 | 4.499, FP16 dense / FP32 state |

Compiled Torch is faster than native fusion at the same FP32 precision in this
fixture. Native mixed precision with fusion takes about **14.8% longer than the
official optimized AMP recipe**, and about 10% longer than the Torch port. The
native improvement therefore narrows the gap without surpassing upstream. TF32
is disabled for the compiled FP32 control and allowed for the authors' AMP
recipe; the native paths do not use TF32. Repeated native fused measurements
across these processes were 6.48--6.50 ms in FP32 and 4.49--4.56 ms with FP16
dense operands. These ordered measurements share caches and are not independent
training runs or confidence intervals.

The official first-block warmup took 1,319.7 seconds for compiled FP32 and
408.0 seconds for AMP, including compilation and autotuning. Subsequent rows
and commands reuse caches. These are observed startup costs, not independently
cold startup comparisons.

Native mixed precision retains FP32 dense outputs, activations, normalization,
adjoints and optimizer state. The authors' Torch autocast recipe rounds more
intermediate values to FP16 and uses a shared scaler, whereas the native path
uses independent actor and critic scales. These numerical recipes are distinct.

Untimed profiles recorded 4,848 kernel events across eight native FP32 calls,
versus 2,584 with fusion. The fused mixed-precision path recorded 3,416 events,
including 820 FP32-to-FP16 casts. Reducing those conversions is a further tuning
opportunity; event counts do not establish runtime shares or predict a speedup.
Counts select the profiler's kernel category, excluding separately classified
memset/copy events. Category assignments differ across variants, so these are
not conserved counts of all GPU or algorithm operations. Profiling restored
learner state exactly and is excluded from the timing table.

All nine focused diagnostic stages passed: dense and normalization formulas,
projection/EMA, dynamic-scaler overflow handling, eager/captured updates, exact
checkpoint restoration and continued graph replay. Complete-learner floating
differences are descriptive, with a loose loss-sanity gate; these checks do not
establish strict Torch/native trajectory equivalence. No environment learning
or policy quality was measured. This backend remains experimental.

Run [the focused diagnostic](../examples/diagnose_cuda_flashsac.py), then
[the upstream comparison](../examples/benchmark_official_flashsac.py) with
`--native-variants fp32 fp32-fused amp amp-fused --profile-native` to reproduce
the native ablations. Use `--compile-fp32` for the compiled FP32 control and
`--recipe amp --allow-tf32` for the authors' AMP control. Pin the official source
commit above and use `--blocks 3 --warmup 20 --updates 100 --actor-calls 100`
for the recorded budgets.
