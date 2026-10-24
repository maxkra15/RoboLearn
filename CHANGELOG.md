# Changelog

## Unreleased

- Reuse consumer-owned FP16 buffers at private native FlashSAC BatchNorm/dense
  edges, preserving FP32 outputs, accumulated adjoints and explicit-conversion rounding.
- Add opt-in `fresh_training_gradients` for verified private residual adjoints,
  with ownership checks, ordinary accumulation/full-clear fallback and host activation
  metadata. Keep complete clearing by default; expose the option in balanced comparisons.

- Align optional Torch FlashSAC compilation with the authors' categorical and
  reward helpers while preserving the eager path and per-transition discounts.

- Add opt-in preparation-time cuBLASLt plan tuning with private scratch buffers,
  fixed captured choices, and recorded candidate timings. Add a balanced
  fresh-process FlashSAC comparison with matched initialization and update counts.

- Batch experimental native FlashSAC adjoint clears with verified persistent
  pointer inventories and reduce BatchNorm backward coefficients once per channel.

- Optimized experimental native FlashSAC with projection/EMA-managed FP16 weight
  copies, vectorized conversions, tiled batch normalization, and cooperative
  categorical heads and target projection.

- Added opt-in FP16 dense compute with FP32 master weights and Adam, dynamic
  loss scaling, and fused normalization/ReLU, parameter projections and target
  EMA for the experimental native FlashSAC backend.
- Added opt-in experimental native CUDA dense forward/backward and fused
  multi-tensor Adam for Warp PPO and Warp FlashSAC, with unchanged architectures,
  loss definitions, optimizer state arrays, and default Warp backend.
- Added standalone numerical diagnostics and synthetic update benchmarks for
  the CUDA backend; performance and learning quality require measured results.
- Load Warp learners lazily so selecting PPO does not import experimental FlashSAC.
- Allow `PPOConfig` imports without installing Torch, Warp, or Warp-NN; preserve
  existing public imports and configuration defaults.
- Emit an explicit experimental-backend warning when constructing Warp FlashSAC.
- Document Isaac Lab logging/checkpoint support, measured FlashSAC bottlenecks,
  and the separate requirements for learning through simulator gradients.

## 0.3.0 - 2026-10-09

- Added an experimental FP32 Warp-NN FlashSAC backend with the authors' residual
  networks, categorical critics, update order, persistent replay and captured
  learning updates. Mixed precision and G1 speed/quality improvements are not
  established.
- Separated the shared FlashSAC configuration from optional Torch imports.
- Added fixed-order BatchNorm, RMSNorm, and LogSoftmax backward reductions to the experimental
  Warp FlashSAC backend, retaining Warp-NN forward statistics and parameters.
- Documented the fixed-budget Cartpole comparison protocol and experimental
  learning/runtime limitations from an audited four-recipe, three-seed cohort.

The initial Cartpole comparison measures a frozen 0.2.0 working snapshot of this
backend. This release metadata bump does not change its learner implementation.

## 0.2.0 - 2026-10-09

- Added opt-in RSL-style Warp PPO settings: ELU networks, directly learned
  Gaussian standard deviation, shuffled minibatches, device-side adaptive KL
  learning rates, separate gradient clipping, and matching value/GAE conventions.
- Added deferred-cadence Warp PPO health metrics and a standalone fixed-rollout
  Torch/Warp numerical diagnostic.
- Added optional unclipped Isaac Lab action passthrough for native Gaussian PPO
  policies, preserving FlashSAC's default normalized action bounds.
- Fixed replay action ownership for the unclipped Isaac Lab adapter: later
  caller mutations no longer alter a returned transition.
- Added an opt-in detached device-tensor metric result to the adapted FlashSAC
  update API, allowing callers to defer host scalar reads until logging.
- Preserve compiled FlashSAC exploration state and returned actions across
  subsequent CUDA graph invocations using independently owned tensor storage.
- Added opt-in tiled Linear gradients for Warp PPO, using split-batch parameter
  reductions while preserving Warp-NN forward layers and checkpoint format.
- Documented the Warp PPO settings and dependency versions used in the G1 runs.
- Added one CPU regression for compiled sampling output ownership; extended the
  existing learner and adapter checks for device metrics and unclipped actions.

## 0.1.0

- Packaged the authors' PyTorch FlashSAC implementation with a tensor API and
  optional dependencies.
- Added an Isaac Lab 3 adapter and training/evaluation example.
- Added WarpNN PPO with full-batch updates, GAE, CUDA capture, and checkpoints.
- Added direct joint capture of MuJoCo Warp physics and PPO updates.
- Added numerical, checkpoint, and capture tests, attribution files, and package CI.
