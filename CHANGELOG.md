# Changelog

## Unreleased

- Add a four-learner stock Franka cable comparison with common full-state
  evaluation, full cable gravity, and stock RSL observation normalization around
  native PPO. Reuse collection, timing, and checkpoint audits without changing
  learner kernels or the task's training curriculum.

- Add optional experimental native CUDA FP32 dense forward/backward and fused
  multi-tensor Adam for Warp PPO and Warp FlashSAC, preserving networks, losses,
  optimizer arrays and the default Warp backend.
- Add native FlashSAC FP16 dense operands with FP32 outputs and optimizer state,
  dynamic loss scaling and checkpointed attempt/skip counters. Fuse normalization,
  ReLU, categorical operations, parameter projections and target EMA.
- Reuse owned FP16 conversion buffers and batch verified gradient clears. Add
  opt-in `fresh_training_gradients` for private residual adjoints, with ownership
  checks, activation metadata and ordinary accumulation/full-clear fallback;
  complete clearing remains the default.
- Add preparation-only cuBLASLt tuning with private scratch, fixed captured plans
  and recorded choices, plus balanced FlashSAC benchmarks with matched initial
  state and update counts.
- Add standalone numerical, capture and checkpoint diagnostics and synthetic
  update benchmarks; performance and learning quality require measured results.
- Align optional Torch FlashSAC compilation with the authors' categorical and
  reward helpers, preserving eager execution and per-transition discounts.
- Keep learner imports lazy and `PPOConfig` independent of optional dependencies;
  explicitly warn that Warp FlashSAC is experimental.
- Document Isaac Lab logging/checkpoints, backend constraints and the separate
  requirements for learning through simulator gradients.

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
