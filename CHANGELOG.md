# Changelog

## Unreleased

- Added opt-in RSL-style Warp PPO settings: ELU networks, directly learned
  Gaussian standard deviation, shuffled minibatches, device-side adaptive KL
  learning rates, separate gradient clipping, and matching value/GAE conventions.
- Added deferred-cadence Warp PPO health metrics and a standalone fixed-rollout
  Torch/Warp numerical diagnostic.
- Added optional unclipped Isaac Lab action passthrough for native Gaussian PPO
  policies, preserving FlashSAC's default normalized action bounds.
- Added an opt-in detached device-tensor metric result to the adapted FlashSAC
  update API, allowing callers to defer host scalar reads until logging.
- Added opt-in tiled Linear gradients for Warp PPO, using split-batch parameter
  reductions while preserving Warp-NN forward layers and checkpoint format.

## 0.1.0

- Packaged the authors' PyTorch FlashSAC implementation with a tensor API and
  optional dependencies.
- Added an Isaac Lab 3 adapter and training/evaluation example.
- Added WarpNN PPO with full-batch updates, GAE, CUDA capture, and checkpoints.
- Added direct joint capture of MuJoCo Warp physics and PPO updates.
- Added numerical, checkpoint, and capture tests, attribution files, and package CI.
