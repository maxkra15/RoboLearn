# Changelog

## Unreleased

- Added opt-in tiled Linear gradients for Warp PPO, using split-batch parameter
  reductions while preserving Warp-NN forward layers and checkpoint format.

## 0.1.0

- Packaged the authors' PyTorch FlashSAC implementation with a tensor API and
  optional dependencies.
- Added an Isaac Lab 3 adapter and training/evaluation example.
- Added WarpNN PPO with full-batch updates, GAE, CUDA capture, and checkpoints.
- Added direct joint capture of MuJoCo Warp physics and PPO updates.
- Added numerical, checkpoint, and capture tests, attribution files, and package CI.
