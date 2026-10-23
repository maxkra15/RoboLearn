# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""PPO configuration without importing the optional Warp runtime."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    hidden_dims: tuple[int, ...] = (64, 64)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    epochs: int = 4
    max_grad_norm: float = 1.0
    initial_std: float = 1.0
    normalize_advantages: bool = True
    clip_value: bool = True
    seed: int = 42
    optimized_linear_backward: bool = False
    activation: str = "tanh"
    std_type: str = "log"
    std_range: tuple[float, float] = (1.0e-6, 1.0e6)
    num_mini_batches: int = 1
    schedule: str = "fixed"
    desired_kl: float = 0.01
    value_loss_scale: float = 0.5
    advantage_sample_std: bool = False
    timeout_bootstrap: str = "next"
    separate_grad_clipping: bool = False
    dense_backend: str = "warp"
