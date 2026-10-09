# Copyright (c) 2026 Holiday Robotics
# Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
# Derived from Holiday-Robot/FlashSAC, revision 87edc9061150ae9e962dd84e6544e27a1554b3ab.

"""Shared FlashSAC configuration without optional learner dependencies."""

from dataclasses import dataclass


@dataclass
class FlashSACConfig:
    """FlashSAC hyperparameters; dimensions and simulator settings belong to the caller.

    Compilation and AMP are opt-in. ``buffer_device=None`` uses ``device``.
    The defaults retain the authors' networks and Isaac Lab update parameters;
    ``learning_rate_decay_step`` counts gradient updates.
    """

    device: str = "cuda:0"
    buffer_device: str | None = None
    seed: int = 0
    normalize_reward: bool = True
    normalized_G_max: float = 5.0
    asymmetric_observation: bool = False
    buffer_max_length: int = 1_000_000
    buffer_min_length: int = 100_000
    sample_batch_size: int = 2048
    learning_rate_init: float = 3e-4
    learning_rate_peak: float = 3e-4
    learning_rate_end: float = 1.5e-4
    learning_rate_warmup_rate: float = 1e-6
    learning_rate_warmup_step: int = 0
    learning_rate_decay_rate: float = 1.0
    learning_rate_decay_step: int = 10_000
    actor_num_blocks: int = 2
    actor_hidden_dim: int = 128
    actor_bc_alpha: float = 0.0
    actor_noise_zeta_mu: float = 2.0
    actor_noise_zeta_max: int = 16
    actor_update_period: int = 2
    critic_num_blocks: int = 2
    critic_hidden_dim: int = 256
    critic_num_bins: int = 101
    critic_min_v: float = -5.0
    critic_max_v: float = 5.0
    critic_target_update_tau: float = 0.01
    temp_initial_value: float = 0.01
    temp_target_sigma: float = 0.15
    temp_target_entropy: float | None = None
    gamma: float = 0.99
    n_step: int = 3
    use_compile: bool = False
    compile_mode: str = "auto"
    use_amp: bool = False
    load_optimizer: bool = True
    load_reward_normalizer: bool = True

    def __post_init__(self):
        if (
            min(
                self.buffer_max_length,
                self.buffer_min_length,
                self.sample_batch_size,
                self.n_step,
                self.actor_update_period,
                self.learning_rate_decay_step,
            )
            <= 0
        ):
            raise ValueError("Buffer sizes, batch size, horizons, and update periods must be positive.")
        if self.buffer_min_length > self.buffer_max_length:
            raise ValueError("Replay warmup cannot exceed buffer capacity.")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1].")
        if self.critic_num_bins < 2 or self.critic_min_v >= self.critic_max_v:
            raise ValueError("The categorical critic needs at least two bins and an increasing support.")

    @property
    def device_type(self) -> str:
        return self.device

    @property
    def buffer_device_type(self) -> str:
        return self.buffer_device or self.device
