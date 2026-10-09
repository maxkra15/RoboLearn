# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Clipped PPO in Warp-NN with fixed-shape, capture-safe optimization.

Original implementation of Schulman et al., *Proximal Policy Optimization
Algorithms* (2017), https://arxiv.org/abs/1707.06347, with generalized advantage
estimation, https://arxiv.org/abs/1506.02438. Neural layers and Adam come from
NVIDIA's Apache-2.0 Warp-NN package; no implementation is copied from RSL-RL.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from warp_nn import nn, optimizers


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


@wp.kernel(enable_backward=False)
def _advance_seed(seed: wp.array(dtype=wp.int32)):
    seed[0] = seed[0] + 1


@wp.kernel(enable_backward=False)
def _sample(
    mean: wp.array2d(dtype=wp.float32),
    value: wp.array2d(dtype=wp.float32),
    log_std: wp.array(dtype=wp.float32),
    scalar_std: int,
    min_std: float,
    max_std: float,
    seed: wp.array(dtype=wp.int32),
    deterministic: int,
    actions: wp.array2d(dtype=wp.float32),
    log_probs: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    sampled_std: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    state = wp.rand_init(seed[0], i)
    log_prob = float(0.0)
    for j in range(mean.shape[1]):
        std = wp.exp(log_std[j])
        action_log_std = log_std[j]
        if scalar_std != 0:
            std = wp.clamp(log_std[j], min_std, max_std)
            action_log_std = wp.log(std)
        if i == 0:
            sampled_std[j] = std
        noise = float(0.0)
        if deterministic == 0:
            noise = wp.randn(state)
        actions[i, j] = mean[i, j] + std * noise
        log_prob = log_prob - 0.5 * (noise * noise + 2.0 * action_log_std + 1.8378770664093453)
    log_probs[i] = log_prob
    values[i] = value[i, 0]


@wp.kernel(enable_backward=False)
def _store(
    step: int,
    obs: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    log_probs: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    rewards: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    next_values: wp.array2d(dtype=wp.float32),
    mean: wp.array2d(dtype=wp.float32),
    sampled_std: wp.array(dtype=wp.float32),
    rollout_obs: wp.array2d(dtype=wp.float32),
    rollout_actions: wp.array2d(dtype=wp.float32),
    rollout_log_probs: wp.array(dtype=wp.float32),
    rollout_values: wp.array(dtype=wp.float32),
    rollout_rewards: wp.array(dtype=wp.float32),
    rollout_terminated: wp.array(dtype=wp.int32),
    rollout_truncated: wp.array(dtype=wp.int32),
    rollout_next_values: wp.array(dtype=wp.float32),
    rollout_means: wp.array2d(dtype=wp.float32),
    rollout_stds: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    k = step * obs.shape[0] + i
    for j in range(obs.shape[1]):
        rollout_obs[k, j] = obs[i, j]
    for j in range(actions.shape[1]):
        rollout_actions[k, j] = actions[i, j]
        rollout_means[k, j] = mean[i, j]
        rollout_stds[k, j] = sampled_std[j]
    rollout_log_probs[k] = log_probs[i]
    rollout_values[k] = values[i]
    rollout_rewards[k] = rewards[i]
    rollout_terminated[k] = terminated[i]
    rollout_truncated[k] = truncated[i]
    rollout_next_values[k] = next_values[i, 0]


@wp.kernel(enable_backward=False)
def _gae(
    horizon: int,
    num_envs: int,
    gamma: float,
    gae_lambda: float,
    current_timeout_bootstrap: int,
    rewards: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    next_values: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    advantages: wp.array(dtype=wp.float32),
    returns: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    advantage = float(0.0)
    for reverse_step in range(horizon):
        k = (horizon - reverse_step - 1) * num_envs + i
        bootstrap = float(1.0)
        continuation = float(1.0)
        if terminated[k] != 0:
            bootstrap = 0.0
        if terminated[k] != 0 or truncated[k] != 0:
            continuation = 0.0
        next_value = next_values[k]
        if current_timeout_bootstrap != 0 and truncated[k] != 0:
            next_value = values[k]
            bootstrap = 1.0
        delta = rewards[k] + gamma * bootstrap * next_value - values[k]
        advantage = delta + gamma * gae_lambda * continuation * advantage
        advantages[k] = advantage
        returns[k] = advantage + values[k]


@wp.kernel(enable_backward=False)
def _advantage_sum(advantages: wp.array(dtype=wp.float32), stats: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(stats, 0, advantages[i])


@wp.kernel(enable_backward=False)
def _advantage_variance(advantages: wp.array(dtype=wp.float32), stats: wp.array(dtype=wp.float32)):
    i = wp.tid()
    centered = advantages[i] - stats[0] / float(advantages.shape[0])
    wp.atomic_add(stats, 1, centered * centered)


@wp.kernel(enable_backward=False)
def _normalize(advantages: wp.array(dtype=wp.float32), stats: wp.array(dtype=wp.float32), sample_std: int):
    i = wp.tid()
    n = float(advantages.shape[0])
    mean = stats[0] / n
    if sample_std != 0:
        variance = stats[1] / wp.max(n - 1.0, 1.0)
        advantages[i] = (advantages[i] - mean) / (wp.sqrt(variance) + 1.0e-8)
    else:
        variance = stats[1] / n
        advantages[i] = (advantages[i] - mean) / wp.sqrt(variance + 1.0e-8)


@wp.kernel(enable_backward=False)
def _shuffle_keys(seed: wp.array(dtype=wp.int32), keys: wp.array(dtype=wp.uint64), indices: wp.array(dtype=wp.int32)):
    i = wp.tid()
    state = wp.rand_init(seed[0], i)
    high = wp.uint64(wp.uint32(wp.randi(state)))
    low = wp.uint64(wp.uint32(wp.randi(state)))
    keys[i] = (high << wp.uint64(32)) | low
    indices[i] = i


@wp.kernel(enable_backward=False)
def _gather_batch(
    offset: int,
    indices: wp.array(dtype=wp.int32),
    observations: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    log_probs: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    advantages: wp.array(dtype=wp.float32),
    returns: wp.array(dtype=wp.float32),
    means: wp.array2d(dtype=wp.float32),
    stds: wp.array2d(dtype=wp.float32),
    batch_observations: wp.array2d(dtype=wp.float32),
    batch_actions: wp.array2d(dtype=wp.float32),
    batch_log_probs: wp.array(dtype=wp.float32),
    batch_values: wp.array(dtype=wp.float32),
    batch_advantages: wp.array(dtype=wp.float32),
    batch_returns: wp.array(dtype=wp.float32),
    batch_means: wp.array2d(dtype=wp.float32),
    batch_stds: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    k = indices[offset + i]
    for j in range(observations.shape[1]):
        batch_observations[i, j] = observations[k, j]
    for j in range(actions.shape[1]):
        batch_actions[i, j] = actions[k, j]
        batch_means[i, j] = means[k, j]
        batch_stds[i, j] = stds[k, j]
    batch_log_probs[i] = log_probs[k]
    batch_values[i] = values[k]
    batch_advantages[i] = advantages[k]
    batch_returns[i] = returns[k]


def _make_kl_kernel(action_dim: int, cfg: PPOConfig):
    @wp.kernel(enable_backward=False, module="unique")
    def kl_kernel(
        mean: wp.array2d(dtype=wp.float32),
        std_parameter: wp.array(dtype=wp.float32),
        old_mean: wp.array2d(dtype=wp.float32),
        old_std: wp.array2d(dtype=wp.float32),
        kl: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        value = float(0.0)
        for j in range(wp.static(action_dim)):
            std = wp.exp(std_parameter[j])
            if wp.static(cfg.std_type == "scalar"):
                std = wp.clamp(std_parameter[j], wp.static(cfg.std_range[0]), wp.static(cfg.std_range[1]))
            variance_ratio = old_std[i, j] / std
            mean_delta = (old_mean[i, j] - mean[i, j]) / std
            # KL(N_old || N_new), with the same diagonal Normal expression as Torch.
            value = value + 0.5 * (variance_ratio * variance_ratio + mean_delta * mean_delta - 1.0)
            value = value - wp.log(variance_ratio)
        wp.atomic_add(kl, 0, value / float(mean.shape[0]))

    wp.set_module_options({"max_unroll": max(action_dim, wp.config.max_unroll)}, module=kl_kernel.module)
    return kl_kernel


@wp.kernel(enable_backward=False)
def _adapt_learning_rate(kl: wp.array(dtype=wp.float32), desired_kl: float, learning_rate: wp.array(dtype=wp.float32)):
    if kl[0] > desired_kl * 2.0:
        learning_rate[0] = wp.max(1.0e-5, learning_rate[0] / 1.5)
    elif kl[0] < desired_kl / 2.0 and kl[0] > 0.0:
        learning_rate[0] = wp.min(1.0e-2, learning_rate[0] * 1.5)


@wp.kernel(enable_backward=False)
def _gradient_sum_squares(gradient: wp.array(dtype=wp.float32), sum_squares: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values = wp.tile_load(gradient, shape=(256,), offset=(i * 256,))
    wp.tile_atomic_add(sum_squares, wp.tile_sum(wp.tile_map(wp.mul, values, values)))


@wp.kernel(enable_backward=False)
def _clip_gradients(gradient: wp.array(dtype=wp.float32), sum_squares: wp.array(dtype=wp.float32), max_norm: float):
    i = wp.tid()
    coefficient = wp.min(1.0, max_norm / (wp.sqrt(sum_squares[0]) + 1.0e-6))
    gradient[i] = gradient[i] * coefficient


def _make_ppo_loss(
    action_dim: int,
    std_type: str = "log",
    std_range: tuple[float, float] = (1.0e-6, 1.0e6),
    value_loss_scale: float = 0.5,
):
    # Unroll the distribution sum: Warp's backward replay must retain the
    # accumulated log probability used by clipping after the action loop.
    @wp.kernel(enable_backward=True, module="unique")
    def loss_kernel(
        mean: wp.array2d(dtype=wp.float32),
        values: wp.array2d(dtype=wp.float32),
        log_std: wp.array(dtype=wp.float32),
        actions: wp.array2d(dtype=wp.float32),
        old_log_probs: wp.array(dtype=wp.float32),
        old_values: wp.array(dtype=wp.float32),
        advantages: wp.array(dtype=wp.float32),
        returns: wp.array(dtype=wp.float32),
        clip_ratio: float,
        value_coefficient: float,
        entropy_coefficient: float,
        clip_value: int,
        loss: wp.array(dtype=wp.float32),
        metrics: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        log_prob = float(0.0)
        entropy = float(0.0)
        for j in range(wp.static(action_dim)):
            action_log_std = log_std[j]
            if wp.static(std_type == "scalar"):
                action_log_std = wp.log(wp.clamp(log_std[j], wp.static(std_range[0]), wp.static(std_range[1])))
            residual = (actions[i, j] - mean[i, j]) * wp.exp(-action_log_std)
            log_prob = log_prob - 0.5 * (residual * residual + 2.0 * action_log_std + 1.8378770664093453)
            entropy = entropy + action_log_std + 1.4189385332046727
        ratio = wp.exp(log_prob - old_log_probs[i])
        policy_loss = -wp.min(
            ratio * advantages[i], wp.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[i]
        )
        error = values[i, 0] - returns[i]
        value_loss = error * error
        if clip_value != 0:
            clipped_value = old_values[i] + wp.clamp(values[i, 0] - old_values[i], -clip_ratio, clip_ratio)
            clipped_error = clipped_value - returns[i]
            value_loss = wp.max(value_loss, clipped_error * clipped_error)
        value_loss = wp.static(value_loss_scale) * value_loss
        scale = 1.0 / float(mean.shape[0])
        wp.atomic_add(loss, 0, scale * (policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy))
        wp.atomic_add(metrics, 0, scale * policy_loss)
        wp.atomic_add(metrics, 1, scale * value_loss)
        wp.atomic_add(metrics, 2, scale * entropy)

    # wp.static is resolved before Warp checks max_unroll. Without this
    # override, larger action spaces still generate a dynamic backward loop.
    wp.set_module_options({"max_unroll": max(action_dim, wp.config.max_unroll)}, module=loss_kernel.module)
    return loss_kernel


def _mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    rng: np.random.Generator,
    optimized_linear_backward: bool = False,
    activation: str = "tanh",
) -> nn.Sequential:
    if optimized_linear_backward:
        from ._linear import TiledLinear

        linear = TiledLinear
    else:
        linear = nn.Linear
    layers = []
    widths = (input_dim, *hidden_dims, output_dim)
    for index, (in_dim, out_dim) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
        layer = linear(in_dim, out_dim, initialize_parameters=False)
        # Simulation callers may disable Warp differentiation globally. Enable
        # it only for the Warp-NN kernels used by this network.
        if not layer._kernel.module.options["enable_backward"]:
            wp.set_module_options({"enable_backward": True}, module=layer._kernel.module)
        bound = 1.0 / math.sqrt(in_dim)
        for parameter in (layer.weight.data, layer.bias.data):
            parameter.assign(rng.uniform(-bound, bound, parameter.shape).astype(np.float32))
        layers.append(layer)
        if index < len(hidden_dims):
            activation_layer = nn.ELU() if activation == "elu" else nn.Tanh()
            for kernel in activation_layer._kernels.values():
                if not kernel.module.options["enable_backward"]:
                    wp.set_module_options({"enable_backward": True}, module=kernel.module)
            layers.append(activation_layer)
    return nn.Sequential(*layers)


class WarpPPO:
    """Fixed-shape continuous-action PPO with optional shuffled minibatches.

    Call :meth:`act`, step the environment, then :meth:`store` for each rollout
    step. ``next_observations`` passed to ``store`` must be the observations
    before an automatic reset, including the terminal observations at timeouts.
    Arrays and kernels remain on one Warp device. ``launch_update`` can be
    included in an outer CUDA capture after ``warmup``.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        num_envs: int,
        horizon: int,
        config: PPOConfig | None = None,
        device: str = "cuda",
    ):
        self.config = config or PPOConfig()
        if min(observation_dim, action_dim, num_envs, horizon, self.config.epochs, *self.config.hidden_dims) < 1:
            raise ValueError("Dimensions, horizon, and epochs must be positive.")
        if self.config.initial_std <= 0:
            raise ValueError("initial_std must be positive.")
        if self.config.activation not in ("tanh", "elu") or self.config.std_type not in ("log", "scalar"):
            raise ValueError("activation must be 'tanh' or 'elu', and std_type must be 'log' or 'scalar'.")
        if self.config.schedule not in ("fixed", "adaptive") or self.config.desired_kl <= 0:
            raise ValueError("schedule must be 'fixed' or 'adaptive', and desired_kl must be positive.")
        if self.config.timeout_bootstrap not in ("next", "current"):
            raise ValueError("timeout_bootstrap must be 'next' or 'current'.")
        if self.config.std_range[0] <= 0 or self.config.std_range[0] > self.config.std_range[1]:
            raise ValueError("std_range must contain positive increasing bounds.")
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("WarpPPO requires a CUDA device.")
        self.num_envs, self.horizon = num_envs, horizon
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.batch_size = num_envs * horizon
        if self.config.num_mini_batches < 1 or self.batch_size % self.config.num_mini_batches:
            raise ValueError("num_mini_batches must be positive and divide the rollout batch size.")
        self.mini_batch_size = self.batch_size // self.config.num_mini_batches
        rng = np.random.default_rng(self.config.seed)
        with wp.ScopedDevice(self.device):
            self.actor = _mlp(
                observation_dim,
                action_dim,
                self.config.hidden_dims,
                rng,
                self.config.optimized_linear_backward,
                self.config.activation,
            )
            self.critic = _mlp(
                observation_dim,
                1,
                self.config.hidden_dims,
                rng,
                self.config.optimized_linear_backward,
                self.config.activation,
            )
            std_initial = self.config.initial_std
            if self.config.std_type == "log":
                std_initial = math.log(std_initial)
            self.distribution_parameter = wp.full(action_dim, std_initial, dtype=wp.float32, requires_grad=True)
            if self.config.std_type == "log":
                self.log_std = self.distribution_parameter
            else:
                self.std_param = self.distribution_parameter
            self.parameters = self.actor.parameters() + self.critic.parameters() + [self.distribution_parameter]
            self.optimizer = optimizers.Adam(
                self.parameters,
                lr=self.config.learning_rate,
                max_norm=None if self.config.separate_grad_clipping else self.config.max_grad_norm,
                disable_graph=True,
                device=self.device,
            )
            self.observations = wp.zeros((self.batch_size, observation_dim), dtype=wp.float32, requires_grad=True)
            self.actions = wp.zeros((self.batch_size, action_dim), dtype=wp.float32)
            self.means = wp.zeros((self.batch_size, action_dim), dtype=wp.float32)
            self.stds = wp.zeros((self.batch_size, action_dim), dtype=wp.float32)
            for name in ("log_probs", "values", "rewards", "next_values", "advantages", "returns"):
                setattr(self, name, wp.zeros(self.batch_size, dtype=wp.float32))
            self.terminated = wp.zeros(self.batch_size, dtype=wp.int32)
            self.truncated = wp.zeros(self.batch_size, dtype=wp.int32)
            self._actions = wp.zeros((num_envs, action_dim), dtype=wp.float32)
            self._log_probs = wp.zeros(num_envs, dtype=wp.float32)
            self._values = wp.zeros(num_envs, dtype=wp.float32)
            self._sampled_std = wp.zeros(action_dim, dtype=wp.float32)
            self._seed = wp.array([self.config.seed], dtype=wp.int32)
            self._stats = wp.zeros(2, dtype=wp.float32)
            self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.metrics = wp.zeros(3, dtype=wp.float32)
            self.kl = wp.zeros(1, dtype=wp.float32)
            if self.config.num_mini_batches > 1:
                self._shuffle_keys = wp.zeros(2 * self.batch_size, dtype=wp.uint64)
                self._shuffle_indices = wp.zeros(2 * self.batch_size, dtype=wp.int32)
                self._batch_observations = wp.zeros(
                    (self.mini_batch_size, observation_dim), dtype=wp.float32, requires_grad=True
                )
                for name in ("actions", "means", "stds"):
                    setattr(self, f"_batch_{name}", wp.zeros((self.mini_batch_size, action_dim), dtype=wp.float32))
                for name in ("log_probs", "values", "advantages", "returns"):
                    setattr(self, f"_batch_{name}", wp.zeros(self.mini_batch_size, dtype=wp.float32))
            else:
                for name in (
                    "observations",
                    "actions",
                    "means",
                    "stds",
                    "log_probs",
                    "values",
                    "advantages",
                    "returns",
                ):
                    setattr(self, f"_batch_{name}", getattr(self, name))
            self._gradient_groups = (
                [p.grad.flatten() for p in self.actor.parameters() + [self.distribution_parameter]],
                [p.grad.flatten() for p in self.critic.parameters()],
            )
            self._group_norms = [wp.zeros(1, dtype=wp.float32), wp.zeros(1, dtype=wp.float32)]
        self._loss_kernel = _make_ppo_loss(
            action_dim, self.config.std_type, self.config.std_range, self.config.value_loss_scale
        )
        self._kl_kernel = _make_kl_kernel(action_dim, self.config)
        self._update_graph = None

    def act(self, observations: wp.array, deterministic: bool = False) -> wp.array:
        """Return raw Gaussian actions in a reusable array, without recording gradients."""
        wp.launch(_advance_seed, dim=1, inputs=[self._seed], device=self.device)
        self._mean = self.actor(observations)
        wp.launch(
            _sample,
            dim=self.num_envs,
            inputs=[
                self._mean,
                self.critic(observations),
                self.distribution_parameter,
                int(self.config.std_type == "scalar"),
                *self.config.std_range,
                self._seed,
                int(deterministic),
            ],
            outputs=[self._actions, self._log_probs, self._values, self._sampled_std],
            device=self.device,
        )
        return self._actions

    def store(
        self,
        step: int,
        observations: wp.array,
        rewards: wp.array,
        terminated: wp.array,
        truncated: wp.array,
        next_observations: wp.array,
    ) -> None:
        """Store the last sampled action and bootstrap from the pre-reset next observation."""
        if not 0 <= step < self.horizon:
            raise ValueError("step must be within the rollout horizon.")
        wp.launch(
            _store,
            dim=self.num_envs,
            inputs=[
                step,
                observations,
                self._actions,
                self._log_probs,
                self._values,
                rewards,
                terminated,
                truncated,
                self.critic(next_observations),
                self._mean,
                self._sampled_std,
            ],
            outputs=[
                self.observations,
                self.actions,
                self.log_probs,
                self.values,
                self.rewards,
                self.terminated,
                self.truncated,
                self.next_values,
                self.means,
                self.stds,
            ],
            device=self.device,
        )

    def compute_returns(self) -> None:
        """Compute GAE, preserving timeout bootstraps and stopping traces at all resets."""
        cfg = self.config
        wp.launch(
            _gae,
            dim=self.num_envs,
            inputs=[
                self.horizon,
                self.num_envs,
                cfg.gamma,
                cfg.gae_lambda,
                int(cfg.timeout_bootstrap == "current"),
                self.rewards,
                self.values,
                self.next_values,
                self.terminated,
                self.truncated,
            ],
            outputs=[self.advantages, self.returns],
            device=self.device,
        )
        if cfg.normalize_advantages:
            self._stats.zero_()
            wp.launch(_advantage_sum, dim=self.batch_size, inputs=[self.advantages, self._stats], device=self.device)
            wp.launch(
                _advantage_variance, dim=self.batch_size, inputs=[self.advantages, self._stats], device=self.device
            )
            wp.launch(
                _normalize,
                dim=self.batch_size,
                inputs=[self.advantages, self._stats, int(cfg.advantage_sample_std)],
                device=self.device,
            )

    def launch_update(self) -> None:
        """Launch the entire PPO update, suitable for inclusion in an external graph.

        Every epoch uses all samples. Shuffling and optional KL-dependent
        learning-rate changes remain on the device with persistent buffers.
        As in RSL-RL, one fresh permutation is reused across rollout epochs.
        """
        cfg = self.config
        self.compute_returns()
        if cfg.num_mini_batches > 1:
            wp.launch(_advance_seed, dim=1, inputs=[self._seed], device=self.device)
            wp.launch(
                _shuffle_keys,
                dim=self.batch_size,
                inputs=[self._seed, self._shuffle_keys, self._shuffle_indices],
                device=self.device,
            )
            wp.utils.radix_sort_pairs(self._shuffle_keys, self._shuffle_indices, self.batch_size)
        for _ in range(cfg.epochs):
            for mini_batch in range(cfg.num_mini_batches):
                if cfg.num_mini_batches > 1:
                    wp.launch(
                        _gather_batch,
                        dim=self.mini_batch_size,
                        inputs=[
                            mini_batch * self.mini_batch_size,
                            self._shuffle_indices,
                            self.observations,
                            self.actions,
                            self.log_probs,
                            self.values,
                            self.advantages,
                            self.returns,
                            self.means,
                            self.stds,
                        ],
                        outputs=[
                            self._batch_observations,
                            self._batch_actions,
                            self._batch_log_probs,
                            self._batch_values,
                            self._batch_advantages,
                            self._batch_returns,
                            self._batch_means,
                            self._batch_stds,
                        ],
                        device=self.device,
                    )
                self.loss.zero_()
                self.metrics.zero_()
                with wp.Tape() as tape:
                    mean = self.actor(self._batch_observations)
                    if cfg.schedule == "adaptive":
                        self.kl.zero_()
                        wp.launch(
                            self._kl_kernel,
                            dim=self.mini_batch_size,
                            inputs=[mean, self.distribution_parameter, self._batch_means, self._batch_stds],
                            outputs=[self.kl],
                            device=self.device,
                            record_tape=False,
                        )
                        wp.launch(
                            _adapt_learning_rate,
                            dim=1,
                            inputs=[self.kl, cfg.desired_kl, self.optimizer._lr],
                            device=self.device,
                            record_tape=False,
                        )
                    wp.launch(
                        self._loss_kernel,
                        dim=self.mini_batch_size,
                        inputs=[
                            mean,
                            self.critic(self._batch_observations),
                            self.distribution_parameter,
                            self._batch_actions,
                            self._batch_log_probs,
                            self._batch_values,
                            self._batch_advantages,
                            self._batch_returns,
                            cfg.clip_ratio,
                            cfg.value_coefficient,
                            cfg.entropy_coefficient,
                            int(cfg.clip_value),
                        ],
                        outputs=[self.loss, self.metrics],
                        device=self.device,
                    )
                tape.backward(self.loss)
                if cfg.separate_grad_clipping:
                    for gradients, sum_squares in zip(self._gradient_groups, self._group_norms, strict=True):
                        sum_squares.zero_()
                        for gradient in gradients:
                            wp.launch_tiled(
                                _gradient_sum_squares,
                                dim=(gradient.size + 255) // 256,
                                inputs=[gradient, sum_squares],
                                device=self.device,
                                block_dim=256,
                            )
                        for gradient in gradients:
                            wp.launch(
                                _clip_gradients,
                                dim=gradient.size,
                                inputs=[gradient, sum_squares, cfg.max_grad_norm],
                                device=self.device,
                            )
                self.optimizer.step()
                tape.zero()

    def health_metrics(self) -> dict[str, float]:
        """Synchronize distribution and optimizer diagnostics at the caller's logging cadence.

        Loss and KL fields describe the last minibatch, and gradient norms
        describe its gradients before separate clipping when that is enabled.
        """
        parameter = self.distribution_parameter.numpy()
        std = np.clip(parameter, *self.config.std_range) if self.config.std_type == "scalar" else np.exp(parameter)
        metrics = self.metrics.numpy()
        result = {
            "std_min": float(std.min()),
            "std_mean": float(std.mean()),
            "std_max": float(std.max()),
            "std_parameter_min": float(parameter.min()),
            "std_parameter_max": float(parameter.max()),
            "learning_rate": float(self.optimizer._lr.numpy()[0]),
            "kl": float(self.kl.numpy()[0]),
            "policy_loss": float(metrics[0]),
            "value_loss": float(metrics[1]),
            "entropy": float(metrics[2]),
            "optimizer_updates": float(self.optimizer._timestep.numpy()[0]),
        }
        if self.config.separate_grad_clipping:
            result["actor_grad_norm"] = float(np.sqrt(self._group_norms[0].numpy()[0]))
            result["critic_grad_norm"] = float(np.sqrt(self._group_norms[1].numpy()[0]))
        return result

    def warmup(self) -> None:
        """Compile learning kernels and allocate caches without changing optimizer state."""
        state = self.state_dict()
        self.launch_update()
        self.load_state_dict(state)

    def update(self, capture: bool = True) -> None:
        """Execute one complete update, capturing and replaying it on CUDA by default."""
        if not capture:
            self.launch_update()
            return
        if self._update_graph is None:
            self.warmup()
            with wp.ScopedCapture(device=self.device) as captured:
                self.launch_update()
            self._update_graph = captured.graph
        wp.capture_launch(self._update_graph)

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return model, Adam, and sampling state; this synchronizes with the device.

        Warp-NN 0.4 does not implement Adam checkpointing. Its moment arrays are
        recorded explicitly here; the Warp-NN version is constrained by the package.
        """
        state = {f"parameter_{i}": p.numpy() for i, p in enumerate(self.parameters)}
        state.update({f"adam_m1_{i}": p.numpy() for i, p in enumerate(self.optimizer._m1)})
        state.update({f"adam_m2_{i}": p.numpy() for i, p in enumerate(self.optimizer._m2)})
        state["adam_timestep"] = self.optimizer._timestep.numpy()
        state["adam_lr"] = self.optimizer._lr.numpy()
        state["seed"] = self._seed.numpy()
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Restore state in place so existing graph pointers remain valid."""
        arrays = {f"parameter_{i}": p for i, p in enumerate(self.parameters)}
        arrays.update({f"adam_m1_{i}": p for i, p in enumerate(self.optimizer._m1)})
        arrays.update({f"adam_m2_{i}": p for i, p in enumerate(self.optimizer._m2)})
        arrays.update(adam_timestep=self.optimizer._timestep, adam_lr=self.optimizer._lr, seed=self._seed)
        if set(state) != set(arrays):
            raise ValueError("Checkpoint parameter or optimizer keys do not match this model.")
        for name, array in arrays.items():
            if state[name].shape != array.shape:
                raise ValueError(f"Checkpoint shape for {name} does not match this model.")
        for name, array in arrays.items():
            array.assign(state[name])

    def save(self, path: str | Path) -> None:
        """Save model and optimizer state to an NPZ checkpoint."""
        with Path(path).open("wb") as file:
            np.savez(file, **self.state_dict())

    def load(self, path: str | Path) -> None:
        """Load an NPZ checkpoint into a model with the same architecture."""
        with np.load(path, allow_pickle=False) as state:
            self.load_state_dict(dict(state))
