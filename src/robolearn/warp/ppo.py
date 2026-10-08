# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Full-batch clipped PPO in Warp-NN.

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


@wp.kernel(enable_backward=False)
def _advance_seed(seed: wp.array(dtype=wp.int32)):
    seed[0] = seed[0] + 1


@wp.kernel(enable_backward=False)
def _sample(
    mean: wp.array2d(dtype=wp.float32),
    value: wp.array2d(dtype=wp.float32),
    log_std: wp.array(dtype=wp.float32),
    seed: wp.array(dtype=wp.int32),
    deterministic: int,
    actions: wp.array2d(dtype=wp.float32),
    log_probs: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    state = wp.rand_init(seed[0], i)
    log_prob = float(0.0)
    for j in range(mean.shape[1]):
        noise = float(0.0)
        if deterministic == 0:
            noise = wp.randn(state)
        actions[i, j] = mean[i, j] + wp.exp(log_std[j]) * noise
        log_prob = log_prob - 0.5 * (noise * noise + 2.0 * log_std[j] + 1.8378770664093453)
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
    rollout_obs: wp.array2d(dtype=wp.float32),
    rollout_actions: wp.array2d(dtype=wp.float32),
    rollout_log_probs: wp.array(dtype=wp.float32),
    rollout_values: wp.array(dtype=wp.float32),
    rollout_rewards: wp.array(dtype=wp.float32),
    rollout_terminated: wp.array(dtype=wp.int32),
    rollout_truncated: wp.array(dtype=wp.int32),
    rollout_next_values: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    k = step * obs.shape[0] + i
    for j in range(obs.shape[1]):
        rollout_obs[k, j] = obs[i, j]
    for j in range(actions.shape[1]):
        rollout_actions[k, j] = actions[i, j]
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
        delta = rewards[k] + gamma * bootstrap * next_values[k] - values[k]
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
def _normalize(advantages: wp.array(dtype=wp.float32), stats: wp.array(dtype=wp.float32)):
    i = wp.tid()
    n = float(advantages.shape[0])
    mean = stats[0] / n
    variance = stats[1] / n
    advantages[i] = (advantages[i] - mean) / wp.sqrt(variance + 1.0e-8)


def _make_ppo_loss(action_dim: int):
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
            residual = (actions[i, j] - mean[i, j]) * wp.exp(-log_std[j])
            log_prob = log_prob - 0.5 * (residual * residual + 2.0 * log_std[j] + 1.8378770664093453)
            entropy = entropy + log_std[j] + 1.4189385332046727
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
        value_loss = 0.5 * value_loss
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
            activation = nn.Tanh()
            for kernel in activation._kernels.values():
                if not kernel.module.options["enable_backward"]:
                    wp.set_module_options({"enable_backward": True}, module=kernel.module)
            layers.append(activation)
    return nn.Sequential(*layers)


class WarpPPO:
    """Fixed-shape continuous-action PPO with full-batch optimization.

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
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("WarpPPO requires a CUDA device.")
        self.num_envs, self.horizon = num_envs, horizon
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.batch_size = num_envs * horizon
        rng = np.random.default_rng(self.config.seed)
        with wp.ScopedDevice(self.device):
            self.actor = _mlp(
                observation_dim, action_dim, self.config.hidden_dims, rng, self.config.optimized_linear_backward
            )
            self.critic = _mlp(observation_dim, 1, self.config.hidden_dims, rng, self.config.optimized_linear_backward)
            self.log_std = wp.full(action_dim, math.log(self.config.initial_std), dtype=wp.float32, requires_grad=True)
            self.parameters = self.actor.parameters() + self.critic.parameters() + [self.log_std]
            self.optimizer = optimizers.Adam(
                self.parameters,
                lr=self.config.learning_rate,
                max_norm=self.config.max_grad_norm,
                disable_graph=True,
                device=self.device,
            )
            self.observations = wp.zeros((self.batch_size, observation_dim), dtype=wp.float32, requires_grad=True)
            self.actions = wp.zeros((self.batch_size, action_dim), dtype=wp.float32)
            for name in ("log_probs", "values", "rewards", "next_values", "advantages", "returns"):
                setattr(self, name, wp.zeros(self.batch_size, dtype=wp.float32))
            self.terminated = wp.zeros(self.batch_size, dtype=wp.int32)
            self.truncated = wp.zeros(self.batch_size, dtype=wp.int32)
            self._actions = wp.zeros((num_envs, action_dim), dtype=wp.float32)
            self._log_probs = wp.zeros(num_envs, dtype=wp.float32)
            self._values = wp.zeros(num_envs, dtype=wp.float32)
            self._seed = wp.array([self.config.seed], dtype=wp.int32)
            self._stats = wp.zeros(2, dtype=wp.float32)
            self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.metrics = wp.zeros(3, dtype=wp.float32)
        self._loss_kernel = _make_ppo_loss(action_dim)
        self._update_graph = None

    def act(self, observations: wp.array, deterministic: bool = False) -> wp.array:
        """Return raw Gaussian actions in a reusable array, without recording gradients."""
        wp.launch(_advance_seed, dim=1, inputs=[self._seed], device=self.device)
        wp.launch(
            _sample,
            dim=self.num_envs,
            inputs=[self.actor(observations), self.critic(observations), self.log_std, self._seed, int(deterministic)],
            outputs=[self._actions, self._log_probs, self._values],
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
            wp.launch(_normalize, dim=self.batch_size, inputs=[self.advantages, self._stats], device=self.device)

    def launch_update(self) -> None:
        """Launch the entire PPO update, suitable for inclusion in an external graph.

        Every epoch uses all rollout samples. No host reads, changing shapes,
        random minibatches, or KL-dependent early exits occur in this update.
        """
        cfg = self.config
        self.compute_returns()
        for _ in range(cfg.epochs):
            self.loss.zero_()
            self.metrics.zero_()
            with wp.Tape() as tape:
                wp.launch(
                    self._loss_kernel,
                    dim=self.batch_size,
                    inputs=[
                        self.actor(self.observations),
                        self.critic(self.observations),
                        self.log_std,
                        self.actions,
                        self.log_probs,
                        self.values,
                        self.advantages,
                        self.returns,
                        cfg.clip_ratio,
                        cfg.value_coefficient,
                        cfg.entropy_coefficient,
                        int(cfg.clip_value),
                    ],
                    outputs=[self.loss, self.metrics],
                    device=self.device,
                )
            tape.backward(self.loss)
            self.optimizer.step()
            tape.zero()

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
