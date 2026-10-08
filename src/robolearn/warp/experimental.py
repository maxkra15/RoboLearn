# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Experimental short-horizon actor gradients through a differentiable simulator.

Inspired by SHAC: Jie Xu, Viktor Makoviychuk, Yashraj Narang, Fabio Ramos,
Wojciech Matusik, Animesh Garg, and Miles Macklin, *Accelerated Policy Learning
with Parallel Differentiable Simulation* (2022), https://arxiv.org/abs/2204.07137.

This original prototype uses deterministic actors and detached n-step critic
targets. It is not a reproduction of SHAC's stochastic policy, TD-lambda critic,
normalization, or target-network schedule. Neural layers and Adam are dependencies
from NVIDIA's Apache-2.0 Warp-NN package. Simulator derivatives are supplied by
the caller; graph capture alone does not provide them.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import warp as wp
from warp_nn import nn, optimizers

from .ppo import _mlp


@dataclass(frozen=True)
class PathwiseConfig:
    """Settings for the experimental fixed-shape actor/critic update."""

    hidden_dims: tuple[int, ...] = (32, 32)
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    critic_epochs: int = 4
    max_grad_norm: float = 1.0
    seed: int = 42


class DifferentiableRollout(Protocol):
    """Out-of-place physics with persistent, unique arrays for every time step.

    ``step`` returns the original reward and an int32 continuation mask: zero
    at true termination, one otherwise. Time limits retain their value bootstrap.
    No reset occurs inside the taped rollout. The environment masks rewards after
    termination and resets failed/timed-out rows in ``after_update`` outside the
    tape. Segment-start state copies must also happen outside the tape.
    """

    num_envs: int
    obs_dim: int
    action_dim: int
    device: wp.Device

    def prepare(self, horizon: int) -> None: ...

    def begin_rollout(self) -> None: ...

    def observe(self, step: int) -> wp.array: ...

    def step(self, step: int, actions: wp.array) -> tuple[wp.array, wp.array]: ...

    def finish_rollout(self) -> wp.array: ...

    def after_update(self) -> None: ...


@wp.kernel(enable_backward=False)
def _continuation(
    previous: wp.array(dtype=wp.int32),
    continuation: wp.array(dtype=wp.int32),
    result: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    result[i] = previous[i] * continuation[i]


@wp.kernel(enable_backward=True)
def _reward_objective(
    rewards: wp.array(dtype=wp.float32),
    alive: wp.array(dtype=wp.int32),
    discount: float,
    loss: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    wp.atomic_add(loss, 0, -discount * float(alive[i]) * rewards[i] / float(rewards.shape[0]))


@wp.kernel(enable_backward=True)
def _bootstrap_objective(
    values: wp.array2d(dtype=wp.float32),
    alive: wp.array(dtype=wp.int32),
    discount: float,
    loss: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    wp.atomic_add(loss, 0, -discount * float(alive[i]) * values[i, 0] / float(values.shape[0]))


@wp.kernel(enable_backward=False)
def _store_detached(
    step: int,
    observations: wp.array2d(dtype=wp.float32),
    rewards: wp.array(dtype=wp.float32),
    continuation: wp.array(dtype=wp.int32),
    alive: wp.array(dtype=wp.int32),
    batch_observations: wp.array2d(dtype=wp.float32),
    batch_rewards: wp.array(dtype=wp.float32),
    batch_continuation: wp.array(dtype=wp.int32),
    batch_alive: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    k = step * observations.shape[0] + i
    for j in range(observations.shape[1]):
        batch_observations[k, j] = observations[i, j]
    batch_rewards[k] = rewards[i]
    batch_continuation[k] = continuation[i]
    batch_alive[k] = alive[i]


@wp.kernel(enable_backward=False)
def _n_step_targets(
    horizon: int,
    num_envs: int,
    gamma: float,
    rewards: wp.array(dtype=wp.float32),
    continuation: wp.array(dtype=wp.int32),
    final_values: wp.array2d(dtype=wp.float32),
    targets: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    value = final_values[i, 0]
    for reverse in range(horizon):
        k = (horizon - reverse - 1) * num_envs + i
        value = rewards[k] + gamma * float(continuation[k]) * value
        targets[k] = value


@wp.kernel(enable_backward=True)
def _critic_objective(
    values: wp.array2d(dtype=wp.float32),
    targets: wp.array(dtype=wp.float32),
    alive: wp.array(dtype=wp.int32),
    loss: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    error = values[i, 0] - targets[i]
    wp.atomic_add(loss, 0, 0.5 * float(alive[i]) * error * error / float(values.shape[0]))


def _independent_caches(network: nn.Sequential) -> nn.Sequential:
    """Share parameters while retaining a distinct activation cache at each step."""
    layers = []
    for layer in network.modules():
        clone = copy(layer)
        # Warp-NN keys cached outputs by shape/dtype, not invocation. Reusing a
        # layer across time would overwrite values needed by the taped backward.
        clone._cache = {}
        layers.append(clone)
    return nn.Sequential(*layers)


class WarpPathwiseActorCritic:
    """Experimental actor/critic learning through short differentiable rollouts.

    Actors output ``tanh`` actions in [-1, 1]; scaling belongs to the environment.
    The actor maximizes discounted native rewards plus a frozen critic endpoint
    value. Critic weights are copied before each rollout into parameter arrays
    without gradients; derivatives through its input still reach the actor.
    Critic fitting uses detached observations and n-step targets, excluding rows
    after termination. Environment state is detached between updates.

    ``launch_update`` contains no host reads. Fixed buffers permit outer CUDA
    capture after a warmup, provided the environment's physics adjoint and reset
    paths support it. Check correctness before relying on captured execution.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        num_envs: int,
        horizon: int,
        config: PathwiseConfig | None = None,
        device: str = "cuda:0",
    ):
        self.config = config or PathwiseConfig()
        cfg = self.config
        if min(observation_dim, action_dim, num_envs, horizon, cfg.critic_epochs, *cfg.hidden_dims) < 1:
            raise ValueError("Dimensions, horizon, and critic epochs must be positive.")
        if not 0.0 <= cfg.gamma <= 1.0 or min(cfg.actor_learning_rate, cfg.critic_learning_rate) <= 0:
            raise ValueError("gamma must be in [0, 1] and learning rates must be positive.")
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("WarpPathwiseActorCritic requires a CUDA device.")
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.num_envs, self.horizon = num_envs, horizon
        self.batch_size = num_envs * horizon
        self._prepared_environment = None
        rng = np.random.default_rng(cfg.seed)
        with wp.ScopedDevice(self.device):
            self.actor = _mlp(observation_dim, action_dim, cfg.hidden_dims, rng)
            final_actor = list(self.actor.modules())[-1]
            final_actor.weight.data.assign(final_actor.weight.data.numpy() * 0.01)
            final_actor.bias.data.zero_()
            bounded_action = nn.Tanh()
            for kernel in bounded_action._kernels.values():
                if not kernel.module.options["enable_backward"]:
                    wp.set_module_options({"enable_backward": True}, module=kernel.module)
            self.actor.register_module("bounded_action", bounded_action)
            self.critic = _mlp(observation_dim, 1, cfg.hidden_dims, rng)
            final_critic = list(self.critic.modules())[-1]
            final_critic.weight.data.zero_()
            final_critic.bias.data.zero_()
            self.frozen_critic = _mlp(observation_dim, 1, cfg.hidden_dims, rng)
            for parameter in self.frozen_critic.parameters():
                parameter.requires_grad = False
            self.actor_parameters = self.actor.parameters()
            self.critic_parameters = self.critic.parameters()
            self.actor_optimizer = optimizers.Adam(
                self.actor_parameters,
                lr=cfg.actor_learning_rate,
                max_norm=cfg.max_grad_norm,
                disable_graph=True,
                device=self.device,
            )
            self.critic_optimizer = optimizers.Adam(
                self.critic_parameters,
                lr=cfg.critic_learning_rate,
                max_norm=cfg.max_grad_norm,
                disable_graph=True,
                device=self.device,
            )
            self.actor_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.critic_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.observations = wp.zeros((self.batch_size, observation_dim), dtype=wp.float32)
            self.rewards = wp.zeros(self.batch_size, dtype=wp.float32)
            self.targets = wp.zeros_like(self.rewards)
            self.continuation = wp.zeros(self.batch_size, dtype=wp.int32)
            self.alive = wp.zeros_like(self.continuation)
            self._alive = [wp.zeros(num_envs, dtype=wp.int32) for _ in range(horizon + 1)]
        self._actors = [_independent_caches(self.actor) for _ in range(horizon)]
        self._evaluation_actor = _independent_caches(self.actor)

    def act(self, observations: wp.array) -> wp.array:
        """Return deterministic bounded actions using a cache separate from training."""
        return self._evaluation_actor(observations)

    def forward_actor(self, environment: DifferentiableRollout) -> wp.Tape:
        """Record the actor objective without performing backward or an update.

        This entry point permits a standalone finite-difference diagnostic with
        fixed initial physical state and critic parameters. Call ``tape.zero()``
        after inspecting gradients. No parameters change in this method.
        """
        if self._prepared_environment is not environment:
            if (environment.num_envs, environment.obs_dim, environment.action_dim, environment.device) != (
                self.num_envs,
                self.observation_dim,
                self.action_dim,
                self.device,
            ):
                raise ValueError("Environment dimensions/device do not match this learner.")
            environment.prepare(self.horizon)
            self._prepared_environment = environment
        for source, frozen in zip(self.critic_parameters, self.frozen_critic.parameters(), strict=True):
            wp.copy(frozen, source)
        environment.begin_rollout()
        self.actor_loss.zero_()
        self._alive[0].fill_(1)
        with wp.Tape() as tape:
            for step in range(self.horizon):
                observations = environment.observe(step)
                rewards, continuation = environment.step(step, self._actors[step](observations))
                wp.launch(
                    _reward_objective,
                    dim=self.num_envs,
                    inputs=[rewards, self._alive[step], self.config.gamma**step],
                    outputs=[self.actor_loss],
                    device=self.device,
                )
                wp.launch(
                    _continuation,
                    dim=self.num_envs,
                    inputs=[self._alive[step], continuation],
                    outputs=[self._alive[step + 1]],
                    device=self.device,
                    record_tape=False,
                )
                wp.launch(
                    _store_detached,
                    dim=self.num_envs,
                    inputs=[step, observations, rewards, continuation, self._alive[step]],
                    outputs=[self.observations, self.rewards, self.continuation, self.alive],
                    device=self.device,
                    record_tape=False,
                )
            self._final_values = self.frozen_critic(environment.finish_rollout())
            wp.launch(
                _bootstrap_objective,
                dim=self.num_envs,
                inputs=[self._final_values, self._alive[-1], self.config.gamma**self.horizon],
                outputs=[self.actor_loss],
                device=self.device,
            )
        return tape

    def launch_update(self, environment: DifferentiableRollout) -> None:
        """Launch one short physics rollout, actor update, and detached critic fit."""
        tape = self.forward_actor(environment)
        tape.backward(self.actor_loss)
        wp.launch(
            _n_step_targets,
            dim=self.num_envs,
            inputs=[
                self.horizon,
                self.num_envs,
                self.config.gamma,
                self.rewards,
                self.continuation,
                self._final_values,
            ],
            outputs=[self.targets],
            device=self.device,
        )
        self.actor_optimizer.step()
        tape.zero()
        for _ in range(self.config.critic_epochs):
            self.critic_loss.zero_()
            with wp.Tape() as critic_tape:
                wp.launch(
                    _critic_objective,
                    dim=self.batch_size,
                    inputs=[self.critic(self.observations), self.targets, self.alive],
                    outputs=[self.critic_loss],
                    device=self.device,
                )
            critic_tape.backward(self.critic_loss)
            self.critic_optimizer.step()
            critic_tape.zero()
        environment.after_update()

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return parameters and Adam state; configuration/physics are not included."""
        state = {}
        for name, parameters, optimizer in (
            ("actor", self.actor_parameters, self.actor_optimizer),
            ("critic", self.critic_parameters, self.critic_optimizer),
        ):
            state.update({f"{name}_parameter_{i}": parameter.numpy() for i, parameter in enumerate(parameters)})
            state.update({f"{name}_adam_m1_{i}": moment.numpy() for i, moment in enumerate(optimizer._m1)})
            state.update({f"{name}_adam_m2_{i}": moment.numpy() for i, moment in enumerate(optimizer._m2)})
            state[f"{name}_adam_timestep"] = optimizer._timestep.numpy()
            state[f"{name}_adam_lr"] = optimizer._lr.numpy()
        return state

    def save(self, path: str | Path) -> None:
        """Save an NPZ checkpoint for inspecting or resuming the same architecture."""
        with Path(path).open("wb") as file:
            np.savez(file, **self.state_dict())

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Restore parameters/Adam in place, retaining pointers held by a graph."""
        arrays = {}
        for name, parameters, optimizer in (
            ("actor", self.actor_parameters, self.actor_optimizer),
            ("critic", self.critic_parameters, self.critic_optimizer),
        ):
            arrays.update({f"{name}_parameter_{i}": parameter for i, parameter in enumerate(parameters)})
            arrays.update({f"{name}_adam_m1_{i}": moment for i, moment in enumerate(optimizer._m1)})
            arrays.update({f"{name}_adam_m2_{i}": moment for i, moment in enumerate(optimizer._m2)})
            arrays[f"{name}_adam_timestep"] = optimizer._timestep
            arrays[f"{name}_adam_lr"] = optimizer._lr
        if set(arrays) != set(state):
            raise ValueError("Checkpoint keys do not match this architecture.")
        for name, array in arrays.items():
            if array.shape != state[name].shape:
                raise ValueError(f"Checkpoint shape for {name} does not match this architecture.")
        for name, array in arrays.items():
            array.assign(state[name])

    def load(self, path: str | Path) -> None:
        """Load a checkpoint into the same architecture/configuration."""
        with np.load(path, allow_pickle=False) as state:
            self.load_state_dict(dict(state))
