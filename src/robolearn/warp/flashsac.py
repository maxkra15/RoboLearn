# SPDX-FileCopyrightText: Copyright (c) 2026 Holiday Robotics
# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Experimental FP32 FlashSAC updates in Warp-NN.

Adapted from Holiday-Robot/FlashSAC, revision
87edc9061150ae9e962dd84e6544e27a1554b3ab, accompanying
https://arxiv.org/abs/2604.04539. The actor, temperature, distributional critic
and target update order follows the authors' implementation. Warp-NN supplies
the neural layers and Adam. FP16 AMP is not implemented by this backend.

Returned actions, sampled batches and device metrics use persistent storage.
Consume or copy them before the next call that reuses the same output. Random
streams and floating point reduction order differ from the Torch backend.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import warp as wp
from warp_nn import optimizers

from robolearn.flashsac.config import FlashSACConfig

from . import _flash_replay
from ._flash_networks import FlashActor, FlashDoubleCritic
from ._flash_replay import WarpFlashReplay, WarpRewardNormalizer


@wp.kernel(enable_backward=False)
def _initialize_rng(seed: int, states: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    states[i] = wp.rand_init(seed, i)


@wp.kernel(enable_backward=False)
def _normal_noise(states: wp.array(dtype=wp.uint32), noise: wp.array2d(dtype=wp.float32)):
    i = wp.tid()
    state = states[i]
    for j in range(noise.shape[1]):
        noise[i, j] = wp.randn(state)
    states[i] = state


@wp.kernel(enable_backward=False)
def _actor_observations(
    observations: wp.array2d(dtype=wp.float32),
    next_observations: wp.array2d(dtype=wp.float32),
    joined: wp.array2d(dtype=wp.float32),
    next_actor: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    joined[i, j] = observations[i, j]
    joined[i + observations.shape[0], j] = next_observations[i, j]
    next_actor[i, j] = next_observations[i, j]


@wp.kernel(enable_backward=False)
def _actor_prefix(observations: wp.array2d(dtype=wp.float32), actor: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    actor[i, j] = observations[i, j]


@wp.kernel
def _sample_policy(
    mean: wp.array2d(dtype=wp.float32),
    log_std: wp.array2d(dtype=wp.float32),
    noise: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    log_probs: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    log_prob = float(0.0)
    for j in range(mean.shape[1]):
        epsilon = noise[i, j]
        raw = mean[i, j] + wp.exp(log_std[i, j]) * epsilon
        actions[i, j] = wp.tanh(raw)
        # Stable softplus(-2*raw) implements the authors' tanh Jacobian.
        value = -2.0 * raw
        softplus = float(0.0)
        if value >= 0.0:
            softplus = value + wp.log(1.0 + wp.exp(-value))
        else:
            softplus = wp.log(1.0 + wp.exp(value))
        correction = 2.0 * (0.6931471805599453 - raw - softplus)
        log_prob = log_prob - 0.5 * epsilon * epsilon - log_std[i, j] - 0.9189385332046727 - correction
    log_probs[i] = log_prob


@wp.kernel
def _first_actions(actions: wp.array2d(dtype=wp.float32), first: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    first[i, j] = actions[i, j]


@wp.kernel(enable_backward=False)
def _temperature_value(log_temperature: wp.array(dtype=wp.float32), alpha: wp.array(dtype=wp.float32)):
    alpha[0] = wp.exp(log_temperature[0])


@wp.kernel(enable_backward=False)
def _actor_diagnostics(
    q: wp.array2d(dtype=wp.float32),
    log_probs: wp.array(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    entropy: wp.array(dtype=wp.float32),
    mean_action: wp.array(dtype=wp.float32),
    q_abs: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n = float(actions.shape[0])
    wp.atomic_add(entropy, 0, -log_probs[i] / n)
    wp.atomic_add(q_abs, 0, wp.abs(wp.min(q[0, i], q[1, i])) / n)
    value = float(0.0)
    for j in range(actions.shape[1]):
        value = value + actions[i, j]
    wp.atomic_add(mean_action, 0, value / (n * float(actions.shape[1])))


@wp.kernel
def _actor_objective(
    q: wp.array2d(dtype=wp.float32),
    log_probs: wp.array(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    replay_actions: wp.array2d(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    q_abs: wp.array(dtype=wp.float32),
    bc_alpha: float,
    loss: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    # torch.minimum splits its derivative equally at an exact tie.
    value = 0.5 * (q[0, i] + q[1, i])
    if q[0, i] < q[1, i]:
        value = q[0, i]
    elif q[1, i] < q[0, i]:
        value = q[1, i]
    objective = alpha[0] * log_probs[i] - value
    if bc_alpha > 0.0:
        error = float(0.0)
        for j in range(actions.shape[1]):
            delta = actions[i, j] - replay_actions[i, j]
            error = error + delta * delta
        objective = objective + bc_alpha * q_abs[0] * error / float(actions.shape[1])
    wp.atomic_add(loss, 0, objective / float(actions.shape[0]))


@wp.kernel
def _temperature_objective(
    log_temperature: wp.array(dtype=wp.float32),
    entropy: wp.array(dtype=wp.float32),
    target_entropy: float,
    loss: wp.array(dtype=wp.float32),
):
    loss[0] = wp.exp(log_temperature[0]) * (entropy[0] - target_entropy)


@wp.kernel(enable_backward=False)
def _critic_inputs(
    observations: wp.array2d(dtype=wp.float32),
    next_observations: wp.array2d(dtype=wp.float32),
    replay_actions: wp.array2d(dtype=wp.float32),
    next_actions: wp.array2d(dtype=wp.float32),
    joined_observations: wp.array2d(dtype=wp.float32),
    joined_actions: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    n = observations.shape[0]
    if j < observations.shape[1]:
        joined_observations[i, j] = observations[i, j]
        joined_observations[n + i, j] = next_observations[i, j]
    if j < replay_actions.shape[1]:
        joined_actions[i, j] = replay_actions[i, j]
        joined_actions[n + i, j] = next_actions[i, j]


@wp.kernel(enable_backward=False)
def _categorical_target(
    q: wp.array2d(dtype=wp.float32),
    log_probs: wp.array3d(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.float32),
    discount: wp.array(dtype=wp.float32),
    next_log_probs: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    min_v: float,
    max_v: float,
    target: wp.array2d(dtype=wp.float32),
    maximum_entropy: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    n = target.shape[0]
    bins = target.shape[1]
    width = (max_v - min_v) / float(bins - 1)
    entropy = alpha[0] * next_log_probs[i]
    wp.atomic_max(maximum_entropy, 0, entropy)
    # argmin selects the first critic at a tie, as in the authors' target.
    selected = int(0)
    if q[1, n + i] < q[0, n + i]:
        selected = 1
    for j in range(bins):
        target[i, j] = 0.0
    for j in range(bins):
        support = min_v + float(j) * width
        value = reward[i] + discount[i] * (support - entropy) * (1.0 - terminated[i])
        location = (wp.clamp(value, min_v, max_v) - min_v) / width
        lower = wp.clamp(int(wp.floor(location)), 0, bins - 1)
        upper = wp.min(lower + 1, bins - 1)
        fraction = location - float(lower)
        probability = wp.exp(log_probs[selected, n + i, j])
        target[i, lower] = target[i, lower] + probability * (1.0 - fraction)
        target[i, upper] = target[i, upper] + probability * fraction


@wp.kernel
def _critic_objective(
    log_probs: wp.array3d(dtype=wp.float32),
    target: wp.array2d(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    critic, i = wp.tid()
    value = float(0.0)
    for j in range(target.shape[1]):
        value = value - target[i, j] * log_probs[critic, i, j]
    wp.atomic_add(loss, 0, value / (2.0 * float(target.shape[0])))


@wp.kernel(enable_backward=False)
def _ema(source: wp.array(dtype=wp.float32), target: wp.array(dtype=wp.float32), tau: float):
    i = wp.tid()
    target[i] = (1.0 - tau) * target[i] + tau * source[i]


@wp.kernel(enable_backward=False)
def _schedule(
    timestep: wp.array(dtype=wp.float32),
    initial: float,
    peak: float,
    end: float,
    warmup_steps: int,
    decay_steps: int,
    learning_rate: wp.array(dtype=wp.float32),
):
    step = timestep[0]
    value = end
    if step < float(warmup_steps):
        value = initial + (peak - initial) * step / float(warmup_steps)
    elif step < float(decay_steps):
        fraction = (step - float(warmup_steps)) / float(decay_steps - warmup_steps)
        value = end + 0.5 * (peak - end) * (1.0 + wp.cos(3.141592653589793 * fraction))
    learning_rate[0] = value


@wp.kernel(enable_backward=False)
def _advance_updates(counters: wp.array(dtype=wp.int64), actor: int):
    counters[0] = counters[0] + wp.int64(1)
    counters[1] = counters[1] + wp.int64(actor)
    counters[2] = counters[2] + wp.int64(actor)


@wp.kernel(enable_backward=False)
def _exploration_repeat(
    state: wp.array(dtype=wp.uint32),
    cdf: wp.array(dtype=wp.float32),
    repetition: wp.array(dtype=wp.int32),
    refresh: wp.array(dtype=wp.int32),
):
    rng = state[0]
    value = wp.randf(rng)
    length = cdf.shape[0]
    for j in range(cdf.shape[0]):
        if value < cdf[j]:
            length = j + 1
            break
    state[0] = rng
    reset = repetition[0] == 0 or repetition[0] >= repetition[1]
    refresh[0] = int(reset)
    if reset:
        repetition[0] = 0
        repetition[1] = length
    repetition[0] = repetition[0] + 1


@wp.kernel(enable_backward=False)
def _exploration_actions(
    mean: wp.array2d(dtype=wp.float32),
    log_std: wp.array2d(dtype=wp.float32),
    state: wp.array(dtype=wp.uint32),
    refresh: wp.array(dtype=wp.int32),
    cached_noise: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    rng = state[i]
    for j in range(mean.shape[1]):
        # Generate a candidate every step even while retaining repeated noise.
        candidate = wp.randn(rng)
        if refresh[0] != 0:
            cached_noise[i, j] = candidate
        actions[i, j] = wp.tanh(mean[i, j] + wp.exp(log_std[i, j]) * cached_noise[i, j])
    state[i] = rng


@wp.kernel(enable_backward=False)
def _deterministic_actions(mean: wp.array2d(dtype=wp.float32), actions: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    actions[i, j] = wp.tanh(mean[i, j])


class WarpFlashSAC:
    """Authors' FlashSAC recipe with persistent FP32 Warp optimization buffers.

    ``update`` performs one critic update; call it twice per vector environment
    step for the paper recipe. The first update and every ``actor_update_period``
    thereafter also update actor and temperature. Call ``prepare(capture=True)`` before timing
    to compile kernels and allocate neural caches without advancing training.

    CUDA capture covers replay sampling and learning, not the caller's simulator
    or Torch MDP. ``launch_update`` can be included in an external graph once
    prepared; the caller then owns host readiness and phase bookkeeping.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        num_envs: int = 1,
        cfg: FlashSACConfig | None = None,
        critic_observation_dim: int | None = None,
        *,
        optimized_linear_backward: bool = True,
    ):
        cfg = cfg or FlashSACConfig()
        if cfg.use_amp:
            raise ValueError("WarpFlashSAC currently implements FP32 only; set use_amp=False.")
        if min(observation_dim, action_dim, num_envs) < 1:
            raise ValueError("Observation, action and environment dimensions must be positive.")
        total_dim = critic_observation_dim or observation_dim
        if total_dim < observation_dim:
            raise ValueError("Critic observations must contain the actor observation prefix.")
        if cfg.temp_initial_value <= 0 or cfg.temp_target_sigma <= 0 or cfg.actor_noise_zeta_max < 1:
            raise ValueError("Temperature and exploration distribution parameters must be positive.")
        if not 0 <= cfg.critic_target_update_tau <= 1:
            raise ValueError("Target EMA tau must lie in [0, 1].")
        if cfg.learning_rate_warmup_step < 0 or cfg.learning_rate_warmup_step >= cfg.learning_rate_decay_step:
            raise ValueError("Learning-rate warmup must be nonnegative and shorter than decay.")
        target_entropy = 0.5 * action_dim * math.log(2.0 * math.pi * math.e * cfg.temp_target_sigma**2)
        self.cfg = self.config = replace(
            cfg, asymmetric_observation=critic_observation_dim is not None, temp_target_entropy=target_entropy
        )
        self.device = wp.get_device(cfg.device)
        self.observation_dim = observation_dim
        self.critic_observation_dim = total_dim
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.batch_size = cfg.sample_batch_size
        self.optimized_linear_backward = optimized_linear_backward
        self._update_step = 0
        self._prepared = False
        self._update_graphs = {}
        self._retained_tapes = []
        with wp.ScopedDevice(self.device):
            self.actor = FlashActor(
                observation_dim,
                action_dim,
                cfg.actor_hidden_dim,
                cfg.actor_num_blocks,
                seed=cfg.seed,
                device=self.device,
                optimized_linear_backward=optimized_linear_backward,
            )
            critic_args = (
                total_dim,
                action_dim,
                cfg.critic_hidden_dim,
                cfg.critic_num_blocks,
                cfg.critic_num_bins,
                cfg.critic_min_v,
                cfg.critic_max_v,
            )
            self.critic = FlashDoubleCritic(
                *critic_args,
                seed=cfg.seed + 1,
                device=self.device,
                optimized_linear_backward=optimized_linear_backward,
            )
            self.target_critic = FlashDoubleCritic(
                *critic_args,
                seed=cfg.seed + 1,
                device=self.device,
                optimized_linear_backward=optimized_linear_backward,
                requires_grad=False,
            )
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.log_temperature = wp.array([math.log(cfg.temp_initial_value)], dtype=wp.float32, requires_grad=True)
            self.actor_optimizer = self._optimizer(self.actor.parameters())
            self.critic_optimizer = self._optimizer(self.critic.parameters())
            self.temperature_optimizer = self._optimizer([self.log_temperature])
            self.update_counters = wp.zeros(3, dtype=wp.int64)
            self._replay_buffer = WarpFlashReplay(total_dim, action_dim, num_envs, self.cfg, self.device)
            self.replay = self._replay_buffer
            self.batch = self.replay.batch
            self.reward_normalizer = (
                WarpRewardNormalizer(num_envs, self.cfg, self.device) if cfg.normalize_reward else None
            )
            b = self.batch_size
            self._actor_observations = wp.zeros((2 * b, observation_dim), dtype=wp.float32)
            self._actor_next_observations = wp.zeros((b, observation_dim), dtype=wp.float32)
            self._critic_observations = wp.zeros((2 * b, total_dim), dtype=wp.float32)
            self._critic_actions = wp.zeros((2 * b, action_dim), dtype=wp.float32)
            self.actor_noise = wp.zeros((2 * b, action_dim), dtype=wp.float32)
            self.next_noise = wp.zeros((b, action_dim), dtype=wp.float32)
            self._actor_rng = wp.zeros(2 * b, dtype=wp.uint32)
            self._next_rng = wp.zeros(b, dtype=wp.uint32)
            self._actor_actions = wp.zeros((2 * b, action_dim), dtype=wp.float32, requires_grad=True)
            self._actor_log_probs = wp.zeros(2 * b, dtype=wp.float32, requires_grad=True)
            self._first_actions = wp.zeros((b, action_dim), dtype=wp.float32, requires_grad=True)
            self._next_actions = wp.zeros((b, action_dim), dtype=wp.float32)
            self._next_log_probs = wp.zeros(b, dtype=wp.float32)
            self.target_probabilities = wp.zeros((b, cfg.critic_num_bins), dtype=wp.float32)
            self.actor_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.critic_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.temperature_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.entropy = wp.zeros(1, dtype=wp.float32)
            self.mean_action = wp.zeros(1, dtype=wp.float32)
            self._q_abs = wp.zeros(1, dtype=wp.float32)
            self.alpha = wp.zeros(1, dtype=wp.float32)
            self.temperature_value = wp.zeros(1, dtype=wp.float32)
            self.maximum_entropy_bonus = wp.zeros(1, dtype=wp.float32)
            self._exploration_rng = wp.zeros(num_envs, dtype=wp.uint32)
            self._repeat_rng = wp.zeros(1, dtype=wp.uint32)
            self._repetition = wp.array([0, 1], dtype=wp.int32)
            self._refresh = wp.zeros(1, dtype=wp.int32)
            self._cached_noise = wp.zeros((num_envs, action_dim), dtype=wp.float32)
            lengths = np.arange(1, cfg.actor_noise_zeta_max + 1, dtype=np.float32)
            probabilities = lengths ** (-cfg.actor_noise_zeta_mu)
            self._zeta_cdf = wp.array(np.cumsum(probabilities / probabilities.sum()), dtype=wp.float32)
            self._action_cache = {}
            for offset, states in (
                (11, self._actor_rng),
                (23, self._next_rng),
                (37, self._exploration_rng),
                (41, self._repeat_rng),
            ):
                wp.launch(_initialize_rng, dim=states.size, inputs=[cfg.seed + offset, states], device=self.device)
        # Preserve parameter-only EMA: target BatchNorm buffers evolve in target forward.
        source = dict(self.critic.named_parameters())
        target = dict(self.target_critic.named_parameters())
        if set(source) != set(target):
            raise ValueError("Online and target critic parameter names differ.")
        self._ema_parameters = [(source[name].data.flatten(), target[name].data.flatten()) for name in source]
        self.metrics = {
            "actor/loss": self.actor_loss,
            "actor/entropy": self.entropy,
            "actor/mean_action": self.mean_action,
            "critic/loss": self.critic_loss,
            "critic/max_entropy_bonus": self.maximum_entropy_bonus,
            "temperature/loss": self.temperature_loss,
            "temperature/value": self.temperature_value,
        }

    def _optimizer(self, parameters):
        return optimizers.Adam(
            parameters,
            lr=self.cfg.learning_rate_init,
            device=self.device,
            disable_graph=True,
            betas=(0.9, 0.999),
            eps=1.0e-8,
        )

    def parameters(self) -> list[wp.array]:
        """Return learnable Warp arrays; target critic parameters are excluded."""
        return self.actor.parameters() + self.critic.parameters() + [self.log_temperature]

    def act(self, observations: wp.array, training: bool = True) -> wp.array:
        """Sample normalized actions; deterministic inference leaves exploration state unchanged."""
        if observations.ndim != 2 or observations.shape[1] < self.observation_dim or observations.dtype != wp.float32:
            raise ValueError("act expects FP32 observations with the configured actor prefix.")
        n = observations.shape[0]
        if training and n != self.num_envs:
            raise ValueError("Training action batches must match num_envs.")
        if n not in self._action_cache:
            self._action_cache[n] = (
                wp.empty((n, self.observation_dim), dtype=wp.float32, device=self.device),
                wp.empty((n, self.action_dim), dtype=wp.float32, device=self.device),
            )
        actor_observations, actions = self._action_cache[n]
        wp.launch(
            _actor_prefix, dim=actor_observations.shape, inputs=[observations, actor_observations], device=self.device
        )
        mean, log_std = self.actor(actor_observations, training=False)
        if training:
            wp.launch(
                _exploration_repeat,
                dim=1,
                inputs=[self._repeat_rng, self._zeta_cdf, self._repetition, self._refresh],
                device=self.device,
            )
            wp.launch(
                _exploration_actions,
                dim=n,
                inputs=[mean, log_std, self._exploration_rng, self._refresh, self._cached_noise, actions],
                device=self.device,
            )
        else:
            wp.launch(_deterministic_actions, dim=actions.shape, inputs=[mean, actions], device=self.device)
        return actions

    def sample_actions(self, interaction_step: int, prev_transition: Mapping, training: bool) -> wp.array:
        """Compatibility entry point; exploration repeat counts advance on the device."""
        return self.act(prev_transition["next_observation"], training=training)

    def observe(self, observations, actions, rewards, terminated, truncated, next_observations) -> None:
        """Store pre-reset transitions; input done flags must be int32 Warp arrays."""
        self.replay.add(observations, actions, rewards, terminated, truncated, next_observations)
        if self.reward_normalizer is not None:
            self.reward_normalizer.update(rewards, terminated, truncated)

    def process_transition(self, transition: Mapping) -> None:
        self.observe(
            *(
                transition[key]
                for key in (
                    "observation",
                    "action",
                    "reward",
                    "terminated",
                    "truncated",
                    "next_observation",
                )
            )
        )

    @property
    def ready(self) -> bool:
        return self.replay.can_sample()

    def can_start_training(self) -> bool:
        return self.ready

    def _schedule_optimizer(self, optimizer) -> None:
        cfg = self.cfg
        wp.launch(
            _schedule,
            dim=1,
            inputs=[
                optimizer._timestep,
                cfg.learning_rate_init,
                cfg.learning_rate_peak,
                cfg.learning_rate_end,
                cfg.learning_rate_warmup_step,
                cfg.learning_rate_decay_step,
                optimizer._lr,
            ],
            device=self.device,
        )

    def _step(self, optimizer) -> None:
        self._schedule_optimizer(optimizer)
        optimizer.step()
        # Saved LR has the same post-step scheduler position as Torch LambdaLR.
        self._schedule_optimizer(optimizer)

    def _update_actor(self, apply_optimizer: bool) -> None:
        self.actor_loss.zero_()
        self.entropy.zero_()
        self.mean_action.zero_()
        self._q_abs.zero_()
        wp.launch(_temperature_value, dim=1, inputs=[self.log_temperature, self.alpha], device=self.device)
        with self.critic.freeze_parameters():
            with wp.Tape() as tape:
                mean, log_std = self.actor(self._actor_observations, training=True)
                wp.launch(
                    _sample_policy,
                    dim=2 * self.batch_size,
                    inputs=[mean, log_std, self.actor_noise],
                    outputs=[self._actor_actions, self._actor_log_probs],
                    device=self.device,
                )
                wp.launch(
                    _first_actions,
                    dim=self._first_actions.shape,
                    inputs=[self._actor_actions],
                    outputs=[self._first_actions],
                    device=self.device,
                )
                q, _ = self.critic(self.batch["observation"], self._first_actions, training=False)
                wp.launch(
                    _actor_diagnostics,
                    dim=self.batch_size,
                    inputs=[q, self._actor_log_probs, self._first_actions, self.entropy, self.mean_action, self._q_abs],
                    device=self.device,
                    record_tape=False,
                )
                wp.launch(
                    _actor_objective,
                    dim=self.batch_size,
                    inputs=[
                        q,
                        self._actor_log_probs,
                        self._first_actions,
                        self.batch["action"],
                        self.alpha,
                        self._q_abs,
                        self.cfg.actor_bc_alpha,
                    ],
                    outputs=[self.actor_loss],
                    device=self.device,
                )
            tape.backward(self.actor_loss)
            if apply_optimizer:
                self._step(self.actor_optimizer)
                self.actor.normalize_parameters()
                tape.zero()
            else:
                self._retained_tapes.append(tape)

    def _update_temperature(self, apply_optimizer: bool) -> None:
        self.temperature_loss.zero_()
        wp.launch(_temperature_value, dim=1, inputs=[self.log_temperature, self.temperature_value], device=self.device)
        with wp.Tape() as tape:
            wp.launch(
                _temperature_objective,
                dim=1,
                inputs=[self.log_temperature, self.entropy, self.cfg.temp_target_entropy],
                outputs=[self.temperature_loss],
                device=self.device,
            )
        tape.backward(self.temperature_loss)
        if apply_optimizer:
            self._step(self.temperature_optimizer)
            tape.zero()
        else:
            self._retained_tapes.append(tape)

    def _update_critic(self, reward: wp.array, apply_optimizer: bool) -> None:
        mean, log_std = self.actor(self._actor_next_observations, training=False)
        wp.launch(
            _sample_policy,
            dim=self.batch_size,
            inputs=[mean, log_std, self.next_noise],
            outputs=[self._next_actions, self._next_log_probs],
            device=self.device,
            record_tape=False,
        )
        wp.launch(_temperature_value, dim=1, inputs=[self.log_temperature, self.alpha], device=self.device)
        wp.launch(
            _critic_inputs,
            dim=(self.batch_size, max(self.critic_observation_dim, self.action_dim)),
            inputs=[
                self.batch["observation"],
                self.batch["next_observation"],
                self.batch["action"],
                self._next_actions,
                self._critic_observations,
                self._critic_actions,
            ],
            device=self.device,
        )
        target_q, target_lp = self.target_critic(self._critic_observations, self._critic_actions, training=True)
        self.maximum_entropy_bonus.fill_(-1.0e30)
        wp.launch(
            _categorical_target,
            dim=self.batch_size,
            inputs=[
                target_q,
                target_lp,
                reward,
                self.batch["terminated"],
                self.batch["discount"],
                self._next_log_probs,
                self.alpha,
                self.cfg.critic_min_v,
                self.cfg.critic_max_v,
                self.target_probabilities,
                self.maximum_entropy_bonus,
            ],
            device=self.device,
        )
        self.critic_loss.zero_()
        with wp.Tape() as tape:
            _, log_probs = self.critic(self._critic_observations, self._critic_actions, training=True)
            wp.launch(
                _critic_objective,
                dim=(2, self.batch_size),
                inputs=[log_probs, self.target_probabilities],
                outputs=[self.critic_loss],
                device=self.device,
            )
        tape.backward(self.critic_loss)
        if apply_optimizer:
            self._step(self.critic_optimizer)
            self.critic.normalize_parameters()
            tape.zero()
        else:
            self._retained_tapes.append(tape)

    def launch_update(
        self,
        do_actor_update: bool = True,
        *,
        sample: bool = True,
        randomize_noise: bool = True,
        apply_optimizer: bool = True,
        normalize_reward: bool = True,
    ) -> None:
        """Launch one static learning phase, without host reads or phase increments.

        Prepare first before external capture. ``do_actor_update`` is a static
        graph choice; device counters and RNG advance each replay. ``sample=False``
        uses ``batch``, ``actor_noise`` and ``next_noise`` for numerical diagnostics.
        """
        if sample:
            self.replay.sample(check_ready=False)
        reward = self.batch["reward"]
        if normalize_reward and self.reward_normalizer is not None:
            reward = self.reward_normalizer.normalize_rewards(reward)
        wp.launch(
            _actor_observations,
            dim=(self.batch_size, self.observation_dim),
            inputs=[
                self.batch["observation"],
                self.batch["next_observation"],
                self._actor_observations,
                self._actor_next_observations,
            ],
            device=self.device,
        )
        if do_actor_update:
            if randomize_noise:
                wp.launch(
                    _normal_noise,
                    dim=2 * self.batch_size,
                    inputs=[self._actor_rng, self.actor_noise],
                    device=self.device,
                )
            self._update_actor(apply_optimizer)
            self._update_temperature(apply_optimizer)
        if randomize_noise:
            wp.launch(_normal_noise, dim=self.batch_size, inputs=[self._next_rng, self.next_noise], device=self.device)
        self._update_critic(reward, apply_optimizer)
        if apply_optimizer:
            for source, target in self._ema_parameters:
                wp.launch(
                    _ema,
                    dim=source.size,
                    inputs=[source, target, self.cfg.critic_target_update_tau],
                    device=self.device,
                )
            wp.launch(_advance_updates, dim=1, inputs=[self.update_counters, int(do_actor_update)], device=self.device)

    def update_from_batch(
        self,
        batch: Mapping,
        *,
        actor_noise=None,
        next_noise=None,
        do_actor_update: bool | None = None,
        apply_optimizer: bool = True,
        normalize_reward: bool = False,
        tensor_metrics: bool = True,
    ) -> dict:
        """Use an explicit fixed batch/noise for eager numerical comparisons.

        Inputs can be Warp or NumPy arrays with configured shapes. Prescribed
        noises are standard Normal samples before the reparameterization, of
        shapes ``(2B, A)`` and ``(B, A)``. With ``apply_optimizer=False``, gradients
        remain available on parameters and weights/counters are unchanged; BN
        running statistics still follow the authors' training forwards.
        """
        for key, destination in self.batch.items():
            self._assign(destination, batch[key], key)
        prescribed = actor_noise is not None or next_noise is not None
        if prescribed and (actor_noise is None or next_noise is None):
            raise ValueError("Supply both actor_noise and next_noise for fixed-noise diagnostics.")
        if prescribed:
            self._assign(self.actor_noise, actor_noise, "actor_noise")
            self._assign(self.next_noise, next_noise, "next_noise")
        actor_phase = (
            self._update_step % self.cfg.actor_update_period == 0 if do_actor_update is None else do_actor_update
        )
        # Clear retained diagnostic gradients before recording another tape.
        self._clear_gradients()
        self.launch_update(
            actor_phase,
            sample=False,
            randomize_noise=not prescribed,
            apply_optimizer=apply_optimizer,
            normalize_reward=normalize_reward,
        )
        if apply_optimizer:
            self._update_step += 1
        return self._metrics(actor_phase, tensor_metrics)

    def prepare(self, *, capture: bool = False) -> None:
        """Precompile inference and both learner phases while preserving training state.

        Warmup uses valid dummy batch arrays, so it works before replay readiness.
        Replay storage is neither copied nor modified. Sampling kernels are
        compiled explicitly before capture; batch scratch is overwritten later.
        """
        if self._prepared:
            if capture:
                self.capture_update()
            return
        state = self.state_dict(include_replay=False)
        try:
            for key, array in self.batch.items():
                array.fill_(self.cfg.gamma**self.cfg.n_step if key == "discount" else 0.0)
            observations = wp.zeros((self.num_envs, self.critic_observation_dim), dtype=wp.float32, device=self.device)
            self.act(observations, training=False)
            self.act(observations, training=True)
            for phase in (True, False):
                self.launch_update(phase, sample=False)
            wp.load_module(module=_flash_replay._sample_indices.module, device=self.device)
            self._prepared = True
        finally:
            self.load_state_dict(state)
        if capture:
            self.capture_update()

    warmup = prepare

    def capture_update(self) -> None:
        """Capture both static learning phases before measurement, without executing them."""
        self.prepare(capture=False)
        if not self.device.is_cuda:
            return
        for phase in (True, False):
            if phase not in self._update_graphs:
                with wp.ScopedCapture(device=self.device) as captured:
                    self.launch_update(phase)
                self._update_graphs[phase] = captured.graph

    def update(self, *, tensor_metrics: bool = False, capture: bool = True) -> dict:
        """Perform one replay update, choosing the actor phase from the host mirror."""
        if not self.ready:
            raise RuntimeError("Replay has not reached buffer_min_length.")
        actor_phase = self._update_step % self.cfg.actor_update_period == 0
        if capture and self.device.is_cuda:
            self.capture_update()
            wp.capture_launch(self._update_graphs[actor_phase])
        else:
            self.launch_update(actor_phase)
        self._update_step += 1
        return self._metrics(actor_phase, tensor_metrics)

    def _metrics(self, actor_phase: bool, tensor_metrics: bool) -> dict:
        values = {key: value for key, value in self.metrics.items() if actor_phase or key.startswith("critic/")}
        return values if tensor_metrics else {key: float(value.numpy()[0]) for key, value in values.items()}

    def get_metrics(self) -> dict:
        """Synchronize diagnostics at an explicit logging boundary."""
        result = self._metrics(True, False)
        result.update(
            actor_optimizer_steps=float(self.actor_optimizer._timestep.numpy()[0]),
            critic_optimizer_steps=float(self.critic_optimizer._timestep.numpy()[0]),
            temperature_optimizer_steps=float(self.temperature_optimizer._timestep.numpy()[0]),
        )
        return result

    def _clear_gradients(self) -> None:
        for tape in self._retained_tapes:
            tape.zero()
        self._retained_tapes.clear()
        for parameter in self.parameters():
            parameter.grad.zero_()

    @staticmethod
    def _assign(destination, value, name: str) -> None:
        if tuple(value.shape) != destination.shape:
            raise ValueError(f"Shape of {name} differs from {destination.shape}.")
        if isinstance(value, wp.array):
            if value.dtype != destination.dtype:
                raise ValueError(f"Dtype of {name} differs from {destination.dtype}.")
            wp.copy(destination, value)
        else:
            destination.assign(np.asarray(value))

    def _state_arrays(self) -> dict:
        arrays = {}
        for name, network in (("actor", self.actor), ("critic", self.critic), ("target_critic", self.target_critic)):
            arrays.update({f"{name}.{key}": value for key, value in network.state_dict().items()})
        for name, optimizer in (
            ("actor", self.actor_optimizer),
            ("critic", self.critic_optimizer),
            ("temperature", self.temperature_optimizer),
        ):
            arrays.update({f"optimizer.{name}.m1.{i}": value for i, value in enumerate(optimizer._m1)})
            arrays.update({f"optimizer.{name}.m2.{i}": value for i, value in enumerate(optimizer._m2)})
            arrays[f"optimizer.{name}.timestep"] = optimizer._timestep
            arrays[f"optimizer.{name}.lr"] = optimizer._lr
        arrays.update(
            {
                "log_temperature": self.log_temperature,
                "update_counters": self.update_counters,
                "rng.actor": self._actor_rng,
                "rng.next": self._next_rng,
                "rng.exploration": self._exploration_rng,
                "rng.repeat": self._repeat_rng,
                "exploration.repetition": self._repetition,
                "exploration.cached_noise": self._cached_noise,
            }
        )
        return arrays

    def state_dict(self, include_replay: bool = True) -> dict[str, np.ndarray]:
        """Snapshot weights, BN buffers, Adam, RNG, normalizer and replay state.

        ``include_replay=False`` omits only storage, retaining replay metadata
        for in-place warmup restoration. It is not a portable replay checkpoint.
        Routine ``save`` marks this omission and resets replay on disk load.
        Warp-NN 0.4 Adam moments are saved explicitly because its checkpoint
        methods are not implemented.
        """
        state = {key: value.numpy().copy() for key, value in self._state_arrays().items()}
        state.update(
            {f"replay.{key}": value for key, value in self.replay.state_dict(include_storage=include_replay).items()}
        )
        if self.reward_normalizer is not None:
            state.update({f"normalizer.{key}": value for key, value in self.reward_normalizer.state_dict().items()})
        state["replay_storage_included"] = np.array([include_replay], dtype=np.bool_)
        return state

    def load_state_dict(self, state: Mapping) -> None:
        """Restore arrays in place, retaining replay population if storage was omitted."""
        arrays = self._state_arrays()
        for key, array in arrays.items():
            if key not in state or tuple(state[key].shape) != array.shape:
                raise ValueError(f"Checkpoint array {key} is missing or has the wrong shape.")
        for key, array in arrays.items():
            self._assign(array, state[key], key)
        replay_state = {key.removeprefix("replay."): value for key, value in state.items() if key.startswith("replay.")}
        self.replay.load_state_dict(replay_state, restore_storage=bool(state["replay_storage_included"][0]))
        if self.reward_normalizer is not None:
            self.reward_normalizer.load_state_dict(
                {
                    key.removeprefix("normalizer."): value
                    for key, value in state.items()
                    if key.startswith("normalizer.")
                }
            )
        self._update_step = int(np.asarray(state["update_counters"])[0])
        self._clear_gradients()

    def save(self, path: str | Path, *, include_replay: bool = False) -> None:
        """Save a NumPy/JSON checkpoint; omitted replay is explicitly reset on load."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "robolearn-warp-flashsac-v1",
            "precision": "float32",
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "critic_observation_dim": self.critic_observation_dim if self.cfg.asymmetric_observation else None,
            "num_envs": self.num_envs,
            "config": asdict(self.cfg),
            "optimized_linear_backward": self.optimized_linear_backward,
            "replay_included": include_replay,
        }
        state = self.state_dict(include_replay=include_replay)
        if not include_replay:
            state = {key: value for key, value in state.items() if not key.startswith("replay.")}
        with (directory / "config.json").open("w") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
        with (directory / "agent.npz").open("wb") as stream:
            np.savez(stream, **state)

    def load(self, path: str | Path) -> None:
        """Load a checkpoint with matching dimensions and restore allocated arrays in place."""
        directory = Path(path)
        metadata = json.loads((directory / "config.json").read_text())
        if metadata.get("format") != "robolearn-warp-flashsac-v1":
            raise ValueError("Unrecognized WarpFlashSAC checkpoint format.")
        if metadata["observation_dim"] != self.observation_dim or metadata["action_dim"] != self.action_dim:
            raise ValueError("Checkpoint observation/action dimensions differ from this agent.")
        saved_critic_dim = metadata["critic_observation_dim"] or metadata["observation_dim"]
        if saved_critic_dim != self.critic_observation_dim:
            raise ValueError("Checkpoint critic observation dimension differs from this agent.")
        with np.load(directory / "agent.npz", allow_pickle=False) as archive:
            state = dict(archive)
        if metadata["replay_included"] and metadata["num_envs"] != self.num_envs:
            raise ValueError("Exact replay continuation requires the original num_envs.")
        if not metadata["replay_included"]:
            self.replay.reset()
            state.update(
                {f"replay.{key}": value for key, value in self.replay.state_dict(include_storage=False).items()}
            )
            if metadata["num_envs"] != self.num_envs:
                # Evaluation can use fewer worlds than training. Fresh exploration
                # state is independent of the loaded policy and optimizer weights.
                state.update(
                    {
                        key: value.numpy()
                        for key, value in self._state_arrays().items()
                        if key
                        in ("rng.exploration", "rng.repeat", "exploration.repetition", "exploration.cached_noise")
                    }
                )
            if self.reward_normalizer is not None:
                state["normalizer.returns"] = np.zeros(self.num_envs, dtype=np.float32)
        if not self.cfg.load_optimizer:
            state.update(
                {key: value.numpy() for key, value in self._state_arrays().items() if key.startswith("optimizer.")}
            )
            state["update_counters"] = self.update_counters.numpy()
        if self.reward_normalizer is not None and not self.cfg.load_reward_normalizer:
            state.update({f"normalizer.{key}": value for key, value in self.reward_normalizer.state_dict().items()})
        self.load_state_dict(state)
