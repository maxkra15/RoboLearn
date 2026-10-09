# Copyright (c) 2026 Holiday Robotics
# Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
# Replay and reward formulas derived from Holiday-Robot/FlashSAC,
# revision 87edc9061150ae9e962dd84e6544e27a1554b3ab.

"""Persistent Warp replay and the FlashSAC authors' reward normalization.

Terminations and timeouts both stop an n-step return. A timeout keeps its
pre-reset next observation and remains eligible for critic bootstrapping.
Kernels use FP32; their reduction order and random stream differ from Torch.
"""

import numpy as np
import warp as wp

from robolearn.flashsac.config import FlashSACConfig


@wp.kernel(enable_backward=False)
def _initialize_rng(seed: int, states: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    states[i] = wp.rand_init(seed, i)


@wp.kernel(enable_backward=False)
def _copy_pending(
    observation: wp.array2d(dtype=wp.float32),
    action: wp.array2d(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    next_observation: wp.array2d(dtype=wp.float32),
    counters: wp.array(dtype=wp.int64),
    pending_observation: wp.array3d(dtype=wp.float32),
    pending_action: wp.array3d(dtype=wp.float32),
    pending_reward: wp.array2d(dtype=wp.float32),
    pending_terminated: wp.array2d(dtype=wp.float32),
    pending_truncated: wp.array2d(dtype=wp.float32),
    pending_next_observation: wp.array3d(dtype=wp.float32),
):
    i, j = wp.tid()
    slot = int(counters[0] % wp.int64(pending_reward.shape[0]))
    if j < observation.shape[1]:
        pending_observation[slot, i, j] = observation[i, j]
        pending_next_observation[slot, i, j] = next_observation[i, j]
    if j < action.shape[1]:
        pending_action[slot, i, j] = action[i, j]
    if j == 0:
        pending_reward[slot, i] = reward[i]
        pending_terminated[slot, i] = float(terminated[i])
        pending_truncated[slot, i] = float(truncated[i])


@wp.kernel(enable_backward=False)
def _commit_pending(
    gamma: float,
    counters: wp.array(dtype=wp.int64),
    pending_observation: wp.array3d(dtype=wp.float32),
    pending_action: wp.array3d(dtype=wp.float32),
    pending_reward: wp.array2d(dtype=wp.float32),
    pending_terminated: wp.array2d(dtype=wp.float32),
    pending_truncated: wp.array2d(dtype=wp.float32),
    pending_next_observation: wp.array3d(dtype=wp.float32),
    observation: wp.array2d(dtype=wp.float32),
    action: wp.array2d(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.float32),
    truncated: wp.array(dtype=wp.float32),
    next_observation: wp.array2d(dtype=wp.float32),
    discount: wp.array(dtype=wp.float32),
):
    i, j = wp.tid()
    n_step = pending_reward.shape[0]
    if counters[0] >= wp.int64(n_step - 1):
        latest = int(counters[0] % wp.int64(n_step))
        oldest = (latest + 1) % n_step
        next_slot = latest
        total = pending_reward[latest, i]
        terminal = pending_terminated[latest, i]
        timeout = pending_truncated[latest, i]
        factor = gamma
        for offset in range(1, n_step):
            slot = (latest - offset + n_step) % n_step
            done = pending_terminated[slot, i] != 0.0 or pending_truncated[slot, i] != 0.0
            if done:
                total = pending_reward[slot, i]
                factor = gamma
                terminal = pending_terminated[slot, i]
                timeout = pending_truncated[slot, i]
                next_slot = slot
            else:
                total = pending_reward[slot, i] + gamma * total
                factor = gamma * factor
        target = int((counters[1] + wp.int64(i)) % wp.int64(reward.shape[0]))
        if j < observation.shape[1]:
            observation[target, j] = pending_observation[oldest, i, j]
            next_observation[target, j] = pending_next_observation[next_slot, i, j]
        if j < action.shape[1]:
            action[target, j] = pending_action[oldest, i, j]
        if j == 0:
            reward[target] = total
            terminated[target] = terminal
            truncated[target] = timeout
            discount[target] = factor


@wp.kernel(enable_backward=False)
def _advance_replay(counters: wp.array(dtype=wp.int64), n_step: int, num_envs: int, capacity: int):
    if counters[0] >= wp.int64(n_step - 1):
        counters[1] = (counters[1] + wp.int64(num_envs)) % wp.int64(capacity)
        counters[2] = wp.min(counters[2] + wp.int64(num_envs), wp.int64(capacity))
    counters[0] = counters[0] + wp.int64(1)


@wp.kernel(enable_backward=False)
def _sample_indices(
    counters: wp.array(dtype=wp.int64), states: wp.array(dtype=wp.uint32), indices: wp.array(dtype=wp.int32)
):
    i = wp.tid()
    state = states[i]
    bound = wp.uint32(counters[2])
    threshold = (wp.uint32(0) - bound) % bound
    value = wp.uint32(wp.randi(state))
    while value < threshold:
        value = wp.uint32(wp.randi(state))
    indices[i] = int(value % bound)
    states[i] = state


@wp.kernel(enable_backward=False)
def _gather_batch(
    indices: wp.array(dtype=wp.int32),
    observation: wp.array2d(dtype=wp.float32),
    action: wp.array2d(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.float32),
    truncated: wp.array(dtype=wp.float32),
    next_observation: wp.array2d(dtype=wp.float32),
    discount: wp.array(dtype=wp.float32),
    batch_observation: wp.array2d(dtype=wp.float32),
    batch_action: wp.array2d(dtype=wp.float32),
    batch_reward: wp.array(dtype=wp.float32),
    batch_terminated: wp.array(dtype=wp.float32),
    batch_truncated: wp.array(dtype=wp.float32),
    batch_next_observation: wp.array2d(dtype=wp.float32),
    batch_discount: wp.array(dtype=wp.float32),
):
    i, j = wp.tid()
    source = indices[i]
    if j < observation.shape[1]:
        batch_observation[i, j] = observation[source, j]
        batch_next_observation[i, j] = next_observation[source, j]
    if j < action.shape[1]:
        batch_action[i, j] = action[source, j]
    if j == 0:
        batch_reward[i] = reward[source]
        batch_terminated[i] = terminated[source]
        batch_truncated[i] = truncated[source]
        batch_discount[i] = discount[source]


def _restore_array(array, value, name: str) -> None:
    value = np.asarray(value)
    if value.shape != array.shape:
        raise ValueError(f"Saved {name} shape {value.shape} differs from {array.shape}.")
    wp.copy(array, wp.array(value, dtype=array.dtype, device=array.device))


class WarpFlashReplay:
    """Uniform replay with persistent n-step staging and sampled batches.

    ``add`` accepts FP32 observations/actions/rewards and int32 done flags.
    Next observations must be the pre-reset state for completed episodes.
    ``sample`` overwrites and returns the same batch arrays on every call.
    All calls use the caller's current Warp stream on ``device``.

    Readiness and length track eager ``add`` calls. Sampling and device state
    advance safely inside a captured learner graph. Captured collection would
    also require its caller to maintain the host readiness bookkeeping.
    """

    _keys = ("observation", "action", "reward", "terminated", "truncated", "next_observation", "discount")

    def __init__(self, observation_dim: int, action_dim: int, num_envs: int, cfg: FlashSACConfig, device="cuda:0"):
        if min(observation_dim, action_dim, num_envs) < 1:
            raise ValueError("Replay dimensions and environment count must be positive.")
        if num_envs > cfg.buffer_max_length or cfg.buffer_max_length > np.iinfo(np.int32).max:
            raise ValueError("Replay capacity must fit a vector step and signed 32-bit sample indices.")
        self.device = wp.get_device(device)
        if cfg.buffer_device is not None and wp.get_device(cfg.buffer_device) != self.device:
            raise ValueError("Captured Warp replay must share its learner's device.")
        self.cfg = cfg
        self.num_envs = num_envs
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.capacity = cfg.buffer_max_length
        self.batch_size = cfg.sample_batch_size
        self._width = max(observation_dim, action_dim)
        self._host_additions = 0
        self._counters = wp.zeros(3, dtype=wp.int64, device=self.device)
        self._rng = wp.zeros(self.batch_size, dtype=wp.uint32, device=self.device)
        self._indices = wp.zeros(self.batch_size, dtype=wp.int32, device=self.device)
        self.storage = self._allocate(self.capacity)
        self.batch = self._allocate(self.batch_size)
        n = cfg.n_step
        self.pending = {
            "observation": wp.zeros((n, num_envs, observation_dim), dtype=wp.float32, device=self.device),
            "action": wp.zeros((n, num_envs, action_dim), dtype=wp.float32, device=self.device),
            "next_observation": wp.zeros((n, num_envs, observation_dim), dtype=wp.float32, device=self.device),
            **{
                key: wp.zeros((n, num_envs), dtype=wp.float32, device=self.device)
                for key in ("reward", "terminated", "truncated")
            },
        }
        wp.launch(_initialize_rng, dim=self.batch_size, inputs=[cfg.seed, self._rng], device=self.device)

    def _allocate(self, count: int) -> dict:
        return {
            key: wp.empty(
                (count, self.observation_dim)
                if key in ("observation", "next_observation")
                else (count, self.action_dim)
                if key == "action"
                else count,
                dtype=wp.float32,
                device=self.device,
            )
            for key in self._keys
        }

    def __len__(self) -> int:
        committed = max(0, self._host_additions - self.cfg.n_step + 1) * self.num_envs
        return min(committed, self.capacity)

    def can_sample(self) -> bool:
        return len(self) >= self.cfg.buffer_min_length

    def reset(self) -> None:
        """Discard replay contents and staging without replacing persistent arrays."""
        self._counters.zero_()
        self._host_additions = 0
        wp.launch(_initialize_rng, dim=self.batch_size, inputs=[self.cfg.seed, self._rng], device=self.device)

    def add(self, observations, actions, rewards, terminated, truncated, next_observations) -> None:
        """Freeze a vector transition and commit its oldest complete n-step return."""
        wp.launch(
            _copy_pending,
            dim=(self.num_envs, self._width),
            inputs=[
                observations,
                actions,
                rewards,
                terminated,
                truncated,
                next_observations,
                self._counters,
                *[self.pending[key] for key in self._keys if key != "discount"],
            ],
            device=self.device,
        )
        wp.launch(
            _commit_pending,
            dim=(self.num_envs, self._width),
            inputs=[
                self.cfg.gamma,
                self._counters,
                *[self.pending[key] for key in self._keys if key != "discount"],
                *[self.storage[key] for key in self._keys],
            ],
            device=self.device,
        )
        wp.launch(
            _advance_replay,
            dim=1,
            inputs=[self._counters, self.cfg.n_step, self.num_envs, self.capacity],
            device=self.device,
        )
        self._host_additions += 1

    def sample(self, *, check_ready: bool = True) -> dict:
        """Sample with replacement; static capture callers can defer the readiness guard."""
        if check_ready and not self.can_sample():
            raise RuntimeError("Replay has not reached the configured warmup length.")
        wp.launch(
            _sample_indices,
            dim=self.batch_size,
            inputs=[self._counters, self._rng, self._indices],
            device=self.device,
        )
        wp.launch(
            _gather_batch,
            dim=(self.batch_size, self._width),
            inputs=[
                self._indices,
                *[self.storage[key] for key in self._keys],
                *[self.batch[key] for key in self._keys],
            ],
            device=self.device,
        )
        return self.batch

    def state_dict(self, *, include_storage: bool = True) -> dict[str, np.ndarray]:
        """Copy continuation state to host outside capture.

        Full replay checkpoints can be large. A metadata-only snapshot permits
        capture preparation to restore counters and RNG while retaining an
        unchanged population via ``load_state_dict(..., restore_storage=False)``.
        Routine policy checkpoints should omit replay entirely, as FlashSAC does.
        """
        counters = self._counters.numpy().copy()
        state = {"counters": counters, "rng": self._rng.numpy().copy()}
        if include_storage:
            count = int(counters[2])
            for key, array in self.storage.items():
                state[f"storage.{key}"] = array[:count].numpy().copy()
        # Empty pending slots are not read until written, but preserve all slots
        # for exact continuation after partially filled n-step staging.
        for key, array in self.pending.items():
            state[f"pending.{key}"] = array.numpy().copy()
        return state

    def load_state_dict(self, state: dict[str, np.ndarray], *, restore_storage: bool = True) -> None:
        """Restore existing arrays; metadata-only restores retain the unchanged population."""
        counters = np.asarray(state["counters"])
        if (
            counters.shape != (3,)
            or int(counters[0]) < 0
            or not 0 <= int(counters[1]) < self.capacity
            or not 0 <= int(counters[2]) <= self.capacity
        ):
            raise ValueError("Saved replay counters are incompatible with this buffer.")
        count = int(counters[2])
        for key, array in self.storage.items():
            name = f"storage.{key}"
            if count and restore_storage:
                if name not in state:
                    raise ValueError("A nonempty replay checkpoint must include storage.")
                _restore_array(array[:count], state[name], name)
        for key, array in self.pending.items():
            _restore_array(array, state[f"pending.{key}"], f"pending.{key}")
        _restore_array(self._rng, state["rng"], "rng")
        _restore_array(self._counters, counters, "counters")
        self._host_additions = int(counters[0])


@wp.kernel(enable_backward=False)
def _update_returns(
    reward: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    gamma: float,
    returns: wp.array(dtype=wp.float32),
    maximum: wp.array(dtype=wp.float32),
    moments: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    previous = returns[i]
    if terminated[i] != 0 or truncated[i] != 0:
        previous = 0.0
    value = gamma * previous + reward[i]
    returns[i] = value
    wp.atomic_max(maximum, 0, wp.abs(value))
    wp.atomic_add(moments, 0, value)


@wp.kernel(enable_backward=False)
def _return_variance(returns: wp.array(dtype=wp.float32), moments: wp.array(dtype=wp.float32)):
    i = wp.tid()
    mean = moments[0] / float(returns.shape[0])
    centered = returns[i] - mean
    wp.atomic_add(moments, 1, centered * centered)


@wp.kernel(enable_backward=False)
def _update_running_moments(moments: wp.array(dtype=wp.float32), count: int, running: wp.array(dtype=wp.float32)):
    sample_count = float(count)
    sample_mean = moments[0] / sample_count
    sample_var = moments[1] / sample_count
    delta = sample_mean - running[0]
    total_count = running[2] + sample_count
    ratio = sample_count / total_count
    mean = running[0] + delta * ratio
    m_a = running[1] * (running[2] + 1.0e-4)
    m_b = sample_var * sample_count
    variance = (m_a + m_b + delta * delta * running[2] * ratio) / total_count
    running[0] = mean
    running[1] = variance
    running[2] = total_count


@wp.kernel(enable_backward=False)
def _normalize_reward(
    rewards: wp.array(dtype=wp.float32),
    running: wp.array(dtype=wp.float32),
    maximum: wp.array(dtype=wp.float32),
    g_max: float,
    normalized: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    denominator = wp.max(wp.sqrt(running[1] + 1.0e-8), maximum[0] / g_max)
    normalized[i] = rewards[i] / denominator


class WarpRewardNormalizer:
    """The authors' return-RMS scaling with eagerly allocated persistent arrays."""

    def __init__(self, num_envs: int, cfg: FlashSACConfig, device="cuda:0"):
        if num_envs < 1 or cfg.normalized_G_max <= 0:
            raise ValueError("Reward normalization requires positive environment count and return scale.")
        self.device = wp.get_device(device)
        self.cfg = cfg
        self.num_envs = num_envs
        self.returns = wp.zeros(num_envs, dtype=wp.float32, device=self.device)
        self.maximum = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.running = wp.array(np.array([0.0, 1.0, 0.0], dtype=np.float32), device=self.device)
        self._moments = wp.zeros(2, dtype=wp.float32, device=self.device)
        self.normalized = wp.empty(cfg.sample_batch_size, dtype=wp.float32, device=self.device)

    def update(self, rewards, terminated, truncated) -> None:
        """Accumulate return statistics once per collected vector transition."""
        self._moments.zero_()
        wp.launch(
            _update_returns,
            dim=self.num_envs,
            inputs=[rewards, terminated, truncated, self.cfg.gamma, self.returns, self.maximum, self._moments],
            device=self.device,
        )
        wp.launch(_return_variance, dim=self.num_envs, inputs=[self.returns, self._moments], device=self.device)
        wp.launch(
            _update_running_moments, dim=1, inputs=[self._moments, self.num_envs, self.running], device=self.device
        )

    def normalize_rewards(self, rewards):
        """Write scaled sampled rewards without modifying replay or raw batch storage."""
        if rewards.shape != self.normalized.shape:
            raise ValueError("Reward batch shape differs from the configured sample batch size.")
        wp.launch(
            _normalize_reward,
            dim=self.cfg.sample_batch_size,
            inputs=[rewards, self.running, self.maximum, self.cfg.normalized_G_max, self.normalized],
            device=self.device,
        )
        return self.normalized

    def state_dict(self) -> dict[str, np.ndarray]:
        """Copy FP32 normalization state to host outside capture."""
        return {
            "returns": self.returns.numpy().copy(),
            "maximum": self.maximum.numpy().copy(),
            "running": self.running.numpy().copy(),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Restore existing arrays, preserving captured graph references."""
        for key in ("returns", "maximum", "running"):
            _restore_array(getattr(self, key), state[key], key)
