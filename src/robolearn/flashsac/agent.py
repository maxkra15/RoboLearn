# Copyright (c) 2026 Holiday Robotics
# Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
# Derived from Holiday-Robot/FlashSAC, revision 87edc9061150ae9e962dd84e6544e27a1554b3ab.

import json
import math
import os
from dataclasses import asdict, replace
from typing import Any, MutableMapping, Optional, cast

import gymnasium as gym
import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler

from robolearn.flashsac._base import BaseAgent
from robolearn.flashsac._types import Tensor
from robolearn.flashsac._utils.network import Network
from robolearn.flashsac._utils.reward_normalization import RewardNormalizer
from robolearn.flashsac._utils.scheduler import warmup_cosine_decay_scheduler
from robolearn.flashsac.config import FlashSACConfig
from robolearn.flashsac.networks import (
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature,
)
from robolearn.flashsac.replay import TorchUniformBuffer
from robolearn.flashsac.updates import (
    update_actor,
    update_critic,
    update_target_network,
    update_temperature,
)


def _init_flashsac_networks(
    actor_observation_dim: int,
    critic_observation_dim: int,
    action_dim: int,
    cfg: FlashSACConfig,
    device: torch.device,
) -> tuple[Network, Network, Network, Network]:
    # Create learning rate schedule
    warmup_cosine_decay_lr = warmup_cosine_decay_scheduler(
        init_value=cfg.learning_rate_init,
        peak_value=cfg.learning_rate_peak,
        end_value=cfg.learning_rate_end,
        warmup_steps=cfg.learning_rate_warmup_step,
        decay_steps=cfg.learning_rate_decay_step,
    )

    # Initialize actor
    actor_net = FlashSACActor(
        num_blocks=cfg.actor_num_blocks,
        input_dim=actor_observation_dim,
        hidden_dim=cfg.actor_hidden_dim,
        action_dim=action_dim,
    ).to(device)

    use_fused = device.type == "cuda" and torch.cuda.is_available()
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=cfg.learning_rate_peak, fused=use_fused)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
        actor_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
    )
    actor = Network(
        network=actor_net,
        optimizer=actor_optimizer,
        scheduler=actor_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
    )
    # Manually compile `get_mean_and_std` function
    if cfg.use_compile:
        actor.network.get_mean_and_std = torch.compile(actor.network.get_mean_and_std, mode=cfg.compile_mode)  # type: ignore

    # Initialize critic
    critic_net = FlashSACDoubleCritic(
        num_blocks=cfg.critic_num_blocks,
        input_dim=critic_observation_dim + action_dim,
        hidden_dim=cfg.critic_hidden_dim,
        num_bins=cfg.critic_num_bins,
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
    ).to(device)

    critic_optimizer = optim.Adam(
        critic_net.parameters(),
        lr=cfg.learning_rate_peak,
        fused=use_fused,
    )
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
        critic_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
    )
    critic = Network(
        network=critic_net,
        optimizer=critic_optimizer,
        scheduler=critic_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
    )

    # Initialize target critic (same as critic but no optimizer)
    target_critic_net = FlashSACDoubleCritic(
        num_blocks=cfg.critic_num_blocks,
        input_dim=critic_observation_dim + action_dim,
        hidden_dim=cfg.critic_hidden_dim,
        num_bins=cfg.critic_num_bins,
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
    ).to(device)
    target_critic_net.load_state_dict(critic_net.state_dict())
    target_critic = Network(
        network=target_critic_net,
        optimizer=None,
        scheduler=None,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
        ema_source=critic,  # wire EMA update source
        ema_tau=cfg.critic_target_update_tau,
    )

    # Initialize temperature
    temp_net = FlashSACTemperature(cfg.temp_initial_value).to(device)
    temp_optimizer = optim.Adam(
        temp_net.parameters(),
        lr=cfg.learning_rate_peak,
        fused=use_fused,
    )
    temp_scheduler = torch.optim.lr_scheduler.LambdaLR(
        temp_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
    )
    temperature = Network(
        network=temp_net,
        optimizer=temp_optimizer,
        scheduler=temp_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=False,
    )

    # normalize network parameters after initialization
    actor.normalize_parameters()
    critic.normalize_parameters()
    target_critic.normalize_parameters()

    return actor, critic, target_critic, temperature


def _build_truncated_zeta_cdf(mu: float, max_n: int) -> torch.Tensor:
    """
    Build the truncated zeta CDF for the given mu and max_n.
    """
    ns = torch.arange(1, max_n + 1, dtype=torch.float32)
    pmf = ns ** (-mu)
    pmf = pmf / torch.sum(pmf)
    cdf = torch.cumsum(pmf, dim=0)
    return cdf


def _sample_integer_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
    """
    Sample an integer from the given CDF.
    Returns a 0-d int32 tensor.
    """
    u = torch.rand((), device=cdf.device)
    idx = torch.argmax((u < cdf).to(torch.int32))
    return (idx + 1).to(torch.int32)


def _sample_flashsac_actions(
    actor: Network,
    noise: torch.Tensor,
    observations: torch.Tensor,
    temperature: float,
    cur_count: torch.Tensor,
    cur_n: torch.Tensor,
    zeta_cdf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample actions with noise repeat logic fully in torch."""
    # forward actor → distribution mean and std
    mean, std = actor.apply(
        "get_mean_and_std",
        observations=observations,
        training=False,
    )
    # return deterministic actions without changing noise sampling params
    if temperature == 0.0:
        actions = torch.tanh(mean)
        return noise, actions, cur_count, cur_n

    # reinit noise after a certain number of steps (only during training)
    reinit = (cur_count == 0) | (cur_count >= cur_n)

    new_noise = torch.randn_like(mean)
    new_n = _sample_integer_from_cdf(zeta_cdf)

    noise = torch.where(reinit, new_noise, noise)
    cur_n = torch.where(reinit, new_n, cur_n)
    cur_count = torch.where(reinit, torch.zeros_like(cur_count), cur_count)

    # sample action
    actions = torch.tanh(mean + std * noise * temperature)

    return noise, actions, cur_count + 1, cur_n


def _update_networks(
    batch: dict[str, torch.Tensor],
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    cfg: FlashSACConfig,
    do_actor_update: bool,
    device: torch.device,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    if do_actor_update:
        # Update actor
        actor_info = update_actor(
            actor=actor,
            critic=critic,
            temperature=temperature,
            batch=batch,  # type: ignore
            bc_alpha=cfg.actor_bc_alpha,
            device=device,
            use_amp=cfg.use_amp,
            grad_scaler=grad_scaler,
        )

        # Update temperature
        temperature_info = update_temperature(
            temperature=temperature,
            entropy=actor_info["actor/entropy"],
            target_entropy=cfg.temp_target_entropy,
        )
    else:
        actor_info = {}
        temperature_info = {}

    # Update critic
    critic_info = update_critic(
        actor=actor,  # updated
        critic=critic,
        target_critic=target_critic,
        temperature=temperature,  # updated
        batch=batch,  # type: ignore
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
        num_bins=cfg.critic_num_bins,
        gamma=cfg.gamma,
        n_step=cfg.n_step,
        device=device,
        use_amp=cfg.use_amp,
        grad_scaler=grad_scaler,
    )

    target_critic_info = update_target_network(
        target_network=target_critic,
    )

    # Merge all info dicts
    update_info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return update_info


def _resolve_compile_mode(mode: str) -> str:
    """Resolve 'auto' compile mode based on the installed torch version."""
    if mode != "auto":
        return mode
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        return "max-autotune"
    return "reduce-overhead"


class FlashSAC(BaseAgent[FlashSACConfig]):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        num_envs: int = 1,
        cfg: FlashSACConfig | None = None,
        critic_observation_dim: int | None = None,
    ):
        """Create a tensor agent for a vectorized environment.

        Actions use normalized [-1, 1] coordinates. If critic_observation_dim
        is supplied, it is the total input dimension; observations concatenate
        actor features first, followed by additional privileged features.
        """
        import numpy as np

        cfg = cfg or FlashSACConfig()
        if min(observation_dim, action_dim, num_envs) <= 0:
            raise ValueError("Observation, action, and environment dimensions must be positive.")
        if num_envs > cfg.buffer_max_length:
            raise ValueError("Replay capacity must fit one vector environment step.")
        total_dim = critic_observation_dim or observation_dim
        if total_dim < observation_dim:
            raise ValueError("Critic observations must contain the actor observation prefix.")
        cfg = replace(cfg, asymmetric_observation=critic_observation_dim is not None)
        torch.manual_seed(cfg.seed)
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(num_envs, total_dim), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(num_envs, action_dim), dtype=np.float32)
        env_info = {"actor_observation_size": (observation_dim,)}
        self._num_envs = num_envs
        self._critic_observation_dim = total_dim
        self._action_dim = action_dim
        self._actor_observation_dim = observation_dim

        temp_target_entropy = 0.5 * self._action_dim * math.log(2 * math.pi * math.e * cfg.temp_target_sigma**2)
        compile_mode = _resolve_compile_mode(cfg.compile_mode)
        cfg = replace(cfg, temp_target_entropy=temp_target_entropy, compile_mode=compile_mode)

        super().__init__(
            observation_space,
            action_space,
            env_info,
            cfg,
        )
        self._cfg = cfg

        device_type = cfg.device_type
        device_type = (
            device_type
            if device_type.startswith("cuda") and ":" in device_type
            else ("cuda:0" if device_type.startswith("cuda") else "cpu")
        )
        self._device = torch.device(device_type)

        # Initialize networks
        (
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
        ) = _init_flashsac_networks(
            actor_observation_dim=self._actor_observation_dim,
            critic_observation_dim=self._critic_observation_dim,
            action_dim=self._action_dim,
            cfg=self._cfg,
            device=self._device,
        )
        self._update_step = 0
        self._sample_actions = (
            torch.compile(_sample_flashsac_actions, mode=compile_mode) if cfg.use_compile else _sample_flashsac_actions
        )

        # Grad scaler for FP16 AMP
        self._grad_scaler = GradScaler(device=self._device.type, enabled=self._cfg.use_amp)

        # Noise repetition (zeta distribution)
        self._zeta_cdf = _build_truncated_zeta_cdf(
            mu=self._cfg.actor_noise_zeta_mu, max_n=self._cfg.actor_noise_zeta_max
        ).to(self._device)
        self._cur_noise_repeat_n = torch.tensor(1, dtype=torch.int32, device=self._device)
        self._cur_noise_repeat_count = torch.tensor(0, dtype=torch.int32, device=self._device)
        action_shape = tuple(action_space.shape) if action_space.shape is not None else ()
        self._cached_noise = torch.randn(action_shape, device=self._device)

        # Reward normalizer
        self.reward_normalizer = None
        if self._cfg.normalize_reward:
            self.reward_normalizer = RewardNormalizer(
                gamma=self._cfg.gamma,
                G_max=self._cfg.normalized_G_max,
                load_rms=self._cfg.load_reward_normalizer,
                device=self._device,
            )

        # Replay buffer
        self._replay_buffer = TorchUniformBuffer(
            observation_space=observation_space,
            action_space=action_space,
            n_step=self._cfg.n_step,
            gamma=self._cfg.gamma,
            max_length=self._cfg.buffer_max_length,
            min_length=self._cfg.buffer_min_length,
            sample_batch_size=self._cfg.sample_batch_size,
            device_type=self._cfg.buffer_device_type,
        )

    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Tensor],
        training: bool,
    ) -> Tensor:
        if training:
            temperature = 1.0
        else:
            temperature = 0.0

        observations = prev_transition["next_observation"]
        if self._cfg.asymmetric_observation:
            observations = observations[:, : self._actor_observation_dim]

        observations = torch.as_tensor(observations, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            (
                self._cached_noise,
                actions,
                self._cur_noise_repeat_count,
                self._cur_noise_repeat_n,
            ) = self._sample_actions(
                actor=self._actor,
                noise=self._cached_noise,
                observations=observations,
                temperature=temperature,
                cur_count=self._cur_noise_repeat_count,
                cur_n=self._cur_noise_repeat_n,
                zeta_cdf=self._zeta_cdf,
            )
            if self._cfg.use_compile:
                # Compiled CUDA graph outputs are reused by later sampling or
                # learner calls. Keep exploration state and returned actions
                # in independently owned storage across those invocations.
                self._cached_noise = self._cached_noise.clone()
                self._cur_noise_repeat_count = self._cur_noise_repeat_count.clone()
                self._cur_noise_repeat_n = self._cur_noise_repeat_n.clone()
                actions = actions.clone()

        return actions

    def process_transition(self, transition: MutableMapping[str, Tensor]) -> None:
        # add to replay buffer
        self._replay_buffer.add(transition)

        # update reward normalizer
        if self._cfg.normalize_reward:
            assert "reward" in transition and self.reward_normalizer is not None
            self.reward_normalizer.update_reward_stats(
                reward=torch.as_tensor(transition["reward"], device=self._device),
                terminated=torch.as_tensor(transition["terminated"], device=self._device),
                truncated=torch.as_tensor(transition["truncated"], device=self._device),
            )

    def can_start_training(self) -> bool:
        return self._replay_buffer.can_sample()

    def update(self, *, tensor_metrics: bool = False) -> dict[str, Any]:
        """Run one replay update and return diagnostics.

        ``tensor_metrics=True`` returns detached device tensors so callers can
        aggregate diagnostics before a single host read at a logging boundary.
        The default retains the existing Python-float result.
        """
        batch = cast(dict[str, torch.Tensor], self._replay_buffer.sample())

        for k, v in batch.items():
            batch[k] = v.to(self._device, non_blocking=True)

        if self._cfg.asymmetric_observation:
            batch["actor_observation"] = batch["observation"][:, : self._actor_observation_dim]
            batch["actor_next_observation"] = batch["next_observation"][:, : self._actor_observation_dim]
        else:
            batch["actor_observation"] = batch["observation"]
            batch["actor_next_observation"] = batch["next_observation"]

        if self._cfg.normalize_reward:
            assert self.reward_normalizer is not None
            # batch["unnormalized_reward"] = batch["reward"].clone()
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])

        # Update step
        _update_info = _update_networks(
            batch=batch,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            cfg=self._cfg,
            do_actor_update=(self._update_step % self._cfg.actor_update_period == 0),
            device=self._device,
            grad_scaler=self._grad_scaler,
        )
        self._update_step += 1

        if tensor_metrics:
            return {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in _update_info.items()
                if not isinstance(value, dict)
            }

        # Convert tensors to floats
        update_info: dict[str, float] = {}
        for key, value in _update_info.items():
            if isinstance(value, torch.Tensor):
                update_info[key] = value.item()
            elif not isinstance(value, dict):
                update_info[key] = float(value)

        return update_info

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        metadata = {
            "observation_dim": self._actor_observation_dim,
            "critic_observation_dim": self._critic_observation_dim if self._cfg.asymmetric_observation else None,
            "action_dim": self._action_dim,
            "config": asdict(self._cfg),
        }
        with open(os.path.join(path, "config.json"), "w") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
        self._actor.save(os.path.join(path, "actor.pt"))
        self._critic.save(os.path.join(path, "critic.pt"))
        self._target_critic.save(os.path.join(path, "target_critic.pt"))
        self._temperature.save(os.path.join(path, "temperature.pt"))
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(os.path.join(path, "reward_normalizer.pt"))

        agent_state: dict[str, Any] = {
            "update_step": self._update_step,
            "grad_scaler_state_dict": self._grad_scaler.state_dict(),
        }
        torch.save(agent_state, os.path.join(path, "agent_state.pt"))

    def save_replay_buffer(self, path: str) -> None:
        self._replay_buffer.save(os.path.join(path, "replay_buffer.pt"))

    def load(self, path: str) -> None:
        load_optimizer = self._cfg.load_optimizer
        self._actor.load(os.path.join(path, "actor.pt"), load_optimizer=load_optimizer)
        self._critic.load(os.path.join(path, "critic.pt"), load_optimizer=load_optimizer)
        self._target_critic.load(os.path.join(path, "target_critic.pt"), load_optimizer=False)
        self._temperature.load(os.path.join(path, "temperature.pt"), load_optimizer=load_optimizer)

        # Load agent-level optimizer state
        if load_optimizer:
            agent_state_path = os.path.join(path, "agent_state.pt")
            assert os.path.exists(agent_state_path)
            agent_state = torch.load(agent_state_path, map_location=self._device)
            self._update_step = agent_state["update_step"]
            self._grad_scaler.load_state_dict(agent_state["grad_scaler_state_dict"])

        if self._cfg.load_reward_normalizer and self.reward_normalizer is not None:
            self.reward_normalizer.load(os.path.join(path, "reward_normalizer.pt"))
            self.reward_normalizer.G_r = torch.zeros(self._num_envs, device=self._device)

    def load_replay_buffer(self, path: str) -> None:
        self._replay_buffer.load(os.path.join(path, "replay_buffer.pt"))

    @property
    def ready(self) -> bool:
        """Whether the replay buffer has enough samples for an update."""
        return self.can_start_training()

    def act(self, observations: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Return normalized actions on the agent device."""
        return self.sample_actions(0, {"next_observation": observations}, training)

    def observe(self, observations, actions, rewards, terminated, truncated, next_observations) -> None:
        """Store transitions; next_observations must precede automatic reset on done rows."""
        self.process_transition(
            {
                "observation": observations,
                "action": actions,
                "reward": rewards,
                "terminated": terminated,
                "truncated": truncated,
                "next_observation": next_observations,
            }
        )

    def get_metrics(self) -> dict[str, Any]:
        return {}
