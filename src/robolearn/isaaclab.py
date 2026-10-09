# Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT

"""Tensor adapter for Isaac Lab's vector environments; the caller owns startup."""

import torch


class IsaacLabEnv:
    """Expose flat actor/critic observations and pre-reset transitions.

    The environment must enable ``cfg.compute_final_obs`` before construction.
    Actions are clipped and scaled exactly once at the boundary. The default
    normalized range is [-1, 1]; ``clip_actions=None`` preserves native policy
    actions for callers matching an unclipped environment contract.
    This adapter does not import Isaac Lab or launch a simulator.
    """

    def __init__(self, env, observation_group="policy", critic_group=None, action_scale=1.0, clip_actions=1.0):
        self.env = env.unwrapped
        if not self.env.cfg.compute_final_obs:
            raise ValueError("Enable env_cfg.compute_final_obs before creating the environment.")
        self.observation_group = observation_group
        self.critic_group = critic_group
        self.device = self.env.device
        self.num_envs = self.env.num_envs
        self.action_scale = torch.as_tensor(action_scale, device=self.device, dtype=torch.float32)
        if clip_actions is not None and clip_actions <= 0:
            raise ValueError("clip_actions must be positive or None.")
        self.clip_actions = clip_actions
        space = self.env.single_observation_space
        for group in (observation_group, critic_group):
            if group and (space[group].shape is None or len(space[group].shape) != 1):
                raise ValueError(f"Observation group {group!r} must be a flat, concatenated vector.")
        self.observation_dim = space[observation_group].shape[0]
        self.critic_observation_dim = self.observation_dim + space[critic_group].shape[0] if critic_group else None
        self.action_dim = self.env.single_action_space.shape[0]

    def _observations(self, groups):
        actor = groups[self.observation_group]
        return torch.cat((actor, groups[self.critic_group]), dim=-1) if self.critic_group else actor

    def reset(self, seed=None):
        """Return observations after a full environment reset."""
        groups, _ = self.env.reset(seed=seed)
        return self._observations(groups).clone()

    def step(self, actions):
        """Return post-reset observations and a frozen replay transition.

        The replay next observation uses the state before reset for finished
        episodes. The returned observation is the state for the next policy call.
        """
        before = self._observations(self.env.obs_buf).clone()
        actions = actions.to(self.device)
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
        groups, reward, terminated, truncated, extras = self.env.step(actions * self.action_scale)
        observations = self._observations(groups).clone()
        done = terminated | truncated
        if "final_obs" not in extras and done.any():
            raise RuntimeError("The environment reset an episode without providing pre-reset final observations.")
        final = self._observations(extras["final_obs"]) if "final_obs" in extras else observations
        replay_next = torch.where(done[:, None], final, observations)
        transition = {
            "observation": before,
            "action": actions,
            "reward": reward.clone(),
            "terminated": terminated.clone(),
            "truncated": truncated.clone(),
            "next_observation": replay_next,
        }
        return observations, transition
