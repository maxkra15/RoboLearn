# Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT

"""Check tensor ownership and episode boundaries without launching a simulator."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from robolearn.isaaclab import IsaacLabEnv


class ReusingVectorEnv:
    """Supply raw vector-environment results in persistent simulator-style buffers."""

    def __init__(self, include_final_obs=True):
        self.unwrapped = self
        self.cfg = SimpleNamespace(compute_final_obs=True)
        self.device = "cpu"
        self.num_envs = 3
        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "critic": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            }
        )
        self.single_action_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.obs_buf = {"policy": torch.zeros(3, 2), "critic": torch.zeros(3, 1)}
        self.final_obs = {"policy": torch.zeros(3, 2), "critic": torch.zeros(3, 1)}
        self.reward = torch.zeros(3)
        self.terminated = torch.zeros(3, dtype=torch.bool)
        self.truncated = torch.zeros(3, dtype=torch.bool)
        self.include_final_obs = include_final_obs

    def reset(self, seed=None):
        self.obs_buf["policy"].copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        self.obs_buf["critic"].copy_(torch.tensor([[10.0], [20.0], [30.0]]))
        return self.obs_buf, {}

    def step(self, actions):
        self.received_actions = actions.clone()
        self.final_obs["policy"].copy_(torch.tensor([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]))
        self.final_obs["critic"].copy_(torch.tensor([[110.0], [120.0], [130.0]]))
        self.obs_buf["policy"].copy_(torch.tensor([[-1.0, -2.0], [-3.0, -4.0], [15.0, 16.0]]))
        self.obs_buf["critic"].copy_(torch.tensor([[-10.0], [-20.0], [130.0]]))
        self.reward.copy_(torch.tensor([1.0, 2.0, 3.0]))
        self.terminated.copy_(torch.tensor([True, False, False]))
        self.truncated.copy_(torch.tensor([False, True, False]))
        extras = {"final_obs": self.final_obs} if self.include_final_obs else {}
        return self.obs_buf, self.reward, self.terminated, self.truncated, extras

    def overwrite_buffers(self):
        for tensor in (*self.obs_buf.values(), *self.final_obs.values(), self.reward):
            tensor.fill_(99.0)
        self.terminated.fill_(False)
        self.truncated.fill_(False)


@pytest.mark.parametrize(("critic_group", "clip_actions"), [(None, 1.0), ("critic", None)])
def test_adapter_preserves_transitions_and_scales_actions_once(critic_group, clip_actions):
    env = ReusingVectorEnv()
    adapter = IsaacLabEnv(env, critic_group=critic_group, action_scale=3.0, clip_actions=clip_actions)
    initial = adapter.reset()
    actions = torch.tensor([[2.0, -2.0], [0.25, -0.5], [0.0, 1.0]])
    observations, transition = adapter.step(actions)
    env.overwrite_buffers()
    actions.fill_(42.0)

    expected_before = torch.tensor([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0], [5.0, 6.0, 30.0]])
    expected_after_reset = torch.tensor([[-1.0, -2.0, -10.0], [-3.0, -4.0, -20.0], [15.0, 16.0, 130.0]])
    expected_replay_next = torch.tensor([[11.0, 12.0, 110.0], [13.0, 14.0, 120.0], [15.0, 16.0, 130.0]])
    if critic_group is None:
        expected_before = expected_before[:, :2]
        expected_after_reset = expected_after_reset[:, :2]
        expected_replay_next = expected_replay_next[:, :2]

    torch.testing.assert_close(initial, expected_before)
    torch.testing.assert_close(transition["observation"], expected_before)
    torch.testing.assert_close(observations, expected_after_reset)
    torch.testing.assert_close(transition["next_observation"], expected_replay_next)
    if clip_actions is None:
        expected_actions = torch.tensor([[2.0, -2.0], [0.25, -0.5], [0.0, 1.0]])
        expected_controls = torch.tensor([[6.0, -6.0], [0.75, -1.5], [0.0, 3.0]])
    else:
        expected_actions = torch.tensor([[1.0, -1.0], [0.25, -0.5], [0.0, 1.0]])
        expected_controls = torch.tensor([[3.0, -3.0], [0.75, -1.5], [0.0, 3.0]])
    torch.testing.assert_close(transition["action"], expected_actions)
    torch.testing.assert_close(env.received_actions, expected_controls)
    torch.testing.assert_close(transition["reward"], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(transition["terminated"], torch.tensor([True, False, False]))
    torch.testing.assert_close(transition["truncated"], torch.tensor([False, True, False]))


def test_adapter_rejects_done_without_terminal_observations():
    adapter = IsaacLabEnv(ReusingVectorEnv(include_final_obs=False))
    adapter.reset()
    with pytest.raises(RuntimeError, match="pre-reset"):
        adapter.step(torch.zeros(3, 2))
