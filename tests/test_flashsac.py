# Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT

"""Behavioral coverage for the Torch agent, replay targets, and checkpoints."""

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch

from robolearn.flashsac import FlashSAC, FlashSACConfig
from robolearn.flashsac.replay import TorchUniformBuffer


@pytest.fixture(scope="module", autouse=True)
def _small_cpu_workloads():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


def _config(**overrides):
    cfg = FlashSACConfig(
        device="cpu",
        use_compile=False,
        use_amp=False,
        buffer_max_length=128,
        buffer_min_length=8,
        sample_batch_size=8,
        actor_num_blocks=1,
        actor_hidden_dim=16,
        critic_num_blocks=1,
        critic_hidden_dim=16,
        critic_num_bins=11,
        learning_rate_warmup_step=0,
        learning_rate_decay_step=100,
        n_step=1,
    )
    return replace(cfg, **overrides)


@pytest.fixture
def eager_compilation(monkeypatch):
    """Use Torch's real compiler wrapper without CPU code generation."""
    compiler = torch.compile

    def eager_backend(function, *, mode=None, **kwargs):
        return compiler(function, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", eager_backend)


def _collect(agent, observations, steps=4):
    for _ in range(steps):
        actions = agent.act(observations)
        next_observations = observations + 0.1 * torch.randn_like(observations)
        rewards = 1.0 - actions.square().sum(dim=-1)
        dones = torch.zeros(len(observations), dtype=torch.bool)
        agent.observe(observations, actions, rewards, dones, dones, next_observations)
        observations = next_observations


def test_actions_are_device_tensors_with_normalized_bounds():
    agent = FlashSAC(3, 2, num_envs=4, cfg=_config())
    observations = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [-0.1, -0.2, -0.3], [0.7, -0.8, 0.9]])

    for training in (True, False):
        actions = agent.act(observations, training=training)
        assert isinstance(actions, torch.Tensor)
        assert actions.device == observations.device
        assert actions.shape == (4, 2)
        assert torch.isfinite(actions).all()
        assert (actions.abs() <= 1.0).all()


def test_replay_updates_change_policy_with_finite_losses():
    torch.manual_seed(10)
    agent = FlashSAC(3, 2, num_envs=4, cfg=_config())
    observations = torch.randn(4, 3)
    before = agent.act(observations, training=False).clone()
    parameters_before = [parameter.detach().clone() for parameter in agent._actor.network.parameters()]
    assert not agent.ready

    _collect(agent, observations)
    assert agent.ready
    for step in range(3):
        tensor_metrics = step == 0
        losses = agent.update(tensor_metrics=tensor_metrics)
        assert losses
        if tensor_metrics:
            assert all(
                isinstance(value, torch.Tensor)
                and value.device == observations.device
                and not value.requires_grad
                and torch.isfinite(value).all()
                for value in losses.values()
            )
        else:
            assert all(isinstance(value, float) and np.isfinite(value) for value in losses.values())

    after = agent.act(observations, training=False)
    assert torch.isfinite(after).all()
    assert not torch.allclose(before, after, rtol=1e-5, atol=1e-7)
    assert any(
        not torch.allclose(before_parameter, after_parameter, rtol=1e-5, atol=1e-7)
        for before_parameter, after_parameter in zip(parameters_before, agent._actor.network.parameters(), strict=True)
    )


@pytest.mark.parametrize("normalize_reward", [False, True])
def test_checkpoint_restores_policy_with_different_environment_count(tmp_path, normalize_reward):
    torch.manual_seed(20)
    cfg = _config(normalize_reward=normalize_reward)
    agent = FlashSAC(3, 2, num_envs=4, cfg=cfg)
    observations = torch.randn(4, 3)
    _collect(agent, observations)
    agent.update()
    # Match the forward batch shape so this checks reload rather than CPU GEMM rounding.
    expected = agent.act(observations[:2], training=False).clone()
    checkpoint = str(tmp_path / "agent")
    agent.save(checkpoint)

    restored = FlashSAC(3, 2, num_envs=2, cfg=cfg)
    restored.load(checkpoint)
    actual = restored.act(observations[:2], training=False)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    # Training after playback-sized loading also exercises per-environment return state.
    _collect(restored, observations[:2])
    losses = restored.update()
    assert all(np.isfinite(value) for value in losses.values())


def test_checkpoints_transfer_between_eager_and_compiled_agents(tmp_path, eager_compilation):
    torch.manual_seed(30)
    cfg = _config(normalize_reward=False)
    trained = FlashSAC(3, 2, num_envs=2, cfg=cfg)
    observations = torch.randn(2, 3)
    _collect(trained, observations)
    trained.update()
    expected = trained.act(observations, training=False).clone()
    eager_checkpoint = str(tmp_path / "eager")
    compiled_checkpoint = str(tmp_path / "compiled")
    trained.save(eager_checkpoint)

    compiled = FlashSAC(3, 2, num_envs=2, cfg=replace(cfg, use_compile=True, compile_mode="default"))
    compiled.load(eager_checkpoint)
    compiled.save(compiled_checkpoint)

    restored = FlashSAC(3, 2, num_envs=2, cfg=cfg)
    restored.load(compiled_checkpoint)
    torch.testing.assert_close(restored.act(observations, training=False), expected, rtol=1e-6, atol=1e-7)


def test_compiled_sampling_owns_reusable_output_storage(eager_compilation):
    """CPU storage contract: retained actions and noise survive producer reuse.

    Reuse real sampler results in persistent buffers to model the compiled
    producer's storage contract. This does not exercise CUDA graph execution.
    """
    cfg = _config(normalize_reward=False, actor_noise_zeta_mu=0.0)
    reference = FlashSAC(3, 2, num_envs=2, cfg=cfg)
    compiled = FlashSAC(3, 2, num_envs=2, cfg=replace(cfg, use_compile=True, compile_mode="default"))
    sampler, buffers = compiled._sample_actions, []

    def reuse_outputs(**kwargs):
        values = sampler(**kwargs)
        if not buffers:
            buffers.extend(value.clone() for value in values)
        else:
            for buffer, value in zip(buffers, values, strict=True):
                buffer.copy_(value)
        return tuple(buffers)

    compiled._sample_actions = reuse_outputs
    observations = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    held_actions, held_expected = None, None
    for step, inputs in enumerate((observations, observations, observations + 1.0)):
        torch.manual_seed(step)
        expected = reference.act(inputs)
        if step == 1:
            # A later compiled operation may overwrite the producer's buffers.
            for buffer in buffers:
                buffer.zero_()
            torch.testing.assert_close(expected, held_expected)
        torch.manual_seed(step)
        actual = compiled.act(inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        if step == 0:
            held_actions, held_expected = actual, expected.clone()
        torch.testing.assert_close(held_actions, held_expected, rtol=1e-6, atol=1e-7)


def test_n_step_replay_stops_at_done_and_preserves_timeout_bootstrap():
    replay = TorchUniformBuffer(
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        n_step=3,
        gamma=0.5,
        max_length=16,
        min_length=3,
        sample_batch_size=3,
        device_type="cpu",
    )
    # Environment 0 times out after two steps, 1 continues, and 2 truly terminates.
    for step, rewards in enumerate(([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [100.0, 4.0, 100.0])):
        replay.add(
            {
                "observation": torch.tensor([[step * 10.0], [step * 10.0 + 1], [step * 10.0 + 2]]),
                "action": torch.zeros(3, 1),
                "reward": torch.tensor(rewards),
                "terminated": torch.tensor([False, False, step == 1]),
                "truncated": torch.tensor([step == 1, False, False]),
                "next_observation": torch.tensor(
                    [[(step + 1) * 10.0], [(step + 1) * 10.0 + 1], [(step + 1) * 10.0 + 2]]
                ),
            }
        )

    batch = replay.sample(np.array([0, 1, 2]))
    torch.testing.assert_close(batch["reward"], torch.tensor([2.0, 3.0, 2.0]))
    torch.testing.assert_close(batch["next_observation"], torch.tensor([[20.0], [31.0], [22.0]]))
    torch.testing.assert_close(batch["discount"], torch.tensor([0.25, 0.125, 0.25]))
    torch.testing.assert_close(batch["terminated"], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(batch["truncated"], torch.tensor([1.0, 0.0, 0.0]))
    # Independent value reference: the timeout bootstraps, while termination suppresses it.
    target = batch["reward"] + batch["discount"] * (1.0 - batch["terminated"]) * batch["next_observation"].squeeze(-1)
    torch.testing.assert_close(target, torch.tensor([7.0, 6.875, 2.0]))
