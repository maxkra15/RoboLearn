# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Numerical PPO behavior and CUDA capture equivalence."""

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("warp_nn")

from robolearn.warp import PPOConfig, WarpPPO  # noqa: E402
from robolearn.warp.ppo import _make_ppo_loss  # noqa: E402


@pytest.fixture
def device():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("Warp-NN uses CUDA tile GEMM; an NVIDIA GPU is required.")
    return "cuda"


def test_gae_distinguishes_termination_timeout_and_rollout_boundary(device):
    agent = WarpPPO(1, 1, 2, 3, PPOConfig(hidden_dims=(), normalize_advantages=False), device)
    reward = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    next_value = np.array([[0.3, 0.4], [9, 10], [11, 12]], dtype=np.float32)
    terminated = np.array([[0, 0], [1, 0], [0, 0]], dtype=np.int32)
    truncated = np.array([[0, 0], [0, 1], [0, 0]], dtype=np.int32)
    for name, data in (
        ("rewards", reward),
        ("values", value),
        ("next_values", next_value),
        ("terminated", terminated),
        ("truncated", truncated),
    ):
        getattr(agent, name).assign(data.flatten())
    expected = np.zeros_like(reward)
    tail = np.zeros(2, dtype=np.float32)
    for t in range(2, -1, -1):
        delta = reward[t] + 0.99 * (1 - terminated[t]) * next_value[t] - value[t]
        tail = delta + 0.99 * 0.95 * (1 - np.maximum(terminated[t], truncated[t])) * tail
        expected[t] = tail
    agent.compute_returns()
    np.testing.assert_allclose(agent.advantages.numpy().reshape(3, 2), expected, rtol=1e-6)
    np.testing.assert_allclose(agent.returns.numpy().reshape(3, 2), expected + value, rtol=1e-6)


def test_advantage_normalization_preserves_small_variance_at_large_offset(device):
    agent = WarpPPO(1, 1, 2, 1, PPOConfig(hidden_dims=(), gamma=0.0), device)
    # E[x^2] - E[x]^2 loses this variance in float32 and produces +/-5000.
    rewards = np.array([10000.0, 10001.0], dtype=np.float32)
    agent.rewards.assign(rewards)
    agent.compute_returns()
    np.testing.assert_allclose(agent.advantages.numpy(), [-1.0, 1.0], rtol=1e-6)
    np.testing.assert_array_equal(agent.returns.numpy(), rewards)


def test_clipped_gaussian_loss_and_gradient_match_reference(device):
    mean_np = np.array([[0.4, 0.1], [-0.3, 0.2], [0.2, -0.2]], dtype=np.float32)
    action_np = np.array([[0.0, 0.5], [0.6, 0.1], [-0.5, 0.4]], dtype=np.float32)
    old_lp_np = np.array([-2.4, -1.8, -2.0], dtype=np.float32)
    advantage_np = np.array([1.0, -0.8, 0.4], dtype=np.float32)
    value_np = np.array([[0.8], [1.0], [-0.5]], dtype=np.float32)
    old_value_np = np.array([0.2, 1.3, -0.4], dtype=np.float32)
    return_np = np.array([0.7, -0.2, 1.0], dtype=np.float32)
    log_std_np = np.array([-0.2, 0.1], dtype=np.float32)
    mean = wp.array(mean_np, device=device, requires_grad=True)
    value = wp.array(value_np, device=device, requires_grad=True)
    log_std = wp.array(log_std_np, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)
    metrics = wp.zeros(3, dtype=wp.float32, device=device)
    with wp.Tape() as tape:
        wp.launch(
            _make_ppo_loss(2),
            dim=3,
            inputs=[
                mean,
                value,
                log_std,
                wp.array(action_np, device=device),
                wp.array(old_lp_np, device=device),
                wp.array(old_value_np, device=device),
                wp.array(advantage_np, device=device),
                wp.array(return_np, device=device),
                0.2,
                0.5,
                0.01,
                1,
            ],
            outputs=[loss, metrics],
            device=device,
        )
    tape.backward(loss)

    def reference(mean_values=mean_np, value_values=value_np, std_values=log_std_np):
        lp = (-0.5 * (((action_np - mean_values) * np.exp(-std_values)) ** 2 + 2 * std_values + np.log(2 * np.pi))).sum(
            axis=1
        )
        ratio = np.exp(lp - old_lp_np)
        policy = -np.minimum(ratio * advantage_np, np.clip(ratio, 0.8, 1.2) * advantage_np).mean()
        clipped_value = old_value_np + np.clip(value_values[:, 0] - old_value_np, -0.2, 0.2)
        value_loss = 0.5 * np.maximum((value_values[:, 0] - return_np) ** 2, (clipped_value - return_np) ** 2).mean()
        entropy = (std_values + 0.5 * np.log(2 * np.pi * np.e)).sum()
        return policy + 0.5 * value_loss - 0.01 * entropy

    np.testing.assert_allclose(loss.numpy()[0], reference(), rtol=1e-5, atol=1e-7)
    for argument, data, gradient in (
        ("mean_values", mean_np, mean.grad),
        ("value_values", value_np, value.grad),
        ("std_values", log_std_np, log_std.grad),
    ):
        numerical_gradient = np.zeros_like(data)
        for index in np.ndindex(data.shape):
            plus, minus = data.copy(), data.copy()
            plus[index] += 1e-3
            minus[index] -= 1e-3
            numerical_gradient[index] = (reference(**{argument: plus}) - reference(**{argument: minus})) / 2e-3
        np.testing.assert_allclose(gradient.numpy(), numerical_gradient, atol=2e-4, rtol=2e-3)


def test_captured_update_matches_eager_and_checkpoint_preserves_graph(device, tmp_path):
    config = PPOConfig(hidden_dims=(16,), epochs=2)
    eager = WarpPPO(3, 2, 8, 4, config, device)
    captured = WarpPPO(3, 2, 8, 4, config, device)
    captured.load_state_dict(eager.state_dict())
    rng = np.random.default_rng(7)
    for name, shape in (("observations", (32, 3)), ("actions", (32, 2)), ("rewards", (32,))):
        samples = rng.normal(size=shape).astype(np.float32)
        getattr(eager, name).assign(samples)
        getattr(captured, name).assign(samples)
    for _ in range(2):
        eager.update(capture=False)
        captured.update()
    for expected, actual in zip(eager.parameters, captured.parameters, strict=True):
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-5, atol=2e-6)
    checkpoint = tmp_path / "ppo.npz"
    captured.save(checkpoint)
    saved = captured.state_dict()
    captured.update()
    captured.load(checkpoint)
    for key, value in saved.items():
        np.testing.assert_array_equal(captured.state_dict()[key], value)
    # The checkpoint restore writes into existing arrays referenced by the graph.
    captured.update()
    eager.update(capture=False)
    for expected, actual in zip(eager.parameters, captured.parameters, strict=True):
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-5, atol=2e-6)
