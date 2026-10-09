# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Compare one controlled RSL-style PPO update in Torch and Warp.

This standalone numerical diagnostic does not launch a simulator. It checks
fixed rollout data, shuffled minibatches, Gaussian KL scheduling, separate
gradient clipping, Adam state, and eager versus captured updates. Torch is an
optional diagnostic dependency; the learner itself does not import it.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import warp as wp

from robolearn.warp import PPOConfig, WarpPPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--linear-backward", choices=("stock", "tiled"), default="tiled")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    wp.init()
    cfg = PPOConfig(
        hidden_dims=(16, 8),
        activation="elu",
        std_type="scalar",
        epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        desired_kl=0.01,
        value_coefficient=1.0,
        value_loss_scale=1.0,
        entropy_coefficient=0.008,
        advantage_sample_std=True,
        timeout_bootstrap="current",
        separate_grad_clipping=True,
        optimized_linear_backward=args.linear_backward == "tiled",
    )
    agent = WarpPPO(7, 3, 16, 4, cfg)
    initial = agent.state_dict()
    rng = np.random.default_rng(29)
    observations = rng.normal(size=(agent.batch_size, 7)).astype(np.float32)
    agent.observations.assign(observations)
    means = agent.actor(agent.observations).numpy()
    values = agent.critic(agent.observations).numpy()[:, 0]
    actions = means + rng.normal(size=means.shape).astype(np.float32)
    log_probs = (-0.5 * ((actions - means) ** 2 + math.log(2.0 * math.pi))).sum(1)
    data = {
        "observations": observations,
        "actions": actions,
        "means": means,
        "stds": np.ones_like(means),
        "values": values,
        "log_probs": log_probs,
        "rewards": rng.normal(0.2, 0.05, size=agent.batch_size).astype(np.float32),
        "next_values": rng.normal(0.0, 0.2, size=agent.batch_size).astype(np.float32),
        "terminated": np.zeros(agent.batch_size, dtype=np.int32),
        "truncated": np.zeros(agent.batch_size, dtype=np.int32),
    }
    data["terminated"][18] = 1
    data["truncated"][39] = 1
    for key, value in data.items():
        getattr(agent, key).assign(value)

    tensor_data = {key: torch.tensor(value, device="cuda") for key, value in data.items()}
    advantages = torch.zeros(agent.batch_size, device="cuda")
    carry = torch.zeros(agent.num_envs, device="cuda")
    for step in reversed(range(agent.horizon)):
        interval = slice(step * agent.num_envs, (step + 1) * agent.num_envs)
        terminal = tensor_data["terminated"][interval].bool()
        timeout = tensor_data["truncated"][interval].bool()
        bootstrap = torch.where(timeout, tensor_data["values"][interval], tensor_data["next_values"][interval])
        bootstrap = torch.where(terminal & ~timeout, 0.0, bootstrap)
        delta = tensor_data["rewards"][interval] + cfg.gamma * bootstrap - tensor_data["values"][interval]
        carry = delta + cfg.gamma * cfg.gae_lambda * (~(terminal | timeout)).float() * carry
        advantages[interval] = carry
    returns = advantages + tensor_data["values"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
    agent.compute_returns()
    returns_error = float(np.max(np.abs(agent.returns.numpy() - returns.cpu().numpy())))
    advantages_error = float(np.max(np.abs(agent.advantages.numpy() - advantages.cpu().numpy())))

    agent.launch_update()
    wp.synchronize()
    eager = agent.state_dict()
    permutation = agent._shuffle_indices.numpy()[: agent.batch_size]
    if not np.array_equal(np.sort(permutation), np.arange(agent.batch_size)):
        raise RuntimeError("The device shuffle did not produce a permutation.")
    agent.load_state_dict(initial)
    for key, value in data.items():
        getattr(agent, key).assign(value)
    agent.update(capture=True)
    wp.synchronize()
    captured = agent.state_dict()
    capture_error = max(float(np.max(np.abs(eager[key] - captured[key]))) for key in eager)

    actor_count = len(agent.actor.parameters())
    critic_count = len(agent.critic.parameters())
    parameters = [
        torch.nn.Parameter(torch.tensor(initial[f"parameter_{i}"], device="cuda")) for i in range(len(agent.parameters))
    ]
    actor = parameters[:actor_count]
    critic = parameters[actor_count : actor_count + critic_count]
    std_parameter = parameters[-1]
    optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate, fused=False)

    def forward(inputs, network):
        value = inputs
        for index in range(0, len(network), 2):
            value = torch.nn.functional.linear(value, network[index], network[index + 1].reshape(-1))
            if index < len(network) - 2:
                value = torch.nn.functional.elu(value)
        return value

    learning_rate = cfg.learning_rate
    last_kl = 0.0
    for _ in range(cfg.epochs):
        for mini_batch in range(cfg.num_mini_batches):
            selected = permutation[mini_batch * agent.mini_batch_size : (mini_batch + 1) * agent.mini_batch_size]
            indices = torch.tensor(selected.astype(np.int64), device="cuda")
            mean = forward(tensor_data["observations"][indices], actor)
            prediction = forward(tensor_data["observations"][indices], critic)[:, 0]
            distribution = torch.distributions.Normal(mean, std_parameter.clamp(*cfg.std_range))
            old_distribution = torch.distributions.Normal(tensor_data["means"][indices], tensor_data["stds"][indices])
            with torch.no_grad():
                last_kl = float(torch.distributions.kl_divergence(old_distribution, distribution).sum(-1).mean())
                if last_kl > 2.0 * cfg.desired_kl:
                    learning_rate = max(1.0e-5, learning_rate / 1.5)
                elif 0.0 < last_kl < cfg.desired_kl / 2.0:
                    learning_rate = min(1.0e-2, learning_rate * 1.5)
                optimizer.param_groups[0]["lr"] = learning_rate
            ratio = (
                distribution.log_prob(tensor_data["actions"][indices]).sum(-1) - tensor_data["log_probs"][indices]
            ).exp()
            policy_loss = torch.maximum(
                -ratio * advantages[indices],
                -ratio.clamp(1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantages[indices],
            ).mean()
            clipped = tensor_data["values"][indices] + (prediction - tensor_data["values"][indices]).clamp(
                -cfg.clip_ratio, cfg.clip_ratio
            )
            value_loss = torch.maximum((prediction - returns[indices]) ** 2, (clipped - returns[indices]) ** 2).mean()
            loss = (
                policy_loss
                + cfg.value_coefficient * value_loss
                - cfg.entropy_coefficient * distribution.entropy().sum(-1).mean()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor + [std_parameter], cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic, cfg.max_grad_norm)
            optimizer.step()

    torch_parameter_error = max(
        float(np.max(np.abs(eager[f"parameter_{i}"] - parameter.detach().cpu().numpy())))
        for i, parameter in enumerate(parameters)
    )
    moment_error = 0.0
    for index, parameter in enumerate(parameters):
        state = optimizer.state[parameter]
        for warp_key, torch_key in (("adam_m1", "exp_avg"), ("adam_m2", "exp_avg_sq")):
            moment_error = max(
                moment_error,
                float(
                    np.max(
                        np.abs(eager[f"{warp_key}_{index}"].reshape(-1) - state[torch_key].cpu().numpy().reshape(-1))
                    )
                ),
            )
    health = agent.health_metrics()
    finite = all(np.isfinite(value).all() for value in captured.values())
    result = {
        "linear_backward": args.linear_backward,
        "returns_max_abs_error": returns_error,
        "advantages_max_abs_error": advantages_error,
        "eager_capture_all_state_max_abs_error": capture_error,
        "torch_warp_parameters_max_abs_error": torch_parameter_error,
        "torch_warp_adam_moments_max_abs_error": moment_error,
        "torch_learning_rate": learning_rate,
        "torch_last_kl": last_kl,
        "warp_health": health,
        "finite": finite,
        "optimizer_updates": cfg.epochs * cfg.num_mini_batches,
        "scope": "Synthetic fixed rollout; reference formulas follow RSL-RL 5.5.1 PPO; no task learning claim.",
    }
    result["passed"] = bool(
        finite
        and returns_error < 1.0e-5
        and advantages_error < 1.0e-5
        and capture_error < 2.0e-5
        and torch_parameter_error < 2.0e-4
        and moment_error < 2.0e-4
        and abs(learning_rate - health["learning_rate"]) < 1.0e-7
        and health["optimizer_updates"] == cfg.epochs * cfg.num_mini_batches
    )
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise RuntimeError("Controlled PPO update did not meet the reported numerical tolerances.")


if __name__ == "__main__":
    main()
