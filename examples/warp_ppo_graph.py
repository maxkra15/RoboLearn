# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Learn a continuous-action bandit with captured Warp-NN PPO updates."""

import argparse

import warp as wp

from robolearn.warp import PPOConfig, WarpPPO


@wp.kernel(enable_backward=False)
def reward(actions: wp.array2d(dtype=wp.float32), rewards: wp.array(dtype=wp.float32)):
    i = wp.tid()
    error = actions[i, 0] - 0.5
    rewards[i] = -error * error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=512)
    args = parser.parse_args()
    wp.init()
    agent = WarpPPO(1, 1, args.num_envs, 1, PPOConfig(learning_rate=3e-3, hidden_dims=(32,), epochs=4))
    obs = wp.ones((args.num_envs, 1), dtype=wp.float32, device=agent.device)
    rewards = wp.zeros(args.num_envs, dtype=wp.float32, device=agent.device)
    terminal = wp.ones(args.num_envs, dtype=wp.int32, device=agent.device)
    timeout = wp.zeros(args.num_envs, dtype=wp.int32, device=agent.device)
    for iteration in range(args.iterations):
        actions = agent.act(obs)
        wp.launch(reward, dim=args.num_envs, inputs=[actions, rewards], device=agent.device)
        agent.store(0, obs, rewards, terminal, timeout, obs)
        agent.update()
        if iteration % 20 == 0 or iteration == args.iterations - 1:
            print(f"iteration={iteration + 1} mean_reward={rewards.numpy().mean():.4f}")
    print(f"deterministic_action={agent.act(obs, deterministic=True).numpy()[0, 0]:.4f}; target=0.5000")


if __name__ == "__main__":
    main()
