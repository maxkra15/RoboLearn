# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Capture policy sampling, MuJoCo Warp physics, rewards, GAE, and PPO together.

The task moves a one-dimensional slider toward x=1. This direct MuJoCo Warp
example demonstrates joint capture; it does not capture an Isaac Lab environment.
MuJoCo Warp is maintained by Google DeepMind and NVIDIA under Apache-2.0.
"""

import argparse
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from robolearn.warp import PPOConfig, WarpPPO

MODEL = """
<mujoco>
  <option timestep="0.05" gravity="0 0 0" iterations="1"/>
  <worldbody>
    <body>
      <joint name="slider" type="slide" axis="1 0 0" damping="0.2"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="slider" ctrlrange="-5 5" ctrllimited="true"/></actuator>
</mujoco>
"""


@wp.kernel(enable_backward=False)
def observe(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    observations: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    observations[i, 0] = qpos[i, 0] - 1.0
    observations[i, 1] = qvel[i, 0]


@wp.kernel(enable_backward=False)
def control(actions: wp.array2d(dtype=wp.float32), controls: wp.array2d(dtype=wp.float32)):
    i = wp.tid()
    controls[i, 0] = wp.clamp(actions[i, 0], -5.0, 5.0)


@wp.kernel(enable_backward=False)
def reward(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    rewards: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    error = qpos[i, 0] - 1.0
    rewards[i] = -error * error - 0.01 * qvel[i, 0] * qvel[i, 0] - 0.001 * ctrl[i, 0] * ctrl[i, 0]


@wp.kernel(enable_backward=False)
def reset(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    simulation_time: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    qpos[i, 0] = 0.0
    qvel[i, 0] = 0.0
    simulation_time[i] = 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--eager", action="store_true", help="Launch the same operations without a CUDA graph.")
    args = parser.parse_args()
    wp.init()
    model_cpu = mujoco.MjModel.from_xml_string(MODEL)
    model = mjw.put_model(model_cpu)
    data = mjw.make_data(model_cpu, nworld=args.num_envs)
    agent = WarpPPO(2, 1, args.num_envs, args.horizon, PPOConfig(hidden_dims=(32,), learning_rate=1e-3))
    obs = wp.zeros((args.num_envs, 2), dtype=wp.float32, device=agent.device)
    next_obs = wp.zeros_like(obs)
    rewards = wp.zeros(args.num_envs, dtype=wp.float32, device=agent.device)
    terminated = wp.zeros(args.num_envs, dtype=wp.int32, device=agent.device)
    truncated = wp.zeros_like(terminated)

    def iteration():
        for step in range(args.horizon):
            wp.launch(observe, dim=args.num_envs, inputs=[data.qpos, data.qvel, obs], device=agent.device)
            actions = agent.act(obs)
            wp.launch(control, dim=args.num_envs, inputs=[actions, data.ctrl], device=agent.device)
            mjw.step(model, data)
            wp.launch(observe, dim=args.num_envs, inputs=[data.qpos, data.qvel, next_obs], device=agent.device)
            wp.launch(reward, dim=args.num_envs, inputs=[data.qpos, data.qvel, data.ctrl, rewards], device=agent.device)
            truncated.fill_(int(step == args.horizon - 1))
            agent.store(step, obs, rewards, terminated, truncated, next_obs)
        wp.launch(reset, dim=args.num_envs, inputs=[data.qpos, data.qvel, data.time], device=agent.device)
        agent.launch_update()

    # Execute one real iteration to compile every forward/backward/physics kernel
    # and allocate all output caches before recording the stable launch sequence.
    iteration()
    wp.synchronize_device(agent.device)
    initial_weights = agent.parameters[0].numpy()
    initial_seed = int(agent.state_dict()["seed"][0])
    if not args.eager:
        with wp.ScopedCapture(device=agent.device) as captured:
            iteration()
    start = time.perf_counter()
    for _ in range(args.iterations):
        if args.eager:
            iteration()
        else:
            wp.capture_launch(captured.graph)
    wp.synchronize_device(agent.device)
    elapsed = time.perf_counter() - start
    weight_change = np.max(np.abs(agent.parameters[0].numpy() - initial_weights))
    if not np.isfinite(agent.loss.numpy()[0]) or weight_change == 0:
        raise RuntimeError("The joint physics/learning iteration did not produce a finite parameter update.")
    if int(agent.state_dict()["seed"][0]) != initial_seed + args.iterations * args.horizon:
        raise RuntimeError("The sampling counter did not advance on every graph replay.")
    mode = "eager" if args.eager else "joint CUDA graph"
    print(f"mode={mode}; iterations={args.iterations}; elapsed_seconds={elapsed:.3f}")
    print(f"transitions_per_second={args.iterations * args.horizon * args.num_envs / elapsed:.0f}")
    print(f"final_step_mean_reward={rewards.numpy().mean():.4f}; max_weight_change={weight_change:.6f}")


if __name__ == "__main__":
    main()
