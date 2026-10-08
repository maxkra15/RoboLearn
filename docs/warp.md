# Warp-NN PPO and CUDA capture

`robolearn.warp.WarpPPO` is an original implementation of continuous-action,
clipped PPO using NVIDIA's [Warp-NN](https://github.com/NVIDIA/warp-nn) layers
and Adam optimizer. The complete learning update is CUDA capturable: GAE,
advantage normalization, policy/value forward passes, clipped objective,
backward passes, gradient clipping, Adam, and gradient reset.

The initial scope is deliberately small: fixed observation/action dimensions,
diagonal Gaussian policies, independent actor and critic MLPs, one NVIDIA GPU,
and full-batch epochs. It has no minibatch shuffle, distributed training,
visual encoder, recurrent policy, or KL-driven early stopping.

## Install and run

```bash
uv sync --extra warp
uv run python examples/warp_ppo_graph.py
```

The bandit example trains the policy toward action `0.5`. It demonstrates
learning and captured updates; it is not a robotics benchmark.

```python
from robolearn.warp import PPOConfig, WarpPPO

agent = WarpPPO(observation_dim=24, action_dim=6, num_envs=256, horizon=32, config=PPOConfig(hidden_dims=(64, 64)))
for step in range(agent.horizon):
    actions = agent.act(observations)
    # Step your environment and retain the observations before automatic reset.
    agent.store(step, observations, rewards, terminated, truncated, next_observations)
agent.update()  # Captures the entire learning update on first use, then replays it.
agent.save("policy.npz")
```

The arrays in this interface are Warp arrays on the agent's device. Observation
arrays have shape `(num_envs, observation_dim)`, rewards have shape `(num_envs,)`,
floating arrays use `wp.float32`, and termination/truncation masks use `wp.int32`.
`act` returns raw Gaussian
actions in a reusable array. Apply action scaling or clipping in the environment;
the stored actions and log probabilities describe the raw Gaussian samples.
Write transformed actions into separate control buffers so `store` retains the raw samples.
Preserve the current observation until `store` has copied it to rollout storage.

For timeouts, `next_observations` must contain the terminal observation before
reset. GAE bootstraps from this observation at a timeout, excludes the bootstrap
after a true termination, and stops the trace across either kind of reset.
The final rollout step bootstraps normally when the episode continues.

NPZ checkpoints save model weights, Adam moments/timestep/learning rate, and the
sampling seed counter. Instantiate the same architecture/configuration before
loading; configuration and rollout/environment state are not in the checkpoint.
The arrays are restored in place so an existing update graph remains valid.
Warp-NN 0.4 does not expose Adam checkpoint methods; this module accesses its
moment arrays explicitly and therefore constrains that dependency's version.

## Capture physics and learning together

The direct MuJoCo Warp example includes policy sampling, actions, physics,
observations, rewards, rollout storage, GAE, and all PPO epochs in one graph:

```bash
uv sync --extra warp --extra mujoco
uv run python examples/mujoco_warp_capture.py
uv run python examples/mujoco_warp_capture.py --eager
```

This slider task is a capture demonstration, with no performance or robotics
benchmark claim. [MuJoCo Warp explicitly supports graph capture of its physics
operations](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#graph-capture).

To compose another native Warp environment with PPO, run one complete iteration
to warm all kernels and caches, then put the fixed rollout and
`agent.launch_update()` inside an outer `wp.ScopedCapture`. Keep the arrays alive,
reuse their addresses and shapes, advance random seeds on the device, and use
device kernels for rewards and resets. Call `wp.capture_launch` to replay.
`agent.warmup()` prepares the learning part without changing optimizer state;
environment/inference kernels still need their own warmup.

## Optional tiled network gradients

Set `PPOConfig(optimized_linear_backward=True)` to replace Warp-NN's generated
Linear backward with separate tiled input-gradient and weight-gradient products.
Weight gradients use fixed split-batch partial buffers and reductions. Forward
layers, activations, PPO losses, Adam, and checkpoint keys stay compatible.
The default retains Warp-NN's generated backward for comparisons.

This path uses Warp-NN 0.4 layer caches and Warp's tape callbacks. Its persistent
scratch buffers are prepared before capture. Numerical diagnostics compare
FP32 reference gradients, complete PPO updates, and eager versus captured replay;
floating-point reduction order can change results slightly. Measure the complete
training workload before selecting it: graph capture and custom kernels do not
guarantee faster matrix multiplication than optimized Torch backends.

Capturing the whole Isaac Lab `env.step()` requires its observation, reward,
event, reset, and bookkeeping paths to support capture. This library demonstrates
joint capture with direct MuJoCo Warp; it does not claim generic joint capture
of Isaac Lab environments.

## FlashSAC in Warp-NN

A Warp-NN implementation of FlashSAC is technically plausible: its compact
actor/critic networks, replay sampling, losses, target updates, and Adam can all
operate on fixed GPU arrays. It would be a separate algorithm port needing
numerical and learning comparisons with the original PyTorch agent. RoboLearn's
FlashSAC implementation currently uses PyTorch.

The potential benefit is fewer Python/kernel launches and composition with
Warp physics. Warp-NN's maintainers explicitly warn that its kernels may not
outperform libraries using cuBLAS/cuDNN. Measure comparable task throughput and
learning curves before making a speed claim. See the
[Warp-NN introduction](https://nvidia.github.io/warp-nn/latest/) and
[Warp graph documentation](https://nvidia.github.io/warp/v1.17/user_guide/runtime.html#graphs).

## Attribution

- PPO: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov,
  [*Proximal Policy Optimization Algorithms* (2017)](https://arxiv.org/abs/1707.06347).
- GAE: John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel,
  [*High-Dimensional Continuous Control Using Generalized Advantage Estimation*
  (2015)](https://arxiv.org/abs/1506.02438).
- Adam: Diederik P. Kingma and Jimmy Ba,
  [*Adam: A Method for Stochastic Optimization* (2014)](https://arxiv.org/abs/1412.6980).
- [Warp-NN](https://github.com/NVIDIA/warp-nn), NVIDIA Corporation,
  maintained by Toni-SM / NVIDIA Isaac Sim, Apache-2.0.
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp),
  Google DeepMind and NVIDIA, Apache-2.0.

The PPO code is newly written from the published objective. Warp-NN and MuJoCo
Warp are dependencies; their code is not vendored in this implementation.
