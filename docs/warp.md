# Warp-NN PPO and CUDA capture

`robolearn.warp.WarpPPO` is an original implementation of continuous-action,
clipped PPO using NVIDIA's [Warp-NN](https://github.com/NVIDIA/warp-nn) layers
and Adam optimizer. The complete learning update is CUDA capturable: GAE,
advantage normalization, policy/value forward passes, clipped objective,
backward passes, gradient clipping, Adam, and gradient reset.

The scope is fixed observation/action dimensions, diagonal Gaussian policies,
independent actor and critic MLPs, and one NVIDIA GPU. Full-batch epochs remain
the default. Optional shuffled minibatches and a Gaussian-KL adaptive learning
rate also run entirely on the device. There is no distributed training,
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

## Match Isaac Lab's stock G1 PPO configuration

The following options reproduce the numerical conventions used by RSL-RL 5.5.1
and Isaac Lab's flat G1 configuration. These are opt-in; existing configuration
defaults and NPZ checkpoint keys remain unchanged.

```python
config = PPOConfig(
    hidden_dims=(256, 128, 128),
    activation="elu",
    std_type="scalar",
    initial_std=1.0,
    std_range=(1e-6, 1e6),
    epochs=5,
    num_mini_batches=4,
    learning_rate=1e-3,
    schedule="adaptive",
    desired_kl=0.01,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    value_coefficient=1.0,
    value_loss_scale=1.0,
    entropy_coefficient=0.008,
    max_grad_norm=1.0,
    advantage_sample_std=True,
    timeout_bootstrap="current",
    separate_grad_clipping=True,
    optimized_linear_backward=True,
)
```

Important numerical conventions:

- Scalar standard deviation is a directly optimized parameter; the distribution
  uses its value clamped to `std_range`, including the clamp's gradient. The
  parameter itself is not projected after Adam. The default log parameterization
  retains its previous unconstrained behavior.
- `value_loss_scale=1.0` uses the full mean squared error. The default `0.5`
  retains the original RoboLearn objective. `value_coefficient` is a separate
  multiplier in the combined policy, value, and entropy objective.
- Sample standard deviation uses Bessel's correction and adds epsilon after
  the square root, matching Torch's default `std()` normalization.
- `timeout_bootstrap="current"` uses `V(s_t)` for a timeout, matching RSL-RL's
  reward bootstrap convention. The default `"next"` uses the pre-reset
  `V(s_{t+1})`. Both stop GAE traces across resets.
- Separate gradient clipping gives the actor plus distribution parameters and
  the critic their own norm limits. Default clipping uses one combined norm.
- One fresh device permutation is reused across all epochs of an update.
  The batch size must divide evenly into `num_mini_batches`. Random 64-bit keys
  are sorted with Warp's radix sort; its temporary storage is warmed before
  capture.
- Before each minibatch update, Gaussian `KL(old || new)` above twice
  `desired_kl` divides the learning rate by 1.5 (minimum `1e-5`). A positive KL
  below half `desired_kl` multiplies it by 1.5 (maximum `1e-2`). There are no
  host scalar reads in the update.

`agent.health_metrics()` synchronizes distribution and optimizer diagnostics
when called. Use it at a logging cadence; it is not part of graph capture.
Losses, KL, and gradient norms describe the most recent minibatch, rather than
epoch averages. Checkpoint loading requires the same parameterization and
configuration as saving.

A standalone fixed-rollout numerical diagnostic compares GAE, a complete
20-minibatch update, Adam state, and eager/captured replay with Torch formulas
following the installed RSL-RL 5.5.1 implementation:

```bash
uv run python examples/diagnose_ppo_update.py --output ppo-update-diagnostic.json
uv run python examples/diagnose_ppo_update.py --linear-backward stock --output ppo-stock-update-diagnostic.json
```

It also requires Torch. This synthetic diagnostic does not establish locomotion
learning quality or comparative training speed.

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
