# FlashSAC with Isaac Lab 3

Install and validate Isaac Lab 3 first. From its checkout, install RoboLearn
into the existing environment; Isaac Lab owns its NumPy, PyTorch, and Gymnasium versions:

```bash
git clone https://github.com/maxkra15/RoboLearn.git /path/to/RoboLearn
uv pip install --no-deps -e /path/to/RoboLearn
uv run --no-sync python /path/to/RoboLearn/examples/isaaclab_flashsac.py \
    --num_envs 16 --steps 2048 --buffer_size 4096 --warmup 256 --batch_size 64
```

The example defaults to `Isaac-Open-Drawer-Franka`, Newton MJWarp, and action
scale 3, matching the action range in the authors' FlashSAC adapter. Other tasks
default to action scale 1; configure bounds suitable for the task. It derives
observation/action dimensions from the environment and enables terminal observations.
For privileged observations, pass `--critic_group critic`; actor features precede
privileged features in the combined observation, as expected by the FlashSAC agent.

Train the drawer task with 1,024 environments:

```bash
uv run --no-sync python /path/to/RoboLearn/examples/isaaclab_flashsac.py \
    --num_envs 1024 --steps 5000192
```

The default network matches the original FlashSAC architecture. The example uses
three-step returns, gamma 0.99, two updates per vector step, AMP on CUDA, and a
1-million-transition replay buffer. Pass `--compile` to compile the networks;
compilation has a startup cost. The authors' benchmark used a 10-million replay
buffer and 50-million transitions.

Evaluate a checkpoint with fewer environments:

```bash
uv run --no-sync python /path/to/RoboLearn/examples/isaaclab_flashsac.py \
    --checkpoint logs/flashsac/<task>/<run>/step5000192 \
    --evaluate_only --num_envs 64
```

The checkpoint includes agent configuration and the task, observation groups,
action scale, physics selection, and environment overrides. The example restores
these by default; task/group/action mismatches fail explicitly. Environment count
and device may change for playback. Add `--viz newton` to watch the policy.
Pass a checkpoint without `--evaluate_only` to resume with restored optimizer and
reward statistics and an empty replay buffer.

This is a standalone external example. RoboLearn's
[Isaac Lab integration branch](https://github.com/maxkra15/IsaacLab/tree/experiment/g1-learning-baselines)
also registers it with `isaaclab train --rl_library robolearn`. That integration
is separate from installing this library into an upstream Isaac Lab checkout.

## Policy actions and deferred diagnostics

`robolearn.isaaclab.IsaacLabEnv` clips policy actions to `[-1, 1]` by default
before applying `action_scale`. Pass `clip_actions=None` for an unclipped
Gaussian PPO policy matching Isaac Lab's native G1 PPO wrapper. The environment's
own joint action scaling remains in effect. FlashSAC keeps its bounded normalized
policy support; see the authors' task-specific
[Isaac Lab integration](https://github.com/Holiday-Robot/FlashSAC/tree/87edc9061150ae9e962dd84e6544e27a1554b3ab)
when choosing an action range.

The adapted FlashSAC API accepts `agent.update(tensor_metrics=True)` to return
detached device metrics. Aggregate these tensors on the device and convert them
to Python scalars at a logging boundary. Copy or accumulate each result before
the next update, because compiled graph outputs can reuse their storage.
The default `agent.update()` retains
Python-float metrics and their immediate scalar reads. This changes diagnostic
handling; the original FlashSAC network updates remain in use.

The FlashSAC paper used Isaac Lab 2.1.0 with PhysX. Isaac Lab 3 tasks and assets
have evolved, and Newton physics differ; these runs demonstrate integration and
learning behavior, not a reproduction of the paper's benchmark.
