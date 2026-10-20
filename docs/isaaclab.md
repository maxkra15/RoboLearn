# Isaac Lab integration

The [current Isaac Lab integration branch](https://github.com/maxkra15/IsaacLab/tree/feature/robolearn-lean-integration-20261020)
is based on upstream [main `159af106`](https://github.com/isaac-sim/IsaacLab/commit/159af106ce06165d7147683695b20d2e3ed63257)
and registers RoboLearn as an RL backend. Its lock pins the cleaned library at
[`994b6d3d`](https://github.com/maxkra15/RoboLearn/commit/994b6d3ddf3b4d764a5a0a0e74b500cfae3fb3ff),
including lazy imports and the experimental Warp FlashSAC warning:

```bash
git clone --branch feature/robolearn-lean-integration-20261020 \
    https://github.com/maxkra15/IsaacLab.git
cd IsaacLab
git checkout d82da26b725039dcacb8cba29d53ebd7b6d45b98
uv sync --locked --extra robolearn --python 3.12.14
```

Use uv 0.13.3 for this lock. Then select a learner through the native CLI:

```bash
uv run --locked --extra robolearn isaaclab train --rl_library robolearn --algorithm warp_ppo \
    --task Isaac-Velocity-Flat-G1 --num_envs 1024 --max_iterations 2050 \
    --seed 0 physics=newton_mjwarp \
    agent.algorithm_cfg.optimized_linear_backward=true

uv run --locked --extra robolearn isaaclab train --rl_library robolearn --algorithm flashsac \
    --task Isaac-Velocity-Flat-G1 --num_envs 1024 --max_iterations 2050 \
    --seed 0 physics=newton_mjwarp
```

The integration owns simulator launch, task configuration, rollout collection,
and checkpoints. RoboLearn owns the learner and tensor adapter. Installing the
library into an upstream checkout does not add these registrations.
The frozen measured stack is pinned below; the library cleanup does not change
its algorithm math or defaults and has not been used for a new training run.
The [G1 guide](g1.md) preserves the older comparison, its exact recipes, common
evaluations, and timing scopes.

## Current main baseline (2026-10-20)

Use Git LFS, Python 3.12.14, and uv 0.13.3 for the recorded dependency lock:

```bash
git clone --branch experiment/robolearn-main-baseline-20261020 \
    https://github.com/maxkra15/IsaacLab.git
cd IsaacLab
git checkout afcabbddb03bdd665bddd7c16ab931dd644cbe34
uv sync --locked --extra robolearn --python 3.12.14
```

The lock pins RoboLearn [`7e872dbd`](https://github.com/maxkra15/RoboLearn/commit/7e872dbda8a0385a7fae227dbc6e61c608aa1696),
Torch 2.11.0, Warp 1.17.0, Warp-NN 0.4.1, Newton
[`71eeba16`](https://github.com/newton-physics/newton/commit/71eeba16a717d4dcf0b9dd7056a7b6f863054dc3),
MuJoCo/MuJoCo Warp 3.15.0, and RSL-RL 5.5.1.
The [baseline report](https://maxkra15.github.io/reports/2026-10-20-g1-main-baselines/) records each run's source, measurements, and limitations.
Native RSL PPO used [`0879f3bc`](https://github.com/maxkra15/IsaacLab/commit/0879f3bc7905932770f34f779cbaf6fb72eef0d5);
the RoboLearn runs used [`afcabbdd`](https://github.com/maxkra15/IsaacLab/commit/afcabbddb03bdd665bddd7c16ab931dd644cbe34).
The follow-up changes only configuration inheritance and its changelog; the MDP,
learner code, and dependency lock are unchanged between those revisions.

The MDP uses Torch observations, rewards, resets, and terminations. Newton
MJWarp captures physics; Warp PPO captures its learner update separately.
RSL PPO uses Torch, and FlashSAC uses the authors' compiled Torch recipe with AMP.
The FP32 Warp-NN FlashSAC backend remains experimental; synthetic learner probes
have a separate scope from the walking comparison.

Compare training-loop time, production process wall time, and the complete
smoke/train/evaluation pipeline separately. Cold compilation and module loading
can dominate short runs, while production processes can reuse smoke caches.
The [2026-10-09 comparison](g1.md) retains its older frozen setup and results.

## Library API and run ownership

Current configuration imports do not load the optional learning runtimes:

```python
from robolearn.flashsac import FlashSACConfig
from robolearn.warp.config import PPOConfig
```

`PPOConfig` is also exported from `robolearn.warp`; its learner classes load lazily.
The frozen baseline above intentionally retains its recorded older library pin.
The lock above records the tested Isaac Lab combination; install runtime extras
within that supported Python/Torch/Warp environment.

| Learner | Metrics supplied to the caller | Core checkpoint support |
| --- | --- | --- |
| Torch FlashSAC | `update(tensor_metrics=True)` returns detached device metrics. | `save`/`load` restore networks, optimizers, schedulers, AMP scaler, and reward statistics. Replay has separate save/load methods. |
| Warp PPO | `metrics` stays on the device; `health_metrics()` returns a synchronized host snapshot. | `save`/`load` store parameters, Adam state, and sampling seed in NPZ. The caller supplies the matching configuration. |
| Experimental Warp FlashSAC | `update(tensor_metrics=True)` returns device metrics; `get_metrics()` returns host values. | JSON/NPZ stores configuration, networks, Adam, RNG, and reward statistics; replay is optional. |

RoboLearn core does not create log writers. The Isaac Lab runner provides basic
logging: console progress and one `metrics.jsonl` record per iteration under
`logs/robolearn/<experiment>/<run>/`. TensorBoard and W&B writers are not wired
into this runner. Host metric reads and checkpoint serialization synchronize;
perform them at logging/checkpoint boundaries outside captured updates.

The integration adds run manifests, saved configurations, and checkpoint JSON
sidecars with dimensions and iteration/transition/update counters. Native
`train --checkpoint` restores this state and starts a fresh environment rollout;
SAC replay is rebuilt by default. `play --checkpoint` restores the saved policy
contract. The `latest` and `best` selectors currently choose the newest compatible
checkpoint; `best` does not rank evaluation scores.

### Remaining upstream integration

- Review and maintain the thin Isaac Lab adapter, task/agent registrations, and
  optional dependency pin in upstream Isaac Lab; installing RoboLearn alone does
  not register the backend.
- Add an optional TensorBoard/W&B bridge at the Isaac runner's logging boundary,
  reusing the existing metrics instead of adding writers to the learner core.
- Define supported policy export formats and register an export backend. The
  current fork supports training and playback, without a RoboLearn `isaaclab export` backend.

## Standalone FlashSAC example

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
[Isaac Lab integration branch](https://github.com/maxkra15/IsaacLab/tree/feature/robolearn-lean-integration-20261020)
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
