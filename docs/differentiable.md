# Experimental learning through physics gradients

`robolearn.warp.experimental.WarpPathwiseActorCritic` connects native simulator
derivatives to policy parameters through short, out-of-place rollouts. It uses
Warp-NN layers and Adam. The regular PPO implementation remains independent.

The actor minimizes

\[
L_\mathrm{actor}=-\frac{1}{N}\sum_i\left[
\sum_{t=0}^{H-1}\gamma^t r(s_t, a_t)
+\gamma^H V_\mathrm{frozen}(s_H)\right],
\qquad a_t=\tanh\pi_\theta(s_t).
\]

Gradients pass through rewards, observations, simulator state transitions, and
the endpoint critic's input. The critic parameters in that actor computation
have no gradients. Critic fitting uses detached observations and n-step targets
from the same rollout. Each policy step has independent activation caches sharing
the actor weights, so later calls cannot overwrite saved forward values.

This is a **prototype inspired by SHAC**, not a reproduction of the published
algorithm. It uses a deterministic actor, n-step targets, and a critic snapshot
each iteration. It does not implement SHAC's stochastic policy, TD-lambda critic,
normalization, or target-network schedule. It has no claimed advantage over PPO.

## Environment contract

The caller supplies the physics implementation. Its adapter must expose:

- `num_envs`, `obs_dim`, `action_dim`, and a Warp `device`.
- `prepare(horizon)` to allocate distinct state/observation/reward arrays per step.
- `begin_rollout()` to detach the current state before recording gradients.
- `observe(step)` returning a gradient-enabled float32 `(num_envs, obs_dim)` array.
- `step(step, actions)` returning gradient-enabled native rewards `(num_envs,)`
  and an int32 continuation mask `(num_envs,)`.
- `finish_rollout()` returning the last pre-reset observation.
- `after_update()` to detach the final state and reset failed/time-limited rows.

Continuation is zero at true termination. Rewards after the first termination
are masked, and critic fitting excludes those rows. Time limits retain a
final-state value bootstrap, a continuing-value approximation commonly used in
RL; reset must occur outside the tape. Resets and discrete termination decisions
are not differentiated. The adapter must respect episode boundaries even when
they fall inside a requested segment, or reject incompatible horizon lengths.

Actions lie in `[-1, 1]`; the adapter applies native force scaling. Reward shaping
and state/action scaling are not supplied by this learner. Preserve the original
MDP and report any changed objective explicitly.

```python
from robolearn.warp.experimental import PathwiseConfig, WarpPathwiseActorCritic

# environment implements the contract above, including actual physics derivatives.
agent = WarpPathwiseActorCritic(
    observation_dim=4,
    action_dim=1,
    num_envs=128,
    horizon=10,
    config=PathwiseConfig(hidden_dims=(32, 32), seed=0),
)
for iteration in range(2000):
    agent.launch_update(environment)
agent.save("pathwise_actor_critic.npz")
```

For a standalone policy-gradient diagnostic, fix the initial physical state and
critic parameters, then call `tape = agent.forward_actor(environment)` and
`tape.backward(agent.actor_loss)`. Inspect `agent.actor_parameters[j].grad`, compare
directional derivatives with central differences, and call `tape.zero()`.
No optimizer runs in `forward_actor`.

Checkpoints contain actor/critic parameters and their Adam state, with fixed
array addresses on reload. Configuration, simulator state, and reset RNG state
are not included, so checkpoint loading does not reproduce an uninterrupted run.

## Capture and physics support

Run a complete warmup before placing `agent.launch_update(environment)` in an
outer Warp CUDA graph. The learner has no host reads during updates, but its
environment must also support capture of the physics backward, persistent
adjoint scratch buffers, state copies, and resets. Capture support must be
measured; recording a graph does not create simulator derivatives.

Released MuJoCo Warp does not provide the physics adjoint required by this
experiment. The Isaac Lab Cartpole prototype uses the separate experimental
[hybrid analytic differentiability implementation](https://github.com/google-deepmind/mujoco_warp/pull/1535),
pinned to commit `357a75d60a56d67d476942a1b6e54b3045ee8e87`.
It is an unmerged dependency with geometry and feature limitations. Contact
derivatives and G1 mesh compatibility require additional validation before
extending the Cartpole result to humanoid locomotion.

## Captured native Cartpole pilot

Each of two training seeds (`0` and `1`) completed 2,000 actor and 8,000 critic updates with
64 environments and horizon 10: **1.28 million nominal transitions**, including
masked slots after true termination. Training retained the native manager task's
reward, force scaling, reset ranges, two physics substeps, and 300-step time limit.

| Evaluation reset seed | Mean native return / maximum 5 | Upright fraction | Timeout survival | Wrapped angle RMS |
| --- | ---: | ---: | ---: | ---: |
| 10000, initial policy | -21.940 | 1.57% | 100% | 2.318 rad |
| 10000, final policy | 4.430 | 81.02% | 89.06% | 0.176 rad |
| 10001, final policy | 4.551 | 84.84% | 92.19% | 0.159 rad |

Each row averages 64 complete native episodes. Upright means wrapped angle
within 0.2 rad; survival only checks timeout without cart-bound failure. A fallen
pole does not terminate, which explains the initial policy's high survival and
poor angular control. The table describes training seed `0`; its seed-10000
return peaked at 4.587 at update 1,000, so this
pilot does not establish monotonic improvement.

A second training run with seed `1` reached return **4.794**, upright **86.20%**,
and timeout survival **98.44%** on evaluation seed `10000`. On fresh evaluation
seed `10001`, it reached **4.814**, **88.32%**, and **98.44%**, respectively
(64 native episodes per evaluation). Its gradient probes and eager/captured parity
also passed, and all final parameters/Adam arrays were finite. Two training
seeds demonstrate this bounded pilot's repeatability; they do not establish
an advantage over another learner or reliable performance in other tasks.

For training seed `0`, uniform-action and final-actor-bias finite-difference probes passed with best
relative errors of **0.130%** and **0.320%**, respectively. All three recorded
epsilon choices were below 1.7%. The initial zero critic made that actor probe
specifically exercise the physics-to-policy gradient. These local probes do not
validate every state, contact, or parameter. Native one-step reward parity was
within `2.8e-9`; observations matched exactly.

One complete eager update and graph replay from restored states agreed within
`2.98e-8` on parameters and `3.73e-9` on Adam state, with zero qpos/qvel error.
The captured graph included physics, observations/rewards, their backward,
actor/critic Adam updates, and detached state/reset kernels. Python logging,
checkpoint serialization, and native evaluations ran outside it.

The training loop plus checkpoint evaluations took **39.4 s** after warmup;
the measured process interval including setup, diagnostics, and final evaluation
was **48.8 s**. The local GPU shared an unrelated workload. These timings do not
establish an isolated throughput result or a speed advantage over PPO.
Training seed `1` took 39.8 s for its training loop plus checkpoint evaluations,
and 48.8 s for the measured process interval.

### Reproduce the experiment

Use the `experiment/differentiable-cartpole` branches of both RoboLearn and
[Isaac Lab](https://github.com/maxkra15/IsaacLab/tree/experiment/differentiable-cartpole).
The [native experiment harness](https://github.com/maxkra15/IsaacLab/blob/experiment/differentiable-cartpole/scripts/benchmarks/try_differentiable_cartpole.py)
owns simulator startup, parity/finite-difference diagnostics, full native episode
evaluation, and checkpoints; the learner remains in this package.

Requirements: Isaac Lab 3 development checkout with its native Cartpole assets,
an NVIDIA CUDA device, Newton `1.6.1`, Warp `1.17.0`, Warp-NN `0.4.x`, and the
pinned experimental MJWarp commit above. From the Isaac Lab checkout, with the
RoboLearn experiment checkout at `../RoboLearn`:

```bash
uv sync --extra robolearn
uv pip install --no-config --no-sources --python .venv/bin/python --reinstall --no-deps -e ../RoboLearn
uv pip install --no-config --no-sources --python .venv/bin/python --reinstall --no-deps \
  'mujoco-warp @ git+https://github.com/etaoxing/mujoco_warp.git@357a75d60a56d67d476942a1b6e54b3045ee8e87'
uv run --no-sync python scripts/benchmarks/try_differentiable_cartpole.py \
  --output logs/differentiable-cartpole/captured-pilot-seed0 \
  --num_envs 64 --horizon 10 --iterations 2000 --seed 0 --capture
uv run --no-sync python scripts/benchmarks/render_differentiable_cartpole.py \
  --input logs/differentiable-cartpole/captured-pilot-seed0/results.json \
  --output logs/differentiable-cartpole/captured-pilot-seed0/report.html
```

The output directory must be new. `--no-sync` keeps these deliberate local
dependency overrides instead of restoring the ordinary locked releases.
The install commands ignore project source overrides so the experimental
MJWarp revision is installed as requested.
Use `--gradient_only` for the native parity and finite-difference probes without
training. The output includes raw results, per-update diagnostics, initial/final
checkpoints, and an offline interactive HTML report.

## Attribution

SHAC was introduced by Jie Xu, Viktor Makoviychuk, Yashraj Narang, Fabio Ramos,
Wojciech Matusik, Animesh Garg, and Miles Macklin in
[*Accelerated Policy Learning with Parallel Differentiable Simulation* (2022)](https://arxiv.org/abs/2204.07137).
The learner is original RoboLearn code inspired by the paper's short-horizon
actor/critic idea; no SHAC implementation is copied.

Warp-NN and Warp are NVIDIA dependencies under Apache-2.0. MuJoCo Warp is an
Apache-2.0 dependency from Google DeepMind, NVIDIA, and project contributors.
The experimental adjoint is developed upstream by Eliot Xing, Eric Heiden,
Miles Macklin, and project contributors;
see the linked pull request for contributor provenance and implementation details.
