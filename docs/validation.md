# Initial validation

Validated on 2026-10-08 with Python 3.12 and an NVIDIA RTX 4090.
These checks demonstrate implementation behavior and integration; they are
single-seed runs, not comparative performance benchmarks or paper reproductions.

## FlashSAC with Isaac Lab

Used Isaac Lab develop commit `abe7db4f38174a8ed4e4f843b28d6fe68f34b578`,
Newton 1.6.1 / MuJoCo Warp 3.12.0, and PyTorch 2.12.0 with CUDA 13.
The [Isaac Lab example](../examples/isaaclab_flashsac.py) ran its default
`Isaac-Open-Drawer-Franka` configuration: 1,024 environments, seed 0,
5,000,192 transitions, 9,568 updates, and a 1-million-transition replay buffer.
Network compilation was disabled; AMP was enabled.

| Check | Mean episode return |
| --- | ---: |
| Initial deterministic policy, 1,024 environments | 8.82 |
| Trained deterministic policy, same evaluation seed | 90.67 |
| Saved checkpoint loaded in 64 environments | 91.59 |

Evaluation collects one complete episode per environment. Playback restored
the saved task, action scaling, observation groups, physics, agent weights,
and reward statistics. See [the training guide](isaaclab.md) for commands.

## WarpNN PPO

Used WarpNN 0.4.0, Warp 1.18.0, and MuJoCo / MuJoCo Warp 3.15.0.

- The [bandit example](../examples/warp_ppo_graph.py) trained for 100 updates:
  sampled mean reward improved from -1.6074 to -0.1036; the deterministic action
  was 0.5324 against the target of 0.5.
- The [joint capture example](../examples/mujoco_warp_capture.py) completed
  50 captured simulation/rollout/PPO iterations after warmup. Both eager and
  captured runs produced finite loss, a maximum first-layer weight change of
  0.254791, and final-step mean reward of -0.0523. Device sampling seeds advanced
  on every replay.
- All 13 local tests passed: nine FlashSAC/adapter tests and four CUDA PPO tests.
  These include numerical PPO gradients, timeout-aware GAE, stable advantage
  normalization, repeated eager/captured update parity, and checkpoint reuse.
- Ruff, wheel/source builds, and the citation schema check passed. Both builds
  retain the upstream FlashSAC license.

The [Warp guide](warp.md) describes the supported scope and requirements for
joint capture. FlashSAC currently uses PyTorch; a WarpNN FlashSAC implementation
and comparative benchmarks are not included.
