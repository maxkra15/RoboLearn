# Attribution and provenance

RoboLearn is an independent integration and implementation project. Algorithm
names refer to the work of their original authors, not an invention of this project.

## FlashSAC

The PyTorch implementation in `src/robolearn/flashsac/` is derived from
[Holiday-Robot/FlashSAC](https://github.com/Holiday-Robot/FlashSAC), commit
`87edc9061150ae9e962dd84e6544e27a1554b3ab`.

Original authors: Donghu Kim, Youngdo Lee, Minho Park, Kinam Kim,
I Made Aswin Nahendra, Takuma Seno, Sehee Min, Daniel Palenicek, Florian Vogt,
Danica Kragic, Jan Peters, Jaegul Choo, and Hojoon Lee.

Paper: *FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for
High-Dimensional Robot Control*, 2026, [arXiv:2604.04539](https://arxiv.org/abs/2604.04539).

The copyright belongs to Holiday Robotics. The original MIT license is retained
in [licenses/FlashSAC.txt](licenses/FlashSAC.txt), in source headers, and in
distributed packages. RoboLearn retains the original network architecture,
categorical critics, weight and feature normalization, entropy adaptation,
reward normalization, and exploration mechanism.

Adaptations in this extraction:

- Removed the original multi-simulator framework, Hydra factories, and JAX-only
  type aliases; added a dimensions-based Python API and configuration defaults.
- Kept policy actions and transitions as PyTorch tensors on the device.
- Made compilation opt-in and stored raw model state for eager/compiled checkpoint interoperability.
- Reset episode accumulators when loading reward statistics into a different environment count.
- Stored effective bootstrap discounts for n-step returns interrupted by timeouts.
- Added an Isaac Lab 3 adapter with pre-reset terminal observations.

RoboLearn's checkpoints support weight and optimizer resume. Replay saving is
optional, and pending n-step transitions, exploration noise, and RNG state are
not preserved for bitwise identical continuation.

FlashSAC builds on Soft Actor-Critic, by Tuomas Haarnoja, Aurick Zhou,
Pieter Abbeel, and Sergey Levine:
[Soft Actor-Critic](https://arxiv.org/abs/1801.01290), 2018.

## PPO and GAE

The Warp PPO implementation is original RoboLearn code implementing:

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov,
  [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347), 2017.
- John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel,
  [*High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438), 2015.

It uses the clipped PPO objective and GAE. Numerical references and capture tests
are provided; this implementation is not a port of RSL-RL source code.

## Experimental short-horizon actor/critic

`src/robolearn/warp/experimental.py` is original RoboLearn code inspired by SHAC:
Jie Xu, Viktor Makoviychuk, Yashraj Narang, Fabio Ramos, Wojciech Matusik,
Animesh Garg, and Miles Macklin,
[*Accelerated Policy Learning with Parallel Differentiable Simulation*](https://arxiv.org/abs/2204.07137), 2022.

It implements a deterministic pathwise actor with short physics rollouts and
critic bootstrapping, plus detached n-step critic targets. It does not reproduce
the complete published SHAC method. No code is copied from SHAC implementations.

The Isaac Lab prototype uses the unmerged MuJoCo Warp
[hybrid analytic differentiability implementation](https://github.com/google-deepmind/mujoco_warp/pull/1535),
commit `357a75d60a56d67d476942a1b6e54b3045ee8e87`, developed upstream by Eliot Xing,
Eric Heiden, Miles Macklin, and project contributors. It remains an Apache-2.0
dependency and is not vendored here.
This adjoint must not be described as a capability of released MuJoCo Warp.

## Dependencies and project structure

[WarpNN](https://github.com/NVIDIA/warp-nn) and
[Warp](https://github.com/NVIDIA/warp) are developed by NVIDIA and contributors
and licensed under Apache-2.0. [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
is developed by its project contributors and licensed under Apache-2.0.
These libraries are dependencies; their source is not vendored here.

Package layout, citation files, optional dependencies, and example integration
were informed by [RSL-RL](https://github.com/leggedrobotics/rsl_rl) and
[RL-Games](https://github.com/isaac-sim/rl_games). Their source code is not copied
into RoboLearn.
