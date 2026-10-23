# RoboLearn

Small reinforcement learning implementations for GPU robot simulation.

| Learner | Use | Status |
| --- | --- | --- |
| `robolearn.flashsac.FlashSAC` | Authors' Torch implementation, including compilation and AMP | Recommended FlashSAC backend |
| `robolearn.warp.WarpPPO` | Fixed buffers and capturable Warp-NN PPO updates | Single-GPU state observations |
| `robolearn.warp.WarpFlashSAC` | FP32 Warp-NN port | Experimental; slower in current learner benchmarks |

Isaac Lab owns simulation, rollout collection, and experiment logging. RoboLearn
owns policy inference, learning, and native learner checkpoints. The package
keeps Torch and Warp optional, and configuration imports do not load either runtime.

RoboLearn is an independent project. FlashSAC was developed by Donghu Kim,
Youngdo Lee, and their coauthors; see [credits](#credits) and [ATTRIBUTION.md](ATTRIBUTION.md).

## Install

Use Python 3.10 or newer. Install only the algorithm you need:

```bash
pip install 'robolearn-rl[flashsac] @ git+https://github.com/maxkra15/RoboLearn.git'
pip install 'robolearn-rl[warp] @ git+https://github.com/maxkra15/RoboLearn.git'
```

The distribution is named `robolearn-rl`; the Python import is `robolearn`.
Source releases are hosted on GitHub; nothing is published to PyPI.

For development:

```bash
git clone https://github.com/maxkra15/RoboLearn.git
cd RoboLearn
uv sync --extra flashsac --extra warp
uv run ruff check .
uv run pytest -q tests/test_flashsac.py tests/test_isaaclab_adapter.py
```

Isaac Lab and MuJoCo Warp are installed separately in their own supported
environments. The library does not install simulators, launchers, or experiment services.

## FlashSAC

```python
from robolearn.flashsac import FlashSAC, FlashSACConfig

agent = FlashSAC(31, 8, num_envs=1024, cfg=FlashSACConfig(device="cuda:0"))
```

The agent exposes tensor actions, transition ingestion, gradient updates, and
checkpointing. See [the Isaac Lab guide](docs/isaaclab.md) and
[example](examples/isaaclab_flashsac.py) for a complete training and evaluation loop.

The experimental `robolearn.warp.WarpFlashSAC` backend uses the same configuration,
authors' residual networks, categorical critics, and update order. It accepts Warp
arrays and captures replay sampling and learning updates. Start with FP32 to check
numerical fidelity; mixed precision is not implemented. See the
[Warp FlashSAC guide](docs/warp_flashsac.md) for scope and validation status.
Construction emits a `RuntimeWarning` identifying the experimental backend;
existing imports and checkpoint formats remain available.

The [Cartpole report](https://maxkra15.github.io/reports/2026-10-09-cartpole-flashsac/)
compares four Torch and Warp recipes with three seeds each. Its
[profiling appendix](https://maxkra15.github.io/reports/2026-10-09-cartpole-flashsac/#profiling)
examines synthetic learner timings separately from the training results.

## WarpNN PPO

See [the Warp guide](docs/warp.md) and [capture example](examples/warp_ppo_graph.py).
Graph capture requires fixed shapes and persistent device buffers. Capturing an
entire simulation and training cycle also requires device implementations of
observations, rewards, resets, and rollout collection.
The PPO implementation supports state observations on one GPU. A
[MuJoCo Warp example](examples/mujoco_warp_capture.py) captures physics, rollout
collection, and PPO updates in one graph. The experimental Warp-NN FlashSAC
backend captures learning separately from the Isaac Lab environment.

## Isaac Lab, logging, and results

The [integration guide](docs/isaaclab.md) covers installation, the native
`isaaclab train/play --rl_library robolearn` adapter, supported metrics, and
checkpoint/resume behavior. That adapter currently lives on our Isaac Lab branch;
installing RoboLearn alone does not register it in upstream Isaac Lab.
JSONL and console logging are available there. TensorBoard and W&B writers are
not wired into that runner.

The [2026-10-20 G1 report](https://maxkra15.github.io/reports/2026-10-20-g1-main-baselines/)
compares RSL PPO, Warp PPO, and Torch FlashSAC across three seeds each. Warp PPO
reduced update time by 26.1% and recorded loop time by 4.5%; its complete
smoke/train/evaluation pipeline took 7.1% longer. Torch FlashSAC passed the walking
gate at earlier sampled checkpoints while taking longer over the full budget.
These are descriptive results for one task on isolated L40 allocations.
The MDP uses Torch and the physics and learner graphs are separate.

The report also includes isolated Warp FlashSAC learner measurements; they do
not establish G1 learning quality for that backend. See the
[performance explanation](docs/warp_flashsac.md#current-performance-2026-10-20),
[historical G1 comparison](docs/g1.md), and [earlier validation](docs/validation.md).

## Differentiable learning

PPO already differentiates its policy and value networks. Using gradients through
physics requires a differentiable simulator and an additional learning objective.
The current G1 MuJoCo Warp setup does not provide that path. See
[differentiable learning](docs/differentiable.md) for the separate experimental
Cartpole prototype, solver limitations, and a minimal route to a hybrid learner.

## Experimental native CUDA learning kernels

The optional [CUDA backend](docs/cuda.md) replaces dense forward/backward and
Adam in both Warp learners, preserving their current networks and losses. It
uses cuBLASLt matrix products plus original CUDA bias-gradient and multi-tensor
Adam kernels, persistent Warp-owned arrays, and CUDA graph capture. It does not
depend on Torch. CUDA development tools are required when first building it.
This is experimental; inspect the numerical and timing results before adopting it.
The [upstream learner comparison](docs/cuda.md#comparison-with-upstream-learners-23-october-2026)
measures actual RSL-RL PPO and official FlashSAC alongside RoboLearn's backends.

## Credits

**FlashSAC** — Donghu Kim, Youngdo Lee, Minho Park, Kinam Kim, I Made Aswin Nahendra,
Takuma Seno, Sehee Min, Daniel Palenicek, Florian Vogt, Danica Kragic,
Jan Peters, Jaegul Choo, and Hojoon Lee.
[Paper](https://arxiv.org/abs/2604.04539) ·
[Official code](https://github.com/Holiday-Robot/FlashSAC) ·
[Project](https://holiday-robot.github.io/FlashSAC/).
The extracted implementation retains Holiday Robotics' MIT copyright and license.

**PPO** — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
RoboLearn's Warp implementation is original code implementing their algorithm.

**WarpNN, Warp, and MuJoCo Warp** — NVIDIA and the respective project contributors.
Their implementations are dependencies, with their own licenses.

Please cite the original algorithm papers when using RoboLearn in research;
BibTeX entries are in [CITATIONS.bib](CITATIONS.bib). The structure was informed
by [RSL-RL](https://github.com/leggedrobotics/rsl_rl) and
[RL-Games](https://github.com/isaac-sim/rl_games).
