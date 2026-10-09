# RoboLearn

Small reinforcement learning implementations for GPU robot simulation.

- **FlashSAC:** a PyTorch library adapted from the authors' official implementation,
  with an Isaac Lab 3 example and an experimental FP32 Warp-NN backend.
- **PPO:** an implementation using [NVIDIA WarpNN](https://nvidia.github.io/warp-nn/),
  with fixed buffers and a CUDA graph capture interface.

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

## Isaac Lab G1 results

The [G1 reproduction guide](docs/g1.md) compares stock RSL-RL PPO, matched Warp
PPO, and the authors' optimized FlashSAC recipe on the same Isaac Lab 3 task.
Each learner used three seeds and 50.38 million transitions per seed on isolated
NVIDIA L40 GPUs. All final forward evaluations passed walking and survival.

- Warp PPO reduced PPO update time by **18.6%**; its production training loop
  was **2.5% faster** because rollout dominated. Preparation costs prevented
  an overall command-pipeline speed win.
- FlashSAC passed the walking gate at earlier sampled checkpoints and achieved
  lower final tracking error, while taking longer to process the full budget.

These are descriptive results for one task. The MDP runs in Torch, with separate
physics and learner graphs. The tested Warp recipe explicitly enables tiled
Linear gradients; FlashSAC uses AMP, compilation, and deferred tensor metrics.
The library keeps existing defaults, so use the documented recipe to reproduce it.

See [earlier validation](docs/validation.md) for the drawer run, checkpoint
playback, and direct CUDA capture checks.

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
