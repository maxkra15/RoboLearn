# Experimental Warp-NN FlashSAC

`robolearn.warp.WarpFlashSAC` ports the authors' FlashSAC architecture and losses
to Warp-NN 0.4 and FP32 Warp kernels. Original authors, paper, upstream revision,
and license are recorded in [ATTRIBUTION.md](../ATTRIBUTION.md). This is an
experimental backend, with no established G1 quality or speed advantage.

## Why FP32 first

Mixed precision can reduce memory traffic and use tensor cores more effectively.
It does not inherently improve the policy's reward or sample efficiency. SAC's
categorical probabilities, entropy temperature, normalization statistics, and
optimizer updates also need numerical care. FP32 is the first implementation so
fixed-data comparisons can isolate porting errors. The current Torch G1 recipe
uses FP16 AMP; comparisons must label that precision difference explicitly.

Warp-NN's built-in training layers use FP32. A mixed precision backend would need
additional layer and optimizer work, followed by numerical and learning checks.
Warp-NN's CUDA capture support does not guarantee faster kernels than optimized
Torch libraries. Measure preparation, warmed actor inference, replay insertion,
warmed updates, and complete training separately.

## API

Install the `warp` extra. Torch is needed only for Isaac Lab integration or the
reference diagnostic, not by the Warp learner itself.
The recorded numerical runs use Warp 1.17.0 and Warp-NN 0.4.0. The custom backward
hooks depend on Warp tape internals and Warp-NN layer caches and Adam state;
review those interfaces when updating dependencies.

```python
from robolearn.flashsac import FlashSACConfig
from robolearn.warp import WarpFlashSAC

cfg = FlashSACConfig(device="cuda:0", use_amp=False, use_compile=False)
agent = WarpFlashSAC(4, 1, num_envs=1024, cfg=cfg)
agent.prepare(capture=True)  # Warm and capture without retaining synthetic updates.
# uniform_actions: caller-provided Warp array sampled uniformly in [-1, 1].
actions = agent.act(observations) if agent.ready else uniform_actions
# Step the environment, retaining pre-reset next observations at episode ends.
agent.observe(observations, actions, rewards, terminated, truncated, next_observations)
if agent.ready:
    for _ in range(2):  # Authors' two replay updates per vector environment step.
        metrics = agent.update(tensor_metrics=True)
        # Copy or accumulate these device metrics before the next update.
agent.save("checkpoint")  # Model/optimizer state; fresh replay on resume by default.
```

Observations and actions are FP32 two-dimensional Warp arrays on the learner's
device. Rewards are one-dimensional FP32 arrays; termination and truncation masks
are `wp.int32`. `act` returns persistent storage: consume its actions before the
next call. Use `training=False` for deterministic evaluation. Replay ingestion
copies transitions into owned storage. Timeouts stop n-step accumulation but
retain bootstrap from the pre-reset next observation; true terminations remove it.
The caller owns uniform-action replay warmup and the number of updates per
environment step. The Isaac Lab runner supplies both. Device metrics also use
persistent storage: copy or accumulate each result before the next update.

The actor and critic normalize concatenated current/next batches during learning,
matching the authors' BatchNorm statistics. Actor, temperature, critic, and target
updates preserve their original order. Actor updates use the configured period;
each `update` counts one critic update. Targets average parameters, with their own
BatchNorm statistics. Captured sampling advances device RNG state on every replay.

BatchNorm, RMSNorm, and LogSoftmax retain Warp-NN's forward calculation and use fixed-order
analytical backward reductions. Initial G1-size experiments found that atomic
gradient reductions could perturb near-zero gradients enough for Adam to amplify
them into different updates. This change targets that source of variation;
complete updates still require numerical and learning validation. Different
floating-point reduction orders and activation boundaries can affect agreement
with Torch even when the mathematical operations match.

## Capture and integration

The backend captures separate actor/temperature/critic and critic-only learning
graphs using persistent arrays. It does not capture an arbitrary environment.
The Isaac Lab adapter exposes `--algorithm warp_flashsac` for registered Cartpole
and flat G1 tasks, using the existing Torch MDP and explicit CUDA stream
dependencies. Physics, observations, rewards, resets, and logging retain their
existing workflow. A whole physics/MDP/learner graph requires a separate native
environment implementation.

NumPy initialization and Warp sampling differ from Torch RNG. Compare shared
initial weights, fixed batches, and fixed noise for numerical fidelity; equal
seeds alone cannot establish identical trajectories. Checkpoints are backend
specific. Saving replay is optional and large; resuming without it deliberately
starts fresh replay. Environment state is not included.
For standalone loading, construct the learner with the checkpoint's saved
`FlashSACConfig` from `config.json` before calling `load`. Loading checks dimensions
and restores arrays; it leaves the learner's configured hyperparameters in effect.
The Isaac Lab runner restores and checks its saved configuration before loading.

## Validation

The standalone `examples/diagnose_warp_flashsac.py` reference compares against
the authors' Torch FP32 implementation. It is a numerical experiment, not a
robotics learning benchmark. Record its output and GPU/runtime versions before
making numerical or performance claims. G1 results in [g1.md](g1.md) describe
the existing Torch FlashSAC backend, not this experimental port.

The initial L40 experiments pass the small-network and replay checks. The full
G1-size experiment passes forward, isolated-gradient, and checkpoint checks but
still fails strict complete optimizer-update and eager/captured agreement checks.
The LogSoftmax change reduced failed capture comparisons from 111 to 6. A focused
actor comparison against FP64 and finite differences at three sensitive coordinates
supports the actor derivatives at that initial state. These strict full-update
failures remain recorded. Learning comparisons allow different Torch and Warp
trajectories and assess native parameter health and common policy evaluations.

## Initial Cartpole comparison

The retrieved 12-run cohort passed SHA256 verification and audits of its native
checkpoints, optimizer state, budgets, shared initial fixtures, and evaluations.
The original comparison JSON remains unchanged; its SHA256 is
`6180178817857404af14ea2dbfe63531fde8e89be348e1755e5e7a3bb5b8cfa2`.

Each recipe uses learner seeds 0/1/2, 1,024 environments, 24 vector steps per
iteration, and 200 iterations: 4,915,200 collected transitions per run. The
recipes share initial weights per seed and the same 128-world evaluation reset
fixture. Evaluations use deterministic actions at iterations 0/50/100/150/200.

| Recipe | Final upright occupancy, seeds 0/1/2 | Final cart-bound survival, seeds 0/1/2 | Mean warmed throughput | Mean training-process wall |
| --- | --- | --- | --- | --- |
| Torch FP32 eager | 7.7% / 85.9% / 37.0% | 0% / 62.5% / 0% | 37,995 transitions/s | 201.3 s |
| Torch FP32 compiled | 96.4% / 92.7% / 7.1% | 98.4% / 100% / 0% | 71,853 transitions/s | 1,088.3 s |
| Torch FP16 AMP compiled | 81.2% / 71.2% / 94.9% | 54.7% / 52.3% / 100% | 72,828 transitions/s | 557.4 s |
| Warp FP32 captured | 96.1% / 8.9% / 14.4% | 100% / 0% / 0% | 15,026 transitions/s | 444.8 s |

The Warp seed outcomes vary substantially, and this captured recipe has lower
warmed throughput than compiled Torch. Graph capture alone does not establish
faster kernels or reliable learning. Three seeds provide descriptive outcomes;
this cohort does not isolate the cause of the differences or establish a general
advantage for a backend. Compilation and preparation costs change the process-wall
comparison: Warp's process is shorter than these compiled Torch recipes, while
eager Torch has the shortest mean process wall for this budget.

All 12 runs have finite, updated actor/critic parameters and finite optimizer
state. Each attempted 9,402 critic and 4,701 actor/temperature updates. AMP seeds
0/1/2 completed 9,400/9,401/9,402 critic updates, with 2/1/0 skipped steps;
all other attempted updates completed. Checkpoint health establishes executable
training, while the policy metrics above show that learning remains inconsistent.

The Cartpole MDP terminates at cart bounds or its 300-step / 5-second time limit;
pole falls do not terminate it. Survival therefore means completing the episode
within the cart bounds, while upright occupancy measures balance separately.
The existing Torch MDP and MJWarp physics are shared; learner capture is separate
from environment execution.

The comparison overrides the API example's defaults with a 131,072-transition
replay capacity, 100,000-transition warmup, batch size 2,048, n-step horizon 3,
two critic updates per vector step, actor period 2, and learning-rate decay over
9,600 attempted critic updates. It retains the authors' 128/256-wide actor/critic,
two residual blocks, 101 categorical bins, and reward normalization. This is one
fixed recipe; larger replay or longer training needs a separate measurement.
Warmed throughput excludes the recorded first-use interval; startup and whole
training-process wall time are separate measurements. Training-process wall
includes imports, environment setup, preparation, learning, checkpoint audits,
and shutdown; separate evaluation subprocesses and OSMO queueing/retrieval are
excluded. Artifact transfer overlapped later training runs. The transfer performed
no GPU profiling or computation. Host/disk effects may affect those timings;
the recorded values have not been adjusted.

The measured working snapshot reports package version 0.2.0 and includes the
unreleased Warp backend. Candidate 0.3.0 packages that API with updated release
metadata. Reproduction depends on the recorded source hashes and recipe, rather
than the version label alone.
