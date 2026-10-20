# Simulator gradients and policy learning

RoboLearn's regular PPO and FlashSAC learners compute gradients of neural-network
losses. They do not differentiate through Isaac Lab's physics or manager MDP.
The [experimental Cartpole branch](https://github.com/maxkra15/RoboLearn/tree/experiment/differentiable-cartpole)
contains a separate pathwise actor/critic prototype; it is not part of this branch
or an option in `PPOConfig`.

## Two gradient estimators

[PPO](https://arxiv.org/abs/1707.06347) differentiates a clipped likelihood-ratio
objective using sampled states, actions and estimated advantages. These samples
are held fixed during the update. Its policy gradient does not require the
derivatives of the environment transition or reward.

A pathwise objective instead differentiates simulated return through actions,
state transitions, observations and rewards into policy parameters. For a short
rollout, one possible objective is
`J = sum_t gamma**t * r(s_t, a_t) + gamma**H * V_frozen(s_H)`.
The endpoint critic's parameters are frozen for the actor update, while its
derivative with respect to the endpoint state remains active. A stochastic actor
can use reparameterized actions with fixed noise for each recorded rollout.

[SHAC](https://arxiv.org/abs/2204.07137), by Jie Xu and coauthors, combines short
differentiable rollouts with a critic to limit long-horizon gradient instability.
Replacing PPO's actor loss with this objective changes the learning algorithm.
Adding a weighted pathwise auxiliary loss to PPO is also an experimental hybrid;
PPO's ratio clipping does not constrain that additional term.

## What the physics libraries support

[Warp autodiff](https://nvidia.github.io/warp/latest/user_guide/differentiability.html)
provides `requires_grad` arrays, `wp.Tape` and generated or custom adjoints. It
cannot supply missing solver derivatives merely because physics uses Warp.
Internal derivatives used by an implicit integrator likewise do not constitute
an end-to-end policy gradient API.

[Newton's solver matrix](https://newton-physics.github.io/newton/latest/solvers/index.html#supported-features)
lists basic differentiability for `SolverFeatherstone` and `SolverSemiImplicit`,
but none for `SolverMuJoCo`, the solver used by the Isaac Lab MJWarp baseline.
Changing solvers changes the simulated dynamics and needs separate validation.
The [MJWarp FAQ](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#features)
also states that released MJWarp does not support Warp autodiff. Its proposed
[analytic adjoint](https://github.com/google-deepmind/mujoco_warp/pull/1535) and
[adjoint dispatch API](https://github.com/google-deepmind/mujoco_warp/pull/1727)
remain open development work as of 20 October 2026; the API proposal alone
registers no physics adjoints.

## Requirements for an experimental adapter

- Snapshot the complete rollout start, including actuator state, action history,
  solver warmstart, episode counters and relevant random state. Detach segment
  boundaries and preserve terminal observations before reset.
- Retain distinct state and network activation buffers for every recorded step.
  A Warp tape retains array references, not historical copies. Reusing a buffer
  can destroy the values needed by backward; longer horizons increase memory.
- Provide derivatives for the full action-to-reward path: action transforms,
  actuators, solver, observations and rewards. Host reads or detached interop
  buffers break this chain unless an explicit derivative bridge reconnects it.
- Treat termination decisions and reset sampling as discrete boundaries. Mask
  rewards after true termination, exclude its value bootstrap, retain the
  pre-reset value for timeouts, and run resets outside the differentiable tape.
- Validate contact behavior separately. Contact-set changes, friction switches,
  geometry queries and clamps can produce discontinuous or unhelpful gradients.
  Newton's experimental [contact reconstruction](https://newton-physics.github.io/newton/latest/concepts/collisions.html#differentiable-contacts)
  uses frozen contact geometry and a local approximation, not derivatives of the
  entire collision search or a guarantee of a differentiable MuJoCo solver.
- Warm up and validate forward/backward before capture. Fixed device buffers and
  persistent adjoint scratch storage must support replay without host work.
  [Graph capture](https://nvidia.github.io/warp/latest/user_guide/runtime.html#graphs)
  reduces dispatch overhead; it neither creates nor validates gradients.

## Existing prototype and a minimal next step

The [prototype documentation](https://github.com/maxkra15/RoboLearn/blob/experiment/differentiable-cartpole/docs/differentiable.md)
and [Isaac Lab harness](https://github.com/maxkra15/IsaacLab/blob/experiment/differentiable-cartpole/scripts/benchmarks/try_differentiable_cartpole.py)
describe a deterministic short-horizon learner inspired by SHAC, not a complete
SHAC reproduction or PPO implementation. It pins the unmerged MJWarp adjoint at
`357a75d60a56d67d476942a1b6e54b3045ee8e87`, adds a native generalized-force
adjoint, and retains per-step physics states and activation caches. Its older
Newton/Warp-NN dependency combination needs an explicit port before reuse with
the current baseline. Its Cartpole diagnostics do not validate G1 contacts.

A bounded next experiment would keep ordinary PPO available and use the separate
adapter on Cartpole first: short recorded segments, a frozen endpoint critic,
and either an independent pathwise learner or an opt-in weighted actor auxiliary.
Compare against the same native MDP and evaluations, counting extra simulator
steps, backward work, memory and preparation. Require local gradient agreement
and eager/captured parity before interpreting training results. No hybrid is
implemented here, and this guide claims no speed or policy-quality advantage.
