# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Fixed-data CUDA diagnostics and PPO timings; no simulator or policy-quality claim.

Dense/Adam reference checks, finite state, counters, scheduler decisions and
checkpoint integrity are mandatory. Twenty-minibatch PPO floating-state budgets
are reported diagnostics unless --strict-ppo-parity is requested. A throughput
measurement with numerical differences does not establish numerical equivalence.
"""

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp
import warp_nn

from robolearn.cuda import CudaAdam, CudaLinear
from robolearn.warp import PPOConfig, WarpPPO
from robolearn.warp._linear import TiledLinear


def error(value, reference):
    difference = value.astype(np.float64) - reference
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "reference_max_abs": float(np.max(np.abs(reference))),
        "relative_rms": float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-12)),
        "finite": bool(np.isfinite(value).all() and np.isfinite(reference).all()),
    }


@wp.kernel(enable_backward=False)
def _record_ppo_schedule(
    learning_rate: wp.array[wp.float32],
    kl: wp.array[wp.float32],
    desired_kl: float,
    cursor: wp.array[wp.int32],
    rates: wp.array[wp.float32],
    decisions: wp.array[wp.int32],
    divergences: wp.array[wp.float32],
):
    index = cursor[0]
    if index < rates.shape[0]:
        rates[index] = learning_rate[0]
        divergences[index] = kl[0]
        decision = int(0)
        if kl[0] > desired_kl * 2.0:
            decision = -1
        elif kl[0] < desired_kl / 2.0 and kl[0] > 0.0:
            decision = 1
        decisions[index] = decision
    cursor[0] = index + 1


class _PPOScheduleTrace:
    """Observe device scheduler state without changing the learner's update math.

    The temporary observer graph is discarded before timing. The production
    graph is subsequently recaptured without these diagnostic kernels.
    """

    def __init__(self, agent):
        self.agent = agent
        self.launch = agent.launch_update
        self.step = agent.optimizer.step
        count = agent.config.epochs * agent.config.num_mini_batches
        self.cursor = wp.zeros(1, dtype=wp.int32, device=agent.device)
        self.rates = wp.zeros(count, dtype=wp.float32, device=agent.device)
        self.decisions = wp.zeros(count, dtype=wp.int32, device=agent.device)
        self.divergences = wp.zeros(count, dtype=wp.float32, device=agent.device)

    def __enter__(self):
        self.agent._update_graph = None
        self.agent.launch_update = self._launch
        self.agent.optimizer.step = self._step
        return self

    def __exit__(self, *_):
        self.agent.launch_update = self.launch
        self.agent.optimizer.step = self.step
        self.agent._update_graph = None

    def _launch(self):
        self.cursor.zero_()
        self.launch()

    def _step(self, *, lr=None):
        wp.launch(
            _record_ppo_schedule,
            dim=1,
            inputs=[
                self.agent.optimizer._lr,
                self.agent.kl,
                self.agent.config.desired_kl,
                self.cursor,
                self.rates,
                self.decisions,
                self.divergences,
            ],
            device=self.agent.device,
        )
        self.step(lr=lr)

    def snapshot(self):
        return {
            "recorded_minibatches": int(self.cursor.numpy()[0]),
            "learning_rates": self.rates.numpy().tolist(),
            "adaptive_decisions": self.decisions.numpy().tolist(),
            "kl": self.divergences.numpy().tolist(),
        }


def _ppo_parity(initial, eager, repeated, captured, schedules, expected_updates):
    """Bound FP32 state drift using scale and an observed eager repeat baseline."""
    absolute_floor, nominal_relative, hard_relative, repeat_multiplier = 2e-5, 1e-4, 1e-3, 4.0
    controls = {}
    expected = {
        "seed": initial["seed"] + 1,
        "adam_timestep": initial["adam_timestep"] + expected_updates,
    }
    for key, value in expected.items():
        controls[key] = all(np.array_equal(state[key], value) for state in (eager, repeated, captured))
    controls["final_learning_rate"] = all(
        np.array_equal(eager["adam_lr"], state["adam_lr"]) for state in (repeated, captured)
    )
    controls["schedule_finite"] = all(
        np.isfinite(schedule["learning_rates"]).all() and np.isfinite(schedule["kl"]).all() for schedule in schedules
    )
    controls["all_minibatches_observed"] = all(
        schedule["recorded_minibatches"] == expected_updates for schedule in schedules
    )
    for field in ("learning_rates", "adaptive_decisions"):
        controls[field] = all(schedules[0][field] == schedule[field] for schedule in schedules[1:])
    controls["finite_all_state_arrays"] = all(
        np.isfinite(value).all() for state in (initial, eager, repeated, captured) for value in state.values()
    )
    arrays, failures = {}, []
    for key in eager:
        if not (key.startswith("parameter_") or key.startswith(("adam_m1_", "adam_m2_"))):
            continue
        reference = eager[key].astype(np.float64)
        baseline = repeated[key].astype(np.float64) - reference
        difference = captured[key].astype(np.float64) - reference
        finite = bool(np.isfinite(reference).all() and np.isfinite(baseline).all() and np.isfinite(difference).all())
        reference_max = float(np.max(np.abs(reference))) if finite else 0.0
        reference_rms = float(np.sqrt(np.mean(reference**2))) if finite else 0.0
        metrics = {}
        accepted = finite
        for name, residual, scale in (
            ("max_abs", lambda value: np.max(np.abs(value)), reference_max),
            ("rms_abs", lambda value: np.sqrt(np.mean(value**2)), reference_rms),
        ):
            baseline_value = float(residual(baseline)) if finite else None
            captured_value = float(residual(difference)) if finite else None
            nominal = absolute_floor + nominal_relative * scale
            hard = absolute_floor + hard_relative * scale
            limit = min(hard, max(nominal, repeat_multiplier * baseline_value)) if finite else nominal
            passed = finite and baseline_value <= hard and captured_value <= limit
            accepted = accepted and passed
            metrics[name] = {
                "eager_repeat": baseline_value,
                "eager_captured": captured_value,
                "reference_scale": scale,
                "nominal_limit": nominal,
                "hard_limit": hard,
                "capture_limit": limit,
                "accepted": bool(passed),
            }
        arrays[key] = {"finite": finite, "accepted": bool(accepted), **metrics}
        if not accepted:
            failures.append(key)
    failures.extend(f"control:{key}" for key, passed in controls.items() if not passed)
    return {
        "accepted": not failures,
        "hard_checks_passed": all(controls.values()),
        "float_array_budgets_passed": all(item["accepted"] for item in arrays.values()),
        "criterion": {
            "absolute_floor": absolute_floor,
            "nominal_relative": nominal_relative,
            "hard_relative": hard_relative,
            "eager_repeat_multiplier": repeat_multiplier,
            "description": (
                "Per-array max and RMS residuals: nominal absolute+relative budget, or up to four times "
                "same-device eager-repeat residual, bounded by the hard absolute+relative budget. "
                "The eager baseline must itself fit the hard budget. RNG/update counters and all scheduler "
                "learning rates/branch decisions must match exactly. This is FP32 numerical agreement, not bit identity."
            ),
        },
        "controls": controls,
        "control_values": {
            key: {
                label: state[key].tolist()
                for label, state in zip(
                    ("initial", "eager", "eager_repeat", "captured"), (initial, eager, repeated, captured), strict=True
                )
            }
            for key in ("seed", "adam_timestep", "adam_lr")
        },
        "schedules": {key: value for key, value in zip(("eager", "eager_repeat", "captured"), schedules, strict=True)},
        "arrays": arrays,
        "offending_keys": failures,
    }


def _write_result(path, result):
    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, list):
            return [finite_json(item) for item in value]
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Failed finite checks must remain serializable. Native NPZ snapshots retain
    # original values; JSON represents nonfinite diagnostic scalars as null.
    temporary.write_text(json.dumps(finite_json(result), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def dense_timing(layer, source, cotangent, calls=50):
    def update():
        for parameter in layer.parameters():
            parameter.grad.zero_()
        source.grad.zero_()
        with wp.Tape() as tape:
            output = layer(source)
        output.grad.zero_()
        tape.backward(grads={output: cotangent})

    update()
    wp.synchronize()
    with wp.ScopedCapture() as captured:
        update()
    for _ in range(5):
        wp.capture_launch(captured.graph)
    wp.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        wp.capture_launch(captured.graph)
    wp.synchronize()
    return (time.perf_counter() - start) / calls


def dense_diagnostics():
    rng = np.random.default_rng(31)
    records = []
    # Actual PPO/Flash shapes, including irregular observations and output heads.
    for batch, inputs, outputs, bias in (
        (1024, 123, 256, True),
        (6144, 128, 37, True),
        (2048, 128, 512, False),
        (4096, 256, 1024, False),
        (2048, 256, 101, False),
    ):
        x = rng.normal(0, 0.2, (batch, inputs)).astype(np.float32)
        weight = rng.normal(0, 0.1, (outputs, inputs)).astype(np.float32)
        cotangent = rng.normal(0, 0.1, (batch, outputs)).astype(np.float32)
        layer = CudaLinear(inputs, outputs, bias=bias, initialize_parameters=False)
        layer.weight.data.assign(weight)
        b = rng.normal(0, 0.1, (outputs, 1)).astype(np.float32) if bias else None
        if bias:
            layer.bias.data.assign(b)
        source = wp.array(x, dtype=wp.float32, requires_grad=True)
        layer.prepare(batch)
        with wp.Tape() as tape:
            result = layer(source)
        forward = result.numpy().copy()
        tape.backward(grads={result: wp.array(cotangent, dtype=wp.float32)})
        x64, w64, dy64 = x.astype(np.float64), weight.astype(np.float64), cotangent.astype(np.float64)
        expected = x64 @ w64.T
        if bias:
            expected += b[:, 0].astype(np.float64)
        comparisons = {
            "forward": error(forward, expected),
            "input_gradient": error(source.grad.numpy(), dy64 @ w64),
            "weight_gradient": error(layer.weight.data.grad.numpy(), dy64.T @ x64),
        }
        if bias:
            comparisons["bias_gradient"] = error(layer.bias.data.grad.numpy(), dy64.sum(0)[:, None])
        passed = all(
            item["finite"] and item["relative_rms"] < 2e-5 and item["max_abs"] < 1e-4 + 2e-5 * item["reference_max_abs"]
            for item in comparisons.values()
        )
        if not passed:
            raise RuntimeError(f"Dense numerical diagnostic failed: {comparisons}")
        adjoint = wp.array(cotangent, dtype=wp.float32)
        timings = {"cuda": dense_timing(layer, source, adjoint)}
        # The frozen online critic must propagate actions without accumulating
        # weight gradients; its original gradient arrays must stay owned/live.
        parameter_grads = [(parameter, parameter.grad) for parameter in layer.parameters()]
        for parameter, _ in parameter_grads:
            parameter.grad = None
        source.grad.zero_()
        with wp.Tape() as frozen_tape:
            frozen_output = layer(source)
        frozen_output.grad.zero_()
        frozen_tape.backward(grads={frozen_output: adjoint})
        frozen_error = error(source.grad.numpy(), dy64 @ w64)
        for parameter, gradient in parameter_grads:
            parameter.grad = gradient
        if not frozen_error["finite"] or frozen_error["relative_rms"] > 2e-5:
            raise RuntimeError(f"Frozen critic input gradient failed: {frozen_error}")
        # Every native dense backward adds to existing adjoints.
        for parameter in layer.parameters():
            parameter.grad.fill_(0.125)
        source.grad.fill_(0.125)
        with wp.Tape() as accumulate_tape:
            accumulation_output = layer(source)
        accumulation_output.grad.zero_()
        accumulate_tape.backward(grads={accumulation_output: adjoint})
        accumulation_error = error(layer.weight.data.grad.numpy(), dy64.T @ x64 + 0.125)
        if not accumulation_error["finite"] or accumulation_error["relative_rms"] > 2e-5:
            raise RuntimeError(f"Gradient accumulation failed: {accumulation_error}")
        reference_layer = TiledLinear(inputs, outputs, bias=bias, initialize_parameters=False)
        reference_layer.weight.data.assign(weight)
        if bias:
            reference_layer.bias.data.assign(b)
        timings["warp_tiled"] = dense_timing(reference_layer, source, adjoint)
        records.append(
            {
                "shape": [batch, inputs, outputs],
                "bias": bias,
                "comparisons": comparisons,
                "frozen_input_gradient": frozen_error,
                "accumulated_weight_gradient": accumulation_error,
                "captured_forward_backward_seconds": timings,
                "timing_scope": "Fixed-weight forward/backward plus gradient clearing; same data, sequential same GPU.",
            }
        )
    return records


def adam_diagnostic():
    rng = np.random.default_rng(17)
    initial = [rng.normal(size=size).astype(np.float32) for size in (1, 37, 257, 4097, 16389)]
    parameters = [wp.array(value, dtype=wp.float32, requires_grad=True) for value in initial]
    optimizer = CudaAdam(parameters, lr=3e-4, disable_graph=True)
    optimizer.prepare()
    reference = [value.astype(np.float64) for value in initial]
    m1 = [np.zeros_like(value) for value in reference]
    m2 = [np.zeros_like(value) for value in reference]
    for step in range(1, 11):
        gradients = [rng.normal(0, 0.05, value.shape).astype(np.float32) for value in initial]
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad.assign(gradient)
        optimizer.step()
        for index, gradient in enumerate(gradients):
            m1[index] = 0.9 * m1[index] + 0.1 * gradient.astype(np.float64)
            m2[index] = 0.999 * m2[index] + 0.001 * gradient.astype(np.float64) ** 2
            reference[index] -= 3e-4 * (m1[index] / (1 - 0.9**step)) / (np.sqrt(m2[index] / (1 - 0.999**step)) + 1e-8)
    comparisons = [
        {
            "parameters": error(parameter.numpy(), value),
            "m1": error(moment1.numpy(), expected1),
            "m2": error(moment2.numpy(), expected2),
        }
        for parameter, value, moment1, moment2, expected1, expected2 in zip(
            parameters, reference, optimizer._m1, optimizer._m2, m1, m2, strict=True
        )
    ]
    counter = float(optimizer._timestep.numpy()[0])
    if counter != 10 or any(
        not item["finite"] or item["max_abs"] > 2e-5 for record in comparisons for item in record.values()
    ):
        raise RuntimeError(f"Adam numerical diagnostic failed: {comparisons}, step={counter}")
    return {"tensors": comparisons, "optimizer_steps": counter}


def ppo_measurements(warmup, updates, checkpoint_dir, progress, *, strict_ppo_parity=False):
    cfg = PPOConfig(
        hidden_dims=(256, 128, 128),
        activation="elu",
        std_type="scalar",
        epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        desired_kl=0.01,
        value_coefficient=1.0,
        value_loss_scale=1.0,
        entropy_coefficient=0.008,
        advantage_sample_std=True,
        timeout_bootstrap="current",
        separate_grad_clipping=True,
        optimized_linear_backward=True,
    )
    rng = np.random.default_rng(29)
    observations = rng.normal(0, 0.2, (24576, 123)).astype(np.float32)
    data = None
    initial = None
    records = []
    reference_eager = None
    for backend in ("warp", "cuda"):
        start = time.perf_counter()
        agent = WarpPPO(123, 37, 1024, 24, replace(cfg, dense_backend=backend))
        if initial is None:
            initial = agent.state_dict()
        else:
            agent.load_state_dict(initial)
        if data is None:
            source = wp.array(observations, dtype=wp.float32)
            means = agent.actor(source).numpy()
            values = agent.critic(source).numpy()[:, 0]
            actions = means + rng.normal(size=means.shape).astype(np.float32)
            data = {
                "observations": observations,
                "means": means,
                "values": values,
                "actions": actions,
                "log_probs": (-0.5 * ((actions - means) ** 2 + math.log(2 * math.pi))).sum(1),
                "stds": np.ones_like(means),
                "rewards": rng.normal(0.2, 0.05, 24576).astype(np.float32),
                "next_values": rng.normal(0, 0.2, 24576).astype(np.float32),
                "terminated": np.zeros(24576, np.int32),
                "truncated": np.zeros(24576, np.int32),
            }
        for key, value in data.items():
            getattr(agent, key).assign(value)
        agent.warmup()
        with _PPOScheduleTrace(agent) as trace:
            # Build the observed graph first. Restore exactly the same public
            # model/optimizer/RNG state before each measured diagnostic update.
            agent.update(capture=True)
            states, schedules = [], []
            for captured_mode in (False, False, True):
                agent.load_state_dict(initial)
                agent.update(capture=captured_mode)
                states.append(agent.state_dict())
                schedules.append(trace.snapshot())
            eager, repeated, captured = states
        expected_updates = cfg.epochs * cfg.num_mini_batches
        parity = _ppo_parity(initial, eager, repeated, captured, schedules, expected_updates)
        parity["dense_backend"] = backend
        parity["strict_ppo_parity"] = strict_ppo_parity
        parity["policy"] = (
            "Exact controls and finite state are mandatory; floating-array budgets are mandatory only with "
            "--strict-ppo-parity. Failed budgets retain accepted=false and do not establish numerical equivalence."
        )
        diagnostics_dir = checkpoint_dir.parent / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        native_snapshot = diagnostics_dir / f"{backend}-ppo-parity.npz"
        np.savez_compressed(
            native_snapshot,
            **{
                f"{label}.{key}": value
                for label, state in zip(
                    ("initial", "eager", "eager_repeat", "captured"), (initial, *states), strict=True
                )
                for key, value in state.items()
            },
        )
        parity["native_state_snapshot"] = str(native_snapshot)
        _write_result(diagnostics_dir / f"{backend}-ppo-parity.json", parity)
        progress("parity", parity)
        if not parity["hard_checks_passed"]:
            failed_controls = [key for key, passed in parity["controls"].items() if not passed]
            raise RuntimeError(f"{backend} PPO mandatory controls/finite checks failed: {failed_controls}")
        if strict_ppo_parity and not parity["accepted"]:
            raise RuntimeError(f"{backend} PPO numerical parity failed: {parity['offending_keys']}")
        capture_error = max(item["max_abs"]["eager_captured"] for item in parity["arrays"].values())
        repeat_error = max(item["max_abs"]["eager_repeat"] for item in parity["arrays"].values())
        if reference_eager is None:
            reference_eager = eager
        one_update_difference = {
            "parameters_max_abs": max(
                float(np.max(np.abs(eager[key] - reference_eager[key])))
                for key in eager
                if key.startswith("parameter_")
            ),
            "moments_max_abs": max(
                float(np.max(np.abs(eager[key] - reference_eager[key])))
                for key in eager
                if key.startswith(("adam_m1_", "adam_m2_"))
            ),
        }
        # Remove all observer kernels from the timed graph. Existing native
        # plans/kernels are warm; the clean graph is rebuilt and state restored.
        agent.load_state_dict(initial)
        agent.update(capture=True)
        agent.load_state_dict(initial)
        for _ in range(warmup):
            agent.update(capture=True)
        wp.synchronize()
        preparation = time.perf_counter() - start
        before = float(agent.optimizer._timestep.numpy()[0])
        start = time.perf_counter()
        for _ in range(updates):
            agent.update(capture=True)
        wp.synchronize()
        seconds = time.perf_counter() - start
        state = agent.state_dict()
        actual = float(agent.optimizer._timestep.numpy()[0]) - before
        attempted = updates * cfg.epochs * cfg.num_mini_batches
        finite = all(np.isfinite(value).all() for value in state.values())
        changed = any(
            not np.array_equal(initial[key], value) for key, value in state.items() if key.startswith("parameter_")
        )
        if not finite or not changed or actual != attempted:
            raise RuntimeError(f"{backend}: finite={finite}, changed={changed}, updates={actual}/{attempted}")
        checkpoint = checkpoint_dir / f"{backend}-ppo.npz"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        agent.save(checkpoint)
        agent.parameters[0].zero_()
        agent.load(checkpoint)
        restored = agent.state_dict()
        if any(not np.array_equal(state[key], restored[key]) for key in state):
            raise RuntimeError(f"{backend} checkpoint restore changed native state")
        agent.update(capture=True)
        if not all(np.isfinite(value).all() for value in agent.state_dict().values()):
            raise RuntimeError(f"{backend} checkpoint graph replay produced nonfinite state")
        records.append(
            {
                "dense_backend": backend,
                "precision": "fp32",
                "captured": True,
                "preparation_seconds": preparation,
                "preparation_scope": (
                    "Construction, fixture creation/assignment (shared fixtures created for the first backend only), "
                    "warmup, eager-repeat/capture numerical diagnostics, diagnostic JSON/NPZ serialization and clean "
                    "graph setup. This is not pure JIT or independently cold startup time. Final checkpoint I/O and "
                    "post-load replay occur after update timing and are excluded from preparation."
                ),
                "update_seconds_per_call": seconds / updates,
                "attempted_optimizer_updates": attempted,
                "actual_optimizer_updates": actual,
                "finite": finite,
                "updated_weights": changed,
                "eager_capture_max_abs_error": capture_error,
                "eager_repeat_max_abs_error": repeat_error,
                "ppo_parity_accepted": parity["accepted"],
                "ppo_parity_hard_checks_passed": parity["hard_checks_passed"],
                "strict_ppo_parity": strict_ppo_parity,
                "parity_diagnostic": str(diagnostics_dir / f"{backend}-ppo-parity.json"),
                "one_update_difference_from_warp": one_update_difference,
                "checkpoint_restored": True,
                "post_load_captured_update": True,
                "health": agent.health_metrics(),
            }
        )
        progress("ppo", records[-1])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument(
        "--strict-ppo-parity",
        action="store_true",
        help="Fail on twenty-minibatch PPO float-array agreement budgets; otherwise report those differences diagnostically.",
    )
    args = parser.parse_args()
    if min(args.warmup, args.updates) < 1:
        parser.error("Warmup and update counts must be positive.")
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        result = {
            "scope": "Synthetic FP32 kernel and PPO diagnostics; no simulator, MDP, or policy-quality comparison.",
            "warp": wp.__version__,
            "warp_nn": warp_nn.__version__,
            "numpy": np.__version__,
            "gpu": wp.get_device("cuda:0").name,
            "warmup": args.warmup,
            "updates": args.updates,
            "strict_ppo_parity": args.strict_ppo_parity,
            "status": "RUNNING",
            "ppo": [],
            "ppo_parity_checks": [],
            "timing": "Sequential same-GPU synchronized wall timings; preparation excluded from warmed updates.",
            "limits": (
                "One shape set/process; cache reuse and order effects are not controlled; no speed superiority inference. "
                "Long PPO floating-state budgets are diagnostic unless strict mode is selected; completed timing with "
                "numerical differences does not establish numerical equivalence."
            ),
        }
        _write_result(args.output, result)

        def progress(kind, record):
            result["ppo_parity_checks" if kind == "parity" else "ppo"].append(record)
            _write_result(args.output, result)

        try:
            result["dense"] = dense_diagnostics()
            _write_result(args.output, result)
            result["adam"] = adam_diagnostic()
            _write_result(args.output, result)
            ppo_measurements(
                args.warmup,
                args.updates,
                args.output.parent / "checkpoints",
                progress,
                strict_ppo_parity=args.strict_ppo_parity,
            )
            result["status"] = (
                "PASS"
                if all(item["accepted"] for item in result["ppo_parity_checks"])
                else "COMPLETED_WITH_NUMERICAL_DIFFERENCES"
            )
        except Exception as exception:
            result["status"] = "FAILED"
            result["failure"] = {"type": type(exception).__name__, "message": str(exception)}
            _write_result(args.output, result)
            raise
        _write_result(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
