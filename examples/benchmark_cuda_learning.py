# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Fixed-data CUDA dense diagnostics and PPO update timings; no simulator or policy-quality claim."""

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
        "finite": bool(np.isfinite(value).all()),
    }


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


def ppo_measurements(warmup, updates, checkpoint_dir):
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
        # Capture and execute once, then restore initial model/optimizer/RNG state.
        agent.update(capture=True)
        agent.load_state_dict(initial)
        agent.update(capture=False)
        eager = agent.state_dict()
        agent.load_state_dict(initial)
        agent.update(capture=True)
        captured = agent.state_dict()
        capture_error = max(float(np.max(np.abs(eager[key] - captured[key]))) for key in eager)
        if capture_error > 2e-5:
            raise RuntimeError(f"{backend} eager/captured state differs: {capture_error}")
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
                "update_seconds_per_call": seconds / updates,
                "attempted_optimizer_updates": attempted,
                "actual_optimizer_updates": actual,
                "finite": finite,
                "updated_weights": changed,
                "eager_capture_max_abs_error": capture_error,
                "one_update_difference_from_warp": one_update_difference,
                "checkpoint_restored": True,
                "post_load_captured_update": True,
                "health": agent.health_metrics(),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--updates", type=int, default=50)
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
            "dense": dense_diagnostics(),
            "adam": adam_diagnostic(),
            "ppo": ppo_measurements(args.warmup, args.updates, args.output.parent / "checkpoints"),
            "timing": "Sequential same-GPU synchronized wall timings; preparation excluded from warmed updates.",
            "limits": "One shape set/process; cache reuse and order effects are not controlled; no speed superiority inference.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
