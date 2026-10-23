# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Standalone CUDA FlashSAC math/control diagnostic; no simulator or speed claim.

FP16 denotes rounded dense operands with FP32 accumulation, activations,
outputs, master parameters and Adam state. Primitive references are strict.
Complete learner floating-state differences are descriptive; control state,
finite updated weights, optimizer progress, checkpoints and loose loss sanity
remain mandatory. Each stage persists its raw result before the next begins.
"""

import argparse
import json
import os
import time
import traceback
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import warp as wp
from benchmark_official_flashsac import write_json


def snapshot(arrays):
    return {name: value.numpy().copy() for name, value in arrays.items()}


def restore(arrays, state):
    for name, array in arrays.items():
        array.assign(state[name])


def difference(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    same_shape = actual.shape == reference.shape
    residual = actual.astype(np.float64) - reference.astype(np.float64) if finite and same_shape else None
    return {
        "shape": list(actual.shape),
        "reference_shape": list(reference.shape),
        "finite": finite,
        "max_abs": float(np.abs(residual).max(initial=0.0)) if residual is not None else None,
        "rms_abs": float(np.sqrt(np.mean(residual**2))) if residual is not None and residual.size else 0.0,
        "reference_max_abs": float(np.abs(reference.astype(np.float64)).max(initial=0.0)) if finite else None,
    }


class Report:
    def __init__(self, output, seed):
        self.output = output
        self.artifacts = output.with_name(output.stem + "-artifacts")
        self.artifacts.mkdir(parents=True, exist_ok=False)
        self.data = {
            "scope": "Standalone primitive math, capture/control and fixed-data FlashSAC updates; no learning-quality claim",
            "seed": seed,
            "source_commit": os.environ.get("SOURCE_COMMIT"),
            "precision": "FP16 dense operands; FP32 accumulation, outputs, activations, master parameters and Adam",
            "stages": {},
        }
        self.flush()

    def flush(self):
        write_json(self.output, self.data)

    @contextmanager
    def stage(self, name):
        row = self.data["stages"][name] = {"status": "running", "checks": {}}
        started = time.perf_counter()
        self.current = row
        self.flush()
        try:
            yield row
        except BaseException:
            row.update(status="failed", traceback=traceback.format_exc())
            raise
        else:
            row["status"] = "passed"
        finally:
            row["diagnostic_wall_seconds"] = time.perf_counter() - started
            self.flush()

    def check(self, name, passed, **details):
        self.current["checks"][name] = {"passed": bool(passed), **details}
        if not passed:
            self.flush()
            raise AssertionError(name)

    def arrays(self, name, actual, reference, *, atol=2e-5, rtol=2e-5, exact=False):
        metrics = difference(actual, reference)
        limit = atol + rtol * metrics["reference_max_abs"] if metrics["finite"] else None
        accepted = (
            np.array_equal(actual, reference)
            if exact
            else metrics["max_abs"] is not None and metrics["max_abs"] <= limit
        )
        self.check(name, metrics["finite"] and accepted, exact=exact, atol=atol, rtol=rtol, limit=limit, **metrics)


def backward(module, source, cotangent):
    source.grad.zero_()
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter.grad.zero_()
    with wp.Tape() as tape:
        output = module(source)
    output.grad.zero_()
    tape.backward(grads={output: cotangent})
    return output


def dense(report, rng):
    from robolearn.cuda import CudaLinear

    x = rng.normal(0, 0.3, (17, 19)).astype(np.float32)
    w = rng.normal(0, 0.2, (23, 19)).astype(np.float32)
    b = rng.normal(0, 0.1, 23).astype(np.float32)
    dy = rng.normal(0, 0.2, (17, 23)).astype(np.float32)
    source = wp.array(x, requires_grad=True)
    cotangent = wp.array(dy)
    layer = CudaLinear(19, 23, initialize_parameters=False, compute_dtype="float16")
    layer.weight.data.assign(w)
    layer.bias.data.assign(b[:, None])
    layer.prepare(17).prepare(17, "float32")
    x16, w16, dy16 = [value.astype(np.float16).astype(np.float64) for value in (x, w, dy)]
    references = {"output": x16 @ w16.T + b, "dx": dy16 @ w16, "dw": dy16.T @ x16, "db": dy.sum(0)[:, None]}
    xf, wf, dyf = [value.astype(np.float32) for value in (x16, w16, dy16)]
    fp32_references = {"output": xf @ wf.T + b, "dx": dyf @ wf, "dw": dyf.T @ xf, "db": dy.sum(0)[:, None]}

    def run():
        return backward(layer, source, cotangent)

    def state(output):
        return {
            "output": output.numpy().copy(),
            "dx": source.grad.numpy().copy(),
            "dw": layer.weight.data.grad.numpy().copy(),
            "db": layer.bias.data.grad.numpy().copy(),
        }

    output = run()
    eager = state(output)
    pointers = (source.ptr, layer.weight.data.ptr, layer.bias.data.ptr)
    with wp.ScopedCapture() as capture:
        output = run()
    wp.capture_launch(capture.graph)
    captured = state(output)
    for name in references:
        report.arrays(f"mixed/{name}", eager[name], references[name], atol=2e-5, rtol=2e-5)
        report.arrays(f"mixed_fp32_reference/{name}", eager[name], fp32_references[name], atol=2e-5, rtol=2e-5)
        report.arrays(f"capture/{name}", captured[name], eager[name], atol=2e-6, rtol=2e-6)
    with layer.precision("float32"):
        result = layer(source).numpy().copy()
    report.arrays("fp32_scope", result, x.astype(np.float64) @ w.astype(np.float64).T + b)
    report.check("precision_restored", layer.compute_dtype == "float16")
    report.check("master_storage_unchanged", pointers == (source.ptr, layer.weight.data.ptr, layer.bias.data.ptr))
    gradients = [parameter.grad for parameter in layer.parameters()]
    try:
        for parameter in layer.parameters():
            parameter.grad = None
        backward(layer, source, cotangent)
        report.arrays("frozen_parameters_dx", source.grad.numpy(), references["dx"])
    finally:
        for parameter, gradient in zip(layer.parameters(), gradients, strict=True):
            parameter.grad = gradient
    shadow = layer.prepare_weight_shadow()
    layer.prepare(9)
    report.check("shared_shadow_across_shapes", shadow.ptr == layer.weight_shadow.ptr)
    layer.set_weight_shadow_managed(True)
    updated_weight = w + np.float32(0.03125)
    layer.weight.data.assign(updated_weight)
    layer.refresh_weight_shadow()
    expected_output = x16 @ updated_weight.astype(np.float16).astype(np.float64).T + b
    report.arrays("managed_shadow/updated_forward", layer(source).numpy(), expected_output)
    with wp.ScopedCapture() as refreshed:
        layer.refresh_weight_shadow()
        refreshed_output = layer(source)
    wp.capture_launch(refreshed.graph)
    report.arrays("managed_shadow/capture", refreshed_output.numpy(), expected_output)
    report.arrays("managed_shadow/rounding", shadow.numpy(), updated_weight.astype(np.float16), exact=True)
    np.savez(
        report.artifacts / "dense.npz",
        **{f"eager_{k}": v for k, v in eager.items()},
        **{f"fp32_reference_{k}": v for k, v in fp32_references.items()},
        **references,
    )


def optimizer_arrays(optimizer, parameter):
    return {
        "parameter": parameter,
        "m1": optimizer._m1[0],
        "m2": optimizer._m2[0],
        "timestep": optimizer._timestep,
        "lr": optimizer._lr,
        **optimizer.state_arrays(),
    }


def scaler(report, rng):
    from robolearn.cuda import CudaAdam

    initial_parameter = rng.normal(0, 0.2, (3, 5)).astype(np.float32)
    gradient = np.linspace(0.01, 0.3, 15, dtype=np.float32).reshape(3, 5)
    parameter = wp.array(initial_parameter, requires_grad=True)
    optimizer = CudaAdam(
        [parameter],
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        disable_graph=True,
        dynamic_loss_scale=True,
        initial_loss_scale=65536.0,
        growth_interval=2,
    ).prepare()
    arrays = optimizer_arrays(optimizer, parameter)
    initial = snapshot(arrays)
    # Warm native launches outside capture, then restore every persistent state array.
    parameter.grad.assign(gradient * 65536)
    optimizer.step()
    restore(arrays, initial)
    with wp.ScopedCapture() as capture:
        optimizer.step()

    def sequence(captured):
        states, gradients = [], []
        for index in range(4):
            scaled = gradient * float(optimizer.loss_scale.numpy()[0])
            if index == 1:
                scaled[0, 0] = np.inf
            parameter.grad.assign(scaled)
            wp.capture_launch(capture.graph) if captured else optimizer.step()
            states.append(snapshot(arrays))
            gradients.append(parameter.grad.numpy().copy())
        return states, gradients

    eager, unscaled = sequence(False)
    g = gradient.astype(np.float64)
    report.arrays("finite_gradient_unscaled", unscaled[0], gradient, exact=True)
    report.arrays("finite_unscale", eager[0]["m1"].reshape(g.shape), 0.1 * g, atol=1e-8, rtol=2e-6)
    report.arrays("finite_second_moment", eager[0]["m2"].reshape(g.shape), 0.001 * g * g, atol=1e-10, rtol=2e-6)
    report.arrays("finite_parameter_update", eager[0]["parameter"], initial_parameter - 3e-4 * g / (np.abs(g) + 1e-8))
    for key in ("parameter", "m1", "m2", "timestep"):
        report.arrays(f"overflow_preserves/{key}", eager[1][key], eager[0][key], exact=True)
    report.arrays("three_updates_m1", eager[3]["m1"].reshape(g.shape), (1 - 0.9**3) * g, atol=1e-8, rtol=2e-6)
    report.arrays("three_updates_m2", eager[3]["m2"].reshape(g.shape), (1 - 0.999**3) * g * g, atol=1e-10, rtol=2e-6)
    expected = (
        (1, 1, 0, 0, 65536.0, 1),
        (1, 2, 1, 1, 32768.0, 0),
        (2, 3, 1, 0, 32768.0, 1),
        (3, 4, 1, 0, 65536.0, 0),
    )
    for index, (actual, attempts, skips, skipped, scale, growth) in enumerate(expected):
        state = eager[index]
        for key, value in (
            ("timestep", actual),
            ("attempted_steps", attempts),
            ("skipped_steps", skips),
            ("step_skipped", skipped),
            ("found_inf", skipped),
            ("loss_scale", scale),
            ("scale_growth_tracker", growth),
        ):
            report.check(
                f"control/{index}/{key}", float(state[key][0]) == value, actual=float(state[key][0]), expected=value
            )
    restore(arrays, initial)
    captured, _ = sequence(True)
    for index in range(4):
        for key in eager[index]:
            report.arrays(f"capture/{index}/{key}", captured[index][key], eager[index][key], exact=True)
    np.savez(
        report.artifacts / "scaler.npz", **{f"step_{i}_{k}": v for i, row in enumerate(eager) for k, v in row.items()}
    )


def projection_ema(report, rng):
    from robolearn.cuda import CudaEMA, CudaProjection

    values = {
        "rows": rng.normal(size=(3, 7)).astype(np.float32),
        "weight": rng.normal(size=7).astype(np.float32),
        "bias": rng.normal(size=7).astype(np.float32),
        "scale": rng.normal(size=7).astype(np.float32),
        "target": rng.normal(size=(3, 7)).astype(np.float32),
    }
    arrays = {name: wp.array(value) for name, value in values.items()}
    row_shadow = wp.empty(arrays["rows"].shape, dtype=wp.float16)
    target_shadow = wp.empty(arrays["target"].shape, dtype=wp.float16)
    projection = CudaProjection(
        rows=[arrays["rows"]],
        row_shadows=[row_shadow],
        affines=[(arrays["weight"], arrays["bias"])],
        scales=[arrays["scale"]],
    ).prepare()
    ema = CudaEMA([arrays["rows"]], [arrays["target"]], 0.125, target_shadows=[target_shadow]).prepare()

    def run():
        projection.step()
        ema.step()

    reference = {name: value.astype(np.float64) for name, value in values.items()}
    reference["rows"] /= np.maximum(np.linalg.norm(reference["rows"], axis=1, keepdims=True), 1e-8)
    factor = np.sqrt(7 / (np.sum(reference["weight"] ** 2 + reference["bias"] ** 2) + 1e-8))
    reference["weight"] *= factor
    reference["bias"] *= factor
    reference["scale"] *= np.sqrt(7 / (np.sum(reference["scale"] ** 2) + 1e-8))
    reference["target"] = 0.875 * reference["target"] + 0.125 * reference["rows"]
    run()
    eager = snapshot(arrays)
    restore(arrays, values)
    with wp.ScopedCapture() as capture:
        run()
    wp.capture_launch(capture.graph)
    captured = snapshot(arrays)
    for name in arrays:
        report.arrays(f"formula/{name}", eager[name], reference[name], atol=2e-6, rtol=2e-6)
        report.arrays(f"capture/{name}", captured[name], eager[name], exact=True)
    report.arrays("projection_shadow", row_shadow.numpy(), captured["rows"].astype(np.float16), exact=True)
    report.arrays("ema_shadow", target_shadow.numpy(), captured["target"].astype(np.float16), exact=True)
    np.savez(report.artifacts / "projection-ema.npz", **eager)


def normalization(report, rng):
    from robolearn.cuda import CudaBatchNorm, CudaRMSNorm

    for kind, relu in (("batch", False), ("batch", True), ("rms", False)):
        label = f"{kind}-relu{int(relu)}"
        x = rng.normal(0, 0.4, (17, 13)).astype(np.float32)
        dy = rng.normal(0, 0.2, x.shape).astype(np.float32)
        weight = rng.uniform(0.5, 1.5, 13).astype(np.float32)
        bias = rng.normal(0, 0.1, 13).astype(np.float32)
        module = CudaBatchNorm(13, eps=1e-5, momentum=0.01, relu=relu) if kind == "batch" else CudaRMSNorm(13, eps=1e-6)
        module.weight.data.assign(weight)
        if kind == "batch":
            module.bias.data.assign(bias)
        initial = snapshot(module.state_dict())
        source, cotangent = wp.array(x, requires_grad=True), wp.array(dy)
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        tw = torch.tensor(weight, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(bias, dtype=torch.float64, requires_grad=True)
        if kind == "batch":
            reference = torch.nn.BatchNorm1d(13, eps=1e-5, momentum=0.01, dtype=torch.float64)
            with torch.no_grad():
                reference.weight.copy_(tw)
                reference.bias.copy_(tb)
            ty = reference(tx)
            if relu:
                ty = torch.relu(ty)
        else:
            ty = tx * torch.rsqrt(tx.square().mean(-1, keepdim=True) + 1e-6) * tw
        ty.backward(torch.tensor(dy, dtype=torch.float64))
        output = backward(module, source, cotangent)
        expected = {
            "output": ty.detach().numpy(),
            "dx": tx.grad.numpy(),
            "dw": reference.weight.grad.numpy() if kind == "batch" else tw.grad.numpy(),
        }
        if kind == "batch":
            expected.update(
                db=reference.bias.grad.numpy(),
                running_mean=reference.running_mean.numpy(),
                running_var=reference.running_var.numpy(),
            )

        def current(output):
            result = {
                "output": output.numpy().copy(),
                "dx": source.grad.numpy().copy(),
                "dw": module.weight.data.grad.numpy().copy(),
            }
            if kind == "batch":
                result.update(
                    db=module.bias.data.grad.numpy().copy(),
                    running_mean=module.running_mean.data.numpy().copy(),
                    running_var=module.running_var.data.numpy().copy(),
                )
            return result

        eager = current(output)
        restore(module.state_dict(), initial)
        with wp.ScopedCapture() as capture:
            backward(module, source, cotangent)
        wp.capture_launch(capture.graph)
        captured = current(output)
        for name in expected:
            report.arrays(f"{label}/torch/{name}", eager[name], expected[name])
            report.arrays(f"{label}/capture/{name}", captured[name], eager[name], atol=2e-6, rtol=2e-6)
        if kind == "batch":
            module.train(False)
            reference.eval()
            tx_eval = torch.tensor(x, dtype=torch.float64, requires_grad=True)
            expected_eval = reference(tx_eval)
            if relu:
                expected_eval = torch.relu(expected_eval)
            expected_eval.backward(torch.tensor(dy, dtype=torch.float64))
            actual_eval = backward(module, source, cotangent)
            report.arrays(f"{label}/eval", actual_eval.numpy(), expected_eval.detach().numpy())
            gradients = [parameter.grad for parameter in module.parameters()]
            try:
                for parameter in module.parameters():
                    parameter.grad = None
                backward(module, source, cotangent)
                report.arrays(f"{label}/frozen_dx", source.grad.numpy(), tx_eval.grad.numpy())
            finally:
                for parameter, gradient in zip(module.parameters(), gradients, strict=True):
                    parameter.grad = gradient
        np.savez(report.artifacts / f"normalization-{label}.npz", **eager)


def categorical(report, rng, bins=101):
    from robolearn.cuda.categorical import CudaCategorical, categorical_target
    from robolearn.warp.flashsac import _categorical_target

    # Irregular dimensions exercise padding and cooperative row reductions.
    batch = 17
    raw = rng.normal(0, 0.4, (batch, bins)).astype(np.float32)
    bias = rng.normal(0, 0.1, bins).astype(np.float32)
    support = np.linspace(-5, 5, bins, dtype=np.float32)
    dlp = rng.normal(0, 0.2, raw.shape).astype(np.float32)
    dq = rng.normal(0, 0.2, batch).astype(np.float32)
    x, b, z = wp.array(raw, requires_grad=True), wp.array(bias, requires_grad=True), wp.array(support)
    seeds = wp.array(dlp), wp.array(dq)
    module = CudaCategorical()
    tx, tb = (
        torch.tensor(raw, dtype=torch.float64, requires_grad=True),
        torch.tensor(bias, dtype=torch.float64, requires_grad=True),
    )
    tlp = torch.log_softmax(tx + tb, dim=-1)
    tq = (tlp.exp() * torch.tensor(support, dtype=torch.float64)).sum(-1)
    (tlp * torch.tensor(dlp, dtype=torch.float64)).sum().add(
        (tq * torch.tensor(dq, dtype=torch.float64)).sum()
    ).backward()
    reference = {
        "log_probs": tlp.detach().numpy(),
        "values": tq.detach().numpy(),
        "dx": tx.grad.numpy(),
        "db": tb.grad.numpy(),
    }

    def run():
        x.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()
        with wp.Tape() as tape:
            values, lp = module(x, b, z)
        wp.copy(lp.grad, seeds[0])
        wp.copy(values.grad, seeds[1])
        tape.backward()
        return values, lp

    def state(outputs):
        values, lp = outputs
        return {
            "log_probs": lp.numpy().copy(),
            "values": values.numpy().copy(),
            "dx": x.grad.numpy().copy(),
            "db": b.grad.numpy().copy(),
        }

    eager = state(run())
    with wp.ScopedCapture() as capture:
        outputs = run()
    wp.capture_launch(capture.graph)
    captured = state(outputs)
    for name in reference:
        report.arrays(f"head/torch/{name}", eager[name], reference[name], atol=3e-5, rtol=3e-5)
        report.arrays(f"head/capture/{name}", captured[name], eager[name], exact=True)
    gradient = b.grad
    try:
        b.grad = None
        run()
        report.arrays("head/frozen_bias_dx", x.grad.numpy(), reference["dx"], atol=3e-5, rtol=3e-5)
    finally:
        b.grad = gradient

    # Match the existing target including ties, boundary clamps and terminations.
    target_logits = rng.normal(size=(2, 2 * batch, bins)).astype(np.float32)
    target_lp = torch.log_softmax(torch.tensor(target_logits, dtype=torch.float64), -1).numpy().astype(np.float32)
    q = (np.exp(target_lp) * support).sum(-1).astype(np.float32)
    q[:, batch + 1] = 0.0
    reward = rng.uniform(-8, 8, batch).astype(np.float32)
    terminated = (np.arange(batch) % 4 == 0).astype(np.float32)
    discount = rng.uniform(0, 1, batch).astype(np.float32)
    next_lp = rng.normal(size=batch).astype(np.float32)
    reward[1], terminated[1], discount[1], next_lp[1] = 0.0, 0.0, 0.9, 0.0
    args = [
        wp.array(value) for value in (q, target_lp, reward, terminated, discount, next_lp, np.array([0.1], np.float32))
    ]
    target, maximum = wp.empty((batch, bins)), wp.empty(1)
    ref_target, ref_maximum = wp.empty_like(target), wp.empty_like(maximum)
    ref_maximum.fill_(-1e30)
    wp.launch(
        _categorical_target,
        dim=batch,
        inputs=[*args, -5.0, 5.0, ref_target, ref_maximum],
    )

    def project():
        maximum.fill_(-1e30)
        categorical_target(*args, -5.0, 5.0, target, maximum)

    project()
    projected = target.numpy().copy()
    report.arrays("target/warp_reference", projected, ref_target.numpy(), atol=3e-6, rtol=3e-6)
    report.arrays("target/maximum_entropy", maximum.numpy(), ref_maximum.numpy(), exact=True)
    report.arrays("target/probability_mass", projected.sum(-1), np.ones(batch), atol=3e-6, rtol=3e-6)
    with wp.ScopedCapture() as capture:
        project()
    wp.capture_launch(capture.graph)
    report.arrays("target/capture", target.numpy(), projected, atol=3e-6, rtol=3e-6)
    np.savez(
        report.artifacts / f"categorical-{bins}.npz",
        **eager,
        **{f"torch_{name}": value for name, value in reference.items()},
        raw=raw,
        bias=bias,
        support=support,
        dlog_probs=dlp,
        dvalues=dq,
        target=projected,
        reference_target=ref_target.numpy(),
        target_q=q,
        target_log_probs=target_lp,
        reward=reward,
        terminated=terminated,
        discount=discount,
        next_log_probs=next_lp,
        alpha=np.array([0.1], np.float32),
    )


def learner_controls(report, label, agent, scheduled):
    for name, count in scheduled.items():
        optimizer = getattr(agent, f"{name}_optimizer")
        actual = int(optimizer._timestep.numpy()[0])
        scaled = getattr(optimizer, "dynamic_loss_scale", False)
        attempts = int(optimizer.attempted_steps.numpy()[0]) if scaled else actual
        skipped = int(optimizer.skipped_steps.numpy()[0]) if scaled else 0
        report.check(f"{label}/{name}/attempts", attempts == count, attempts=attempts, expected=count)
        report.check(f"{label}/{name}/accounting", actual + skipped == attempts, actual=actual, skipped=skipped)
        report.check(f"{label}/{name}/actual_progress", actual > 0, completed_updates=actual)


def flash_recipe(report, rng, *, amp, fused, g1=False, updates=8, batch_size=32):
    from robolearn.flashsac import FlashSACConfig
    from robolearn.warp import WarpFlashSAC

    observation_dim, action_dim, num_envs = (123, 37, 1024) if g1 else (9, 3, 4)
    cfg = FlashSACConfig(
        device=str(wp.get_device()),
        seed=report.data["seed"],
        normalize_reward=False,
        use_amp=amp,
        use_compile=False,
        actor_hidden_dim=128 if g1 else 16,
        critic_hidden_dim=256 if g1 else 32,
        actor_num_blocks=2 if g1 else 1,
        critic_num_blocks=2 if g1 else 1,
        critic_num_bins=101 if g1 else 11,
        sample_batch_size=batch_size,
        buffer_max_length=max(4 * batch_size, 8 * num_envs),
        buffer_min_length=num_envs,
    )
    agents = [
        WarpFlashSAC(observation_dim, action_dim, num_envs, cfg, dense_backend="cuda", fused_updates=fused)
        for _ in range(2)
    ]
    eager, captured = agents
    label = f"{'g1' if g1 else 'small'}-amp{int(amp)}-fused{int(fused)}"
    for _ in range(cfg.n_step + 2):
        observation = wp.array(rng.normal(0, 0.3, (num_envs, observation_dim)).astype(np.float32))
        actions = wp.array(rng.uniform(-0.3, 0.3, (num_envs, action_dim)).astype(np.float32))
        reward = wp.array(rng.uniform(-0.1, 0.1, num_envs).astype(np.float32))
        flags = wp.zeros(num_envs, dtype=wp.int32)
        eager.observe(observation, actions, reward, flags, flags, observation)
    for agent in agents:
        agent.prepare(capture=False)
    initial = eager.state_dict(include_replay=True)
    captured.load_state_dict(initial)
    fixed = {
        "observation": rng.normal(0, 0.3, (batch_size, observation_dim)).astype(np.float32),
        "next_observation": rng.normal(0, 0.3, (batch_size, observation_dim)).astype(np.float32),
        "action": rng.uniform(-0.3, 0.3, (batch_size, action_dim)).astype(np.float32),
        "reward": rng.uniform(-0.1, 0.1, batch_size).astype(np.float32),
        "terminated": (np.arange(batch_size) % 7 == 0).astype(np.float32),
        "truncated": (np.arange(batch_size) % 5 == 0).astype(np.float32),
        "discount": np.full(batch_size, cfg.gamma**cfg.n_step, dtype=np.float32),
    }
    actor_noise = rng.normal(0, 0.3, (2 * batch_size, action_dim)).astype(np.float32)
    next_noise = rng.normal(0, 0.3, (batch_size, action_dim)).astype(np.float32)
    for key, value in fixed.items():
        captured.batch[key].assign(value)
    captured.actor_noise.assign(actor_noise)
    captured.next_noise.assign(next_noise)
    graphs = {}
    for phase in (True, False):
        with wp.ScopedCapture() as capture:
            captured.launch_update(phase, sample=False, randomize_noise=False, normalize_reward=False)
        graphs[phase] = capture.graph
    captured.load_state_dict(initial)

    def loss_sanity(index):
        expected, actual = eager.get_metrics(), captured.get_metrics()
        row = {}
        for name in ("actor/loss", "critic/loss", "temperature/loss"):
            residual = abs(actual[name] - expected[name])
            limit = 1e-3 + 0.05 * abs(expected[name])
            report.check(
                f"{label}/loss/{index}/{name}",
                np.isfinite(residual) and residual <= limit,
                eager=expected[name],
                captured=actual[name],
                max_abs=residual,
                limit=limit,
            )
            row[name] = {"eager": expected[name], "captured": actual[name], "max_abs": residual}
        return row

    phases = [index % cfg.actor_update_period == 0 for index in range(updates)]
    losses = []
    for phase in phases:
        eager.update_from_batch(fixed, actor_noise=actor_noise, next_noise=next_noise, do_actor_update=phase)
        wp.capture_launch(graphs[phase])
        captured._update_step += 1
        losses.append(loss_sanity(len(losses)))
    states = [agent.state_dict(include_replay=True) for agent in agents]
    report.check(f"{label}/state_keys", set(states[0]) == set(states[1]))
    drift = {name: difference(states[1][name], value) for name, value in states[0].items()}
    report.current["floating_state_drift"] = drift
    report.current["losses"] = losses
    for name, value in states[0].items():
        control = value.dtype.kind in "biu" or name.endswith((".timestep", ".lr", ".loss_scale"))
        if control:
            report.arrays(f"{label}/control/{name}", states[1][name], value, exact=True)
    scheduled = {"actor": sum(phases), "temperature": sum(phases), "critic": len(phases)}
    for name, agent, state in zip(("eager", "captured"), agents, states, strict=True):
        report.check(f"{label}/{name}/finite", all(np.isfinite(value).all() for value in state.values()))
        changed = sum(
            not np.array_equal(value, initial[key])
            for key, value in state.items()
            if key.startswith(("actor.", "critic.")) and ".running_" not in key
        )
        report.check(f"{label}/{name}/updated_weights", changed > 0, changed_arrays=changed)
        learner_controls(report, f"{label}/{name}", agent, scheduled)
        checkpoint = report.artifacts / f"{label}-{name}"
        pointers = {key: value.ptr for key, value in agent._state_arrays().items()}
        agent.save(checkpoint, include_replay=True)
        agent.load_state_dict(initial)
        agent.load(checkpoint)
        report.check(
            f"{label}/{name}/checkpoint_storage_retained",
            pointers == {key: value.ptr for key, value in agent._state_arrays().items()},
        )
        restored = agent.state_dict(include_replay=True)
        for key, value in state.items():
            report.arrays(f"{label}/{name}/checkpoint/{key}", restored[key], value, exact=True)
    # Reuse the original graphs after load to check pointer/scaler continuity.
    phase = eager._update_step % cfg.actor_update_period == 0
    report.check(f"{label}/continuation/phase", eager._update_step == captured._update_step == updates)
    eager.update_from_batch(fixed, actor_noise=actor_noise, next_noise=next_noise, do_actor_update=phase)
    wp.capture_launch(graphs[phase])
    captured._update_step += 1
    report.current["post_load_loss"] = loss_sanity("post_load")
    continued = [agent.state_dict(include_replay=True) for agent in agents]
    continued_schedule = {name: count + (1 if name == "critic" else int(phase)) for name, count in scheduled.items()}
    for name, agent, before, after in zip(("eager", "captured"), agents, states, continued, strict=True):
        report.check(f"{label}/{name}/continuation_finite", all(np.isfinite(value).all() for value in after.values()))
        report.check(
            f"{label}/{name}/continuation_counter",
            int(after["update_counters"][0]) == int(before["update_counters"][0]) + 1 == agent._update_step,
        )
        progress = {
            key: int(after[f"optimizer.{key}.timestep"][0] - before[f"optimizer.{key}.timestep"][0])
            for key in scheduled
        }
        report.check(f"{label}/{name}/continuation_optimizer_progress", sum(progress.values()) > 0, updates=progress)
        changed = sum(
            not np.array_equal(value, before[key])
            for key, value in after.items()
            if key.startswith(("actor.", "critic.")) and ".running_" not in key
        )
        report.check(f"{label}/{name}/continuation_updated_weights", changed > 0, changed_arrays=changed)
        learner_controls(report, f"{label}/{name}/continuation", agent, continued_schedule)
    for key, value in continued[0].items():
        if value.dtype.kind in "biu" or key.endswith((".timestep", ".lr", ".loss_scale")):
            report.arrays(f"{label}/continuation_control/{key}", continued[1][key], value, exact=True)
    report.current["post_load_state_drift"] = {
        key: difference(continued[1][key], value) for key, value in continued[0].items()
    }
    np.savez_compressed(
        report.artifacts / f"{label}-fixture.npz", **fixed, actor_noise=actor_noise, next_noise=next_noise
    )
    report.current.update(
        amp=amp,
        fused=fused,
        batch_size=batch_size,
        observation_dim=observation_dim,
        action_dim=action_dim,
        scheduled_updates=scheduled,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-g1", action="store_true")
    parser.add_argument("--g1-updates", type=int, default=8)
    parser.add_argument("--g1-batch-size", type=int, default=2048)
    args = parser.parse_args()
    report = Report(args.output, args.seed)
    try:
        wp.init()
        device = wp.get_device(args.device)
        if not device.is_cuda or args.g1_updates < 2 or args.g1_batch_size < 2:
            raise ValueError("A CUDA device, at least two updates and batch size >=2 are required.")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        report.data["runtime"] = {name: version(name) for name in ("torch", "warp-lang", "warp-nn", "robolearn-rl")}
        rng = np.random.default_rng(args.seed)
        with wp.ScopedDevice(device):
            for name, function in (
                ("dense", dense),
                ("scaler", scaler),
                ("projection-ema", projection_ema),
                ("normalization", normalization),
                ("categorical", categorical),
                ("categorical-wide", lambda report, rng: categorical(report, rng, bins=257)),
            ):
                with report.stage(name):
                    function(report, rng)
            for amp in (False, True):
                for fused in (False, True):
                    with report.stage(f"small-amp{int(amp)}-fused{int(fused)}"):
                        flash_recipe(report, rng, amp=amp, fused=fused)
            if not args.skip_g1:
                with report.stage("g1-shaped-amp-fused"):
                    flash_recipe(
                        report,
                        rng,
                        amp=True,
                        fused=True,
                        g1=True,
                        updates=args.g1_updates,
                        batch_size=args.g1_batch_size,
                    )
        report.data["status"] = "passed"
    except BaseException:
        report.data.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        report.flush()
        print(json.dumps({"status": report.data.get("status"), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
