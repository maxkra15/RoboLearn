# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Compare Warp FlashSAC networks with the vendored authors' FP32 reference.

This standalone diagnostic uses common weights, fixed inputs and fixed Gaussian
noise. It checks training/evaluation outputs, batch-normalization state, actor
gradients through frozen critics and categorical critic gradients. It is not a
learning experiment or a timing benchmark. Production actor, temperature,
critic and target updates are also compared, including Adam and CUDA capture.
For production network dimensions:

    python examples/diagnose_warp_flashsac.py --g1 --batch-size 2048 --output parity.json

The reference is Holiday-Robot/FlashSAC revision
87edc9061150ae9e962dd84e6544e27a1554b3ab, preserved in robolearn.flashsac.
Torch is required only by this diagnostic; the Warp backend does not depend on it.
"""

import argparse
import json
import math
import traceback
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import warp as wp

from robolearn.flashsac import updates as torch_updates
from robolearn.flashsac.agent import _init_flashsac_networks, _update_networks
from robolearn.flashsac.config import FlashSACConfig
from robolearn.flashsac.networks import FlashSACActor, FlashSACDoubleCritic
from robolearn.warp._flash_networks import FlashActor, FlashDoubleCritic
from robolearn.warp.flashsac import WarpFlashSAC


@wp.kernel
def _fixed_actions(
    mean: wp.array2d(dtype=wp.float32),
    log_std: wp.array2d(dtype=wp.float32),
    noise: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    log_probs: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    log_prob = float(0.0)
    for j in range(mean.shape[1]):
        raw = mean[i, j] + wp.exp(log_std[i, j]) * noise[i, j]
        actions[i, j] = wp.tanh(raw)
        x = -2.0 * raw
        softplus = x
        if x <= 20.0:
            softplus = wp.log(1.0 + wp.exp(x))
        jacobian = 2.0 * (0.6931471805599453 - raw - softplus)
        log_prob = log_prob - 0.5 * (noise[i, j] * noise[i, j] + 2.0 * log_std[i, j] + 1.8378770664093453) - jacobian
    log_probs[i] = log_prob


@wp.kernel
def _current_actions(all_actions: wp.array2d(dtype=wp.float32), actions: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    actions[i, j] = all_actions[i, j]


@wp.kernel
def _actor_loss(
    log_probs: wp.array(dtype=wp.float32),
    qs: wp.array2d(dtype=wp.float32),
    temperature: float,
    loss: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    q = qs[0, i]
    if qs[1, i] < q:
        q = qs[1, i]
    elif qs[1, i] == q:
        q = 0.5 * (qs[0, i] + qs[1, i])
    wp.atomic_add(loss, 0, (temperature * log_probs[i] - q) / float(qs.shape[1]))


@wp.kernel
def _critic_loss(
    log_probs: wp.array3d(dtype=wp.float32),
    target: wp.array2d(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    q, i, j = wp.tid()
    wp.atomic_add(loss, 0, -target[i, j] * log_probs[q, i, j] / float(2 * target.shape[0]))


def _snapshot(module) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy().copy() for key, value in module.state_dict().items()}


def _critic_source(name: str) -> tuple[int, str]:
    _, index, name = name.split(".", 2)
    name = (
        name.replace(".w.w.weight", ".w.weight")
        .replace(".w1.w.weight", ".w1.weight")
        .replace(".w2.w.weight", ".w2.weight")
    )
    return int(index), name


class Comparison:
    def __init__(self, atol: float, rtol: float):
        self.atol = atol
        self.rtol = rtol
        self.checks = {}
        self.policies = {
            "operators_and_gradients": {"mode": "elementwise", "atol": atol, "rtol": rtol},
            "integer_bool_rng_counters_and_checkpoint": {"mode": "exact"},
            "adam_first_moment": {"mode": "tensor_max_norm", "atol": 1.0e-10, "rtol": 5.0e-4},
            "adam_second_moment": {"mode": "tensor_max_norm", "atol": 1.0e-14, "rtol": 5.0e-4},
            "captured_adam_moments": {"mode": "tensor_max_norm", "rtol": 1.0e-4, "atol": "same moment floors"},
            "captured_other_float_state": {"mode": "tensor_max_norm", "atol": 2.0e-6, "rtol": 1.0e-6},
            "learning_rates": {"mode": "tensor_max_norm", "atol": 1.0e-12, "rtol": 1.0e-6},
            "parameter_update_deltas": {"mode": "tensor_max_norm", "atol": 5.0e-7, "rtol": 0.02},
            "explanation": (
                "Tensor bounds use max(abs(actual-reference)) <= atol + rtol*max(abs(reference)). "
                "Moment floors are separate from weight tolerances, so erasing small moment tensors fails. "
                "Update deltas use shared initial weights, avoiding a large baseline weight hiding a skipped update. "
                "The 2% delta margin accommodates FP32 reduction and near-zero-gradient Adam sensitivity; "
                "the absolute floor accommodates a few FP32 weight rounding units. Integer state is never approximate."
            ),
        }

    def arrays(self, name, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        same_shape = actual.shape == expected.shape
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64)) if same_shape and finite else None
        scale = float(np.abs(expected.astype(np.float64)).max(initial=0.0)) if finite else None
        maximum = float(error.max(initial=0.0)) if error is not None else None
        mode, atol, rtol = "elementwise", self.atol, self.rtol
        if (
            actual.dtype.kind in "biu"
            or expected.dtype.kind in "biu"
            or name.endswith(("/step", ".timestep"))
            or name.endswith((".terminated", ".truncated", ".repetition"))
            or "counters" in name
            or "/rng." in name
            or name.startswith(("checkpoint_roundtrip/", "initial_actor/", "initial_critic/", "frozen_critic/"))
        ):
            mode, atol, rtol = "exact", 0.0, 0.0
        elif name.endswith("/exp_avg_sq") or ".m2." in name:
            mode, atol, rtol = "tensor_max_norm", 1.0e-14, 5.0e-4
        elif name.endswith("/exp_avg") or ".m1." in name:
            mode, atol, rtol = "tensor_max_norm", 1.0e-10, 5.0e-4
        elif "/update_delta/" in name:
            mode, atol, rtol = "tensor_max_norm", 5.0e-7, 0.02
        elif name.endswith(("/lr", ".lr")):
            mode, atol, rtol = "tensor_max_norm", 1.0e-12, 1.0e-6
        elif name.startswith("captured_update_"):
            mode, atol, rtol = "tensor_max_norm", 2.0e-6, 1.0e-6
        if name.startswith("captured_update_") and (".m1." in name or ".m2." in name):
            rtol = 1.0e-4
        allowed = atol + rtol * scale if scale is not None else None
        passed = False
        if same_shape and finite:
            if mode == "exact":
                passed = bool(np.array_equal(actual, expected))
            elif mode == "tensor_max_norm":
                passed = maximum <= allowed
            else:
                passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
        self.checks[name] = {
            "passed": passed,
            "comparison_mode": mode,
            "atol": atol,
            "rtol": rtol,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "finite": finite,
            "max_absolute_error": maximum,
            "reference_max_absolute_value": scale,
            "tensor_relative_max_error": maximum / scale if maximum is not None and scale else None,
            "allowed_tensor_max_absolute_error": allowed if mode == "tensor_max_norm" else None,
        }

    def condition(self, name: str, passed: bool):
        self.checks[name] = {"passed": bool(passed)}

    @property
    def passed(self):
        return all(check["passed"] for check in self.checks.values())


def _compare_state(comparison, label, warp_model, torch_model, critic=False, buffers_only=False):
    reference = _snapshot(torch_model)
    for name, array in warp_model.state_dict().items():
        if buffers_only and not ("running_mean" in name or "running_var" in name):
            continue
        if critic:
            index, source = _critic_source(name)
            expected = reference[source].reshape(-1) if source == "predictor.bin_values" else reference[source][index]
        else:
            expected = reference[name]
        comparison.arrays(f"{label}/{name}", array.numpy(), expected)


def _compare_gradients(comparison, label, warp_model, torch_model, critic=False):
    reference = dict(torch_model.named_parameters())
    for name, parameter in warp_model.named_parameters():
        if critic:
            index, source = _critic_source(name)
            expected = reference[source].grad
            expected = expected[index] if expected is not None else None
        else:
            expected = reference[name].grad
        actual = parameter.data.grad
        comparison.condition(f"{label}/{name}/gradient_present", actual is not None and expected is not None)
        if actual is not None and expected is not None:
            comparison.arrays(f"{label}/{name}", actual.numpy(), expected.detach().cpu().numpy())


def _compare_update_deltas(comparison, label, warp_model, torch_model, initial, critic=False):
    reference = _snapshot(torch_model)
    for name, parameter in warp_model.named_parameters():
        if critic:
            index, source = _critic_source(name)
            start, expected = initial[source][index], reference[source][index]
        else:
            start, expected = initial[name], reference[name]
        start = start.astype(np.float64)
        actual_delta = parameter.data.numpy().astype(np.float64) - start
        reference_delta = expected.astype(np.float64) - start
        comparison.arrays(f"{label}/update_delta/{name}", actual_delta, reference_delta)


def _compare_optimizer(comparison, label, warp_model, warp_optimizer, reference, critic=False):
    torch_parameters = dict(reference.network.named_parameters())
    for i, (name, _) in enumerate(warp_model.named_parameters()):
        if critic:
            index, source = _critic_source(name)
        else:
            index, source = None, name
        state = reference.optimizer.state[torch_parameters[source]]
        for moment, actual in (("exp_avg", warp_optimizer._m1[i]), ("exp_avg_sq", warp_optimizer._m2[i])):
            expected = state[moment]
            if index is not None:
                expected = expected[index]
            comparison.arrays(
                f"{label}/{name}/{moment}", actual.numpy().reshape(expected.shape), expected.cpu().numpy()
            )
        comparison.arrays(
            f"{label}/{name}/step", warp_optimizer._timestep.numpy(), state["step"].cpu().numpy().reshape(1)
        )
    comparison.arrays(label + "/lr", warp_optimizer._lr.numpy(), np.array([reference.optimizer.param_groups[0]["lr"]]))


def _full_updates(args, comparison, actor_initial, critic_initial, obs_np, act_np, noise_np):
    """Use both implementations' production updates, with prescribed randomness."""
    b = args.batch_size
    device = torch.device(args.device)
    cfg = FlashSACConfig(
        device=args.device,
        seed=args.seed,
        normalize_reward=False,
        use_amp=False,
        use_compile=False,
        buffer_max_length=max(4 * b, 16),
        buffer_min_length=2,
        sample_batch_size=b,
        actor_num_blocks=args.blocks,
        actor_hidden_dim=args.actor_hidden_dim,
        critic_num_blocks=args.blocks,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_bins=args.bins,
    )
    cfg = replace(
        cfg, temp_target_entropy=0.5 * args.action_dim * math.log(2.0 * math.pi * math.e * cfg.temp_target_sigma**2)
    )
    actor, critic, target, temperature = _init_flashsac_networks(
        args.observation_dim, args.observation_dim, args.action_dim, cfg, device
    )
    for network, values in ((actor, actor_initial), (critic, critic_initial), (target, critic_initial)):
        network.network.load_state_dict({key: torch.tensor(value, device=device) for key, value in values.items()})
    eager = WarpFlashSAC(
        args.observation_dim, args.action_dim, 2, cfg, optimized_linear_backward=args.optimized_linear_backward
    )
    for network, values in (
        (eager.actor, actor_initial),
        (eager.critic, critic_initial),
        (eager.target_critic, critic_initial),
    ):
        network.load_author_state_dict(values)
    eager.prepare(capture=False)
    initial = eager.state_dict(include_replay=False)
    fixed = {
        "observation": obs_np[:b],
        "next_observation": obs_np[b:],
        "action": act_np[:b],
        "reward": np.linspace(-0.3, 0.7, b, dtype=np.float32),
        "terminated": (np.arange(b) % 7 == 0).astype(np.float32),
        "truncated": (np.arange(b) % 5 == 0).astype(np.float32),
        "discount": np.power(cfg.gamma, 1 + np.arange(b) % 3).astype(np.float32),
    }
    torch_batch = {key: torch.tensor(value, device=device) for key, value in fixed.items()}
    torch_batch["actor_observation"] = torch_batch["observation"]
    torch_batch["actor_next_observation"] = torch_batch["next_observation"]
    expected_states = []
    for phase_index, actor_phase in enumerate((True, False)):
        queue = [torch.tensor(noise_np, device=device)] if actor_phase else []
        queue.append(torch.tensor(noise_np[:b], device=device))

        def fixed_rsample(distribution, sample_shape=torch.Size()):
            if sample_shape or not queue:
                raise RuntimeError("Unexpected reference sampling call.")
            noise = queue.pop(0)
            if noise.shape != distribution.loc.shape:
                raise RuntimeError("Reference sampling shape differs from fixed noise.")
            return distribution.loc + distribution.scale * noise

        targets = []
        reference_target = torch_updates._compute_categorical_td_target

        def record_target(*values, **kwargs):
            result = reference_target(*values, **kwargs)
            targets.append(result.detach().cpu().numpy().copy())
            return result

        with (
            patch.object(torch.distributions.Normal, "rsample", fixed_rsample),
            patch.object(torch_updates, "_compute_categorical_td_target", record_target),
        ):
            reference_metrics = _update_networks(
                torch_batch, actor, critic, target, temperature, cfg, actor_phase, device, None
            )
        comparison.condition(f"full_update_{phase_index}/all_fixed_noise_consumed", not queue)
        actual_metrics = eager.update_from_batch(
            fixed, actor_noise=noise_np, next_noise=noise_np[:b], do_actor_update=actor_phase, normalize_reward=False
        )
        comparison.condition(f"full_update_{phase_index}/one_categorical_target", len(targets) == 1)
        if targets:
            comparison.arrays(
                f"full_update_{phase_index}/categorical_target", eager.target_probabilities.numpy(), targets[0]
            )
        comparison.arrays(
            f"full_update_{phase_index}/update_counters",
            eager.update_counters.numpy(),
            np.array([phase_index + 1, 1, 1]),
        )
        for key, expected in reference_metrics.items():
            comparison.arrays(
                f"full_update_{phase_index}/metric/{key}",
                actual_metrics[key].numpy(),
                expected.detach().cpu().numpy().reshape(1),
            )
        for name, actual, reference, twin in (
            ("actor", eager.actor, actor, False),
            ("critic", eager.critic, critic, True),
            ("target", eager.target_critic, target, True),
        ):
            _compare_state(comparison, f"full_update_{phase_index}/{name}", actual, reference.network, critic=twin)
            _compare_update_deltas(
                comparison,
                f"full_update_{phase_index}/{name}",
                actual,
                reference.network,
                critic_initial if twin else actor_initial,
                critic=twin,
            )
        _compare_optimizer(
            comparison, f"full_update_{phase_index}/actor_adam", eager.actor, eager.actor_optimizer, actor
        )
        _compare_optimizer(
            comparison,
            f"full_update_{phase_index}/critic_adam",
            eager.critic,
            eager.critic_optimizer,
            critic,
            critic=True,
        )
        comparison.arrays(
            f"full_update_{phase_index}/log_temperature",
            eager.log_temperature.numpy(),
            temperature.network.log_temp.detach().cpu().numpy(),
        )
        temperature_initial = np.array([math.log(cfg.temp_initial_value)], dtype=np.float32).astype(np.float64)
        comparison.arrays(
            f"full_update_{phase_index}/temperature/update_delta/log_temperature",
            eager.log_temperature.numpy().astype(np.float64) - temperature_initial,
            temperature.network.log_temp.detach().cpu().numpy().astype(np.float64) - temperature_initial,
        )
        temp_state = temperature.optimizer.state[temperature.network.log_temp]
        for key, actual in (
            ("exp_avg", eager.temperature_optimizer._m1[0]),
            ("exp_avg_sq", eager.temperature_optimizer._m2[0]),
            ("step", eager.temperature_optimizer._timestep),
        ):
            comparison.arrays(
                f"full_update_{phase_index}/temperature_adam/{key}",
                actual.numpy(),
                temp_state[key].cpu().numpy().reshape(-1),
            )
        comparison.arrays(
            f"full_update_{phase_index}/target_probability_mass",
            eager.target_probabilities.numpy().sum(axis=1),
            np.ones(b),
        )
        comparison.condition(
            f"full_update_{phase_index}/target_probabilities_nonnegative",
            bool((eager.target_probabilities.numpy() >= 0).all()),
        )
        expected_states.append(eager.state_dict(include_replay=False))

    for name, model in (("actor", eager.actor), ("critic", eager.critic)):
        comparison.condition(
            f"full_update/{name}_weights_updated",
            any(
                np.any(parameter.data.numpy() != initial[f"{name}.{key}"])
                for key, parameter in model.named_parameters()
            ),
        )
    checkpoint_state = eager.state_dict(include_replay=False)
    with TemporaryDirectory(prefix="robolearn-flash-parity-") as directory:
        eager.save(directory, include_replay=False)
        eager.load_state_dict(initial)
        eager.load(directory)
        restored = eager.state_dict(include_replay=False)
        for key, expected in checkpoint_state.items():
            comparison.arrays(f"checkpoint_roundtrip/{key}", restored[key], expected)

    if not wp.get_device(args.device).is_cuda:
        return "not executed: CUDA graph capture requires a CUDA device"
    captured = WarpFlashSAC(
        args.observation_dim, args.action_dim, 2, cfg, optimized_linear_backward=args.optimized_linear_backward
    )
    captured.prepare(capture=False)
    captured.load_state_dict(initial)
    for key, value in fixed.items():
        captured._assign(captured.batch[key], value, key)
    captured._assign(captured.actor_noise, noise_np, "actor_noise")
    captured._assign(captured.next_noise, noise_np[:b], "next_noise")
    graphs = []
    for actor_phase in (True, False):
        with wp.ScopedCapture(device=captured.device) as capture:
            captured.launch_update(actor_phase, sample=False, randomize_noise=False, normalize_reward=False)
        graphs.append(capture.graph)
    for phase_index, graph in enumerate(graphs):
        wp.capture_launch(graph)
        actual_state = captured.state_dict(include_replay=False)
        for key, expected in expected_states[phase_index].items():
            comparison.arrays(f"captured_update_{phase_index}/{key}", actual_state[key], expected)
    return "executed: full actor/temperature/critic/EMA phase followed by critic/EMA-only phase"


def diagnose(args, comparison):
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(args.seed + 1)
    batch = args.batch_size
    obs_np = rng.normal(size=(2 * batch, args.observation_dim)).astype(np.float32)
    act_np = np.tanh(rng.normal(size=(2 * batch, args.action_dim))).astype(np.float32)
    noise_np = rng.normal(size=(2 * batch, args.action_dim)).astype(np.float32)
    target_np = rng.uniform(0.1, 1.0, size=(batch, args.bins)).astype(np.float32)
    target_np /= target_np.sum(axis=1, keepdims=True)
    device = torch.device(args.device)
    torch_actor = FlashSACActor(args.blocks, args.observation_dim, args.actor_hidden_dim, args.action_dim).to(device)
    torch_critic = FlashSACDoubleCritic(
        args.blocks, args.observation_dim + args.action_dim, args.critic_hidden_dim, args.bins, -5.0, 5.0
    ).to(device)
    with torch.no_grad():
        for model in (torch_actor, torch_critic):
            for module in model.modules():
                if hasattr(module, "normalize_parameters"):
                    module.normalize_parameters()
    actor_initial, critic_initial = _snapshot(torch_actor), _snapshot(torch_critic)
    warp_actor = FlashActor(
        args.observation_dim,
        args.action_dim,
        args.actor_hidden_dim,
        args.blocks,
        seed=args.seed,
        device=args.device,
        optimized_linear_backward=args.optimized_linear_backward,
    )
    warp_critic = FlashDoubleCritic(
        args.observation_dim,
        args.action_dim,
        args.critic_hidden_dim,
        args.blocks,
        args.bins,
        seed=args.seed,
        device=args.device,
        optimized_linear_backward=args.optimized_linear_backward,
    )
    torch_obs = torch.tensor(obs_np, device=device, requires_grad=True)
    torch_actions = torch.tensor(act_np, device=device, requires_grad=True)
    warp_obs = wp.array(obs_np, device=args.device, requires_grad=True)
    warp_actions = wp.array(act_np, device=args.device, requires_grad=True)

    def reset():
        torch_actor.load_state_dict({key: torch.tensor(value, device=device) for key, value in actor_initial.items()})
        torch_critic.load_state_dict({key: torch.tensor(value, device=device) for key, value in critic_initial.items()})
        torch_actor.zero_grad(set_to_none=True)
        torch_critic.zero_grad(set_to_none=True)
        warp_actor.load_author_state_dict(actor_initial)
        warp_critic.load_author_state_dict(critic_initial)
        for model in (warp_actor, warp_critic):
            for parameter in model.parameters():
                parameter.grad.zero_()
        torch_obs.grad = None
        torch_actions.grad = None
        warp_obs.grad.zero_()
        warp_actions.grad.zero_()

    reset()
    _compare_state(comparison, "initial_actor", warp_actor, torch_actor)
    _compare_state(comparison, "initial_critic", warp_critic, torch_critic, critic=True)
    for training in (False, True, False):
        label = f"forward_{len(comparison.checks)}_{'train' if training else 'eval'}"
        with torch.no_grad():
            expected_mean, expected_std = torch_actor.get_mean_and_std(torch_obs, training=training)
            expected_q, expected_info = torch_critic(torch_obs, torch_actions, training=training)
        actual_mean, actual_log_std = warp_actor(warp_obs, training=training)
        actual_q, actual_log_probs = warp_critic(warp_obs, warp_actions, training=training)
        comparison.arrays(label + "/mean", actual_mean.numpy(), expected_mean.cpu().numpy())
        comparison.arrays(label + "/log_std", actual_log_std.numpy(), expected_std.log().cpu().numpy())
        comparison.arrays(label + "/q", actual_q.numpy(), expected_q.cpu().numpy())
        comparison.arrays(label + "/log_probs", actual_log_probs.numpy(), expected_info["log_prob"].cpu().numpy())
        _compare_state(comparison, label + "/actor_bn", warp_actor, torch_actor, buffers_only=True)
        _compare_state(comparison, label + "/critic_bn", warp_critic, torch_critic, critic=True, buffers_only=True)

    reset()
    noise_torch = torch.tensor(noise_np, device=device)
    mean, std = torch_actor.get_mean_and_std(torch_obs, training=True)
    raw = mean + std * noise_torch
    actions = raw.tanh()
    actions.retain_grad()
    log_prob = torch.distributions.Normal(mean, std).log_prob(raw)
    log_prob -= 2.0 * (math.log(2.0) - raw - torch.nn.functional.softplus(-2.0 * raw))
    log_prob = log_prob.sum(dim=-1)
    torch_critic.requires_grad_(False)
    qs, _ = torch_critic(torch_obs[:batch], actions[:batch], training=False)
    torch_loss = (0.01 * log_prob[:batch] - torch.minimum(qs[0], qs[1])).mean()
    torch_loss.backward()
    torch_critic.requires_grad_(True)

    all_actions = wp.empty((2 * batch, args.action_dim), device=args.device, requires_grad=True)
    current_actions = wp.empty((batch, args.action_dim), device=args.device, requires_grad=True)
    # Match Torch's retain_grad(): Warp otherwise consumes this intermediate
    # adjoint when the prefix-copy kernel propagates it to all_actions.
    current_actions.retain_grad = True
    log_probs = wp.empty(2 * batch, device=args.device, requires_grad=True)
    noise = wp.array(noise_np, device=args.device)
    current_obs = warp_obs[:batch]
    loss = wp.zeros(1, device=args.device, requires_grad=True)
    critic_gradients = [parameter.grad for parameter in warp_critic.parameters()]
    for gradient in critic_gradients:
        gradient.fill_(0.125)
    with warp_critic.freeze_parameters():
        with wp.Tape() as tape:
            mean, log_std = warp_actor(warp_obs, training=True)
            wp.launch(
                _fixed_actions,
                dim=2 * batch,
                inputs=[mean, log_std, noise],
                outputs=[all_actions, log_probs],
                device=args.device,
            )
            wp.launch(
                _current_actions,
                dim=current_actions.shape,
                inputs=[all_actions],
                outputs=[current_actions],
                device=args.device,
            )
            qs, _ = warp_critic(current_obs, current_actions, training=False)
            wp.launch(_actor_loss, dim=batch, inputs=[log_probs, qs, 0.01], outputs=[loss], device=args.device)
        tape.backward(loss)
    comparison.arrays("actor_gradient/loss", loss.numpy(), torch_loss.detach().cpu().numpy().reshape(1))
    _compare_gradients(comparison, "actor_gradient/parameters", warp_actor, torch_actor)
    comparison.arrays("actor_gradient/observation", warp_obs.grad.numpy(), torch_obs.grad.cpu().numpy())
    comparison.arrays("actor_gradient/actions", current_actions.grad.numpy(), actions.grad[:batch].cpu().numpy())
    comparison.condition(
        "actor_gradient/nonzero_critic_action_gradient", bool(np.any(np.abs(current_actions.grad.numpy()) > 0.0))
    )
    for i, (parameter, gradient) in enumerate(zip(warp_critic.parameters(), critic_gradients, strict=True)):
        comparison.condition(f"frozen_critic/{i}/same_gradient_storage", parameter.grad.ptr == gradient.ptr)
        comparison.arrays(
            f"frozen_critic/{i}/untouched_gradient", gradient.numpy(), np.full(gradient.shape, 0.125, np.float32)
        )
    _compare_state(comparison, "actor_gradient/actor_bn", warp_actor, torch_actor, buffers_only=True)
    tape.zero()

    reset()
    _, info = torch_critic(torch_obs, torch_actions, training=True)
    target_torch = torch.tensor(target_np, device=device)
    torch_loss = -(target_torch.unsqueeze(0) * info["log_prob"][:, :batch]).sum(dim=-1).mean()
    torch_loss.backward()
    loss.zero_()
    target = wp.array(target_np, device=args.device)
    with wp.Tape() as tape:
        _, predicted_log_probs = warp_critic(warp_obs, warp_actions, training=True)
        wp.launch(
            _critic_loss,
            dim=(2, batch, args.bins),
            inputs=[predicted_log_probs, target],
            outputs=[loss],
            device=args.device,
        )
    tape.backward(loss)
    comparison.arrays("critic_gradient/loss", loss.numpy(), torch_loss.detach().cpu().numpy().reshape(1))
    _compare_gradients(comparison, "critic_gradient/parameters", warp_critic, torch_critic, critic=True)
    comparison.arrays("critic_gradient/observation", warp_obs.grad.numpy(), torch_obs.grad.cpu().numpy())
    comparison.arrays("critic_gradient/action", warp_actions.grad.numpy(), torch_actions.grad.cpu().numpy())
    _compare_state(comparison, "critic_gradient/bn", warp_critic, torch_critic, critic=True, buffers_only=True)
    tape.zero()
    return _full_updates(args, comparison, actor_initial, critic_initial, obs_np, act_np, noise_np)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--observation-dim", type=int, default=11)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--actor-hidden-dim", type=int, default=16)
    parser.add_argument("--critic-hidden-dim", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--bins", type=int, default=11)
    parser.add_argument("--g1", action="store_true")
    parser.add_argument("--optimized-linear-backward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float, default=2.0e-4)
    parser.add_argument("--rtol", type=float, default=2.0e-3)
    args = parser.parse_args()
    if args.g1:
        args.observation_dim, args.action_dim = 123, 37
        args.actor_hidden_dim, args.critic_hidden_dim = 128, 256
        args.blocks, args.bins = 2, 101
    if args.batch_size < 2 or args.atol < 0 or args.rtol < 0:
        parser.error("Batch size must be at least two and tolerances nonnegative.")
    comparison = Comparison(args.atol, args.rtol)
    result = {
        "scope": "FP32 network, gradient and fixed-batch production update parity; no simulator or learning-speed claim",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "reference_revision": "87edc9061150ae9e962dd84e6544e27a1554b3ab",
        "versions": {"torch": torch.__version__, "warp": wp.__version__, "warp_nn": version("warp-nn")},
        "checks": comparison.checks,
        "comparison_policies": comparison.policies,
    }
    try:
        result["capture"] = diagnose(args, comparison)
        result["passed"] = comparison.passed
    except Exception:
        result["passed"] = False
        result["exception"] = traceback.format_exc()
    result["check_count"] = len(comparison.checks)
    result["failed_checks"] = [name for name, check in comparison.checks.items() if not check["passed"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {args.output}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
