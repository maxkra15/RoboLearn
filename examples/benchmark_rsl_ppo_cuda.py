# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Compare actual RSL-RL PPO with native CUDA PPO on a shared synthetic rollout.

This measures complete learner implementations, including their control flow,
not policy quality or isolated GEMMs. RSL-RL is used through its public API;
its algorithm, optimizer, minibatch generator and metric extraction are unchanged.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import warp as wp
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from robolearn.warp import PPOConfig, WarpPPO

ENVS, HORIZON, OBSERVATIONS, ACTIONS = 1024, 24, 123, 37
STEPS_PER_CALL = 20


@wp.kernel(enable_backward=False)
def _store_last_values(values: wp.array2d(dtype=wp.float32), next_values: wp.array(dtype=wp.float32), offset: int):
    index = wp.tid()
    next_values[offset + index] = values[index, 0]


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def array_hash(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def finite_state(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def same_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_state(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def observation(value, device):
    return TensorDict({"policy": torch.as_tensor(value, device=device)}, batch_size=[ENVS], device=device)


def make_rsl(device, *, mixed=False):
    sample = observation(np.zeros((ENVS, OBSERVATIONS), np.float32), device)
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(
        sample,
        groups,
        "actor",
        ACTIONS,
        hidden_dims=(256, 128, 128),
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": "rsl_rl.modules:GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
            "std_range": (1e-6, 1e6),
        },
    )
    critic = MLPModel(
        sample, groups, "critic", 1, hidden_dims=(256, 128, 128), activation="elu", obs_normalization=False
    )
    storage = RolloutStorage("rl", ENVS, HORIZON, sample, [ACTIONS], device)
    return PPO(
        actor,
        critic,
        storage,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.008,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        normalize_advantage_per_mini_batch=False,
        use_mixed_precision=mixed,
        device=device,
    )


def model_arrays(agent):
    arrays = {}
    for model_name, model in (("actor", agent._raw_actor), ("critic", agent._raw_critic)):
        for index, layer in enumerate(item for item in model.mlp if isinstance(item, torch.nn.Linear)):
            arrays[f"{model_name}_{index}_weight"] = layer.weight.detach().cpu().numpy().copy()
            arrays[f"{model_name}_{index}_bias"] = layer.bias.detach().cpu().numpy().copy().reshape(-1, 1)
    arrays["std"] = agent._raw_actor.distribution.std_param.detach().cpu().numpy().copy()
    return arrays


def make_fixture(agent, seed, device):
    """Collect the fixed synthetic data through stock act/process_env_step APIs."""
    rng = np.random.default_rng(seed)
    observations = rng.normal(0, 0.2, (HORIZON + 1, ENVS, OBSERVATIONS)).astype(np.float32)
    rewards = rng.normal(0.2, 0.05, (HORIZON, ENVS)).astype(np.float32)
    terminated = np.zeros((HORIZON, ENVS), np.int32)
    truncated = np.zeros_like(terminated)
    terminated[7, ::17] = 1
    truncated[13, ::19] = 1
    torch.manual_seed(seed)
    with torch.no_grad():
        for step in range(HORIZON):
            agent.act(observation(observations[step], device))
            agent.process_env_step(
                observation(observations[step + 1], device),
                torch.as_tensor(rewards[step], device=device),
                torch.as_tensor(terminated[step] | truncated[step], device=device),
                {"time_outs": torch.as_tensor(truncated[step], device=device)},
            )
    storage = agent.storage
    with torch.no_grad():
        last_obs = observation(observations[-1], device)
        initial_last_values = agent.critic(last_obs).detach().cpu().numpy().copy()
        agent.compute_returns(last_obs)
    values = storage.values.detach().cpu().numpy().reshape(HORIZON, ENVS).copy()
    next_values = np.zeros_like(values)
    next_values[:-1] = values[1:]
    data = {
        "observations": observations[:-1].reshape(-1, OBSERVATIONS),
        "last_observations": observations[-1],
        "initial_last_values": initial_last_values,
        "initial_returns": storage.returns.detach().cpu().numpy().reshape(-1).copy(),
        "initial_advantages": storage.advantages.detach().cpu().numpy().reshape(-1).copy(),
        "actions": storage.actions.detach().cpu().numpy().reshape(-1, ACTIONS).copy(),
        "means": storage.distribution_params[0].detach().cpu().numpy().reshape(-1, ACTIONS).copy(),
        "stds": storage.distribution_params[1].detach().cpu().numpy().reshape(-1, ACTIONS).copy(),
        "log_probs": storage.actions_log_prob.detach().cpu().numpy().reshape(-1).copy(),
        "values": values.reshape(-1),
        "next_values": next_values.reshape(-1),
        "rewards": rewards.reshape(-1),
        "rsl_bootstrapped_rewards": storage.rewards.detach().cpu().numpy().reshape(-1).copy(),
        "terminated": terminated.reshape(-1),
        "truncated": truncated.reshape(-1),
    }
    if not finite_state(data):
        raise RuntimeError("The stock-collected common fixture is nonfinite")
    return data


def fill_rsl(agent, data, device):
    storage = agent.storage
    storage.observations["policy"].copy_(
        torch.as_tensor(data["observations"].reshape(HORIZON, ENVS, OBSERVATIONS), device=device)
    )
    for destination, key in (
        (storage.actions, "actions"),
        (storage.values, "values"),
        (storage.actions_log_prob, "log_probs"),
        (storage.rewards, "rsl_bootstrapped_rewards"),
    ):
        destination.copy_(torch.as_tensor(data[key].reshape(destination.shape), device=device))
    storage.dones.copy_(
        torch.as_tensor((data["terminated"] | data["truncated"]).reshape(HORIZON, ENVS, 1), device=device)
    )
    storage.distribution_params = tuple(
        torch.as_tensor(data[key].reshape(HORIZON, ENVS, ACTIONS), device=device).clone() for key in ("means", "stds")
    )
    storage.step = HORIZON


def rsl_steps(agent):
    parameters = [parameter for group in agent.optimizer.param_groups for parameter in group["params"]]
    return [int(agent.optimizer.state.get(parameter, {}).get("step", 0)) for parameter in parameters]


def summarize_blocks(blocks):
    values = [block["seconds_per_full_update"] for block in blocks]
    return {
        "mean_seconds_per_full_update": statistics.mean(values),
        "median_seconds_per_full_update": statistics.median(values),
        "min_seconds_per_full_update": min(values),
        "max_seconds_per_full_update": max(values),
        "variation_scope": "Repeated timing blocks in one ordered process, not independent devices or a confidence interval",
    }


def benchmark_rsl(args, common, fixture, *, compile_mode=None, mixed=False, tf32=False):
    label = f"rsl_{'bf16' if mixed else 'fp32'}_{'tf32_allowed' if tf32 else 'strict'}_{compile_mode or 'eager'}"
    print(f"START {label}: construct/compile", flush=True)
    start = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    agent = make_rsl(args.device, mixed=mixed)
    agent.load(copy.deepcopy(common), None, strict=True)
    fill_rsl(agent, fixture, args.device)
    initial_weights = model_arrays(agent)
    initial_hash = array_hash(initial_weights)
    last_obs = observation(fixture["last_observations"], args.device)
    agent.compile(compile_mode)

    def update():
        # Stock update clears only the write cursor. Reuse immutable old-policy
        # buffers and recompute returns using the current final critic bootstrap.
        agent.storage.step = HORIZON
        with torch.no_grad():
            agent.compute_returns(last_obs)
        return agent.update()

    # Charge lazy compiler/optimizer initialization to explicit startup, then
    # reset all learning state before every warm/timed block.
    update()
    torch.cuda.synchronize(args.device)
    startup = time.perf_counter() - start
    blocks = []
    for block in range(args.blocks):
        agent.load(copy.deepcopy(common), None, strict=True)
        torch.manual_seed(args.seed + block)
        reset_exact = array_hash(model_arrays(agent)) == initial_hash and all(step == 0 for step in rsl_steps(agent))
        if not reset_exact or agent.learning_rate != 1e-3:
            raise RuntimeError(f"{label}: initial model/Adam/learning-rate reset failed")
        start = time.perf_counter()
        for _ in range(args.warmup):
            update()
        torch.cuda.synchronize(args.device)
        warm_seconds = time.perf_counter() - start
        before = rsl_steps(agent)
        start = time.perf_counter()
        for _ in range(args.updates):
            losses = update()
        torch.cuda.synchronize(args.device)
        seconds = time.perf_counter() - start
        after = rsl_steps(agent)
        actual = [end - begin for begin, end in zip(before, after, strict=True)]
        finite = finite_state(agent.save()) and finite_state(losses)
        changed = any(not np.array_equal(value, initial_weights[key]) for key, value in model_arrays(agent).items())
        if (
            not finite
            or not changed
            or len(actual) != len(initial_weights)
            or any(count != args.updates * STEPS_PER_CALL for count in actual)
        ):
            raise RuntimeError(f"{label}: finite={finite}, changed={changed}, actual Adam steps={actual}")
        blocks.append(
            {
                "block": block,
                "initial_state_restored": reset_exact,
                "warmup_seconds": warm_seconds,
                "timed_seconds": seconds,
                "seconds_per_full_update": seconds / args.updates,
                "attempted_optimizer_updates": args.updates * STEPS_PER_CALL,
                "actual_optimizer_updates_per_parameter": actual,
                "optimizer_steps_before": before,
                "optimizer_steps_after": after,
                "optimizer_parameter_count": len(actual),
                "finite": finite,
                "updated_weights": changed,
                "last_stock_losses": losses,
                "final_learning_rate": agent.learning_rate,
            }
        )
        print(f"DONE {label} block {block}: {1000 * seconds / args.updates:.3f} ms/full update", flush=True)
    checkpoint = args.output.parent / "checkpoints" / f"{label}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    saved = copy.deepcopy(agent.save())
    torch.save(saved, checkpoint)
    with torch.no_grad():
        next(agent._raw_actor.parameters()).zero_()
    agent.load(torch.load(checkpoint, map_location=args.device, weights_only=True), None, strict=True)
    restored = same_state(saved, agent.save())
    before = rsl_steps(agent)
    update()
    torch.cuda.synchronize(args.device)
    replay_counts = [end - begin for begin, end in zip(before, rsl_steps(agent), strict=True)]
    if not restored or not finite_state(agent.save()) or any(count != STEPS_PER_CALL for count in replay_counts):
        raise RuntimeError(f"{label}: checkpoint restore or post-load update failed")
    return {
        "backend": label,
        "implementation": "Unmodified rsl_rl.algorithms.PPO 5.5.1 public compute_returns/update APIs",
        "precision": "bf16 autocast, FP32 parameters/Adam" if mixed else "fp32",
        "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32_allowed": bool(torch.backends.cudnn.allow_tf32),
        "isaaclab_default_recipe": tf32 and not mixed and compile_mode is None,
        "compile_mode": compile_mode,
        "compiled_scope": "Official PPO.compile wraps actor/critic only; optimizer/control flow remain stock",
        "captured_full_update": False,
        "startup_seconds": startup,
        "parameter_array_count": len(initial_weights),
        "initial_weights_sha256": initial_hash,
        "blocks": blocks,
        **summarize_blocks(blocks),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_restored_exactly": restored,
        "post_load_optimizer_updates_per_parameter": replay_counts,
        "post_load_finite": True,
    }


def benchmark_cuda(args, weights, fixture):
    print("START cuda_fp32: construct/eager warmup/capture", flush=True)
    start = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = PPOConfig(
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
        seed=args.seed,
        dense_backend="cuda",
    )
    agent = WarpPPO(OBSERVATIONS, ACTIONS, ENVS, HORIZON, config, device=args.device)
    assigned = {}
    for name, model in (("actor", agent.actor), ("critic", agent.critic)):
        parameters = model.parameters()
        for index in range(len(parameters) // 2):
            for offset, suffix in enumerate(("weight", "bias")):
                key = f"{name}_{index}_{suffix}"
                parameters[index * 2 + offset].assign(weights[key])
                assigned[key] = parameters[index * 2 + offset].numpy()
    agent.distribution_parameter.assign(weights["std"])
    assigned["std"] = agent.distribution_parameter.numpy()
    if array_hash(assigned) != array_hash(weights):
        raise RuntimeError("Native and RSL initial weights are not identical")
    initial = agent.state_dict()
    for key, value in fixture.items():
        if hasattr(agent, key) and key != "last_observations":
            getattr(agent, key).assign(value)
    last_obs = wp.array(fixture["last_observations"], dtype=wp.float32, device=agent.device)
    native_last_values = agent.critic(last_obs).numpy()
    last_value_error = float(np.max(np.abs(native_last_values - fixture["initial_last_values"])))
    if not np.allclose(native_last_values, fixture["initial_last_values"], atol=2e-5, rtol=1e-4):
        raise RuntimeError(f"Native initial critic bootstrap differs from RSL: max abs {last_value_error}")
    wp.launch(
        _store_last_values,
        dim=ENVS,
        inputs=[agent.critic(last_obs), agent.next_values, (HORIZON - 1) * ENVS],
        device=agent.device,
    )
    agent.compute_returns()
    initial_gae = {}
    for name in ("returns", "advantages"):
        native = getattr(agent, name).numpy().astype(np.float64)
        reference = fixture[f"initial_{name}"].astype(np.float64)
        difference = native - reference
        maximum = float(np.max(np.abs(difference)))
        finite = bool(np.isfinite(native).all() and np.isfinite(reference).all())
        initial_gae[name] = {
            "max_abs_difference": maximum,
            "rms_abs_difference": float(np.sqrt(np.mean(difference**2))),
            "reference_max_abs": float(np.max(np.abs(reference))),
            "finite": finite,
            "max_abs_gate": 2e-4,
            "accepted": finite and maximum <= 2e-4,
        }
        if not initial_gae[name]["accepted"]:
            raise RuntimeError(f"Native initial {name} versus stock RSL failed: {initial_gae[name]}")

    def launch():
        last_values = agent.critic(last_obs)
        wp.launch(
            _store_last_values,
            dim=ENVS,
            inputs=[last_values, agent.next_values, (HORIZON - 1) * ENVS],
            device=agent.device,
        )
        agent.launch_update()

    # Warm the complete wrapper, including its inference batch shape and CUDA
    # dense backward/optimizer plans, before capturing persistent pointers.
    launch()
    wp.synchronize_device(agent.device)
    agent.load_state_dict(initial)
    with wp.ScopedCapture(device=agent.device) as capture:
        launch()
    wp.synchronize_device(agent.device)
    startup = time.perf_counter() - start
    blocks = []

    def update():
        wp.capture_launch(capture.graph)
        # Match the public RSL update's host-returning API. Its metrics average
        # all minibatches; native health reports the final minibatch and norms.
        # The two extraction implementations and reduction semantics differ.
        return agent.health_metrics()

    for block in range(args.blocks):
        state = {key: value.copy() for key, value in initial.items()}
        state["seed"][:] = args.seed + block
        agent.load_state_dict(state)
        reset = agent.state_dict()
        reset_exact = all(np.array_equal(value, reset[key]) for key, value in state.items())
        if not reset_exact:
            raise RuntimeError("Native model/Adam/learning-rate/RNG reset failed")
        start = time.perf_counter()
        for _ in range(args.warmup):
            update()
        wp.synchronize_device(agent.device)
        warm_seconds = time.perf_counter() - start
        before = float(agent.optimizer._timestep.numpy()[0])
        start = time.perf_counter()
        for _ in range(args.updates):
            health = update()
        wp.synchronize_device(agent.device)
        seconds = time.perf_counter() - start
        final = agent.state_dict()
        after = float(final["adam_timestep"][0])
        finite = finite_state(final) and finite_state(health)
        changed = any(
            not np.array_equal(value, initial[key]) for key, value in final.items() if key.startswith("parameter_")
        )
        if not finite or not changed or after - before != args.updates * STEPS_PER_CALL:
            raise RuntimeError(f"Native CUDA: finite={finite}, changed={changed}, actual Adam steps={after - before}")
        blocks.append(
            {
                "block": block,
                "initial_state_restored": reset_exact,
                "warmup_seconds": warm_seconds,
                "timed_seconds": seconds,
                "seconds_per_full_update": seconds / args.updates,
                "attempted_optimizer_updates": args.updates * STEPS_PER_CALL,
                "actual_optimizer_updates": after - before,
                "optimizer_steps_before": before,
                "optimizer_steps_after": after,
                "finite": finite,
                "updated_weights": changed,
                "last_host_metrics": health,
            }
        )
        print(f"DONE cuda_fp32 block {block}: {1000 * seconds / args.updates:.3f} ms/full update", flush=True)
    checkpoint = args.output.parent / "checkpoints" / "cuda_fp32.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    agent.save(checkpoint)
    agent.parameters[0].zero_()
    agent.load(checkpoint)
    reloaded = agent.state_dict()
    restored = all(np.array_equal(value, reloaded[key]) for key, value in final.items())
    before = float(agent.optimizer._timestep.numpy()[0])
    wp.capture_launch(capture.graph)
    wp.synchronize_device(agent.device)
    post_load = agent.state_dict()
    replay_count = float(post_load["adam_timestep"][0]) - before
    if not restored or not finite_state(post_load) or replay_count != STEPS_PER_CALL:
        raise RuntimeError("Native CUDA checkpoint restore or captured post-load update failed")
    return {
        "backend": "cuda_fp32",
        "implementation": "RoboLearn WarpPPO loss/control kernels with cuBLASLt dense and fused native Adam",
        "precision": "fp32, cuBLASLt CUBLAS_COMPUTE_32F_PEDANTIC, TF32 disabled",
        "tf32_allowed": False,
        "captured_full_update": True,
        "config": asdict(config),
        "startup_seconds": startup,
        "parameter_array_count": len(agent.parameters),
        "initial_critic_bootstrap_max_abs_difference": last_value_error,
        "initial_gae_comparison_with_stock_rsl": initial_gae,
        "initial_weights_sha256": array_hash(assigned),
        "blocks": blocks,
        **summarize_blocks(blocks),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_restored_exactly": restored,
        "post_load_optimizer_updates": replay_count,
        "post_load_finite": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--rsl-compile", choices=("none", "default", "max-autotune-no-cudagraphs"), default="default")
    parser.add_argument(
        "--rsl-bf16", action="store_true", help="Also measure stock RSL BF16 autocast (same compile mode)"
    )
    args = parser.parse_args()
    if min(args.updates, args.blocks) < 1 or args.warmup < 0:
        parser.error("updates/blocks must be positive; warmup must be nonnegative")
    if importlib.metadata.version("rsl-rl-lib") != "5.5.1":
        raise RuntimeError("This API benchmark requires the Isaac Lab pinned rsl-rl-lib==5.5.1")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    wp.init()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "RUNNING",
        "benchmark": "Actual stock RSL-RL versus native CUDA PPO; fixed synthetic old-policy data",
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("robolearn-rl", "rsl-rl-lib", "tensordict", "torch", "warp-lang", "warp-nn")
        },
        "sources_sha256": {
            name: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
            for name, cls in (
                ("rsl_ppo", PPO),
                ("rsl_model", MLPModel),
                ("rsl_storage", RolloutStorage),
                ("native", WarpPPO),
            )
        },
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": {"name": torch.cuda.get_device_name(args.device), "index": torch.cuda.current_device()},
        "protocol": {
            "environments": ENVS,
            "horizon": HORIZON,
            "observations": OBSERVATIONS,
            "actions": ACTIONS,
            "hidden_dims": [256, 128, 128],
            "activation": "elu",
            "epochs": 5,
            "mini_batches": 4,
            "minibatch_size": ENVS * HORIZON // 4,
            "warm_full_updates_per_block": args.warmup,
            "timed_full_updates_per_block": args.updates,
            "timing_blocks": args.blocks,
            "optimizer_steps_per_full_update": STEPS_PER_CALL,
            "seed": args.seed,
            "torch_cpu_threads": torch.get_num_threads(),
            "timed_optimizer_steps_per_backend": args.blocks * args.updates * STEPS_PER_CALL,
            "scope": (
                "One full update includes current final-critic bootstrap, GAE, sample-std whole-rollout advantage "
                "normalization, one shuffle reused for five epochs and 20 actor+critic Adam minibatches. "
                "Excludes rollout/action collection, storage filling, model reset, checkpoint I/O and post-timing audits. "
                "Stock RSL adaptive-KL Python decisions, optimizer and per-minibatch .item() metrics remain timed; "
                "native device control, full-update graph and one health_metrics() host extraction per call are timed. "
                "RSL host losses average all 20 minibatches; native host health describes the final minibatch, "
                "optimizer count and clipping norms, so metric reduction/extraction costs are not identical."
            ),
            "startup_scope": (
                "Per-row construction, fixture/state copy, official compile wrapping plus first full execution, "
                "or native full eager warmup and graph capture. Rows share one ordered process and GPU/compiler "
                "caches; these are not independent cold-start measurements. Common fixture creation precedes rows."
            ),
            "limitations": [
                "Synthetic data has no simulator, task reward, learning curve or walking-quality conclusion.",
                "Each block starts from identical initial weights, empty Adam state and initial learning rate.",
                "Warmup updates precede timed calls. Repeated reuse of one old-policy rollout is a timing fixture.",
                "Common weights/data do not imply common RNG, permutations, scheduler decisions or weight trajectories.",
                "Different reduction orders/precision and native/Torch optimizer implementations can change trajectories.",
                "One device and three repeated blocks describe local variation, not statistical hardware evidence.",
                "Stock BF16 autocast, when enabled, is a different precision recipe with FP32 parameters/Adam.",
            ],
        },
        "records": [],
    }
    write_result(args.output, result)
    try:
        print("START shared stock rollout collection", flush=True)
        source = make_rsl(args.device)
        common = copy.deepcopy(source.save())
        weights = model_arrays(source)
        fixture = make_fixture(source, args.seed, args.device)
        fixture_path = args.output.parent / "shared-fixture.npz"
        np.savez_compressed(fixture_path, **fixture, **{f"initial_{key}": value for key, value in weights.items()})
        result["shared_fixture"] = {
            "path": str(fixture_path),
            "sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "data_arrays_sha256": array_hash(fixture),
            "initial_weights_sha256": array_hash(weights),
            "collection": "Actual stock RSL act/process_env_step; raw and stock timeout-bootstrapped rewards retained",
        }
        del source
        # Isaac Lab's stock entrypoint allows TF32; preserve that actual default
        # and also provide a controlled strict-FP32 row versus native pedantic.
        rows = [(None, False, True), (None, False, False)]
        compile_mode = None if args.rsl_compile == "none" else args.rsl_compile
        if compile_mode is not None:
            rows.append((compile_mode, False, False))
        if args.rsl_bf16:
            rows.append((compile_mode, True, False))
        for mode, mixed, tf32 in rows:
            result["records"].append(benchmark_rsl(args, common, fixture, compile_mode=mode, mixed=mixed, tf32=tf32))
            write_result(args.output, result)
        result["records"].append(benchmark_cuda(args, weights, fixture))
        native = result["records"][-1]["mean_seconds_per_full_update"]
        result["rsl_time_over_native_time"] = {
            row["backend"]: row["mean_seconds_per_full_update"] / native for row in result["records"][:-1]
        }
        result["measurement_order"] = [row["backend"] for row in result["records"]]
        result["status"] = "COMPLETED"
        write_result(args.output, result)
    except Exception as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        write_result(args.output, result)
        raise


if __name__ == "__main__":
    main()
