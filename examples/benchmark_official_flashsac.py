# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Compare pristine official FlashSAC and RoboLearn learner APIs on synthetic data.

Install benchmark-only gymnasium, omegaconf and CPU jax/jaxlib alongside the
FlashSAC/Warp extras. Supply an unmodified checkout or Git archive of OFFICIAL_SHA
using --official-root. Run --recipe fp32 and --recipe amp in separate processes.
No simulator is imported, and timings do not measure learning quality.
"""

import argparse
import ast
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, fields, replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp
import warp_nn

from robolearn.flashsac import FlashSAC, FlashSACConfig
from robolearn.warp import WarpFlashSAC

OFFICIAL_SHA = "87edc9061150ae9e962dd84e6544e27a1554b3ab"
OFFICIAL_URL = "https://github.com/Holiday-Robot/FlashSAC"
# SHA256 of the pristine pinned files, including every module imported by this
# standalone learner. This also verifies archives without a .git directory.
OFFICIAL_FILES = {
    "LICENSE": "5e7a0ca5f1955091e476af95148f32b48b58feb851a36e09b5c89a0c05bd4421",
    "README.md": "19f19f72039eb18fbd316c6f7217f07f5a28e0d685b0c57dd4a25f8efea6d8d3",
    "pyproject.toml": "6849d5f4e5ea4eaea56145ca139e905b716d58e02807f2baccd62489afef1f3c",
    "flash_rl/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "flash_rl/types.py": "0c1944097870ab47f5180a09ab979923a4de22b39b4a1314f35c4135bc278cd1",
    "flash_rl/agents/__init__.py": "921bf41fd77b2efdd7714dc966f69a0fbb3208f6feb4cb9088f7b1289cca81b8",
    "flash_rl/agents/base_agent.py": "d58f98e02eed8887c16f5f7d83d97f16e4a7ac0d104dc40d1809bf3832a72dfc",
    "flash_rl/agents/flashSAC/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "flash_rl/agents/flashSAC/agent.py": "e1bb9c49cdf1b72507705403801a5dc993471a89f69dbcee973f14949895595a",
    "flash_rl/agents/flashSAC/network.py": "4068098466f195fac2c7d91e3cb1abe0c4daf42c4d1cd85f3940dca9edfd4c98",
    "flash_rl/agents/flashSAC/layer.py": "7af7b2954c22a5ae56595b8d3139017e2a44b17dbd63c2f54c4e65dbd17ef7cc",
    "flash_rl/agents/flashSAC/update.py": "6134612e4ad43089dd44ce06cb6644546e864a3de13bb12525cb631e65aa30ef",
    "flash_rl/agents/utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "flash_rl/agents/utils/distribution.py": "30c2987d80238ccc9c59dd6f7a9dd4056ed01773901d3d08a0db8c40f61b76ed",
    "flash_rl/agents/utils/network.py": "02b1e8c27ae2d75ba39969a5ea6b21e1bc596dbab71fdd6c5dc9cd5d7aab1d69",
    "flash_rl/agents/utils/reward_normalization.py": "b673f5fba40468f6bb183d3cc82daa05111e95407ccb6b73de854f338c7b70b8",
    "flash_rl/agents/utils/scheduler.py": "b3732509dbe01a01d76b3018d062386287bf7cb669973e2a465192c584030811",
    "flash_rl/buffers/__init__.py": "b2dac97cf5043c9b533c694035194d9a7eb3a31056255f50e45fd54a70331438",
    "flash_rl/buffers/base_buffer.py": "75a067a6d8a3d4d6098961a557d78bbb42e73bfd5e6e1e37b41c9d70fb263699",
    "flash_rl/buffers/numpy_buffer.py": "aa7fdac8d09120e96afde15b4e1cb4d6c59f8e670fdd98c13d7d0ca05d925e81",
    "flash_rl/buffers/torch_buffer.py": "663ec003992ebda3c029c963eb93db8b902bc666cdd76f451e7cb348d37439a9",
}
REPLAY_ATTRIBUTES = {
    "observation": "_observations",
    "next_observation": "_next_observations",
    "action": "_actions",
    "reward": "_rewards",
    "terminated": "_terminateds",
    "truncated": "_truncateds",
}


def write_json(path, value):
    def safe(item):
        if isinstance(item, dict):
            return {key: safe(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [safe(value) for value in item]
        if isinstance(item, float) and not np.isfinite(item):
            return {"nonfinite": str(item)}
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def compile_sites(path):
    tree = ast.parse(path.read_text())
    return {
        "decorated_functions": [
            {"function": node.name, "line": node.lineno}
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(ast.unparse(decorator).startswith("torch.compile") for decorator in node.decorator_list)
        ],
        "compile_call_lines": [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.compile"
        ],
    }


def import_official(root):
    root = root.expanduser().resolve(strict=True)
    hashes = {}
    for name, expected in OFFICIAL_FILES.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise RuntimeError(f"Official source must contain a regular file: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[name] = actual
        if actual != expected:
            raise RuntimeError(f"Official source differs from {OFFICIAL_SHA}: {name}")
    if any(name == "flash_rl" or name.startswith("flash_rl.") for name in sys.modules):
        raise RuntimeError("Import official FlashSAC in a fresh process to avoid another flash_rl package.")
    sys.path.insert(0, str(root))
    start = time.perf_counter()
    module = importlib.import_module("flash_rl.agents.flashSAC.agent")
    import_seconds = time.perf_counter() - start
    imported = {}
    for name, loaded in tuple(sys.modules.items()):
        if name == "flash_rl" or name.startswith("flash_rl."):
            path = Path(loaded.__file__).resolve()
            relative = path.relative_to(root).as_posix()
            if relative not in OFFICIAL_FILES:
                raise RuntimeError(f"Unexpected official dependency module: {relative}")
            imported[name] = relative
    return module, {
        "repository": OFFICIAL_URL,
        "commit": OFFICIAL_SHA,
        "main_head_observed_on_2026_10_23": OFFICIAL_SHA,
        "root": str(root),
        "source_files_sha256": hashes,
        "imported_modules": imported,
        "source_patches": [],
        "runtime_monkeypatches": [],
        "authors_launcher_precision_policy": {
            "source": f"{OFFICIAL_URL}/blob/{OFFICIAL_SHA}/train.py#L57-L61",
            "source_sha256": "bb33137c5dd7e6cc778860b933997ecb3579a1f4551ae9db38529962947b1943",
            "allow_tf32": True,
            "float32_matmul_precision": "high",
            "cudnn_benchmark": True,
        },
        "import_seconds": import_seconds,
        "compatibility": (
            "Standalone learner only. Upstream declares Python >=3.10,<3.12; this benchmark's modern "
            "runtime is outside that matrix. No Isaac Lab or simulator launcher is imported. CPU JAX "
            "is imported unchanged solely by upstream's ndarray type alias."
        ),
        "compile_sites": {
            name: compile_sites(root / name)
            for name in OFFICIAL_FILES
            if name.endswith(".py")
            and "@torch.compile" in (root / name).read_text()
            or name == "flash_rl/agents/utils/network.py"
        },
    }


def raw_module(network):
    module = network.network
    return getattr(module, "_orig_mod", module)


def torch_state(agent):
    return {
        f"{name}.{key}": value.detach().cpu().numpy().copy()
        for name in ("actor", "critic", "target_critic", "temperature")
        for key, value in raw_module(getattr(agent, f"_{name}")).state_dict().items()
    }


def initial_weights(path, reference):
    """Load a canonical export without coercing its architecture, dtypes or values."""
    path = path.expanduser().resolve(strict=True)
    payload = path.read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if len(archive.files) != len(reference) or set(archive.files) != set(reference):
            raise RuntimeError("Initial-weight keys must exactly match the pristine official network state.")
        state = {}
        for key, expected in reference.items():
            value = archive[key]
            if value.shape != expected.shape or value.dtype != expected.dtype or not np.isfinite(value).all():
                raise RuntimeError(f"Initial weight {key!r} must match the official shape/dtype and be finite.")
            state[key] = np.array(value, copy=True, order="C")
    return state, {
        "kind": "canonical_npz",
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "array_count": len(state),
        "validated_keys_shapes_dtypes_and_finiteness": True,
    }


def load_common_weights(agent, state):
    if isinstance(agent, WarpFlashSAC):
        for name in ("actor", "critic", "target_critic"):
            getattr(agent, name).load_author_state_dict(
                {key.removeprefix(name + "."): value for key, value in state.items() if key.startswith(name + ".")}
            )
        agent.log_temperature.assign(state["temperature.log_temp"])
        loaded = {f"actor.{key}": value.numpy().copy() for key, value in agent.actor.state_dict().items()}
        for name in ("critic", "target_critic"):
            children = [critic.state_dict() for critic in getattr(agent, name).critics]
            if len(children) != 2 or set(children[0]) != set(children[1]):
                raise RuntimeError("Native readback requires matching twin-critic states.")
            for key in children[0]:
                source = (
                    key.replace(".w.w.weight", ".w.weight")
                    .replace(".w1.w.weight", ".w1.weight")
                    .replace(".w2.w.weight", ".w2.weight")
                )
                values = [child[key].numpy().copy() for child in children]
                if key == "predictor.bin_values":
                    if array_hash(values[0]) != array_hash(values[1]):
                        raise RuntimeError("Native twin-critic categorical supports differ.")
                    value = values[0].reshape(state[f"{name}.{source}"].shape)
                else:
                    value = np.stack(values, axis=0)
                loaded[f"{name}.{source}"] = value
        loaded["temperature.log_temp"] = agent.log_temperature.numpy().copy()
    else:
        for name in ("actor", "critic", "target_critic", "temperature"):
            network = raw_module(getattr(agent, f"_{name}"))
            network.load_state_dict(
                {
                    key.removeprefix(name + "."): torch.as_tensor(value, device=next(network.parameters()).device)
                    for key, value in state.items()
                    if key.startswith(name + ".")
                }
            )
        loaded = torch_state(agent)
    hashes = {key: array_hash(value) for key, value in loaded.items()}
    if hashes != {key: array_hash(value) for key, value in state.items()}:
        raise RuntimeError("Common model weights/buffers were not loaded exactly.")
    return {"exactly_equal": True, "array_count": len(loaded), "hashes": hashes}


def parameter_state(agent):
    if isinstance(agent, WarpFlashSAC):
        return {
            **{f"actor.{name}": array.data.numpy().copy() for name, array in agent.actor.named_parameters()},
            **{f"critic.{name}": array.data.numpy().copy() for name, array in agent.critic.named_parameters()},
            "temperature.log_temp": agent.log_temperature.numpy().copy(),
        }
    return {
        f"{name}.{key}": value.detach().cpu().numpy().copy()
        for name in ("actor", "critic", "temperature")
        for key, value in raw_module(getattr(agent, f"_{name}")).named_parameters()
    }


def optimizer_steps(agent):
    if isinstance(agent, WarpFlashSAC):
        return {
            name: int(optimizer._timestep.numpy()[0])
            for name, optimizer in (
                ("actor", agent.actor_optimizer),
                ("critic", agent.critic_optimizer),
                ("temperature", agent.temperature_optimizer),
            )
        }
    steps = {}
    for name in ("actor", "critic", "temperature"):
        optimizer = getattr(agent, f"_{name}").optimizer
        values = [
            float(optimizer.state.get(parameter, {}).get("step", 0))
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if len(set(values)) != 1 or any(not value.is_integer() for value in values):
            raise RuntimeError(f"Inconsistent optimizer per-parameter step counters: {name}")
        steps[name] = int(values[0])
    return steps


def fresh_optimizer_state(agent):
    """Record that a worker starts before any learner or optimizer update."""
    native = isinstance(agent, WarpFlashSAC)
    record = {"agent_update_calls": agent._update_step, "optimizer_steps": optimizer_steps(agent)}
    if native:
        optimizers = (agent.actor_optimizer, agent.critic_optimizer, agent.temperature_optimizer)
        moments = [array for optimizer in optimizers for array in (*optimizer._m1, *optimizer._m2)]
        record["adam_moment_arrays"] = len(moments)
        record["adam_moments_zero"] = all(not np.count_nonzero(array.numpy()) for array in moments)
        record["amp_state"] = native_amp_state(agent)
        scaler_fresh = all(
            state["attempted_steps"] == state["skipped_steps"] == state["scale_growth_tracker"] == 0
            for state in record["amp_state"].values()
        )
        fresh = record["adam_moments_zero"] and scaler_fresh
    else:
        record["optimizer_state_entries"] = {
            name: len(getattr(agent, f"_{name}").optimizer.state) for name in ("actor", "critic", "temperature")
        }
        record["amp_state"] = agent._grad_scaler.state_dict()
        fresh = (
            not any(record["optimizer_state_entries"].values()) and record["amp_state"].get("_growth_tracker", 0) == 0
        )
    record["accepted"] = bool(fresh and agent._update_step == 0 and not any(record["optimizer_steps"].values()))
    if not record["accepted"]:
        raise RuntimeError("Worker learner/Adam/scaler state is not fresh before warmup")
    return record


def completed_update_budget(agent, calls, actor_period):
    """Audit cumulative complete work outside the timed update interval."""
    actor_calls = (calls + actor_period - 1) // actor_period
    expected = {"critic": calls, "actor": actor_calls, "temperature": actor_calls}
    actual = optimizer_steps(agent)
    record = {
        "expected_agent_update_calls": calls,
        "actual_agent_update_calls": agent._update_step,
        "expected_optimizer_steps": expected,
        "actual_optimizer_steps": actual,
        "accepted": agent._update_step == calls and actual == expected,
    }
    if isinstance(agent, WarpFlashSAC):
        record["device_update_counters_critic_actor_temperature"] = agent.update_counters.numpy().tolist()
        record["amp_state"] = native_amp_state(agent)
        record["accepted"] = bool(
            record["accepted"]
            and record["device_update_counters_critic_actor_temperature"] == [calls, actor_calls, actor_calls]
            and all(
                state["attempted_steps"] == expected[name] and state["skipped_steps"] == 0
                for name, state in record["amp_state"].items()
            )
        )
    return record


def replay_state(agent):
    buffer = agent._replay_buffer
    count = len(buffer)
    if isinstance(agent, WarpFlashSAC):
        return {key: array[:count].numpy().copy() for key, array in buffer.storage.items()}
    result = {
        key: getattr(buffer, name)[:count].detach().cpu().numpy().copy() for key, name in REPLAY_ATTRIBUTES.items()
    }
    if hasattr(buffer, "_discounts"):
        result["discount"] = buffer._discounts[:count].detach().cpu().numpy().copy()
    return result


def audit_replay(actual, reference, gamma_n):
    records = {}
    for key in (*REPLAY_ATTRIBUTES, "discount"):
        expected = reference.get(key)
        if expected is None:
            expected = np.full(reference["reward"].shape, gamma_n, dtype=np.float32)
        value = actual.get(key, np.full(expected.shape, gamma_n, dtype=np.float32))
        finite = bool(np.isfinite(value).all())
        error = float(np.max(np.abs(value.astype(np.float64) - expected)))
        # Native FP32 n-step accumulation can round in a different order.
        accepted = value.shape == expected.shape and finite and np.allclose(value, expected, atol=2e-7, rtol=2e-6)
        records[key] = {
            "shape": list(value.shape),
            "sha256": array_hash(value),
            "reference_sha256": array_hash(expected),
            "max_absolute_difference": error,
            "finite": finite,
            "accepted": bool(accepted),
        }
    return {"accepted": all(record["accepted"] for record in records.values()), "fields": records}


def timed(call, count, device):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(count):
        call()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / count


def clock_snapshot():
    """Sample container-visible devices outside the measured update interval."""
    fields = "uuid,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,power.limit,utilization.gpu"
    sample = {"observed_unix": time.time(), "scope": "Container-visible devices; outside update timing"}
    try:
        process = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        sample.update(status="RECORDED" if process.returncode == 0 else "UNAVAILABLE", exit_code=process.returncode)
        if process.returncode == 0:
            sample["devices"] = [
                dict(zip(fields.split(","), [item.strip() for item in row], strict=True))
                for row in csv.reader(process.stdout.splitlines())
            ]
        else:
            sample["reason"] = process.stderr.strip()[:500] or "nvidia-smi returned a nonzero exit code"
    except (OSError, subprocess.TimeoutExpired, csv.Error, ValueError) as error:
        sample.update(status="UNAVAILABLE", error=type(error).__name__, reason=str(error))
    return sample


def audit_parameters(initial, final):
    groups = {}
    for name in ("actor", "critic", "temperature"):
        keys = [key for key in initial if key.startswith(name + ".")]
        finite = all(np.isfinite(final[key]).all() for key in keys)
        changed = sum(not np.array_equal(initial[key], final[key]) for key in keys)
        groups[name] = {
            "parameter_arrays": len(keys),
            "changed_parameter_arrays": changed,
            "finite": bool(finite),
            "updated": changed > 0,
            "maximum_absolute_change": max(
                float(np.max(np.abs(final[key].astype(np.float64) - initial[key]))) for key in keys
            ),
        }
    return {"accepted": all(row["finite"] and row["updated"] for row in groups.values()), "groups": groups}


def run_backend(agent, name, args, transitions, common_weights, reference_replay, result, artifact_root, startup):
    device = torch.device(args.device)
    native = isinstance(agent, WarpFlashSAC)
    cfg = agent.cfg if native else agent._cfg
    row = {
        "backend": name,
        "status": "PREPARING",
        "precision": "fp16_dense_fp32_state" if native and cfg.use_amp else "fp16_amp" if cfg.use_amp else "fp32",
        "fused_updates": native and agent.fused_updates,
        "cuda_autotune": bool(native and getattr(agent, "cuda_autotune", False)),
        "resolved_config": asdict(cfg),
        "network_compile": not native and cfg.use_compile,
        "captured_update": native,
        "metric_cadence": "Every update returns the same 7 actor-phase / 2 critic-phase scalar metrics on CPU.",
        "actor_timing_scope": "Public action API, normalized actions copied to a CPU NumPy array on every call.",
        "block_protocol": (
            "Continuous learner/optimizer/RNG state; each block has its own warmup. "
            "Blocks describe timing variation and are not independent seeds."
        ),
        "startup_seconds_scope": "Construction/loading/ingestion/preparation plus first block warmup only.",
        "blocks": [],
        **startup,
    }
    result["records"].append(row)
    write_json(args.output, result)
    start = time.perf_counter()
    row["common_initial_weights_audit"] = load_common_weights(agent, common_weights)
    torch.cuda.synchronize(device)
    row["common_weight_loading_seconds"] = time.perf_counter() - start
    initial = parameter_state(agent)
    np.savez(artifact_root / f"{name}-initial-parameters.npz", **initial)
    start = time.perf_counter()
    flag_owners = []
    for transition in transitions:
        if native:
            flags = {key: transition[key].to(torch.int32) for key in ("terminated", "truncated")}
            flag_owners.append(flags)
            transition = {
                key: wp.from_torch(flags[key], dtype=wp.int32)
                if key in flags
                else wp.from_torch(value, dtype=wp.float32)
                for key, value in transition.items()
            }
        agent.process_transition(transition)
    torch.cuda.synchronize(device)
    row["replay_fill_seconds"] = time.perf_counter() - start
    population = replay_state(agent)
    row["replay_transitions"] = len(agent._replay_buffer)
    if reference_replay is None:
        reference_replay = population
    row["replay_audit"] = audit_replay(population, reference_replay, cfg.gamma**cfg.n_step)
    write_json(args.output, result)
    if not row["replay_audit"]["accepted"]:
        raise RuntimeError(f"Shared replay audit failed: {name}")
    start = time.perf_counter()
    if native:
        agent.prepare(capture=True)
    torch.cuda.synchronize(device)
    row["preparation_seconds"] = time.perf_counter() - start
    row["initial_optimizer_state"] = fresh_optimizer_state(agent)
    observations = transitions[-1]["next_observation"]
    native_observations = wp.from_torch(observations, dtype=wp.float32) if native else None

    def actor():
        if native:
            return agent.act(native_observations).numpy()
        if name == "official_flashsac":
            return agent.sample_actions(0, {"next_observation": observations}, training=True)
        return agent.act(observations).detach().cpu().numpy()

    last_metrics = {}
    metric_keys = set()

    def update():
        nonlocal last_metrics
        last_metrics = agent.update()  # Preserve default per-call scalar synchronization for all three APIs.
        metric_keys.add(tuple(sorted(last_metrics)))
        return last_metrics

    torch.manual_seed(args.seed + 100)
    measured_metric_keys = set()
    for block_index in range(args.blocks):
        block = {"index": block_index, "status": "WARMUP", "warmup_update_calls": args.warmup}
        row["blocks"].append(block)
        write_json(args.output, result)
        start = time.perf_counter()
        for _ in range(args.warmup):
            actor()
            update()
        torch.cuda.synchronize(device)
        block["warmup_seconds"] = time.perf_counter() - start
        if args.require_complete_updates:
            warmup_calls = block_index * (args.warmup + args.updates) + args.warmup
            block["post_warmup_budget"] = completed_update_budget(agent, warmup_calls, cfg.actor_update_period)
            write_json(args.output, result)
            if not block["post_warmup_budget"]["accepted"]:
                raise RuntimeError(f"Incomplete warmup optimizer/update budget: {name}")
        block["actor_seconds_per_call"] = timed(actor, args.actor_calls, device)
        steps_before = optimizer_steps(agent)
        amp_before = native_amp_state(agent) if native else {}
        device_counters_before = agent.update_counters.numpy().copy() if native else None
        first_phase = agent._update_step
        metric_keys.clear()
        if args.record_clocks:
            block["gpu_clocks_before"] = clock_snapshot()
        block["update_seconds_per_call"] = timed(update, args.updates, device)
        if args.record_clocks:
            block["gpu_clocks_after"] = clock_snapshot()
        steps_after = optimizer_steps(agent)
        actor_calls = sum((first_phase + index) % cfg.actor_update_period == 0 for index in range(args.updates))
        attempts = {"actor": actor_calls, "critic": args.updates, "temperature": actor_calls}
        actual = {key: steps_after[key] - steps_before[key] for key in steps_before}
        block.update(
            attempted_optimizer_updates=attempts,
            actual_optimizer_updates=actual,
            skipped_optimizer_updates={key: attempts[key] - actual[key] for key in attempts},
            optimizer_steps_before=steps_before,
            optimizer_steps_after=steps_after,
            agent_update_calls_before=first_phase,
            agent_update_calls_after=agent._update_step,
            exact_update_call_count=agent._update_step - first_phase == args.updates,
            final_metrics={key: float(value) for key, value in last_metrics.items()},
            metric_key_sets=[list(keys) for keys in sorted(metric_keys)],
            status="MEASURED",
        )
        if native:
            device_counters_after = agent.update_counters.numpy().copy()
            block["device_update_counters_before_critic_actor_temperature"] = device_counters_before.tolist()
            block["device_update_counters_after_critic_actor_temperature"] = device_counters_after.tolist()
            block["exact_device_update_counts"] = bool(
                np.array_equal(device_counters_after - device_counters_before, [args.updates, actor_calls, actor_calls])
            )
            amp_after = native_amp_state(agent)
            block["amp_state_after"] = amp_after
            block["amp_counters_match_optimizer_counts"] = all(
                amp_after[key]["attempted_steps"] - amp_before[key]["attempted_steps"] == attempts[key]
                and amp_after[key]["skipped_steps"] - amp_before[key]["skipped_steps"] == attempts[key] - actual[key]
                for key in amp_after
            )
        if args.require_complete_updates:
            total_calls = (block_index + 1) * (args.warmup + args.updates)
            block["post_measurement_budget"] = completed_update_budget(agent, total_calls, cfg.actor_update_period)
            write_json(args.output, result)
            if not block["post_measurement_budget"]["accepted"]:
                raise RuntimeError(f"Incomplete measured optimizer/update budget: {name}")
        measured_metric_keys.update(metric_keys)
        write_json(args.output, result)
    attempts = {
        key: sum(block["attempted_optimizer_updates"][key] for block in row["blocks"])
        for key in ("actor", "critic", "temperature")
    }
    actual = {
        key: sum(block["actual_optimizer_updates"][key] for block in row["blocks"])
        for key in ("actor", "critic", "temperature")
    }
    final = parameter_state(agent)
    np.savez(artifact_root / f"{name}-final-parameters.npz", **final)
    if native and agent.fused_updates and hasattr(agent, "_zero_plan_status"):
        row["gradient_zero_plans"] = {key: dict(value) for key, value in agent._zero_plan_status.items()}
        if set(row["gradient_zero_plans"]) != {"actor", "critic", "temperature"} or not all(
            value["mode"] == "native" for value in row["gradient_zero_plans"].values()
        ):
            raise RuntimeError("Fused native benchmark requires active grouped gradient clears for all phases")
    if native:
        if row["cuda_autotune"] != args.cuda_autotune:
            raise RuntimeError("Native learner autotuning selection differs from the requested flag")
        metadata = getattr(agent, "cuda_tuning_metadata", None)
        row["cuda_tuning_metadata"] = metadata() if metadata is not None else {}
        if args.cuda_autotune and not row["cuda_tuning_metadata"]:
            raise RuntimeError("Requested native autotuning has no recorded plan metadata")
    row.update(
        actor_seconds_per_call=float(np.median([block["actor_seconds_per_call"] for block in row["blocks"]])),
        update_seconds_per_call=float(np.median([block["update_seconds_per_call"] for block in row["blocks"]])),
        timing_aggregation="Median of block means; raw synchronized block means are retained.",
        attempted_optimizer_updates=attempts,
        actual_optimizer_updates=actual,
        skipped_optimizer_updates={key: attempts[key] - actual[key] for key in attempts},
        all_attempted_optimizer_updates_completed=attempts == actual,
        final_metrics={key: float(value) for key, value in last_metrics.items()},
        metric_key_sets=[list(keys) for keys in sorted(measured_metric_keys)],
        learned_parameters=audit_parameters(initial, final),
        checkpoint_directory=str(artifact_root / name),
        warmup_seconds=row["blocks"][0]["warmup_seconds"],
        additional_block_warmup_seconds=sum(block["warmup_seconds"] for block in row["blocks"][1:]),
    )
    row["startup_seconds"] = sum(
        row[key]
        for key in (
            "construction_seconds",
            "common_weight_loading_seconds",
            "replay_fill_seconds",
            "preparation_seconds",
            "warmup_seconds",
        )
    )
    finite_metrics = all(np.isfinite(value) for block in row["blocks"] for value in block["final_metrics"].values())
    expected_key_sets = {
        ("critic/loss", "critic/max_entropy_bonus"),
        (
            "actor/entropy",
            "actor/loss",
            "actor/mean_action",
            "critic/loss",
            "critic/max_entropy_bonus",
            "temperature/loss",
            "temperature/value",
        ),
    }
    row["checks"] = {
        "finite_metrics": bool(finite_metrics),
        "finite_updated_parameters": row["learned_parameters"]["accepted"],
        "same_scalar_metric_cadence": measured_metric_keys == expected_key_sets,
        "agent_update_calls": all(block["exact_update_call_count"] for block in row["blocks"]),
        "native_device_update_calls": all(block.get("exact_device_update_counts", True) for block in row["blocks"]),
        "valid_optimizer_counts": all(0 <= actual[key] <= attempts[key] for key in attempts),
        "all_optimizer_groups_made_measured_progress": all(actual[key] > 0 for key in attempts),
        "temperature_updates_completed": actual["temperature"] == attempts["temperature"],
        "fp32_updates_completed": cfg.use_amp or attempts == actual,
        "native_amp_counters": all(block.get("amp_counters_match_optimizer_counts", True) for block in row["blocks"]),
    }
    row["status"] = "PASSED" if all(row["checks"].values()) else "FAILED"
    write_json(args.output, result)
    if row["status"] != "PASSED":
        raise RuntimeError(f"Learner/count/metric checks failed: {name}")
    # Native format, optimizer moments, BN state and counters are preserved;
    # replay fixture is stored separately rather than duplicating its capacity.
    agent.save(str(artifact_root / name))
    row["checkpoint_saved"] = True
    if native and args.profile_native:
        row["profile"] = profile_native(agent, artifact_root / (name + "-trace.json"), args.updates)
    write_json(args.output, result)
    return reference_replay


def native_amp_state(agent):
    return {
        name: {key: value.numpy().tolist()[0] for key, value in optimizer.state_arrays().items()}
        for name, optimizer in (("actor", agent.actor_optimizer), ("critic", agent.critic_optimizer))
        if optimizer.dynamic_loss_scale
    }


def profile_native(agent, path, measured_calls):
    """Record device kernels outside timing, restoring the entire learner afterward."""
    before = agent.state_dict(include_replay=False)
    trace = {"calls": min(8, measured_calls), "included_in_update_timing": False}
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            with torch.profiler.record_function("native_flashsac_graph_updates"):
                for _ in range(trace["calls"]):
                    agent.update()
                wp.synchronize_device(agent.device)
        profiler.export_chrome_trace(str(path))
        events = json.loads(path.read_text()).get("traceEvents", [])
        totals = defaultdict(lambda: {"calls": 0, "microseconds": 0.0})
        for event in events:
            if event.get("cat") == "kernel" and event.get("ph") == "X":
                group = totals[event["name"]]
                group["calls"] += 1
                group["microseconds"] += event.get("dur", 0.0)
        trace["device_kernel_totals"] = sorted(totals.items(), key=lambda row: row[1]["microseconds"], reverse=True)
        trace["status"] = "RECORDED" if totals else "NO_DEVICE_KERNEL_EVENTS"
    except Exception as error:
        # Profiling availability is separate from numerical/optimizer validity.
        trace.update(status="UNAVAILABLE", error=f"{type(error).__name__}: {error}")
    finally:
        agent.load_state_dict(before)
    after = agent.state_dict(include_replay=False)
    trace["learner_state_restored_exactly"] = all(np.array_equal(value, after[key]) for key, value in before.items())
    if not trace["learner_state_restored_exactly"]:
        raise RuntimeError("Profiling changed the recorded learner state after restoration")
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, default=os.environ.get("OFFICIAL_FLASHSAC_ROOT"))
    parser.add_argument("--recipe", choices=("fp32", "amp"), default="fp32")
    parser.add_argument("--compile-fp32", action="store_true", help="Compile Torch networks without enabling AMP.")
    parser.add_argument(
        "--backend",
        choices=("all", "official", "torch-port", "native"),
        default="all",
        help="Select one learner for a fresh-process worker; the default compares all learners.",
    )
    parser.add_argument("--cuda-autotune", action="store_true", help="Opt into native CUDA dense-plan autotuning.")
    parser.add_argument("--record-clocks", action="store_true", help="Sample GPU clocks outside each timed block.")
    parser.add_argument(
        "--require-complete-updates",
        action="store_true",
        help="Require all warmup and measured optimizer attempts to complete without AMP skips.",
    )
    parser.add_argument(
        "--native-variants", nargs="+", choices=("fp32", "fp32-fused", "amp", "amp-fused"), default=["fp32"]
    )
    parser.add_argument(
        "--profile-native", action="store_true", help="Export an untimed CUDA kernel trace per native row."
    )
    parser.add_argument(
        "--initial-weights", type=Path, help="Load a common-initial-weights.npz export for exact initialization reuse."
    )
    parser.add_argument(
        "--only-native",
        action="store_true",
        help="Time native variants only; construct pristine official FlashSAC for weights and replay reference.",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Use authors' launcher Torch TF32/high/cuDNN-benchmark policy; native stays FP32.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--observations", type=int, default=123)
    parser.add_argument("--actions", type=int, default=37)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=None, help="Per block: default 10 for FP32, 20 for AMP.")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--actor-calls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.only_native and args.backend not in ("all", "native"):
        parser.error("--only-native cannot be combined with a different --backend selection.")
    selected = "native" if args.only_native else args.backend
    if args.cuda_autotune and selected in ("official", "torch-port"):
        parser.error("--cuda-autotune requires a native backend row.")
    if args.warmup is None:
        args.warmup = 20 if args.recipe == "amp" else 10
    if args.official_root is None:
        parser.error("Pass --official-root or set OFFICIAL_FLASHSAC_ROOT.")
    if (
        min(args.observations, args.actions, args.num_envs, args.batch_size, args.warmup, args.actor_calls, args.blocks)
        < 1
    ):
        parser.error("Dimensions and warmup/actor counts must be positive.")
    if args.updates < 2:
        parser.error("At least two measured updates are needed to check both actor phases.")
    artifact_root = args.output.parent / (args.output.stem + "-artifacts")
    artifact_root.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "RUNNING",
        "scope": "Synthetic standalone learner components; no physics, rollout learning, or policy-quality results.",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "backend_selection": selected,
        "benchmark_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "records": [],
        "timing": "Synchronized wall time amortized over calls; default scalar metrics on every update in all rows.",
        "startup_scope": (
            "Construction, common-weight loading, replay ingestion, preparation, and warmup are separate. "
            "Shared fixture generation/export, source import/verification, CPU replay audits, and checkpoint "
            "writing are excluded. Compiler caches are retained; "
            + (
                "only the selected learner is timed; official construction/loading/ingestion supplies the reference."
                if selected != "all"
                else "fixed row order is official, port, native."
            )
        ),
        "limitations": [
            "No terminal or timeout fixtures: upstream and RoboLearn timeout-discount semantics differ.",
            "Common replay population and initial weights do not imply identical RNG draws or trained trajectories.",
            "Torch FP32 network compilation is controlled by --compile-fp32; official helpers always compile.",
            "Native FP16 dense operands retain FP32 accumulation, outputs, activations, master weights and Adam.",
            "Native mixed precision therefore differs from Torch autocast intermediate rounding.",
            "Per-call CPU action copies are included in every actor row; simulator/device-only rollout performance is unmeasured.",
            "Single-process descriptive timings include Python/default logging overhead and cache/order effects.",
            "Timing blocks continue from the previous block and are not independent training seeds.",
            (
                "Torch allows TF32/high matmul as in the authors' launcher; native FP32 remains pedantic."
                if args.allow_tf32
                else "Torch TF32 is disabled/highest matmul for the controlled FP32 comparison."
            ),
            "GEMM/reduction orders differ; equal nominal precision does not establish bitwise IEEE equivalence.",
        ],
    }
    write_json(args.output, result)
    try:
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.benchmark = args.allow_tf32
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("This component comparison requires a CUDA device.")
        device = torch.device(args.device)
        official_module, result["official_provenance"] = import_official(args.official_root)
        result["runtime"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "warp": wp.__version__,
            "warp_nn": warp_nn.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "native_compute_types": {"fp32": "CUBLAS_COMPUTE_32F_PEDANTIC", "fp16": "CUBLAS_COMPUTE_32F"},
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "benchmark_dependencies": {
                name: importlib.metadata.version(name) for name in ("gymnasium", "omegaconf", "jax", "jaxlib")
            },
        }
        package_file = Path(importlib.import_module("robolearn").__file__).resolve(strict=True)
        package_root = package_file.parent
        expected_root = os.environ.get("ROBOLEARN_EXPECTED_PACKAGE_ROOT")
        result["robolearn_provenance"] = {
            "imported_package_file": str(package_file),
            "imported_package_root": str(package_root),
            "expected_library_commit": os.environ.get("ROBOLEARN_LIBRARY_COMMIT"),
            "commit_provenance": "Caller-declared ROBOLEARN_LIBRARY_COMMIT; imported source hashes are recorded separately.",
            "expected_package_root": expected_root,
            "expected_package_root_matches": (
                package_root == Path(expected_root).expanduser().resolve(strict=True) if expected_root else None
            ),
        }
        if result["robolearn_provenance"]["expected_package_root_matches"] is False:
            raise RuntimeError("Imported RoboLearn package differs from ROBOLEARN_EXPECTED_PACKAGE_ROOT.")
        result["robolearn_sources_sha256"] = {
            str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package_root.rglob("*"))
            if path.suffix in (".py", ".cu", ".cuh") and path.is_file()
        }
        wp.init()
        cfg = FlashSACConfig(
            device=args.device,
            seed=args.seed,
            buffer_max_length=max(65536, 16 * args.num_envs),
            buffer_min_length=args.num_envs,
            sample_batch_size=args.batch_size,
            use_compile=args.recipe == "amp" or args.compile_fp32,
            use_amp=args.recipe == "amp",
            learning_rate_decay_step=98400,
        )
        result["config"] = asdict(cfg)
        config = asdict(cfg)
        config["device_type"] = config.pop("device")
        config["buffer_device_type"] = config.pop("buffer_device") or args.device
        if set(config) != {field.name for field in fields(official_module.FlashSACConfig)}:
            raise RuntimeError("Official and RoboLearn configuration fields do not map exactly.")
        official_cfg = official_module.FlashSACConfig(**config)
        torch.manual_seed(args.seed)
        start = time.perf_counter()
        official = official_module.FlashSACAgent(
            gym.spaces.Box(-np.inf, np.inf, shape=(args.num_envs, args.observations), dtype=np.float32),
            gym.spaces.Box(-1.0, 1.0, shape=(args.num_envs, args.actions), dtype=np.float32),
            {"actor_observation_size": (args.observations,)},
            official_cfg,
        )
        torch.cuda.synchronize(device)
        official_startup = {"construction_seconds": time.perf_counter() - start}
        common_weights = torch_state(official)
        if args.initial_weights is not None:
            common_weights, result["common_initial_weights_source"] = initial_weights(
                args.initial_weights, common_weights
            )
            load_common_weights(official, common_weights)
        else:
            result["common_initial_weights_source"] = {"kind": "pristine_official_initialization"}
        np.savez(artifact_root / "common-initial-weights.npz", **common_weights)
        result["common_initial_weights"] = {key: array_hash(value) for key, value in common_weights.items()}
        rng = np.random.default_rng(args.seed + 1000)
        fixture = []
        for _ in range(16):
            fixture.append(
                {
                    "observation": rng.standard_normal((args.num_envs, args.observations), dtype=np.float32),
                    "next_observation": rng.standard_normal((args.num_envs, args.observations), dtype=np.float32),
                    "action": rng.uniform(-1, 1, (args.num_envs, args.actions)).astype(np.float32),
                    "reward": rng.standard_normal(args.num_envs, dtype=np.float32) * np.float32(0.1),
                    "terminated": np.zeros(args.num_envs, dtype=np.bool_),
                    "truncated": np.zeros(args.num_envs, dtype=np.bool_),
                }
            )
        flat_fixture = {
            f"{index}.{key}": value for index, transition in enumerate(fixture) for key, value in transition.items()
        }
        np.savez_compressed(artifact_root / "transition-fixture.npz", **flat_fixture)
        result["transition_fixture"] = {key: array_hash(value) for key, value in flat_fixture.items()}
        transitions = [{key: torch.as_tensor(value, device=device) for key, value in row.items()} for row in fixture]
        result["resolved_configs"] = {"official": asdict(official._cfg)}
        if selected in ("torch-port", "native"):
            start = time.perf_counter()
            for transition in transitions:
                official.process_transition(transition)
            torch.cuda.synchronize(device)
            reference_replay = replay_state(official)
            reference_steps = optimizer_steps(official)
            result["official_reference"] = {
                **official_startup,
                "replay_fill_seconds": time.perf_counter() - start,
                "replay_transitions": len(official._replay_buffer),
                "replay_hashes": {key: array_hash(value) for key, value in reference_replay.items()},
                "agent_update_calls": official._update_step,
                "optimizer_steps": reference_steps,
                "timed_or_warmup_updates": False,
            }
            if official._update_step != 0 or any(reference_steps.values()):
                raise RuntimeError("Reference-only official learner unexpectedly updated.")
        else:
            reference_replay = run_backend(
                official,
                "official_flashsac",
                args,
                transitions,
                common_weights,
                None,
                result,
                artifact_root,
                official_startup,
            )
        if selected in ("all", "torch-port"):
            start = time.perf_counter()
            port = FlashSAC(args.observations, args.actions, num_envs=args.num_envs, cfg=cfg)
            torch.cuda.synchronize(device)
            port_startup = {"construction_seconds": time.perf_counter() - start}
            run_backend(
                port,
                "robolearn_torch_port",
                args,
                transitions,
                common_weights,
                reference_replay,
                result,
                artifact_root,
                port_startup,
            )
            result["resolved_configs"]["robolearn_torch"] = asdict(port._cfg)
        source_root = Path(importlib.import_module("robolearn.flashsac.agent").__file__).parent
        result["robolearn_compile_sites"] = {
            str(path.relative_to(source_root)): compile_sites(path)
            for path in source_root.rglob("*.py")
            if "torch.compile" in path.read_text()
        }
        if selected in ("all", "native"):
            stream = wp.Stream(args.device)
            mirror = wp.stream_to_torch(stream)
            stream.wait_stream(wp.get_stream(args.device))
            mirror.wait_stream(torch.cuda.current_stream(device))
            with wp.ScopedStream(stream, sync_enter=False), torch.cuda.stream(mirror):
                for variant in dict.fromkeys(args.native_variants):
                    start = time.perf_counter()
                    native = WarpFlashSAC(
                        args.observations,
                        args.actions,
                        num_envs=args.num_envs,
                        cfg=replace(cfg, use_compile=False, use_amp=variant.startswith("amp")),
                        dense_backend="cuda",
                        fused_updates=variant.endswith("-fused"),
                        **({"cuda_autotune": True} if args.cuda_autotune else {}),
                    )
                    torch.cuda.synchronize(device)
                    native_startup = {"construction_seconds": time.perf_counter() - start}
                    name = (
                        "robolearn_native_cuda"
                        if variant == "fp32"
                        else "robolearn_native_cuda_" + variant.replace("-", "_")
                    )
                    run_backend(
                        native,
                        name,
                        args,
                        transitions,
                        common_weights,
                        reference_replay,
                        result,
                        artifact_root,
                        native_startup,
                    )
                    result["resolved_configs"][name] = asdict(native.cfg)
        result["status"] = "PASSED"
        write_json(args.output, result)
        print(json.dumps({"status": result["status"], "output": str(args.output)}, allow_nan=False))
    except BaseException as error:
        result["status"] = "FAILED"
        result["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        write_json(args.output, result)
        raise


if __name__ == "__main__":
    main()
