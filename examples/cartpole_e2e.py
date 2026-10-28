# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Run a frozen Isaac Lab Cartpole learning comparison.

Use an installed Isaac Lab environment, the reviewed protocol JSON and pristine
FlashSAC source. ``fixture`` exports shared initial policies and evaluation state;
``smoke`` exercises each selected learner before ``run`` starts fresh training
and evaluation processes in the allocation's prescribed order. Smokes never enter production
results. Newton physics and native learning use separate graphs; the Torch MDP
and collection loop are not one captured physics/learning graph.

RSL-RL executes its actual PPO implementation. FlashSAC executes the authors'
unmodified agent (87edc906) or RoboLearn's experimental native implementation.
Algorithm credit: Schulman et al., arXiv:1707.06347; Kim et al., arXiv:2604.04539.
The driver owns experiment files; it adds no simulator dependency to the library.
Protocols can select official multi-step replay and a full-budget LR horizon;
native multi-step timeout semantics remain a separate comparison. Full-budget
decay uses the declared production budget even during a short smoke.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

PROCESS_STARTED = time.perf_counter()
RECIPES = ("rsl_ppo", "cuda_ppo", "official_flashsac", "cuda_flashsac")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def flash_recipe_settings(spec):
    """Resolve explicit replay and schedule choices without changing old defaults."""
    settings = spec["recipes"]["official_flashsac"]
    n_step = settings.get("n_step", 1)
    require(type(n_step) is int and n_step > 0, "FlashSAC n_step must be a positive integer")
    require(
        n_step == 1 or spec["schedule"]["recipe_order"] == ["official_flashsac"],
        "Multi-step official/native timeout discounts differ; select official FlashSAC alone for this recipe",
    )
    schedule = settings.get("learning_rate_schedule", "fixed")
    require(schedule in ("fixed", "full_budget"), "Unknown FlashSAC learning-rate schedule")
    decay = settings["learning_rate_decay_step"]
    if schedule == "full_budget":
        expected = (
            spec["budget"]["iterations"]
            * spec["budget"]["rollout_steps"]
            * settings["critic_update_calls_per_vector_step_after_warmup"]
        )
        require(decay == expected, "Full-budget decay must match nominal vector steps times critic update calls")
    require(type(decay) is int and decay > 0, "FlashSAC decay must be a positive update count")
    return {"n_step": n_step, "learning_rate_schedule": schedule, "learning_rate_decay_step": decay}


def read_spec(args):
    spec = json.loads(args.protocol.read_text())
    require(spec["mdp"]["task"] == "Isaac-Cartpole", "This example implements the reviewed manager Cartpole contract")
    require(spec["mdp"]["action_boundary_clip"] == [-1.0, 1.0], "The common action boundary must clip to [-1,1]")
    flash_recipe_settings(spec)
    recipes = spec["schedule"]["recipe_order"]
    require(
        bool(recipes)
        and len(recipes) == len(set(recipes))
        and set(recipes) == set(spec["recipes"])
        and all(recipe in RECIPES for recipe in recipes),
        "Protocol recipe population differs from the selected implementations",
    )
    require(
        all(sorted(row) == list(range(len(recipes))) for row in spec["schedule"]["williams_rows"]),
        "Every execution-order row must contain every selected recipe exactly once",
    )
    require(args.recipe is None or args.recipe in recipes, "Worker recipe is absent from the protocol")
    require(args.allocation in spec["schedule"], "Allocation is absent from the protocol")
    require(
        args.phase != "fixture" or "canonical_fixtures" not in spec,
        "This protocol reuses frozen canonical fixtures; regeneration is not permitted",
    )
    checkpointing = spec.get("checkpointing", {})
    if checkpointing.get("full_replay_in_final_continuation_checkpoint", False):
        require(
            checkpointing["continuation_iteration"] == spec["budget"]["iterations"]
            and spec["budget"]["iterations"] in spec["budget"]["checkpoint_iterations"]
            and all(recipe.endswith("flashsac") for recipe in recipes),
            "Full replay export requires the final scored FlashSAC checkpoint",
        )
        name = checkpointing["continuation_directory_name"]
        require(name not in ("", ".", "..") and Path(name).name == name, "Unsafe continuation directory name")
    replay_sample_indices(args, spec)
    return spec


def tree_map(value, function):
    return (
        {key: tree_map(item, function) for key, item in value.items()} if isinstance(value, dict) else function(value)
    )


def array_hash(value):
    """Match benchmark_official_flashsac: SHA256(dtype text + shape text + C bytes)."""
    import numpy as np

    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def state_hashes(state):
    return {key: array_hash(value) for key, value in state.items()}


def replay_sample_indices(args, spec):
    """Read prespecified replay slots without consuming a training RNG stream."""
    import numpy as np

    checkpointing = spec.get("checkpointing", {})
    settings = checkpointing.get("final_replay_sample", {})
    if not settings.get("enabled", False):
        return None
    require(
        spec["schedule"]["recipe_order"] == ["official_flashsac"]
        and not checkpointing.get("full_replay_in_final_continuation_checkpoint", False)
        and settings["iteration"] == spec["budget"]["iterations"]
        and settings["iteration"] in spec["budget"]["checkpoint_iterations"],
        "Bounded replay export requires the final official-only policy checkpoint without full replay export",
    )
    name = settings["indices_file"]
    require(name not in ("", ".", "..") and Path(name).name == name, "Unsafe replay indices filename")
    path = args.protocol.parent / name
    require(sha(path) == settings["indices_file_sha256"], "Replay indices file differs from the protocol")
    frozen = json.loads(path.read_text())
    require(
        frozen["population"] == settings["population"] == spec["recipes"]["official_flashsac"]["buffer_capacity"]
        and len(frozen["batches"]) == settings["batches"] == len(settings["replay_index_seeds"])
        and len(settings["indices_sha256"]) == settings["batches"]
        and len(set(settings["replay_index_seeds"])) == settings["batches"]
        and settings["tensor_fields"]
        == ["observation", "next_observation", "action", "reward", "terminated", "truncated"]
        and settings["tensor_dtype"] == "float32"
        and settings["observation_dimensions"] == spec["mdp"]["observation_dimensions"]
        and settings["action_dimensions"] == spec["mdp"]["action_dimensions"],
        "Replay sample population or batch count differs",
    )
    for batch, seed, digest in zip(
        frozen["batches"], settings["replay_index_seeds"], settings["indices_sha256"], strict=True
    ):
        require(all(type(value) is int for value in batch["indices"]), "Replay indices must be integers")
        indices = np.asarray(batch["indices"], dtype=np.int64)
        require(
            batch["seed"] == seed
            and indices.shape == (settings["batch_size"],)
            and bool(((indices >= 0) & (indices < frozen["population"])).all())
            and array_hash(indices) == batch["indices_sha256"] == digest,
            "Replay sample indices differ from their frozen identity",
        )
    return settings, frozen["batches"]


def export_replay_sample(learner, args, spec, iteration, transitions, saved):
    """Archive owned sampled arrays, not the learning buffer or a resume checkpoint."""
    import numpy as np
    import torch

    started = time.perf_counter()
    settings, batches = replay_sample_indices(args, spec)
    require(learner.recipe == "official_flashsac", "Only the pristine official replay sampler is supported")
    buffer = learner.agent._replay_buffer
    population, current = len(buffer), buffer._current_idx
    pending = len(buffer._n_step_transitions)
    require(
        population == settings["population"]
        and current == settings["expected_current_idx"]
        and pending == learner.config["n_step"],
        "Final live replay counters differ from the declared export",
    )
    cpu_rng, device_rng = torch.get_rng_state(), torch.cuda.get_rng_state(learner.device)
    directory = args.output / "final-replay-sample"
    directory.mkdir(parents=True, exist_ok=False)
    record = {
        "iteration": iteration,
        "transitions": transitions,
        "protocol_sha256": sha(args.protocol),
        "source_commit": spec["source"]["robolearn_measured_commit"],
        "path": str(directory.relative_to(args.output)),
        "model_hashes": saved["model_hashes"],
        "optimizer": saved["optimizer"],
        "configuration": learner.config,
        "flash_recipe_resolution": flash_recipe_settings(spec),
        "num_envs": learner.num_envs,
        "replay_num_in_buffer": population,
        "replay_current_idx": current,
        # The deque retains its oldest, already committed vector batch as well.
        "replay_deque_length": pending,
        "unsaved_pending_vector_steps": max(0, pending - 1),
        "indices_file_sha256": settings["indices_file_sha256"],
        "full_replay_included": False,
        "resumable": False,
        "scope": "Finite sampled slots only; unsampled storage, pending deque and global RNG are not archived",
        "batches": [],
    }
    with learner.scope():
        for batch in batches:
            arrays = {"indices": np.asarray(batch["indices"], dtype=np.int64)}
            # The authors' public sampler accepts explicit slots and does not draw RNG.
            selected = buffer.sample(arrays["indices"])
            require(set(selected) == set(settings["tensor_fields"]), "Unexpected official replay sample fields")
            for key, tensor in selected.items():
                array = tensor.detach().cpu().numpy().copy()
                width = {
                    "observation": settings["observation_dimensions"],
                    "next_observation": settings["observation_dimensions"],
                    "action": settings["action_dimensions"],
                }.get(key)
                shape = (settings["batch_size"], width) if width is not None else (settings["batch_size"],)
                require(
                    array.dtype == np.float32 and array.shape == shape and bool(np.isfinite(array).all()),
                    f"Invalid sampled replay array: {key}",
                )
                arrays[key] = array
            name = f"batch-{batch['seed']}.npz"
            np.savez(directory / name, **arrays)
            record["batches"].append(
                {
                    "seed": batch["seed"],
                    "indices_sha256": batch["indices_sha256"],
                    "file": name,
                    "arrays": {
                        key: {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": array_hash(value)}
                        for key, value in arrays.items()
                    },
                }
            )
    require(
        (len(buffer), buffer._current_idx, len(buffer._n_step_transitions)) == (population, current, pending)
        and state_hashes(learner.model()) == saved["model_hashes"]
        and learner.counts() == saved["optimizer"]
        and torch.equal(cpu_rng, torch.get_rng_state())
        and torch.equal(device_rng, torch.cuda.get_rng_state(learner.device)),
        "Replay sample export changed training state or RNG",
    )
    write_json(directory / "manifest.json", record)
    files = sorted(directory.iterdir())
    record["files_sha256"] = {path.name: sha(path) for path in files}
    record["files_bytes"] = {path.name: path.stat().st_size for path in files}
    record["write_seconds"] = time.perf_counter() - started
    record["process_write_finished_seconds"] = time.perf_counter() - PROCESS_STARTED
    return record


def finite(value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def same(left, right):
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def stock_config(task="Isaac-Cartpole"):
    from isaaclab_tasks.utils import load_cfg_from_registry

    return load_cfg_from_registry(task, "rsl_rl_cfg_entry_point").to_dict()


def make_rsl(num_envs, observations, actions, device, config):
    import torch
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from tensordict import TensorDict

    sample = TensorDict({"policy": torch.zeros((num_envs, observations), device=device)}, batch_size=[num_envs])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor_cfg, critic_cfg, algorithm_cfg = (copy.deepcopy(config[key]) for key in ("actor", "critic", "algorithm"))
    for item in (actor_cfg, critic_cfg, algorithm_cfg):
        item.pop("class_name")
    # PPO.construct_algorithm consumes this model-construction option before PPO.__init__.
    require(
        not algorithm_cfg.pop("share_cnn_encoders", None),
        "The state based comparison requires independent MLPs without shared CNN encoders",
    )
    actor_cfg["distribution_cfg"]["class_name"] = "rsl_rl.modules:GaussianDistribution"
    actor = MLPModel(sample, groups, "actor", actions, **actor_cfg)
    critic = MLPModel(sample, groups, "critic", 1, **critic_cfg)
    storage = RolloutStorage("rl", num_envs, config["num_steps_per_env"], sample, [actions], device)
    return PPO(actor, critic, storage, device=device, **algorithm_cfg)


def flash_config(spec, seed, device, smoke=False):
    from robolearn.flashsac import FlashSACConfig

    settings = spec["recipes"]["official_flashsac"]
    resolved = flash_recipe_settings(spec)
    return FlashSACConfig(
        device=device,
        seed=seed,
        gamma=settings["gamma"],
        n_step=resolved["n_step"],
        buffer_max_length=(
            spec.get("smoke", {}).get("flash_buffer_capacity", settings["buffer_capacity"])
            if smoke
            else settings["buffer_capacity"]
        ),
        buffer_min_length=(
            spec.get("smoke", {}).get("flash_buffer_warmup_transitions", 512)
            if smoke
            else settings["buffer_warmup_transitions"]
        ),
        sample_batch_size=(
            spec.get("smoke", {}).get("flash_sample_batch_size", 256) if smoke else settings["batch_size"]
        ),
        learning_rate_decay_step=resolved["learning_rate_decay_step"],
        use_compile=True,
        use_amp=True,
    )


def official_config(module, config):
    values = asdict(config)
    values["device_type"] = values.pop("device")
    values["buffer_device_type"] = values.pop("buffer_device") or values["device_type"]
    return module.FlashSACConfig(**values)


class Learner:
    """Small collector bridge; algorithm implementations and optimizers stay native."""

    def __init__(self, args, spec, num_envs, observations=4, actions=1, *, evaluation=False):
        import gymnasium as gym
        import numpy as np
        import torch
        import warp as wp
        from benchmark_official_flashsac import import_official
        from tensordict import TensorDict

        from robolearn.warp import PPOConfig, WarpFlashSAC, WarpPPO

        self.args, self.spec, self.recipe = args, spec, args.recipe
        self.is_ppo = self.recipe in ("rsl_ppo", "cuda_ppo")
        self.native = self.recipe.startswith("cuda")
        self.num_envs, self.device, self.TensorDict = num_envs, args.device, TensorDict
        self.stream = wp.get_stream(args.device) if self.native else None
        self.provenance = {}
        torch.manual_seed(args.seed)
        if self.is_ppo:
            self.config = stock_config(spec["mdp"]["task"])
            require(self.config["num_steps_per_env"] == spec["budget"]["rollout_steps"], "Stock PPO horizon differs")
            if self.native:
                algorithm, actor = self.config["algorithm"], self.config["actor"]
                config = PPOConfig(
                    hidden_dims=tuple(actor["hidden_dims"]),
                    activation=actor["activation"],
                    initial_std=actor["distribution_cfg"]["init_std"],
                    std_type=actor["distribution_cfg"]["std_type"],
                    std_range=tuple(actor["distribution_cfg"]["std_range"]),
                    epochs=algorithm["num_learning_epochs"],
                    num_mini_batches=algorithm["num_mini_batches"],
                    learning_rate=algorithm["learning_rate"],
                    schedule=algorithm["schedule"],
                    desired_kl=algorithm["desired_kl"],
                    gamma=algorithm["gamma"],
                    gae_lambda=algorithm["lam"],
                    clip_ratio=algorithm["clip_param"],
                    value_coefficient=algorithm["value_loss_coef"],
                    value_loss_scale=1.0,
                    entropy_coefficient=algorithm["entropy_coef"],
                    clip_value=algorithm["use_clipped_value_loss"],
                    max_grad_norm=algorithm["max_grad_norm"],
                    advantage_sample_std=True,
                    timeout_bootstrap="current",
                    separate_grad_clipping=True,
                    optimized_linear_backward=True,
                    dense_backend="cuda",
                    seed=args.seed,
                )
                self.agent = WarpPPO(
                    observations, actions, num_envs, self.config["num_steps_per_env"], config, args.device
                )
            else:
                self.agent = make_rsl(num_envs, observations, actions, args.device, self.config)
        else:
            config = flash_config(spec, args.seed, args.device, args.smoke)
            config.load_optimizer = not evaluation
            config.load_reward_normalizer = not evaluation
            if self.native:
                config.use_compile = False
                self.agent = WarpFlashSAC(
                    observations,
                    actions,
                    num_envs,
                    config,
                    dense_backend="cuda",
                    fused_updates=True,
                    fresh_training_gradients=True,
                )
            else:
                module, self.provenance = import_official(args.official_root)
                self.agent = module.FlashSACAgent(
                    gym.spaces.Box(-np.inf, np.inf, shape=(num_envs, observations), dtype=np.float32),
                    gym.spaces.Box(-1.0, 1.0, shape=(num_envs, actions), dtype=np.float32),
                    {"actor_observation_size": (observations,)},
                    official_config(module, config),
                )
            self.config = asdict(self.agent.cfg if self.native else self.agent._cfg)
        self.flags = {
            key: torch.empty(num_envs, dtype=torch.int32, device=args.device) for key in ("terminated", "truncated")
        }
        self.attempts = {"actor": 0, "critic": 0, "temperature": 0}

    @contextmanager
    def scope(self):
        import torch
        import warp as wp

        if self.native:
            require(
                bool(self.stream.cuda_stream)
                and torch.cuda.current_stream(self.device).cuda_stream == self.stream.cuda_stream,
                "Native collection and learning must use the GPU worker's shared Torch/Warp stream",
            )
            with wp.ScopedStream(self.stream, sync_enter=False):
                yield
        else:
            yield

    def obs(self, observations):
        return self.TensorDict({"policy": observations}, batch_size=[len(observations)])

    def actions(self, observations, deterministic=False):
        import torch
        import warp as wp

        with torch.no_grad(), self.scope():
            if self.recipe == "rsl_ppo":
                return (
                    self.agent.actor(self.obs(observations), stochastic_output=False)
                    if deterministic
                    else self.agent.act(self.obs(observations))
                )
            if self.recipe == "cuda_ppo":
                return wp.to_torch(self.agent.act(wp.from_torch(observations), deterministic=deterministic))
            if self.native:
                return wp.to_torch(self.agent.act(wp.from_torch(observations), training=not deterministic))
            # The pristine public API returns CPU NumPy actions; retain and record
            # its host transfer rather than replacing the authors' sampler.
            values = self.agent.sample_actions(0, {"next_observation": observations}, training=not deterministic)
            return torch.as_tensor(values, device=self.device)

    @property
    def ready(self):
        return self.is_ppo or (self.agent.ready if self.native else self.agent.can_start_training())

    def observe(self, before, after, transition, step):
        import torch
        import warp as wp

        with torch.no_grad(), self.scope():
            if self.recipe == "rsl_ppo":
                self.agent.process_env_step(
                    self.obs(after),
                    transition["reward"],
                    transition["terminated"] | transition["truncated"],
                    {"time_outs": transition["truncated"]},
                )
            elif self.native:
                for key, array in self.flags.items():
                    array.copy_(transition[key])
                values = {
                    key: wp.from_torch(self.flags[key] if key in self.flags else value)
                    for key, value in transition.items()
                }
                if self.is_ppo:
                    self.agent.store(
                        step,
                        wp.from_torch(before),
                        values["reward"],
                        values["terminated"],
                        values["truncated"],
                        values["next_observation"],
                    )
                else:
                    self.agent.process_transition(values)
            else:
                self.agent.process_transition(transition)

    def update(self, last_observations=None):
        import torch

        with self.scope():
            if self.is_ppo:
                if self.native:
                    self.agent.update(capture=True)
                    metrics = self.agent.health_metrics()
                else:
                    with torch.no_grad():
                        self.agent.compute_returns(self.obs(last_observations))
                    metrics = self.agent.update()
                count = self.config["algorithm"]["num_learning_epochs"] * self.config["algorithm"]["num_mini_batches"]
                self.attempts["actor"] += count
                self.attempts["critic"] += count
            else:
                call = self.attempts["critic"]
                metrics = self.agent.update()
                self.attempts["critic"] += 1
                if call % self.spec["recipes"]["official_flashsac"]["actor_update_period"] == 0:
                    self.attempts["actor"] += 1
                    self.attempts["temperature"] += 1
        require(finite(metrics), "Nonfinite learner metric")
        return metrics

    def counts(self):
        from benchmark_official_flashsac import optimizer_steps
        from benchmark_rsl_ppo_cuda import rsl_steps

        if self.is_ppo:
            values = [int(self.agent.optimizer._timestep.numpy()[0])] if self.native else rsl_steps(self.agent)
            require(len(set(values)) == 1, "PPO optimizer steps differ across parameters")
            actual = {"actor": values[0], "critic": values[0], "temperature": 0}
        else:
            actual = optimizer_steps(self.agent)
        skipped = {key: self.attempts[key] - actual[key] for key in self.attempts}
        require(all(value >= 0 for value in skipped.values()), "Completed optimizer count exceeds attempts")
        return {"attempted": dict(self.attempts), "completed": actual, "skipped": skipped}

    def model(self):
        from benchmark_official_flashsac import torch_state
        from benchmark_rsl_ppo_cuda import model_arrays

        if not self.is_ppo:
            if self.native:
                values = {
                    f"{name}.{key}": value.numpy().copy()
                    for name, network in (
                        ("actor", self.agent.actor),
                        ("critic", self.agent.critic),
                        ("target_critic", self.agent.target_critic),
                    )
                    for key, value in network.state_dict().items()
                }
                values["log_temperature"] = self.agent.log_temperature.numpy().copy()
                return values
            return torch_state(self.agent)
        if not self.native:
            return model_arrays(self.agent)
        values = {}
        for name, model in (("actor", self.agent.actor), ("critic", self.agent.critic)):
            for index, parameter in enumerate(model.parameters()):
                values[f"{name}_{index // 2}_{'weight' if index % 2 == 0 else 'bias'}"] = parameter.numpy().copy()
        values["std"] = self.agent.distribution_parameter.numpy().copy()
        return values

    def initialize(self, fixture):
        import numpy as np
        import torch
        from benchmark_official_flashsac import load_common_weights

        with np.load(fixture, allow_pickle=False) as archive:
            values = dict(archive)
        with self.scope():
            if not self.is_ppo:
                return load_common_weights(self.agent, values)
            if self.native:
                for name, model in (("actor", self.agent.actor), ("critic", self.agent.critic)):
                    for index, parameter in enumerate(model.parameters()):
                        parameter.assign(values[f"{name}_{index // 2}_{'weight' if index % 2 == 0 else 'bias'}"])
                self.agent.distribution_parameter.assign(values["std"])
            else:
                for name, model in (("actor", self.agent._raw_actor), ("critic", self.agent._raw_critic)):
                    layers = [item for item in model.mlp if isinstance(item, torch.nn.Linear)]
                    for index, layer in enumerate(layers):
                        layer.weight.data.copy_(torch.as_tensor(values[f"{name}_{index}_weight"], device=self.device))
                        layer.bias.data.copy_(
                            torch.as_tensor(values[f"{name}_{index}_bias"].reshape(-1), device=self.device)
                        )
                self.agent._raw_actor.distribution.std_param.data.copy_(
                    torch.as_tensor(values["std"], device=self.device)
                )
        require(state_hashes(self.model()) == state_hashes(values), "PPO paired initial states differ")
        return {"exactly_equal": True, "hashes": state_hashes(values)}

    def prepare(self):
        if not self.native:
            return
        with self.scope():
            if self.is_ppo:
                state = self.agent.state_dict()
                self.agent.warmup()
                self.agent.update(capture=True)
                self.agent.load_state_dict(state)
            else:
                self.agent.prepare(capture=True)

    def fresh(self):
        from benchmark_official_flashsac import fresh_gradient_snapshot, validate_fresh_gradient_snapshot

        if self.recipe != "cuda_flashsac":
            return {}
        value = fresh_gradient_snapshot(self.agent)
        validate_fresh_gradient_snapshot(value)
        return value

    def save(self, directory, replay=False):
        import torch

        directory.mkdir(parents=True, exist_ok=False)
        with self.scope():
            if self.recipe == "rsl_ppo":
                torch.save(self.agent.save(), directory / "agent.pt")
            elif self.recipe == "cuda_ppo":
                self.agent.save(directory / "agent.npz")
            elif self.native:
                self.agent.save(directory, include_replay=replay)
            else:
                self.agent.save(str(directory))
                if replay:
                    self.agent.save_replay_buffer(str(directory))
        write_json(directory / "collector.json", {"counts": self.counts(), "fresh": self.fresh()})
        return {path.name: sha(path) for path in sorted(directory.iterdir()) if path.is_file()}

    def load(self, directory, replay=False):
        import torch

        with self.scope():
            if self.recipe == "rsl_ppo":
                self.agent.load(
                    torch.load(directory / "agent.pt", map_location=self.device, weights_only=True), None, strict=True
                )
            elif self.recipe == "cuda_ppo":
                self.agent.load(directory / "agent.npz")
            else:
                self.agent.load(str(directory))
                if replay and not self.native:
                    self.agent.load_replay_buffer(str(directory))
        self.attempts = json.loads((directory / "collector.json").read_text())["counts"]["attempted"]


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        result.update(flatten(item, name) if isinstance(item, dict) else {name: item})
    return result


@contextmanager
def gpu_worker_stream(args):
    """Keep physics, Torch collection and learning on one capture-capable stream."""
    if args.phase not in ("fixture", "train-worker", "evaluate-worker"):
        yield
        return

    import torch
    import warp as wp

    wp.init()
    stream = wp.Stream(args.device)
    mirror = wp.stream_to_torch(stream)
    stream.wait_stream(wp.get_stream(args.device))
    mirror.wait_stream(torch.cuda.current_stream(args.device))
    with wp.ScopedStream(stream, sync_enter=False), torch.cuda.stream(mirror):
        try:
            yield
        finally:
            # Keep both owners alive until all work, including cleanup, completes.
            wp.synchronize_stream(stream)


def runtime(args):
    """Configure the authors'/Isaac Torch policy; native GEMMs stay pedantic."""
    import rsl_rl
    import torch
    import warp as wp

    import robolearn

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    wp.init()
    require(importlib.metadata.version("rsl-rl-lib") == "5.5.1", "This comparison requires stock RSL-RL 5.5.1")
    source_root = Path(__file__).resolve().parents[1]
    imported_root = Path(robolearn.__file__).resolve().parents[1]
    require(imported_root == source_root / "src", "The imported RoboLearn must belong to this driver checkout")
    inventory = {
        path.relative_to(source_root).as_posix(): sha(path)
        for path in sorted((source_root / "src" / "robolearn").rglob("*"))
        if path.is_file() and path.suffix in (".py", ".cu", ".cuh")
    }
    telemetry = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    return {
        "library_files_sha256": inventory,
        "rsl_rl_files_sha256": {
            path.relative_to(Path(rsl_rl.__file__).resolve().parent).as_posix(): sha(path)
            for path in sorted(Path(rsl_rl.__file__).resolve().parent.rglob("*.py"))
        },
        "driver_sha256": sha(Path(__file__)),
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "warp-lang", "warp-nn", "rsl-rl-lib", "numpy")
        },
        "gpu_telemetry": telemetry.stdout.strip(),
        "gpu_telemetry_exit_code": telemetry.returncode,
        "torch_allow_tf32": True,
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "native_dense_math": "FP32 pedantic; Flash AMP uses FP16 operands and FP32 output/accumulation",
        "official_runtime_limit": "Python 3.12 is outside upstream's declared Python >=3.10,<3.12 matrix; source is unmodified.",
    }


@contextmanager
def environment(args, spec, num_envs, seed):
    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab.app import launch_simulation
    from isaaclab.paths import ISAACLAB_ROOT
    from isaaclab_tasks.utils import parse_env_cfg

    from robolearn.isaaclab import IsaacLabEnv

    started = time.perf_counter()
    require(
        Path(isaaclab_tasks.__file__).resolve().is_relative_to(args.isaaclab_root),
        "Isaac Lab tasks must import from the frozen checkout",
    )
    require(
        Path(ISAACLAB_ROOT).resolve() == args.isaaclab_root,
        "The imported Isaac Lab root differs from the frozen checkout",
    )
    cfg = parse_env_cfg(
        spec["mdp"]["task"],
        device=args.device,
        num_envs=num_envs,
        overrides=["physics=newton_mjwarp", "visualizer=none"],
    )
    cfg.compute_final_obs = True
    cfg.seed = seed
    require(math.isclose(cfg.sim.dt, spec["mdp"]["physics_dt"]), "Unexpected stock physics timestep")
    require(cfg.decimation == spec["mdp"]["control_decimation"], "Unexpected stock control decimation")
    require(all(not viewer.backend for viewer in cfg.sim.visualizer_cfgs), "Headless comparison must disable viewers")
    task_root = args.isaaclab_root / "source/isaaclab_tasks/isaaclab_tasks/core/cartpole"
    task_hashes = {
        path.relative_to(args.isaaclab_root).as_posix(): sha(path) for path in sorted(task_root.rglob("*.py"))
    }
    asset_root = Path(os.environ["ISAACSIM_ASSET_ROOT"])
    asset = asset_root / spec["mdp"]["cartpole_asset_relative_path"]
    require(asset.is_file(), f"Missing approved Cartpole USD: {asset}")
    metadata = {
        "task": spec["mdp"]["task"],
        "physics": "newton_mjwarp",
        "renderer": "none",
        "visualizer": "none",
        "task_files_sha256": task_hashes,
        "runtime_contract_files_sha256": {
            name: sha(args.isaaclab_root / name)
            for name in (
                "uv.lock",
                "source/isaaclab_assets/isaaclab_assets/robots/cartpole.py",
                "source/isaaclab/isaaclab/envs/manager_based_rl_env.py",
                "source/isaaclab/isaaclab/envs/manager_based_env.py",
                "source/isaaclab/isaaclab/envs/mdp/observations.py",
                "source/isaaclab/isaaclab/envs/mdp/rewards.py",
                "source/isaaclab/isaaclab/envs/mdp/terminations.py",
                "source/isaaclab/isaaclab/envs/mdp/events.py",
            )
        },
        "expected_isaaclab_commit": spec["source"]["isaaclab_commit"],
        "asset_root_environment": str(asset_root),
        "resolved_asset": str(cfg.scene.robot.spawn.usd_path),
        "asset_files_sha256": {
            path.relative_to(asset_root).as_posix(): sha(path)
            for path in sorted(asset_root.rglob("*"))
            if path.is_file()
        },
        "stock_cfg_overrides": {
            "num_envs": num_envs,
            "compute_final_obs": True,
            "seed": seed,
            "clip_actions": 1.0,
            "init_at_random_ep_len": False,
        },
    }
    with launch_simulation(cfg):
        require(not cfg.sim.visualizer_cfgs, "Runtime preparation enabled an unexpected visualizer")
        env = gym.make(spec["mdp"]["task"], cfg=cfg).unwrapped
        wrapper = IsaacLabEnv(env, clip_actions=1.0)
        require((wrapper.observation_dim, wrapper.action_dim) == (4, 1), "Unexpected Cartpole MDP dimensions")
        metadata["startup_seconds"] = time.perf_counter() - started
        try:
            yield wrapper, metadata
        finally:
            env.close()


def fixture(args, spec):
    import isaaclab_tasks  # noqa: F401
    import numpy as np
    import torch
    from benchmark_official_flashsac import import_official, raw_module
    from benchmark_rsl_ppo_cuda import model_arrays

    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "PREPARING", "runtime": runtime(args), "policies": {}}
    module, official_provenance = import_official(args.official_root)
    metadata["official_source"] = official_provenance
    for seed in spec["budget"]["seeds"]:
        torch.manual_seed(seed)
        ppo = make_rsl(1, 4, 1, "cpu", stock_config())
        ppo_state = model_arrays(ppo)
        torch.manual_seed(seed)
        config = flash_config(spec, seed, "cpu")
        config.use_compile = config.use_amp = False
        networks = module._init_flashsac_networks(4, 4, 1, official_config(module, config), torch.device("cpu"))
        flash_state = {
            f"{name}.{key}": value.detach().numpy().copy()
            for name, network in zip(("actor", "critic", "target_critic", "temperature"), networks, strict=True)
            for key, value in raw_module(network).state_dict().items()
        }
        for name, state in (("ppo", ppo_state), ("flashsac", flash_state)):
            require(finite(state), "Nonfinite canonical initial state")
            filename = f"{name}-seed{seed}.npz"
            np.savez(args.output / filename, **state)
            metadata["policies"][filename] = {"sha256": sha(args.output / filename), "arrays": state_hashes(state)}
    with environment(args, spec, spec["evaluation"]["episodes"], spec["evaluation"]["common_seed"]) as (
        wrapper,
        env_metadata,
    ):
        wrapper.reset(seed=spec["evaluation"]["common_seed"])
        observations = wrapper.reset(seed=spec["evaluation"]["common_seed"])
        require(
            torch.count_nonzero(wrapper.env.scene["robot"].actuators.target_command.velocity.torch).item() == 0,
            "Ordinary effort-controlled Cartpole resets must keep a zero joint velocity target",
        )
        metadata["evaluation_zero_velocity_target_verified"] = True
        metadata["evaluation_observation_hash"] = array_hash(observations.cpu().numpy())
        state = tree_map(wrapper.env.scene.get_state(is_relative=True), lambda value: value.detach().cpu().clone())
        torch.save(state, args.output / "evaluation-state.pt")
        metadata["evaluation_state_hashes"] = state_hashes(
            {name: value.numpy() for name, value in flatten(state).items()}
        )
        metadata["environment"] = env_metadata
    metadata["status"] = "COMPLETED"
    metadata["protocol_sha256"] = sha(args.protocol)
    write_json(args.output / "fixtures.json", metadata)
    files = sorted(path for path in args.output.rglob("*") if path.is_file())
    manifest = args.output / "SHA256SUMS"
    manifest.write_text("".join(f"{sha(path)}  {path.relative_to(args.output).as_posix()}\n" for path in files))
    write_json(
        args.output / "fixture-ready.json",
        {"status": "FIXTURES_READY", "manifest_sha256": sha(manifest), "protocol_sha256": sha(args.protocol)},
    )
    print(f"FIXTURES_READY {args.output} {sha(manifest)}", flush=True)


def verify_fixtures(directory, protocol, *, zero_velocity_target=True):
    spec = json.loads(protocol.read_text())
    adopted = spec.get("canonical_fixtures")
    ready = json.loads((directory / "fixture-ready.json").read_text())
    manifest_hash = sha(directory / "SHA256SUMS")
    require(ready["status"] == "FIXTURES_READY", "Canonical fixture producer was incomplete")
    require(ready["manifest_sha256"] == manifest_hash, "Canonical fixture manifest mismatch")
    require(
        ready["protocol_sha256"] == (adopted["protocol_sha256"] if adopted else sha(protocol)),
        "Canonical fixture protocol mismatch",
    )
    expected = {"SHA256SUMS", "fixture-ready.json"}
    for row in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = row.split("  ", 1)
        require(
            name not in expected and not Path(name).is_absolute() and ".." not in Path(name).parts,
            "Duplicate/unsafe fixture path",
        )
        expected.add(name)
        path = directory / name
        require(not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()), "Unsafe fixture path")
        require(path.is_file() and sha(path) == digest, f"Canonical fixture hash mismatch: {name}")
    require(all(not path.is_symlink() for path in directory.rglob("*")), "Canonical fixtures must not contain symlinks")
    require(
        {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()} == expected,
        "Canonical fixture inventory mismatch",
    )
    metadata = json.loads((directory / "fixtures.json").read_text())
    require(
        metadata["status"] == "COMPLETED"
        and metadata["protocol_sha256"] == ready["protocol_sha256"]
        and (not zero_velocity_target or metadata.get("evaluation_zero_velocity_target_verified") is True),
        "Canonical fixture metadata or stock zero velocity target differs",
    )
    if adopted:
        require(manifest_hash == adopted["manifest_sha256"], "Adopted fixture manifest differs")
        require(metadata["runtime"]["driver_sha256"] == adopted["driver_sha256"], "Adopted fixture source differs")
        require(
            {name: sha(directory / name) for name in expected} == adopted["files_sha256"]
            and {name: (directory / name).stat().st_size for name in expected} == adopted["files_bytes"],
            "Adopted fixture whole inventory differs from the reviewed original",
        )
    return metadata


def checkpoint(learner, args, iteration, transitions, loop_seconds, *, continuation_directory=None):
    started = time.perf_counter()
    directory = args.output / "checkpoints" / f"iteration-{iteration:04d}"
    hashes = learner.save(directory)
    finished = time.perf_counter()
    record = {
        "iteration": iteration,
        "transitions": transitions,
        "path": str(directory.relative_to(args.output)),
        "files_sha256": hashes,
        "model_hashes": state_hashes(learner.model()),
        "process_write_finished_seconds": finished - PROCESS_STARTED,
        "training_loop_seconds": loop_seconds + (finished - started if iteration else 0.0),
        "optimizer": learner.counts(),
        "fresh": learner.fresh(),
        "write_seconds": time.perf_counter() - started,
    }
    if continuation_directory is not None:
        # Keep the scored policy checkpoint portable to the 128-world evaluator;
        # full native replay continuation requires the original training worlds.
        continuation_started = time.perf_counter()
        continuation_path = args.output / continuation_directory
        continuation_files = learner.save(continuation_path, replay=True)
        continuation_finished = time.perf_counter()
        require(state_hashes(learner.model()) == record["model_hashes"], "Replay export changed the final policy")
        require(learner.counts() == record["optimizer"], "Replay export changed the final optimizer counters")
        record["continuation"] = {
            "iteration": iteration,
            "transitions": transitions,
            "path": str(continuation_path.relative_to(args.output)),
            "files_sha256": continuation_files,
            "files_bytes": {name: (continuation_path / name).stat().st_size for name in continuation_files},
            "model_hashes": record["model_hashes"],
            "optimizer": record["optimizer"],
            "replay_included": True,
            "num_envs": learner.num_envs,
            "write_seconds": continuation_finished - continuation_started,
            "process_write_finished_seconds": continuation_finished - PROCESS_STARTED,
        }
        record["write_seconds"] = time.perf_counter() - started
    return record


def collect_iteration(learner, wrapper, observations, horizon, uniform_generator, *, task_diagnostics=None):
    """Time host API sections without extra per-step CUDA barriers."""
    import torch

    collection, learning, eligible, actions_finite = 0.0, 0.0, 0, True
    reward_sum = torch.zeros((), device=learner.device)
    terminated_sum = torch.zeros((), device=learner.device, dtype=torch.int64)
    truncated_sum = torch.zeros_like(terminated_sum)
    diagnostics = learner.spec.get("diagnostics", {}).get("training_actions_and_reward_terms", False)
    if diagnostics:
        action_sums = torch.zeros((3, wrapper.action_dim), device=learner.device)
        reward_manager = wrapper.env.reward_manager
        reward_terms = torch.zeros(len(reward_manager.active_terms), device=learner.device)
    metrics = {}
    if task_diagnostics is not None:
        task_diagnostics.begin_iteration()
    for step in range(horizon):
        started = time.perf_counter()
        with torch.no_grad():
            if not learner.is_ppo and not learner.ready:
                actions = (
                    torch.rand(
                        (wrapper.num_envs, wrapper.action_dim), device=learner.device, generator=uniform_generator
                    )
                    * 2
                    - 1
                )
            else:
                actions = learner.actions(observations)
            # This device reduction is read once at the iteration boundary.
            valid = torch.isfinite(actions).all()
            actions_finite = valid if isinstance(actions_finite, bool) else actions_finite & valid
            after, transition = wrapper.step(actions)
            if task_diagnostics is not None:
                task_diagnostics.observe(transition)
            if diagnostics:
                policy_actions = transition["action"]
                action_sums[0] += policy_actions.sum(dim=0)
                action_sums[1] += policy_actions.square().sum(dim=0)
                action_sums[2] += (policy_actions.abs() > 1).sum(dim=0)
                # This diagnostic buffer contains weighted rates; rewards include step_dt.
                reward_terms += reward_manager._step_reward.sum(dim=0) * wrapper.env.step_dt
            reward_sum += transition["reward"].sum()
            terminated_sum += transition["terminated"].sum()
            truncated_sum += transition["truncated"].sum()
            actions_finite &= torch.isfinite(after).all() & torch.isfinite(transition["reward"]).all()
            actions_finite &= torch.isfinite(transition["next_observation"]).all()
            learner.observe(observations, after, transition, step)
            observations = after
        collection += time.perf_counter() - started
        if not learner.is_ppo and learner.ready:
            eligible += 1
            started = time.perf_counter()
            for _ in range(
                learner.spec["recipes"]["official_flashsac"]["critic_update_calls_per_vector_step_after_warmup"]
            ):
                # Actor/temperature update less often; keep their latest observed metrics.
                metrics.update(learner.update())
            learning += time.perf_counter() - started
    if learner.is_ppo:
        started = time.perf_counter()
        metrics = learner.update(observations)
        learning += time.perf_counter() - started
    torch.cuda.synchronize(learner.device)
    require(bool(actions_finite.item()), "Nonfinite collection action")
    samples = horizon * wrapper.num_envs
    phases = {
        "collection_host_seconds": collection,
        "learning_host_seconds": learning,
        "eligible_update_vector_steps": eligible,
        "mean_transition_reward": float(reward_sum.item()) / samples,
        "terminated_episodes": int(terminated_sum.item()),
        "truncated_episodes": int(truncated_sum.item()),
    }
    if diagnostics:
        phases["training_actions"] = {
            "mean": (action_sums[0] / samples).tolist(),
            "rms": (action_sums[1] / samples).sqrt().tolist(),
            "abs_gt_one_fraction": (action_sums[2] / samples).tolist(),
        }
        phases["mean_weighted_reward_terms"] = dict(
            zip(reward_manager.active_terms, (reward_terms / samples).tolist(), strict=True)
        )
    if task_diagnostics is not None:
        phases["training_cable_state"] = task_diagnostics.finish_iteration()
    return (
        observations,
        metrics,
        phases,
    )


def payload(directory):
    """Read native files for a smoke reload audit; never coerce tensor devices."""
    import numpy as np
    import torch

    result = {}
    for path in sorted(directory.iterdir()):
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                result[path.name] = dict(archive)
        elif path.suffix == ".pt":
            result[path.name] = torch.load(path, map_location="cpu", weights_only=False)
    return result


def train_worker(
    args,
    spec,
    *,
    environment_factory=environment,
    learner_factory=Learner,
    fixture_validator=verify_fixtures,
    training_diagnostic_factory=None,
):
    import numpy as np
    import torch
    from benchmark_official_flashsac import fresh_optimizer_state

    fixtures = fixture_validator(args.fixtures, args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "STARTED",
        "recipe": args.recipe,
        "seed": args.seed,
        "smoke": args.smoke,
        "protocol_sha256": sha(args.protocol),
        "runtime": runtime(args),
        "training_curve": [],
        "checkpoints": [],
    }
    write_json(args.output / "train.json", result)

    num_envs = spec.get("smoke", {}).get("num_envs", 64) if args.smoke else spec["mdp"]["num_envs"]
    iterations = 8 if args.smoke else spec["budget"]["iterations"]
    checkpoints = [0, iterations] if args.smoke else spec["budget"]["checkpoint_iterations"]
    horizon = spec["budget"]["rollout_steps"]
    timing = {
        "learner_constructor_seconds": 0.0,
        "prepare_seconds": 0.0,
        "training_loop_seconds": 0.0,
        "checkpoint_seconds": 0.0,
        "metrics_io_seconds": 0.0,
    }
    with environment_factory(args, spec, num_envs, args.seed) as (wrapper, metadata):
        result["environment"] = metadata
        timing["environment_startup_seconds"] = metadata["startup_seconds"]
        started = time.perf_counter()
        learner = learner_factory(args, spec, num_envs, wrapper.observation_dim, wrapper.action_dim)
        timing["learner_constructor_seconds"] = time.perf_counter() - started
        policy = f"{'ppo' if learner.is_ppo else 'flashsac'}-seed{args.seed}.npz"
        audit = learner.initialize(args.fixtures / policy)
        audit["canonical_state_hashes"] = fixtures["policies"][policy]["arrays"]
        require(
            audit["hashes"] == audit["canonical_state_hashes"],
            "Initial policy readback differs from the canonical fixture",
        )
        result["initial_weights_audit"] = audit
        initial = learner.model()
        np.savez(args.output / "initial-parameters.npz", **initial)
        result["initial_model_hashes"] = state_hashes(initial)
        started = time.perf_counter()
        learner.prepare()
        torch.cuda.synchronize(args.device)
        timing["prepare_seconds"] = time.perf_counter() - started
        require(state_hashes(learner.model()) == state_hashes(initial), "Preparation changed the initial policy")
        require(not any(learner.counts()["completed"].values()), "Preparation left nonzero optimizer counters")
        result["initial_optimizer_state"] = learner.counts() if learner.is_ppo else fresh_optimizer_state(learner.agent)
        result["configuration"] = learner.config
        if not learner.is_ppo:
            result["flash_recipe_resolution"] = {
                **flash_recipe_settings(spec),
                "schedule_step_unit": "Each optimizer's attempted updates, including AMP skips",
                "shared_horizon": "The same decay horizon is used by all three optimizers; actor/temperature run less often",
                "timeout_bootstrap": "Official uses gamma**n_step; native uses the effective replay horizon",
                "official_replay_pending_saved": False,
                "maximum_unsaved_pending_vector_steps": learner.config["n_step"] - 1 if not learner.native else 0,
            }
        result["official_source"] = learner.provenance
        result["fresh_after_prepare"] = learner.fresh()
        # Reset after all compilation/warmup, with no randomized episode progress.
        observations = wrapper.reset(seed=args.seed)
        uniform_generator = torch.Generator(device=args.device).manual_seed(args.seed + 90000)
        task_diagnostics = None
        if spec.get("diagnostics", {}).get("training_cable_state", False):
            require(training_diagnostic_factory is not None, "No task-specific training diagnostic collector")
            task_diagnostics = training_diagnostic_factory(wrapper, spec)
            result["training_cable_state_scope"] = task_diagnostics.scope
        record = checkpoint(learner, args, 0, 0, 0.0)
        timing["checkpoint_seconds"] += record["write_seconds"]
        result["checkpoints"].append(record)
        loop_started = time.perf_counter()
        eligible_total = 0
        for iteration in range(1, iterations + 1):
            iteration_started = time.perf_counter()
            observations, metrics, phases = collect_iteration(
                learner, wrapper, observations, horizon, uniform_generator, task_diagnostics=task_diagnostics
            )
            eligible_total += phases["eligible_update_vector_steps"]
            transitions = iteration * horizon * num_envs
            record = {
                "iteration": iteration,
                "transitions": transitions,
                **phases,
                "metrics": metrics,
                "optimizer": learner.counts(),
            }
            if iteration in checkpoints:
                settings = spec.get("checkpointing", {})
                continuation = (
                    settings["continuation_directory_name"]
                    if not args.smoke
                    and settings.get("full_replay_in_final_continuation_checkpoint", False)
                    and iteration == settings["continuation_iteration"] == iterations
                    and not learner.is_ppo
                    else None
                )
                saved = checkpoint(
                    learner,
                    args,
                    iteration,
                    transitions,
                    time.perf_counter() - loop_started,
                    continuation_directory=continuation,
                )
                if "continuation" in saved:
                    result["final_continuation"] = saved.pop("continuation")
                sample = settings.get("final_replay_sample", {})
                if not args.smoke and sample.get("enabled", False) and iteration == sample["iteration"] == iterations:
                    result["final_replay_sample"] = export_replay_sample(
                        learner, args, spec, iteration, transitions, saved
                    )
                    saved["write_seconds"] += result["final_replay_sample"]["write_seconds"]
                timing["checkpoint_seconds"] += saved["write_seconds"]
                result["checkpoints"].append(saved)
            record["iteration_seconds"] = time.perf_counter() - iteration_started
            result["training_curve"].append(record)
            started = time.perf_counter()
            with (args.output / "training-curve.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            timing["metrics_io_seconds"] += time.perf_counter() - started
            print(
                f"TRAIN {args.recipe} seed={args.seed} iteration={iteration}/{iterations} transitions={transitions}",
                flush=True,
            )
        timing["training_loop_seconds"] = time.perf_counter() - loop_started
        final = learner.model()
        np.savez(args.output / "final-parameters.npz", **final)
        updated = {
            name: any(
                not np.array_equal(initial[key], final[key])
                for key in initial
                if key.startswith(name) and key.endswith((".weight", ".bias", "_weight", "_bias"))
            )
            for name in ("actor", "critic")
        }
        result["health"] = {"finite": finite(final), **{f"updated_{name}": value for name, value in updated.items()}}
        require(
            result["health"]["finite"] and all(updated.values()),
            "Learner did not produce finite changed actor/critic weights",
        )
        result["final_model_hashes"] = state_hashes(final)
        result["optimizer"] = learner.counts()
        require(
            min(result["optimizer"]["completed"][name] for name in ("actor", "critic")) > 0,
            "Neither attempted-only nor entirely skipped learning is a completed training run",
        )
        result["fresh_final"] = learner.fresh()
        result["transitions"], result["iterations"] = iterations * horizon * num_envs, iterations
        result["horizon"], result["num_envs"] = horizon, num_envs
        result["eligible_update_vector_steps"] = eligible_total
        if args.smoke:
            saved = args.output / "continuation-before"
            learner.save(saved, replay=not learner.is_ppo)
            before_payload, before_counts = payload(saved), learner.counts()
            before_model = state_hashes(learner.model())
            observations, _, _ = collect_iteration(
                learner, wrapper, observations, horizon, uniform_generator, task_diagnostics=task_diagnostics
            )
            perturbed_counts = learner.counts()
            require(
                perturbed_counts["completed"]["critic"] > before_counts["completed"]["critic"],
                "Smoke perturbation did not advance the optimizer",
            )
            require(state_hashes(learner.model()) != before_model, "Smoke perturbation did not change the model")
            learner.load(saved, replay=not learner.is_ppo)
            learner.save(args.output / "continuation-reloaded", replay=not learner.is_ppo)
            require(
                same(before_payload, payload(args.output / "continuation-reloaded")),
                "Smoke native checkpoint/scaler reload mismatch",
            )
            require(before_counts == learner.counts(), "Smoke checkpoint counter mismatch")
            observations, _, _ = collect_iteration(
                learner, wrapper, observations, horizon, uniform_generator, task_diagnostics=task_diagnostics
            )
            after_counts = learner.counts()
            require(
                after_counts["completed"]["actor"] > before_counts["completed"]["actor"]
                and after_counts["completed"]["critic"] > before_counts["completed"]["critic"],
                "Smoke continuation did not update both networks",
            )
            require(finite(learner.model()), "Nonfinite smoke continuation")
            result["continuation"] = {
                "checkpoint_reload_exact": True,
                "optimizer_scaler_payload_equal": True,
                "before": before_counts,
                "perturbed": perturbed_counts,
                "after": after_counts,
                "finite": True,
                "extra_iterations_outside_reported_budget": 2,
                "scope": "Native public checkpoint payload restored after deliberate model/optimizer mutation. Environment, Torch global RNG and exact continuation trajectories are not claimed restored.",
            }
    timing["process_seconds"] = time.perf_counter() - PROCESS_STARTED
    result["timing"], result["status"] = timing, "COMPLETED"
    result["timing_scope"] = {
        "collection_host_seconds": "Policy inference + public action API/CPU transfer + common action boundary + env.step + transition observe/insertion. Includes host waits; not pure physics.",
        "learning_host_seconds": "Actual optimizer/update API and scalar metrics; interleaved Flash calls excluded from collection section. Includes host waits.",
        "primary": "Synchronized iteration/whole-process wall; phase host/dispatch durations are not GPU-only timings and do not include the final iteration barrier.",
        "nested": "Checkpoint/metrics IO are inside training_loop_seconds, including any final full-replay or sampled-replay export. Portable policy availability timestamps precede that separate export. Preparation includes native kernel build, warmup and capture, but official lazy compilation remains in constructor/first collection/updates.",
    }
    write_json(args.output / "train.json", result)


def evaluation_metrics(values, settings):
    import numpy as np

    episodes, horizon = settings["episodes"], settings["horizon_steps"]
    active_steps = int(values["length"].sum())

    def mean(name):
        return float(values[name].sum() / max(active_steps, 1))

    metrics = {
        "full_upright_time_fraction": float(values["full_upright_count"].sum() / (episodes * horizon)),
        "settled_upright_time_fraction": float(
            values["settled_upright_count"].sum() / (episodes * (horizon - settings["settling_steps"]))
        ),
        "active_only_upright_fraction": mean("full_upright_count"),
        "pole_angle_abs_mean": mean("angle_abs_sum"),
        "pole_angle_rms": math.sqrt(mean("angle_sq_sum")),
        "unwrapped_pole_angle_rms": math.sqrt(mean("raw_angle_sq_sum")),
        "cart_position_abs_mean": mean("cart_abs_sum"),
        "cart_position_rms": math.sqrt(mean("cart_sq_sum")),
        "return_mean": float(values["return"].mean()),
        "return_std": float(values["return"].std()),
        "episode_length_mean": float(values["length"].mean()),
        "clean_timeout_fraction": float(values["clean_timeout"].mean()),
        "raw_timeout_fraction": float(values["raw_timeout"].mean()),
        "cart_bound_fraction": float(values["terminated"].mean()),
        "simultaneous_timeout_cart_bound_fraction": float(np.mean(values["terminated"] & values["raw_timeout"])),
        "action_rms": math.sqrt(mean("action_sq_sum")),
        "action_saturation_fraction": mean("action_saturated_count"),
        "nonfinite_action_count": int(values["nonfinite_action_count"].sum()),
        "active_first_episode_steps": active_steps,
    }
    gate = settings["primary_quality_gate"]
    metrics["primary_quality_pass"] = bool(
        metrics["clean_timeout_fraction"] >= gate["clean_timeout_fraction_min"]
        and metrics["settled_upright_time_fraction"] >= gate["settled_upright_time_fraction_min"]
        and metrics["nonfinite_action_count"] <= gate["nonfinite_action_count_max"]
    )
    return metrics


def evaluate_episode(learner, wrapper, state, settings):
    import torch

    groups, _ = wrapper.env.reset_to(state, seed=settings["common_seed"], is_relative=True)
    robot = wrapper.env.scene["robot"]
    # reset_to also seeds PD targets from the saved velocities. Effort-controlled
    # Cartpole must retain the ordinary zero target for its kd=10 cart damping.
    robot.actuators.target_command.set_velocity_index(
        value=torch.zeros_like(state["articulation"]["robot"]["joint_velocity"])
    )
    observations = wrapper._observations(groups).clone()
    observation_hash = array_hash(observations.cpu().numpy())
    actual_state = tree_map(wrapper.env.scene.get_state(is_relative=True), lambda value: value.detach().cpu().clone())
    hashes = state_hashes({name: value.numpy() for name, value in flatten(actual_state).items()})
    expected = state_hashes({name: value.detach().cpu().numpy() for name, value in flatten(state).items()})
    require(hashes == expected, "Canonical evaluation state was not restored bit exactly")
    cart = robot.find_joints("slider_to_cart")[0]
    pole = robot.find_joints("cart_to_pole")[0]
    require(len(cart) == len(pole) == 1, "Cart/pole joint lookup was ambiguous")
    require(len(robot.joint_names) * 2 == observations.shape[1], "Joint observation layout changed")
    count = wrapper.num_envs
    active = torch.ones(count, dtype=torch.bool, device=learner.device)
    integers = (
        "length",
        "full_upright_count",
        "settled_upright_count",
        "action_saturated_count",
        "nonfinite_action_count",
    )
    moments = (
        "return",
        "angle_abs_sum",
        "angle_sq_sum",
        "raw_angle_sq_sum",
        "cart_abs_sum",
        "cart_sq_sum",
        "action_sq_sum",
    )
    values = {key: torch.zeros(count, dtype=torch.int64, device=learner.device) for key in integers}
    values.update({key: torch.zeros(count, dtype=torch.float64, device=learner.device) for key in moments})
    values.update({key: torch.zeros_like(active) for key in ("clean_timeout", "raw_timeout", "terminated")})
    finite_actions = torch.ones((), dtype=torch.bool, device=learner.device)
    with torch.no_grad():
        for step in range(settings["horizon_steps"]):
            actions = learner.actions(observations, deterministic=True)
            finite_actions &= torch.isfinite(actions).all()
            observations, transition = wrapper.step(actions)
            post = transition["next_observation"]
            angle, position = post[:, pole[0]].double(), post[:, cart[0]].double()
            wrapped = torch.atan2(torch.sin(angle), torch.cos(angle))
            upright = active & (wrapped.abs() < settings["upright_angle_radians"])
            values["length"] += active
            values["full_upright_count"] += upright
            if step >= settings["settling_steps"]:
                values["settled_upright_count"] += upright
            executed = transition["action"].double()
            per_step = {
                "return": transition["reward"].double(),
                "angle_abs_sum": wrapped.abs(),
                "angle_sq_sum": wrapped.square(),
                "raw_angle_sq_sum": angle.square(),
                "cart_abs_sum": position.abs(),
                "cart_sq_sum": position.square(),
                "action_sq_sum": executed.square().sum(dim=-1),
                "action_saturated_count": (executed.abs() >= 1.0).sum(dim=-1),
                "nonfinite_action_count": (~torch.isfinite(actions)).sum(dim=-1),
            }
            for key, value in per_step.items():
                values[key] += torch.where(active, value, 0)
            done = active & (transition["terminated"] | transition["truncated"])
            values["terminated"] |= done & transition["terminated"]
            values["raw_timeout"] |= done & transition["truncated"]
            values["clean_timeout"] |= done & transition["truncated"] & ~transition["terminated"]
            active &= ~done
    torch.cuda.synchronize(learner.device)
    require(bool(finite_actions.item()), "Nonfinite deterministic evaluation action")
    require(not bool(active.any().item()), "First episode did not finish within the common 300-step horizon")
    return {key: value.cpu().numpy() for key, value in values.items()}, hashes, observation_hash


def optimizer_completed(learner):
    from benchmark_official_flashsac import optimizer_steps
    from benchmark_rsl_ppo_cuda import rsl_steps

    if not learner.is_ppo:
        return optimizer_steps(learner.agent)
    return int(learner.agent.optimizer._timestep.numpy()[0]) if learner.native else rsl_steps(learner.agent)


def evaluate_worker(
    args,
    spec,
    *,
    environment_factory=environment,
    learner_factory=Learner,
    fixture_validator=verify_fixtures,
    episode_evaluator=evaluate_episode,
    aggregate_metrics=evaluation_metrics,
):
    import numpy as np
    import torch

    fixture_metadata = fixture_validator(args.fixtures, args.protocol)
    training = json.loads((args.train_root / "train.json").read_text())
    require(training["status"] == "COMPLETED", "Cannot evaluate incomplete training")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "STARTED",
        "recipe": args.recipe,
        "seed": args.seed,
        "smoke": args.smoke,
        "runtime": runtime(args),
        "protocol_sha256": sha(args.protocol),
        "evaluations": [],
    }
    settings = spec["evaluation"]
    state = torch.load(args.fixtures / "evaluation-state.pt", map_location=args.device, weights_only=True)
    with environment_factory(args, spec, settings["episodes"], settings["common_seed"]) as (wrapper, metadata):
        result["environment"] = metadata
        wrapper.reset(seed=settings["common_seed"])
        started = time.perf_counter()
        learner = learner_factory(
            args, spec, settings["episodes"], wrapper.observation_dim, wrapper.action_dim, evaluation=True
        )
        constructor_seconds = time.perf_counter() - started
        if not learner.is_ppo:
            result["configuration"] = learner.config
            result["flash_recipe_resolution"] = flash_recipe_settings(spec)
        if args.recipe == "rsl_ppo":
            learner.agent.eval_mode()
        for saved in training["checkpoints"]:
            directory = args.train_root / saved["path"]
            require(
                {path.name: sha(path) for path in directory.iterdir() if path.is_file()} == saved["files_sha256"],
                "Checkpoint files changed before evaluation",
            )
            started = time.perf_counter()
            learner.load(directory)
            torch.cuda.synchronize(args.device)
            load_seconds = time.perf_counter() - started
            before, counts_before = state_hashes(learner.model()), optimizer_completed(learner)
            require(
                before == saved["model_hashes"], "Loaded policy state differs from the actual saved checkpoint model"
            )
            started = time.perf_counter()
            values, hashes, observation_hash = episode_evaluator(learner, wrapper, state, settings)
            evaluation_seconds = time.perf_counter() - started
            require(
                hashes == fixture_metadata["evaluation_state_hashes"],
                "Evaluation fixture hashes differ across checkpoints",
            )
            require(
                observation_hash == fixture_metadata["evaluation_observation_hash"],
                "Canonical evaluation observations differ",
            )
            require(state_hashes(learner.model()) == before, "Evaluation changed model or BN running state")
            require(optimizer_completed(learner) == counts_before, "Evaluation performed optimizer work")
            filename = f"episodes-{saved['iteration']:04d}.npz"
            started = time.perf_counter()
            np.savez(args.output / filename, **values)
            metrics = aggregate_metrics(values, settings)
            require(finite(metrics), "Nonfinite evaluation aggregate")
            record = {
                "iteration": saved["iteration"],
                "transitions": saved["transitions"],
                "checkpoint_files_sha256": saved["files_sha256"],
                "loaded_model_hashes": before,
                "initial_state_hashes": hashes,
                "initial_observation_hash": observation_hash,
                "per_episode_file": filename,
                "per_episode_sha256": sha(args.output / filename),
                "metrics": metrics,
                "timing": {
                    "checkpoint_load_seconds": load_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "metrics_io_seconds": time.perf_counter() - started,
                },
                "checkpoint_process_write_finished_seconds": saved["process_write_finished_seconds"],
                "training_loop_seconds": saved["training_loop_seconds"],
                "inference_model_unchanged": True,
                "optimizer_unchanged": True,
            }
            result["evaluations"].append(record)
            write_json(args.output / "evaluate.json", result)
            print(
                f"EVAL {args.recipe} seed={args.seed} checkpoint={saved['iteration']} quality={metrics['primary_quality_pass']}",
                flush=True,
            )
    result["status"] = "COMPLETED"
    result["timing"] = {
        "environment_startup_seconds": metadata["startup_seconds"],
        "learner_constructor_seconds": constructor_seconds,
        "process_seconds": time.perf_counter() - PROCESS_STARTED,
    }
    write_json(args.output / "evaluate.json", result)


def worker_command(args, phase, recipe, seed, output, train_root=None, *, entrypoint=None):
    command = [
        sys.executable,
        str(Path(__file__).resolve() if entrypoint is None else entrypoint),
        phase,
        "--protocol",
        str(args.protocol),
        "--allocation",
        args.allocation,
        "--isaaclab-root",
        str(args.isaaclab_root),
        "--official-root",
        str(args.official_root),
        "--fixtures",
        str(args.fixtures),
        "--output",
        str(output),
        "--recipe",
        recipe,
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]
    if args.phase == "smoke":
        command.append("--smoke")
    if train_root is not None:
        command.extend(["--train-root", str(train_root)])
    return command


def child(args, command, log):
    started = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=args.isaaclab_root, stdout=stream, stderr=subprocess.STDOUT)
        code = process.wait()
    record = {
        "command": command,
        "exit_code": code,
        "wall_seconds": time.perf_counter() - started,
        "log": str(log.relative_to(args.output)),
        "chronological_started_seconds": started - PROCESS_STARTED,
    }
    return record


def controller(args, spec, *, fixture_validator=verify_fixtures, entrypoint=None):
    fixture_validator(args.fixtures, args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    smoke = args.phase == "smoke"
    if not smoke:
        prior = json.loads((args.output.parent / "smoke" / "comparison.json").read_text())
        require(
            prior["status"] == "COMPLETED"
            and [row["recipe"] for row in prior["smokes"]] == spec["schedule"]["recipe_order"]
            and prior["protocol_sha256"] == sha(args.protocol),
            "Production requires the completed selected-recipe smoke first",
        )
    seeds = [0] if smoke else spec["schedule"][args.allocation].get("seeds", spec["budget"]["seeds"])
    schedule = []
    for seed in seeds:
        row_index = 0 if smoke else spec["schedule"][args.allocation]["williams_row_by_seed"][seed]
        indices = (
            range(len(spec["schedule"]["recipe_order"])) if smoke else spec["schedule"]["williams_rows"][row_index]
        )
        for position, index in enumerate(indices):
            schedule.append(
                {
                    "seed": seed,
                    "recipe": spec["schedule"]["recipe_order"][index],
                    "position": position,
                    "williams_row": row_index,
                }
            )
    result = {
        "status": "RUNNING",
        "allocation": args.allocation,
        "protocol_sha256": sha(args.protocol),
        "fixture_manifest_sha256": sha(args.fixtures / "SHA256SUMS"),
        "schedule": schedule,
        "smokes": [],
        "runs": [],
        "chronology": [],
        "cache_scope": "Fresh worker processes, shared on-disk compiler/native caches. Smokes run before production; all startup and chronological positions retained.",
    }
    destination = args.output / "comparison.json"
    write_json(destination, result)
    for entry in schedule:
        recipe, seed = entry["recipe"], entry["seed"]
        relative = Path("runs") / f"{recipe}-seed{seed}"
        run_root = args.output / relative
        run_root.mkdir(parents=True)
        train_root, eval_root = run_root / "train", run_root / "evaluate"
        train_process = child(
            args,
            worker_command(args, "train-worker", recipe, seed, train_root, entrypoint=entrypoint),
            run_root / "train.log",
        )
        result["chronology"].append({**entry, "phase": "training", **train_process})
        write_json(destination, result)
        require(train_process["exit_code"] == 0, f"Training failed; retained log: {run_root / 'train.log'}")
        eval_process = child(
            args,
            worker_command(args, "evaluate-worker", recipe, seed, eval_root, train_root, entrypoint=entrypoint),
            run_root / "evaluate.log",
        )
        result["chronology"].append({**entry, "phase": "evaluation", **eval_process})
        write_json(destination, result)
        require(eval_process["exit_code"] == 0, f"Evaluation failed; retained log: {run_root / 'evaluate.log'}")
        train = json.loads((train_root / "train.json").read_text())
        evaluation = json.loads((eval_root / "evaluate.json").read_text())
        require(train["status"] == evaluation["status"] == "COMPLETED", "Worker result was incomplete")
        if smoke:
            require(train["continuation"]["checkpoint_reload_exact"], "Smoke continuation missing")
        row = {
            **entry,
            "training_process_seconds": train_process["wall_seconds"],
            "evaluation_process_seconds": eval_process["wall_seconds"],
            "end_to_end_seconds": train_process["wall_seconds"] + eval_process["wall_seconds"],
            "train_json": (relative / "train/train.json").as_posix(),
            "eval_json": (relative / "evaluate/evaluate.json").as_posix(),
        }
        result["smokes" if smoke else "runs"].append(row)
        write_json(destination, result)
        print(f"RUN_READY allocation={args.allocation} recipe={recipe} seed={seed} smoke={smoke}", flush=True)
    result["status"], result["protocol_wall_seconds"] = "COMPLETED", time.perf_counter() - PROCESS_STARTED
    write_json(destination, result)


def parse_args(description=None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("phase", choices=("fixture", "smoke", "run", "train-worker", "evaluate-worker"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--isaaclab-root",
        type=Path,
        default=os.environ.get("ISAACLAB_ROOT"),
        required="ISAACLAB_ROOT" not in os.environ,
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=os.environ.get("OFFICIAL_FLASHSAC_ROOT"),
        required="OFFICIAL_FLASHSAC_ROOT" not in os.environ,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--allocation", default="A", help="Allocation key in the protocol schedule")
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-root", type=Path)
    args = parser.parse_args()
    for name in ("protocol", "isaaclab_root", "official_root", "output", "fixtures", "train_root"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())
    if args.phase != "fixture" and args.fixtures is None:
        parser.error("--fixtures must point to the shared, verified canonical directory")
    if args.phase.endswith("worker") and args.recipe is None:
        parser.error("Workers require --recipe")
    if args.phase == "evaluate-worker" and args.train_root is None:
        parser.error("Evaluation requires --train-root")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    try:
        specification = read_spec(arguments)
        with gpu_worker_stream(arguments):
            {
                "fixture": fixture,
                "train-worker": train_worker,
                "evaluate-worker": evaluate_worker,
                "smoke": controller,
                "run": controller,
            }[arguments.phase](arguments, specification)
    except Exception as error:
        write_json(
            arguments.output / "failure.json",
            {
                "status": "FAILED",
                "phase": arguments.phase,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "process_seconds": time.perf_counter() - PROCESS_STARTED,
            },
        )
        raise
