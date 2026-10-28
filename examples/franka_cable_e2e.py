# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Compare four learners on the stock state based Franka cable lift task.

Reuse the Cartpole comparison's collector, process timing and checkpoint audits.
The task adapter preserves stock coupled physics, actions, rewards and training
curricula. Evaluation uses common full states and full cable gravity. The native
PPO bridge reuses RSL-RL's observation normalizer without changing learning kernels.

Algorithm credit: Schulman et al., arXiv:1707.06347; Kim et al., arXiv:2604.04539.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import cartpole_e2e as core
from _ppo_observation_normalization import NativePPOObservationNormalization


def normalizer_arrays(agent):
    return {
        f"{role}.{key}": value.detach().cpu().numpy().copy()
        for role, model in (("actor", agent._raw_actor), ("critic", agent._raw_critic))
        for key, value in model.obs_normalizer.state_dict().items()
    }


class CableLearner(core.Learner):
    """Match stock symmetric PPO normalization outside the captured update."""

    def __init__(self, args, spec, num_envs, observations, actions, *, evaluation=False):
        super().__init__(args, spec, num_envs, observations, actions, evaluation=evaluation)
        self.normalization = None
        self._normalized_action_input = None
        if self.is_ppo:
            core.require(
                self.config["obs_groups"] == {"actor": ["policy"], "critic": ["policy"]}
                and all(
                    self.config["actor"][key] == self.config["critic"][key] for key in ("hidden_dims", "activation")
                ),
                "The native bridge requires the stock symmetric state based PPO recipe",
            )
            core.require(
                all(self.config[role]["obs_normalization"] for role in ("actor", "critic")),
                "The stock cable PPO recipe requires actor and critic normalization",
            )
            if self.native:
                self.normalization = NativePPOObservationNormalization(
                    spec["budget"]["rollout_steps"], num_envs, observations, args.device, evaluation=evaluation
                )
        self.provenance["comparison_harness_files_sha256"] = {
            name: core.sha(Path(__file__).parent / name)
            for name in (
                "franka_cable_e2e.py",
                "_franka_cable_environment.py",
                "_ppo_observation_normalization.py",
                "cartpole_e2e.py",
            )
        }

    def actions(self, observations, deterministic=False):
        if self.normalization is not None:
            observations = self.normalization.normalize(observations)
            self._normalized_action_input = observations
        return super().actions(observations, deterministic=deterministic)

    def observe(self, before, after, transition, step):
        if self.normalization is not None:
            self.normalization.record_raw(step, before)
            core.require(self._normalized_action_input is not None, "PPO transition must follow policy inference")
            before, self._normalized_action_input = self._normalized_action_input, None
            transition = {
                **transition,
                "next_observation": self.normalization.normalize(transition["next_observation"]),
            }
        return super().observe(before, after, transition, step)

    def update(self, last_observations=None):
        metrics = super().update(last_observations)
        if self.normalization is not None:
            self.normalization.update_after_learning()
        return metrics

    def model(self):
        values = super().model()
        if self.is_ppo:
            stats = self.normalization.model_arrays() if self.native else normalizer_arrays(self.agent)
            values.update({f"normalization.{key}": value for key, value in stats.items()})
        return values

    def save(self, directory, replay=False):
        super().save(directory, replay=replay)
        if self.normalization is not None:
            self.normalization.save_sidecar(directory / "observation-normalization.npz")
        return {path.name: core.sha(path) for path in sorted(directory.iterdir()) if path.is_file()}

    def load(self, directory, replay=False):
        super().load(directory, replay=replay)
        if self.normalization is not None:
            self.normalization.load_sidecar(directory / "observation-normalization.npz")


def read_spec(args):
    spec = json.loads(args.protocol.read_text())
    core.require(spec["mdp"]["task"] == "Isaac-Lift-Cable-Franka", "Expected the stock state based cable task")
    core.require(spec["mdp"]["action_boundary_clip"] is None, "The stock cable task uses unclipped policy actions")
    core.require(spec["recipes"]["official_flashsac"]["n_step"] == 1, "Expected one-step timeout targets")
    recipes = spec["schedule"]["recipe_order"]
    core.require(
        recipes and len(set(recipes)) == len(recipes) and set(recipes) <= set(core.RECIPES),
        "Expected a nonempty unique selection of reviewed learners",
    )
    core.require(args.allocation in spec["schedule"], "Unknown allocation")
    core.require(
        all(sorted(row) == list(range(len(recipes))) for row in spec["schedule"]["williams_rows"]),
        "Incomplete execution order",
    )
    return spec


def verify_fixtures(directory, protocol):
    metadata = core.verify_fixtures(directory, protocol, zero_velocity_target=False)
    core.require(metadata["coupled_state_restore_verified"], "Cable fixture state restore was not verified")
    core.require(metadata["full_cable_gravity_verified"], "Cable fixture did not verify full gravity")
    return metadata


def fixture(args, spec):
    import isaaclab_tasks  # noqa: F401
    import numpy as np
    import torch
    from _franka_cable_environment import build_environment, restore, snapshot
    from benchmark_official_flashsac import import_official, raw_module
    from benchmark_rsl_ppo_cuda import model_arrays

    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "PREPARING", "runtime": core.runtime(args), "policies": {}}
    module, metadata["official_source"] = import_official(args.official_root)
    observations, actions = spec["mdp"]["observation_dimensions"], spec["mdp"]["action_dimensions"]
    # The shared smoke uses seed zero even when production uses new seed cohorts.
    for seed in sorted({0, *spec["budget"]["seeds"]}):
        torch.manual_seed(seed)
        ppo = core.make_rsl(1, observations, actions, "cpu", core.stock_config(spec["mdp"]["task"]))
        ppo_state = model_arrays(ppo)
        ppo_state.update({f"normalization.{key}": value for key, value in normalizer_arrays(ppo).items()})
        torch.manual_seed(seed)
        config = core.flash_config(spec, seed, "cpu")
        config.use_compile = config.use_amp = False
        networks = module._init_flashsac_networks(
            observations, observations, actions, core.official_config(module, config), torch.device("cpu")
        )
        flash_state = {
            f"{name}.{key}": value.detach().numpy().copy()
            for name, network in zip(("actor", "critic", "target_critic", "temperature"), networks, strict=True)
            for key, value in raw_module(network).state_dict().items()
        }
        for name, state in (("ppo", ppo_state), ("flashsac", flash_state)):
            core.require(core.finite(state), "Nonfinite canonical policy")
            filename = f"{name}-seed{seed}.npz"
            np.savez(args.output / filename, **state)
            metadata["policies"][filename] = {
                "sha256": core.sha(args.output / filename),
                "arrays": core.state_hashes(state),
            }
    settings = spec["evaluation"]
    with build_environment(args, spec, settings["episodes"], settings["common_seed"], evaluation=True) as (
        wrapper,
        env,
        env_metadata,
    ):
        wrapper.reset(seed=settings["common_seed"])
        wrapper.reset(seed=settings["common_seed"])
        state = snapshot(env)
        torch.save(
            core.tree_map(state, lambda value: value.detach().cpu().clone()), args.output / "evaluation-state.pt"
        )
        observations = restore(env, state)
        restored = snapshot(env)
        hashes = core.state_hashes({name: value.detach().cpu().numpy() for name, value in core.flatten(state).items()})
        core.require(
            hashes
            == core.state_hashes(
                {name: value.detach().cpu().numpy() for name, value in core.flatten(restored).items()}
            ),
            "Full cable state did not restore exactly",
        )
        metadata.update(
            evaluation_state_hashes=hashes,
            evaluation_observation_hash=core.array_hash(observations.detach().cpu().numpy()),
            coupled_state_restore_verified=True,
            full_cable_gravity_verified=env_metadata["full_cable_gravity_verified"],
            environment=env_metadata,
        )
    metadata.update(status="COMPLETED", protocol_sha256=core.sha(args.protocol))
    core.write_json(args.output / "fixtures.json", metadata)
    files = sorted(path for path in args.output.rglob("*") if path.is_file())
    manifest = args.output / "SHA256SUMS"
    manifest.write_text("".join(f"{core.sha(path)}  {path.relative_to(args.output).as_posix()}\n" for path in files))
    core.write_json(
        args.output / "fixture-ready.json",
        {"status": "FIXTURES_READY", "manifest_sha256": core.sha(manifest), "protocol_sha256": core.sha(args.protocol)},
    )
    print(f"FIXTURES_READY {args.output} {core.sha(manifest)}", flush=True)


def main():
    from _franka_cable_environment import environment, evaluate_episode, evaluation_metrics

    args = core.parse_args(description=__doc__)
    try:
        spec = read_spec(args)
        common = {"learner_factory": CableLearner, "fixture_validator": verify_fixtures}
        with core.gpu_worker_stream(args):
            if args.phase == "fixture":
                fixture(args, spec)
            elif args.phase == "train-worker":
                core.train_worker(args, spec, environment_factory=environment, **common)
            elif args.phase == "evaluate-worker":
                core.evaluate_worker(
                    args,
                    spec,
                    environment_factory=environment,
                    episode_evaluator=evaluate_episode,
                    aggregate_metrics=evaluation_metrics,
                    **common,
                )
            else:
                core.controller(args, spec, fixture_validator=verify_fixtures, entrypoint=Path(__file__).resolve())
    except Exception as error:
        core.write_json(
            args.output / "failure.json",
            {
                "status": "FAILED",
                "phase": args.phase,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "process_seconds": time.perf_counter() - core.PROCESS_STARTED,
            },
        )
        raise


if __name__ == "__main__":
    main()
