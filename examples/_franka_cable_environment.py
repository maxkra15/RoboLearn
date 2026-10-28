# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Stock Franka cable environment, canonical reset state and first-episode metrics.

Scene/manager state is restored explicitly. Public scene writes and forward
cold-reset Newton solver history; this is not historical solver/RNG continuation.
The fixture proves initial physical/manager state and observation identity.
"""

import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from cartpole_e2e import array_hash, flatten, require, sha, state_hashes, tree_map

TASK = "Isaac-Lift-Cable-Franka"
PHYSICS = "newton_mjwarp_vbd_proxy"


def _gravity(env):
    import warp as wp

    return wp.to_torch(env.sim.backend_manager.backend.model.gravity)


def observation_layout(env):
    """Resolve raw observation term slices from the actual manager contract."""
    manager = env.observation_manager
    result, offset = {}, 0
    for name, shape in zip(manager.active_terms["policy"], manager.group_obs_term_dim["policy"], strict=True):
        width = math.prod(shape)
        result[name] = (offset, offset + width)
        offset += width
    require(
        list(result) == ["joint_pos", "joint_vel", "cable_segment_positions", "target_position", "actions"]
        and offset == 69
        and result["cable_segment_positions"][1] - result["cable_segment_positions"][0] == 36
        and result["target_position"][1] - result["target_position"][0] == 7,
        "The registered state cable observation contract differs",
    )
    return result


@contextmanager
def build_environment(args, spec, num_envs, seed, evaluation=False):
    """Yield (wrapper, environment, metadata), keeping simulation ownership local."""
    import gymnasium as gym
    import isaaclab_tasks
    import torch
    import warp as wp
    from isaaclab.app import launch_simulation
    from isaaclab.paths import ISAACLAB_ROOT
    from isaaclab_tasks.utils import parse_env_cfg
    from newton import BodyFlags

    from robolearn.isaaclab import IsaacLabEnv

    started = time.perf_counter()
    require(spec["mdp"]["task"] == TASK, "Expected the stock state-based Franka cable task")
    action_scale = float(spec["mdp"].get("policy_action_scale", 1.0))
    require(math.isfinite(action_scale) and action_scale > 0, "Policy action scale must be finite and positive")
    require(
        action_scale == 1.0 or args.recipe in (None, "official_flashsac", "cuda_flashsac"),
        "Nonunit action gain is a separate SAC control, not a matched PPO recipe",
    )
    require(Path(ISAACLAB_ROOT).resolve() == args.isaaclab_root, "Isaac Lab root differs from the frozen checkout")
    require(Path(isaaclab_tasks.__file__).resolve().is_relative_to(args.isaaclab_root), "Unexpected task import root")
    cfg = parse_env_cfg(
        TASK, device=args.device, num_envs=num_envs, overrides=[f"physics={PHYSICS}", "visualizer=none"]
    )
    cfg.compute_final_obs, cfg.seed = True, seed
    cfg.commands.cable_pose.debug_vis = False
    require(math.isclose(cfg.sim.dt, spec["mdp"]["physics_dt"]), "Stock physics timestep differs")
    require(cfg.decimation == spec["mdp"]["control_decimation"], "Stock control decimation differs")
    require(all(not viewer.backend for viewer in cfg.sim.visualizer_cfgs), "Headless comparison enabled a viewer")
    if evaluation:
        settings = spec["evaluation"]
        cfg.curriculum.gravity = cfg.curriculum.action_rate = None
        gravity = settings["gravity"]
        cfg.sim.gravity = tuple(gravity)
        cfg.events.variable_gravity.params["gravity_distribution_params"] = (list(gravity), list(gravity))
        cfg.rewards.action_rate.weight = settings["action_rate_weight"]
    task_root = args.isaaclab_root / "source/isaaclab_tasks/isaaclab_tasks/core/lift"
    asset_root = Path(os.environ["ISAACSIM_ASSET_ROOT"])
    metadata = {
        "task": TASK,
        "physics": PHYSICS,
        "renderer": "none",
        "visualizer": "none",
        "task_files_sha256": {
            path.relative_to(args.isaaclab_root).as_posix(): sha(path) for path in sorted(task_root.rglob("*.py"))
        },
        "runtime_contract_files_sha256": {"uv.lock": sha(args.isaaclab_root / "uv.lock")},
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
            "clip_actions": None,
            "init_at_random_ep_len": False,
            "command_debug_vis": False,
        },
        "evaluation_final_curriculum": evaluation,
        "full_cable_gravity_verified": False,
        "snapshot_scope": "scene, commands, targets, actions, clocks, gravity, sticky success; solver cold reset, not RNG continuation",
        "policy_action_scale": action_scale,
    }
    with launch_simulation(cfg):
        require(not cfg.sim.visualizer_cfgs, "Runtime preparation enabled an unexpected visualizer")
        env = gym.make(TASK, cfg=cfg).unwrapped
        try:
            wrapper = IsaacLabEnv(env, clip_actions=None, action_scale=action_scale)
            require((wrapper.observation_dim, wrapper.action_dim) == (69, 8), "Unexpected cable policy dimensions")
            require(env.scene["robot"].is_fixed_base, "Height reconstruction requires the stock fixed-base Franka")
            solver = env.sim.backend_manager.backend.solver
            entries = {
                name: {
                    "view_body_count": int(solver.view(name).body_count),
                    "view_shape_count": int(solver.view(name).shape_count),
                    "state_body_shape": list(solver.entry_state(name).body_q.shape),
                    "solver_class": type(solver.solver(name)).__name__,
                }
                for name in solver.entry_names()
            }
            require(
                set(entries) == {"rigid", "cable"} and all(item["view_body_count"] > 0 for item in entries.values()),
                "Empty coupled solver entry",
            )
            proxy_count = int(((wp.to_torch(solver.view("cable").body_flags) & int(BodyFlags.PROXY)) != 0).sum().item())
            require(proxy_count > 0, "The cable entry contains no actual rigid-body proxies")
            metadata["coupling"] = {
                "solver_class": type(solver).__name__,
                "entries": entries,
                "cable_view_proxy_body_count": proxy_count,
                "rigid_to_cable_contacts_allocated": solver.get_proxy_contacts("rigid", "cable") is not None,
                "nonempty_proxy_selection_required_by_stock_constructor": True,
            }
            metadata["observation_layout"] = observation_layout(env)
            if evaluation:
                expected = torch.tensor(spec["evaluation"]["gravity"], device=env.device, dtype=torch.float32)
                require(bool((_gravity(env) == expected).all().item()), "Actual Newton gravity is not full gravity")
                metadata["full_cable_gravity_verified"] = True
            metadata["startup_seconds"] = time.perf_counter() - started
            yield wrapper, env, metadata
        finally:
            env.close()


@contextmanager
def environment(args, spec, num_envs, seed):
    """Compatibility wrapper for the common collector's environment factory."""
    evaluation = args.phase in ("fixture", "evaluate-worker")
    with build_environment(args, spec, num_envs, seed, evaluation=evaluation) as (wrapper, _env, metadata):
        yield wrapper, metadata


def snapshot(env):
    """Copy canonical physical and task state; every leaf is an owned tensor."""
    import torch

    command, actions = env.command_manager.get_term("cable_pose"), env.action_manager
    targets = env.scene["robot"].actuators.target_command
    state = {
        "scene": env.scene.get_state(is_relative=True),
        "command": {
            name: getattr(command, name)
            for name in ("pose_command_b", "pose_command_w", "time_left", "command_counter")
        },
        "command_metrics": command.metrics,
        "targets": {name: getattr(targets, name).torch for name in ("position", "velocity", "effort")},
        "action": {"current": actions.action, "previous": actions.prev_action},
        "action_terms": {
            name: {key: getattr(actions.get_term(name), key) for key in ("raw_actions", "processed_actions")}
            for name in actions.active_terms
        },
        "clock": {
            "episode_length": env.episode_length_buf,
            "common_step": torch.tensor(env.common_step_counter, dtype=torch.int64),
            "simulation_step": torch.tensor(env._sim_step_counter, dtype=torch.int64),
        },
        "gravity": _gravity(env),
        "goal_succeeded": env.reward_manager.get_term_cfg("cable_goal_tracking").func._succeeded,
    }
    return tree_map(state, lambda value: value.detach().clone())


def restore(env, state):
    """Restore a fixture and return freshly recomputed raw policy observations."""
    import torch
    from newton import ModelFlags

    state = tree_map(state, lambda value: value.to(env.device))
    env.common_step_counter = int(state["clock"]["common_step"].item())
    env._sim_step_counter = int(state["clock"]["simulation_step"].item())
    env.reset_to(state["scene"], is_relative=True)
    with torch.no_grad():
        command = env.command_manager.get_term("cable_pose")
        for name, value in state["command"].items():
            getattr(command, name).copy_(value)
        for name, value in state["command_metrics"].items():
            command.metrics[name].copy_(value)
        targets = env.scene["robot"].actuators.target_command
        for name, value in state["targets"].items():
            getattr(targets, f"set_{name}_index")(value=value)
        env.action_manager.action.copy_(state["action"]["current"])
        env.action_manager.prev_action.copy_(state["action"]["previous"])
        for name, buffers in state["action_terms"].items():
            for key, value in buffers.items():
                getattr(env.action_manager.get_term(name), key).copy_(value)
        env.episode_length_buf.copy_(state["clock"]["episode_length"])
        env.reward_manager.get_term_cfg("cable_goal_tracking").func._succeeded.copy_(state["goal_succeeded"])
        _gravity(env).copy_(state["gravity"])
        env.sim.backend_manager.add_model_change(ModelFlags.MODEL_PROPERTIES)
        env.scene.write_data_to_sim()
        env.sim.forward()
        full_gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32, device=env.device)
        require(bool((_gravity(env) == full_gravity).all().item()), "Restored actual gravity differs from full gravity")
        env.obs_buf = env.observation_manager.compute(update_history=True)
        return env.obs_buf["policy"].clone()


def evaluate_episode(learner, wrapper, state, settings):
    """Score first episodes from raw pre-reset observations, not reset scenes."""
    import torch
    from isaaclab.utils.math import quat_apply

    env = wrapper.env
    env.seed(settings["common_seed"])
    observations = restore(env, state)
    actual = snapshot(env)
    hashes = state_hashes({name: value.cpu().numpy() for name, value in flatten(actual).items()})
    expected = state_hashes({name: value.cpu().numpy() for name, value in flatten(state).items()})
    require(hashes == expected, "Canonical cable/manager state was not restored bit exactly")
    initial_hash = array_hash(observations.cpu().numpy())
    layout, count = observation_layout(env), wrapper.num_envs
    root = state["scene"]["articulation"]["robot"]["root_pose"].to(env.device)
    initial_middle_height = state["scene"]["cable_object"]["cable"]["segment_pose"][
        :, settings["goal_segment_index"], 2
    ].to(env.device)
    active = torch.ones(count, dtype=torch.bool, device=env.device)
    integers = (
        "length",
        "success_steps",
        "longest_hold",
        "final_hold",
        "nonfinite_action_count",
        "action_saturated_count",
    )
    floats = ("return", "distance_sum", "middle_height_sum", "mean_height_sum", "minimum_height_sum", "action_sq_sum")
    values = {name: torch.zeros(count, dtype=torch.int64, device=env.device) for name in integers}
    values.update({name: torch.zeros(count, dtype=torch.float64, device=env.device) for name in floats})
    for name in ("ever_success", "final_success", "clean_timeout", "raw_timeout", "terminated", "ever_lifted"):
        values[name] = torch.zeros_like(active)
    for name in ("final_distance", "minimum_distance"):
        values[name] = torch.full((count,), float("inf"), dtype=torch.float64, device=env.device)
    values["middle_height_max"] = initial_middle_height.double().clone()
    values["middle_height_gain_max"] = torch.zeros(count, dtype=torch.float64, device=env.device)
    for name in env.termination_manager.active_terms:
        values[f"failure.{name}"] = torch.zeros_like(active)
    hold = torch.zeros(count, dtype=torch.int64, device=env.device)
    finite_actions = torch.ones((), dtype=torch.bool, device=env.device)
    expected_gravity = torch.tensor(settings["gravity"], device=env.device, dtype=torch.float32)
    require(bool((_gravity(env) == expected_gravity).all().item()), "Restored evaluation gravity is not full gravity")
    with torch.no_grad():
        for _step in range(settings["horizon_steps"]):
            actions = learner.actions(observations, deterministic=True)
            finite_actions &= torch.isfinite(actions).all()
            observations, transition = wrapper.step(actions)
            raw = transition["next_observation"]
            begin, end = layout["cable_segment_positions"]
            segments = raw[:, begin:end].reshape(count, 12, 3)
            begin, _end = layout["target_position"]
            distance = (segments[:, settings["goal_segment_index"]] - raw[:, begin : begin + 3]).norm(dim=-1)
            world = quat_apply(root[:, None, 3:].expand(-1, 12, -1).reshape(-1, 4), segments.reshape(-1, 3))
            height = world.reshape(count, 12, 3)[..., 2] + root[:, None, 2]
            middle_height = height[:, settings["goal_segment_index"]]
            success = active & (distance < settings["success_threshold_m"])
            hold = torch.where(success, hold + 1, 0)
            values["length"] += active
            values["success_steps"] += success
            values["ever_success"] |= success
            values["longest_hold"] = torch.maximum(values["longest_hold"], hold)
            gain = middle_height - initial_middle_height
            values["ever_lifted"] |= active & (middle_height > settings["lift_height_threshold_m"])
            values["middle_height_max"] = torch.where(
                active, torch.maximum(values["middle_height_max"], middle_height), values["middle_height_max"]
            )
            values["middle_height_gain_max"] = torch.maximum(
                values["middle_height_gain_max"], torch.where(active, gain, 0)
            )
            values["minimum_distance"] = torch.where(
                active, torch.minimum(values["minimum_distance"], distance), values["minimum_distance"]
            )
            values["final_distance"] = torch.where(active, distance, values["final_distance"])
            per_step = {
                "return": transition["reward"],
                "distance_sum": distance,
                "middle_height_sum": middle_height,
                "mean_height_sum": height.mean(-1),
                "minimum_height_sum": height.min(-1).values,
                "action_sq_sum": transition["action"].square().sum(-1),
                "nonfinite_action_count": (~torch.isfinite(actions)).sum(-1),
                "action_saturated_count": (transition["action"].abs() >= 1).sum(-1),
            }
            for name, value in per_step.items():
                values[name] += torch.where(active, value, 0)
            done = active & (transition["terminated"] | transition["truncated"])
            values["final_success"] |= done & success
            values["final_hold"] = torch.where(done, hold, values["final_hold"])
            values["terminated"] |= done & transition["terminated"]
            values["raw_timeout"] |= done & transition["truncated"]
            values["clean_timeout"] |= done & transition["truncated"] & ~transition["terminated"]
            for name in env.termination_manager.active_terms:
                values[f"failure.{name}"] |= done & env.termination_manager.get_term(name)
            active &= ~done
    require(not bool(active.any().item()), "First cable episode exceeded the common evaluation horizon")
    require(bool(finite_actions.item()), "Nonfinite deterministic cable action")
    return {name: value.cpu().numpy() for name, value in values.items()}, hashes, initial_hash


def evaluation_metrics(values, settings):
    """Aggregate physical task outcomes separately from survival and returns."""
    steps = int(values["length"].sum())
    metrics = {
        "ever_goal_fraction": float(values["ever_success"].mean()),
        "final_success_fraction": float(values["final_success"].mean()),
        "final_hold_fraction": float(
            np.mean(
                (values["length"] == settings["horizon_steps"]) & (values["final_hold"] >= settings["final_hold_steps"])
            )
        ),
        "goal_occupancy_fraction": float(
            values["success_steps"].sum() / (settings["episodes"] * settings["horizon_steps"])
        ),
        "active_only_goal_occupancy_fraction": float(values["success_steps"].sum() / max(steps, 1)),
        "goal_distance_active_mean": float(values["distance_sum"].sum() / max(steps, 1)),
        "final_goal_distance_mean": float(values["final_distance"].mean()),
        "minimum_goal_distance_mean": float(values["minimum_distance"].mean()),
        "return_mean": float(values["return"].mean()),
        "return_std": float(values["return"].std()),
        "episode_length_mean": float(values["length"].mean()),
        "clean_timeout_fraction": float(values["clean_timeout"].mean()),
        "raw_timeout_fraction": float(values["raw_timeout"].mean()),
        "failure_fraction": float(values["terminated"].mean()),
        "lift_fraction": float(values["ever_lifted"].mean()),
        "middle_height_max_mean": float(values["middle_height_max"].mean()),
        "middle_height_gain_max_mean": float(values["middle_height_gain_max"].mean()),
        "mean_cable_height_active_mean": float(values["mean_height_sum"].sum() / max(steps, 1)),
        "minimum_cable_height_active_mean": float(values["minimum_height_sum"].sum() / max(steps, 1)),
        "cable_out_of_bounds_fraction": float(values["failure.cable_out_of_bounds"].mean()),
        "ee_below_table_fraction": float(values["failure.ee_below_table"].mean()),
        "joint_velocity_failure_fraction": float(values["failure.joint_vel_out_of_limit"].mean()),
        "simultaneous_failure_timeout_fraction": float(np.mean(values["terminated"] & values["raw_timeout"])),
        "action_rms": math.sqrt(float(values["action_sq_sum"].sum() / max(steps * 8, 1))),
        "action_abs_ge_one_fraction": float(values["action_saturated_count"].sum() / max(steps * 8, 1)),
        "nonfinite_action_count": int(values["nonfinite_action_count"].sum()),
        "active_first_episode_steps": steps,
    }
    for key in ("middle", "mean", "minimum"):
        metrics[f"cable_{key}_height_mean"] = float(values[f"{key}_height_sum"].sum() / max(steps, 1))
    metrics.update(
        {
            name.replace("failure.", "termination_") + "_fraction": float(value.mean())
            for name, value in values.items()
            if name.startswith("failure.")
        }
    )
    metrics["primary_quality_pass"] = None
    return metrics
