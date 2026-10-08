# Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT

"""Train or evaluate RoboLearn FlashSAC from an installed Isaac Lab 3 environment."""

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch

from robolearn.flashsac import FlashSAC, FlashSACConfig
from robolearn.isaaclab import IsaacLabEnv


def evaluate(adapter, agent, seed):
    """Measure one complete deterministic episode per environment."""
    observations = adapter.reset(seed)
    active = torch.ones(adapter.num_envs, dtype=torch.bool, device=adapter.device)
    returns = torch.zeros(adapter.num_envs, device=adapter.device)
    for _ in range(adapter.env.max_episode_length):
        observations, transition = adapter.step(agent.act(observations, training=False))
        returns += transition["reward"] * active
        active &= ~(transition["terminated"] | transition["truncated"])
        if not active.any():
            break
    return float(returns.mean())


def main():
    """Launch a task, train, and write a policy with reproducible configuration."""
    import isaaclab_tasks  # noqa: F401
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5_000_192, help="Total environment transitions.")
    parser.add_argument("--physics", default=None)
    parser.add_argument("--action_scale", type=float, default=None)
    parser.add_argument("--observation_group", default=None)
    parser.add_argument("--critic_group", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--warmup", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--updates_per_step", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--log_dir", type=Path)
    parser.add_argument("--env_override", action="append", default=None)
    add_launcher_args(parser)
    args = parser.parse_args()
    if min(args.steps, args.num_envs, args.updates_per_step) <= 0:
        parser.error("Steps, environments, and update count must be positive.")
    if args.evaluate_only and not args.checkpoint:
        parser.error("--evaluate_only requires --checkpoint.")
    saved_environment = {}
    if args.checkpoint:
        saved_environment = json.loads((args.checkpoint / "environment.json").read_text())
    defaults = {
        "task": "Isaac-Open-Drawer-Franka",
        "physics": "newton_mjwarp",
        "observation_group": "policy",
        "critic_group": None,
        "env_override": [],
    }
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, saved_environment.get(name, default))
    if args.action_scale is None:
        args.action_scale = saved_environment.get(
            "action_scale", 3.0 if args.task == "Isaac-Open-Drawer-Franka" else 1.0
        )
    # The environment contract is distinct from algorithm state and travels with each checkpoint.
    environment = {name: getattr(args, name) for name in (*defaults, "action_scale")}
    if saved_environment:
        for name in ("task", "observation_group", "critic_group", "action_scale"):
            if environment[name] != saved_environment[name]:
                parser.error(f"Checkpoint {name}={saved_environment[name]!r} does not match {environment[name]!r}.")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        overrides=[f"physics={args.physics}", *args.env_override],
    )
    env_cfg.compute_final_obs = True
    env_cfg.seed = args.seed
    env_cfg.sim.render_interval = env_cfg.decimation
    iterations = math.ceil(args.steps / args.num_envs)
    cfg = FlashSACConfig(
        device=args.device,
        seed=args.seed,
        use_compile=args.compile,
        use_amp=args.device.startswith("cuda"),
        buffer_max_length=args.buffer_size,
        buffer_min_length=args.warmup,
        sample_batch_size=args.batch_size,
        learning_rate_decay_step=iterations * args.updates_per_step,
    )
    if args.checkpoint:
        saved = json.loads((args.checkpoint / "config.json").read_text())
        cfg = FlashSACConfig(**saved["config"])
        cfg.device = args.device
        cfg.buffer_device = args.device
        cfg.buffer_max_length = args.buffer_size
        cfg.use_compile = args.compile
        cfg.load_optimizer = not args.evaluate_only
    log_dir = args.log_dir or Path("logs") / "flashsac" / args.task / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=False)
    arguments = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    (log_dir / "run.json").write_text(json.dumps({"arguments": arguments, "config": asdict(cfg)}, indent=2) + "\n")
    print(f"Artifacts: {log_dir.resolve()}", flush=True)
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            adapter = IsaacLabEnv(env, args.observation_group, args.critic_group, args.action_scale)
            agent = FlashSAC(
                adapter.observation_dim,
                adapter.action_dim,
                adapter.num_envs,
                cfg,
                critic_observation_dim=adapter.critic_observation_dim,
            )
            if args.checkpoint:
                agent.load(str(args.checkpoint))
            initial = evaluate(adapter, agent, args.seed + 10_000)
            print(f"Initial mean return: {initial:.4f}", flush=True)
            if args.evaluate_only:
                (log_dir / "result.json").write_text(json.dumps({"return_mean": initial}, indent=2) + "\n")
                return
            observations = adapter.reset(args.seed)
            adapter.env.episode_length_buf[:] = torch.randint_like(
                adapter.env.episode_length_buf, high=adapter.env.max_episode_length
            )
            updates = 0
            start = time.monotonic()
            with (log_dir / "metrics.jsonl").open("w", buffering=1) as metrics_file:
                for iteration in range(1, iterations + 1):
                    actions = (
                        agent.act(observations)
                        if agent.ready
                        else (2 * torch.rand((adapter.num_envs, adapter.action_dim), device=adapter.device) - 1)
                    )
                    observations, transition = adapter.step(actions)
                    agent.process_transition(transition)
                    metrics = {}
                    if agent.ready:
                        for _ in range(args.updates_per_step):
                            metrics = agent.update()
                            updates += 1
                    if iteration % 100 == 0 or iteration == iterations:
                        metrics.update(
                            step=iteration * adapter.num_envs,
                            updates=updates,
                            reward_mean=float(transition["reward"].mean()),
                            elapsed_s=time.monotonic() - start,
                        )
                        if not all(math.isfinite(value) for value in metrics.values()):
                            raise RuntimeError(f"Non-finite training metric: {metrics}")
                        metrics_file.write(json.dumps(metrics) + "\n")
                        print(
                            f"steps={metrics['step']} updates={updates} reward={metrics['reward_mean']:.4f}", flush=True
                        )
                    if iteration % 1000 == 0 or iteration == iterations:
                        checkpoint = log_dir / f"step{iteration * adapter.num_envs}"
                        agent.save(str(checkpoint))
                        (checkpoint / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
            final = evaluate(adapter, agent, args.seed + 10_000)
            result = {
                "initial_return": initial,
                "final_return": final,
                "steps": iterations * args.num_envs,
                "updates": updates,
                "elapsed_s": time.monotonic() - start,
            }
            (log_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"Final mean return: {final:.4f}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
