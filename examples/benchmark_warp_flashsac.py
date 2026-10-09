# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Measure startup and warmed FlashSAC components; this is not a learning-quality benchmark."""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import warp as wp
import warp_nn

from robolearn.flashsac import FlashSAC, FlashSACConfig
from robolearn.warp import WarpFlashSAC


def timed(call, count, device):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(count):
        call()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / count


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
    result = {}
    for name, network in (("actor", agent._actor), ("critic", agent._critic), ("temperature", agent._temperature)):
        optimizer = network.optimizer
        parameter = optimizer.param_groups[0]["params"][0]
        step = optimizer.state.get(parameter, {}).get("step", 0)
        result[name] = int(step.item() if isinstance(step, torch.Tensor) else step)
    return result


def measured_updates(call, agent, count, device):
    before = optimizer_steps(agent)
    first_phase = agent._update_step
    seconds = timed(call, count, device)
    after = optimizer_steps(agent)
    actor_calls = (
        sum((first_phase + i) % agent.cfg.actor_update_period == 0 for i in range(count))
        if isinstance(agent, WarpFlashSAC)
        else sum((first_phase + i) % agent._cfg.actor_update_period == 0 for i in range(count))
    )
    attempted = {"actor": actor_calls, "critic": count, "temperature": actor_calls}
    actual = {name: after[name] - before[name] for name in before}
    return {
        "update_seconds_per_call": seconds,
        "attempted_optimizer_updates": attempted,
        "actual_optimizer_updates": actual,
        "skipped_optimizer_updates": {name: attempted[name] - actual[name] for name in attempted},
        "all_attempted_optimizer_updates_completed": attempted == actual,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--observations", type=int, default=123)
    parser.add_argument("--actions", type=int, default=37)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--actor-calls", type=int, default=100)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-amp", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.updates, args.actor_calls) < 1:
        parser.error("Measurement and warmup counts must be positive.")
    torch.set_num_threads(8)
    device = torch.device(args.device)
    torch.manual_seed(0)
    wp.init()
    cfg = FlashSACConfig(
        device=args.device,
        buffer_max_length=max(65536, 16 * args.num_envs),
        buffer_min_length=args.num_envs,
        sample_batch_size=args.batch_size,
        use_compile=args.torch_compile,
        use_amp=args.torch_amp,
        learning_rate_decay_step=98400,
    )
    start = time.perf_counter()
    reference = FlashSAC(args.observations, args.actions, num_envs=args.num_envs, cfg=cfg)
    torch.cuda.synchronize(device)
    torch_construct = time.perf_counter() - start

    def raw(network):
        return {k: v.detach().cpu().numpy() for k, v in network.state_dict().items()}

    actor_state = raw(reference._actor._raw_network)
    critic_state = raw(reference._critic._raw_network)
    target_state = raw(reference._target_critic._raw_network)
    transitions = []
    for _ in range(16):
        transition = {
            "observation": torch.randn((args.num_envs, args.observations), device=device),
            "next_observation": torch.randn((args.num_envs, args.observations), device=device),
            "action": torch.rand((args.num_envs, args.actions), device=device) * 2 - 1,
            "reward": torch.randn(args.num_envs, device=device) * 0.1,
            "terminated": torch.zeros(args.num_envs, dtype=torch.bool, device=device),
            "truncated": torch.zeros(args.num_envs, dtype=torch.bool, device=device),
        }
        transitions.append(transition)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for transition in transitions:
        reference.process_transition(transition)
    torch.cuda.synchronize(device)
    torch_replay_fill = time.perf_counter() - start
    observations = transitions[-1]["next_observation"]

    last_information = {}

    def torch_update():
        nonlocal last_information
        last_information = reference.update(tensor_metrics=True)
        return last_information

    start = time.perf_counter()
    for _ in range(args.warmup):
        reference.act(observations)
        torch_update()
    torch.cuda.synchronize(device)
    torch_warmup = time.perf_counter() - start
    torch_actor_seconds = timed(lambda: reference.act(observations), args.actor_calls, device)
    torch_update_results = measured_updates(torch_update, reference, args.updates, device)
    for network in (reference._actor, reference._critic, reference._temperature):
        if not all(torch.isfinite(parameter).all().item() for parameter in network._raw_network.parameters()):
            raise RuntimeError("Torch learner produced nonfinite parameters.")
    records = [
        {
            "backend": "torch",
            "precision": "fp16_amp" if args.torch_amp else "fp32",
            "compiled": args.torch_compile,
            "actor_compiled": args.torch_compile,
            "actor_precision": "fp32",
            "construction_seconds": torch_construct,
            "common_weight_loading_seconds": 0.0,
            "replay_fill_seconds": torch_replay_fill,
            "preparation_seconds": 0.0,
            "warmup_seconds": torch_warmup,
            "startup_seconds": torch_construct + torch_replay_fill + torch_warmup,
            "actor_seconds_per_call": torch_actor_seconds,
            **torch_update_results,
            "final_metrics": {key: float(value.detach().cpu()) for key, value in last_information.items()},
        }
    ]

    # Each backend starts from the same initial weights and temperature. This is a
    # throughput comparison, not a claim of identical sampled trajectories.
    stream = wp.Stream(args.device)
    torch_mirror = wp.stream_to_torch(stream)
    stream.wait_stream(wp.get_stream(args.device))
    for captured in (False, True):
        with wp.ScopedStream(stream, sync_enter=False), torch.cuda.stream(torch_mirror):
            start = time.perf_counter()
            agent = WarpFlashSAC(
                args.observations,
                args.actions,
                num_envs=args.num_envs,
                cfg=replace(cfg, use_compile=False, use_amp=False),
            )
            torch.cuda.synchronize(device)
            construction = time.perf_counter() - start
            start = time.perf_counter()
            agent.actor.load_author_state_dict(actor_state)
            agent.critic.load_author_state_dict(critic_state)
            agent.target_critic.load_author_state_dict(target_state)
            torch.cuda.synchronize(device)
            weight_loading = time.perf_counter() - start
            start = time.perf_counter()
            for transition in transitions:
                flags = {k: v.to(torch.int32) for k, v in transition.items() if k in ("terminated", "truncated")}
                agent.process_transition(
                    {
                        k: wp.from_torch(flags[k], dtype=wp.int32) if k in flags else wp.from_torch(v, dtype=wp.float32)
                        for k, v in transition.items()
                    }
                )
            torch.cuda.synchronize(device)
            replay_fill = time.perf_counter() - start
            start = time.perf_counter()
            agent.prepare(capture=captured)
            torch.cuda.synchronize(device)
            preparation = time.perf_counter() - start
            warp_obs = wp.from_torch(observations, dtype=wp.float32)

            def update():
                return agent.update(tensor_metrics=True, capture=captured)

            start = time.perf_counter()
            for _ in range(args.warmup):
                agent.act(warp_obs)
                update()
            torch.cuda.synchronize(device)
            warmup = time.perf_counter() - start
            actor_seconds = timed(lambda: agent.act(warp_obs), args.actor_calls, device)
            update_results = measured_updates(update, agent, args.updates, device)
            records.append(
                {
                    "backend": "warp_nn",
                    "precision": "fp32",
                    "captured": captured,
                    "actor_captured": False,
                    "actor_precision": "fp32",
                    "construction_seconds": construction,
                    "common_weight_loading_seconds": weight_loading,
                    "replay_fill_seconds": replay_fill,
                    "preparation_seconds": preparation,
                    "warmup_seconds": warmup,
                    "startup_seconds": construction + weight_loading + replay_fill + preparation + warmup,
                    "actor_seconds_per_call": actor_seconds,
                    **update_results,
                    "final_metrics": agent.get_metrics(),
                }
            )
            if not all(np.isfinite(parameter.numpy()).all() for parameter in agent.parameters()):
                raise RuntimeError("Warp learner produced nonfinite parameters.")
    result = {
        "scope": "Isolated learner components; no simulator, MDP, or walking evaluation.",
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "warp": wp.__version__,
        "warp_nn": warp_nn.__version__,
        "config": asdict(cfg),
        "arguments": {**vars(args), "output": str(args.output)},
        "timing": "Synchronized wall time, amortized over calls; preparation reported separately.",
        "update_unit": "One critic/replay update, actor and temperature every second call; two calls per G1 vector step.",
        "startup_scope": (
            "Construction, common-weight loading, replay ingestion, preparation and warmup. "
            "Shared random fixtures and Torch-to-NumPy initial-weight export excluded. "
            "Torch compilation occurs during warmup; Warp compilation occurs during construction/preparation/warmup."
        ),
        "limits": "Different RNG and optimizer state; descriptive single-process measurements, not quality results.",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(records, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
