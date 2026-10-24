# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Compare four FlashSAC recipes in balanced rounds of fresh worker processes.

This orchestrator imports no learner runtime. Each worker uses the unmodified
official agent or an explicit RoboLearn backend on the same synthetic fixture.
Run the complete protocol on separate allocations to inspect reproducibility;
rounds in one allocation are descriptive timing repetitions, not training seeds.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

RECIPES = {
    "official_amp": ("official", False, "official_flashsac"),
    "torch_port_amp": ("torch-port", False, "robolearn_torch_port"),
    "native_amp": ("native", False, "robolearn_native_cuda_amp_fused"),
    "native_amp_tuned": ("native", True, "robolearn_native_cuda_amp_fused"),
}
OFFICIAL = "87edc9061150ae9e962dd84e6544e27a1554b3ab"


def write_json(path, value):
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


def distribution(values):
    require(values and all(math.isfinite(x) and x > 0 for x in values), "Nonpositive/nonfinite timing or ratio")
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "values": values}


def schedule(rounds, seed):
    """Shuffle Williams-design labels and row order independently per four rounds."""
    generator = random.Random(seed)
    labels = list(RECIPES)
    result, audits = [], []
    for block in range(rounds // 4):
        assignment = labels.copy()
        generator.shuffle(assignment)
        orders = [[assignment[(value + shift) % 4] for value in (0, 1, 3, 2)] for shift in range(4)]
        generator.shuffle(orders)
        positions = {
            label: [sum(order[position] == label for order in orders) for position in range(4)] for label in labels
        }
        predecessors = {
            f"{a}->{b}": sum((a, b) in list(zip(order, order[1:])) for order in orders)
            for a, b in itertools.permutations(labels, 2)
        }
        require(all(counts == [1] * 4 for counts in positions.values()), "Unbalanced positions")
        require(all(count == 1 for count in predecessors.values()), "Unbalanced immediate predecessors")
        audits.append(
            {
                "williams_block": block,
                "label_assignment": assignment,
                "position_counts": positions,
                "immediate_predecessor_counts": predecessors,
            }
        )
        first_round = len(result)
        result.extend(
            {"round": first_round + index, "williams_block": block, "order": order}
            for index, order in enumerate(orders)
        )
    return result, audits


def validate_worker(source, label, args, common):
    backend, tuned, raw_backend = RECIPES[label]
    require(source["status"] == "PASSED" and len(source["records"]) == 1, "Worker failed or returned multiple rows")
    require(source["backend_selection"] == backend, "Worker backend selection differs")
    row = source["records"][0]
    require(
        row["backend"] == raw_backend and row["status"] == "PASSED" and all(row["checks"].values()),
        "Worker controls failed",
    )
    require(row["cuda_autotune"] == tuned, "Actual autotuning selection differs")
    require(row["initial_optimizer_state"]["accepted"], "Worker optimizer/scaler/counters were not fresh")
    require(
        row["common_initial_weights_audit"]["exactly_equal"] and row["replay_audit"]["accepted"],
        "Worker initialization/replay audit failed",
    )
    require(row["common_initial_weights_audit"]["hashes"] == source["common_initial_weights"], "Readback hashes differ")
    require(len(source["common_initial_weights"]) == 89, "Canonical architecture must expose 89 state arrays")
    upstream = source["official_provenance"]
    require(
        upstream["commit"] == OFFICIAL and not upstream["source_patches"] and not upstream["runtime_monkeypatches"],
        "Official implementation differs",
    )
    if "official_reference" in source:
        reference = source["official_reference"]
        require(
            reference["agent_update_calls"] == 0 and not any(reference["optimizer_steps"].values()),
            "Untimed official reference trained",
        )
    if common:
        for key in ("common_initial_weights", "transition_fixture", "robolearn_sources_sha256"):
            require(source[key] == common[key], f"Worker shared {key} differs")
        require(source["benchmark_source"]["sha256"] == common["benchmark_source"]["sha256"], "Worker driver changed")
        for key in ("python", "torch", "warp", "warp_nn", "gpu"):
            require(source["runtime"][key] == common["runtime"][key], f"Worker runtime {key} differs")
    protocol = source["arguments"]
    for key in ("warmup", "updates", "actor_calls", "seed", "observations", "actions", "num_envs", "batch_size"):
        require(protocol[key] == getattr(args, key), f"Worker {key} budget differs")
    require(
        protocol["blocks"] == 1
        and protocol["recipe"] == "amp"
        and protocol["allow_tf32"]
        and protocol["require_complete_updates"],
        "Worker precision/block scope differs",
    )
    require(
        len(row["blocks"]) == 1 and row["checkpoint_saved"] and row["learned_parameters"]["accepted"],
        "Worker output/checkpoint incomplete",
    )
    block = row["blocks"][0]
    expected = {"critic": args.updates, "actor": args.updates // 2, "temperature": args.updates // 2}
    require(
        block["exact_update_call_count"] and row["attempted_optimizer_updates"] == expected,
        "Worker update attempt count differs",
    )
    require(
        block["post_warmup_budget"]["accepted"] and block["post_measurement_budget"]["accepted"],
        "Worker cumulative warmup/full-work budget differs",
    )
    for key in expected:
        actual = row["actual_optimizer_updates"][key]
        require(
            actual == block["optimizer_steps_after"][key] - block["optimizer_steps_before"][key],
            "Optimizer step delta differs",
        )
        require(
            actual == expected[key] and row["skipped_optimizer_updates"][key] == 0,
            "Equal-work protocol requires every optimizer attempt to complete with zero AMP skips",
        )
    if tuned:
        require(row["cuda_tuning_metadata"], "Tuned worker lacks algorithm selection metadata")
    return row


def run_worker(label, phase, directory, args, result, common, canonical, round_index=None, position=None):
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{label}.json"
    backend, tuned, _ = RECIPES[label]
    worker = Path(__file__).with_name("benchmark_official_flashsac.py").resolve()
    command = [
        sys.executable,
        str(worker),
        "--official-root",
        str(args.official_root),
        "--backend",
        backend,
        "--recipe",
        "amp",
        "--allow-tf32",
        "--native-variants",
        "amp-fused",
        "--blocks",
        "1",
        "--record-clocks",
        "--require-complete-updates",
    ]
    for key in (
        "device",
        "warmup",
        "updates",
        "actor_calls",
        "seed",
        "observations",
        "actions",
        "num_envs",
        "batch_size",
    ):
        command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    if canonical is not None:
        command += ["--initial-weights", str(canonical)]
    if tuned:
        command.append("--cuda-autotune")
    command += ["--output", str(output)]
    receipt = {
        "label": label,
        "phase": phase,
        "round": round_index,
        "position": position,
        "argv": command,
        "worker_json": str(output),
        "started_unix": time.time(),
        "status": "RUNNING",
    }
    result["workers"].append(receipt)
    write_json(args.output, result)
    print(f"BALANCED_WORKER_START phase={phase} round={round_index} position={position} label={label}", flush=True)
    with output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        receipt["pid"] = process.pid
        write_json(args.output, result)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        receipt["exit_code"] = process.wait()
    receipt["finished_unix"] = time.time()
    receipt["wall_seconds"] = receipt["finished_unix"] - receipt["started_unix"]
    receipt["status"] = "COMPLETED" if receipt["exit_code"] == 0 else "FAILED"
    write_json(args.output, result)
    require(receipt["exit_code"] == 0, f"Worker failed: {phase}/{label}")
    source = json.loads(output.read_text())
    row = validate_worker(source, label, args, common)
    checkpoint = output.parent / (output.stem + "-artifacts") / row["backend"]
    names = (
        ("agent.npz", "config.json")
        if backend == "native"
        else ("actor.pt", "critic.pt", "target_critic.pt", "temperature.pt", "agent_state.pt")
    )
    require(all((checkpoint / name).is_file() for name in names), "Worker checkpoint payload is missing")
    if canonical is not None:
        require(
            source["common_initial_weights_source"]["sha256"] == sha(canonical), "Canonical input file hash differs"
        )
    receipt.update(
        worker_json_sha256=sha(output),
        status="VERIFIED",
        update_seconds_per_call=row["update_seconds_per_call"],
        actor_seconds_per_call=row["actor_seconds_per_call"],
        startup_seconds=row["startup_seconds"],
        warmup_seconds=row["warmup_seconds"],
        attempted=row["attempted_optimizer_updates"],
        completed=row["actual_optimizer_updates"],
        skipped=row["skipped_optimizer_updates"],
        clock_samples={key: row["blocks"][0][key] for key in ("gpu_clocks_before", "gpu_clocks_after")},
    )
    if row.get("cuda_tuning_metadata"):
        receipt["tuning_metadata_sha256"] = hashlib.sha256(
            json.dumps(row["cuda_tuning_metadata"], sort_keys=True).encode()
        ).hexdigest()
    write_json(args.output, result)
    print(
        f"BALANCED_WORKER_COMPLETE phase={phase} round={round_index} label={label} seconds={row['update_seconds_per_call']:.9f}",
        flush=True,
    )
    return source, receipt


def summarize(result, args):
    paired = []
    for item in result["schedule"]:
        rows = {
            row["label"]: row
            for row in result["workers"]
            if row["phase"] == "measurement" and row["round"] == item["round"]
        }
        require(set(rows) == set(RECIPES), "Incomplete paired round")
        times = {label: row["update_seconds_per_call"] for label, row in rows.items()}
        paired.append(
            {
                "round": item["round"],
                "order": item["order"],
                "update_seconds_per_call": times,
                "native_tuned_over_official_time": times["native_amp_tuned"] / times["official_amp"],
                "native_over_official_time": times["native_amp"] / times["official_amp"],
                "native_tuned_over_untuned_time": times["native_amp_tuned"] / times["native_amp"],
            }
        )
    ratios = [row["native_tuned_over_official_time"] for row in paired]
    wins = sum(value < 1 for value in ratios)
    return {
        "paired_rounds": paired,
        "per_recipe_seconds": {
            label: distribution([row["update_seconds_per_call"][label] for row in paired]) for label in RECIPES
        },
        "paired_ratios": {
            key: distribution([row[key] for row in paired])
            for key in (
                "native_tuned_over_official_time",
                "native_over_official_time",
                "native_tuned_over_untuned_time",
            )
        },
        "engineering_target": {
            "minimum_tuned_native_wins": args.target_min_wins,
            "maximum_median_tuned_native_over_official_time": args.target_median_ratio,
            "observed_tuned_native_wins": wins,
            "observed_rounds": len(paired),
            "observed_median_paired_ratio": statistics.median(ratios),
            "met": wins >= args.target_min_wins and statistics.median(ratios) <= args.target_median_ratio,
            "interpretation": "Descriptive engineering target on this allocation; not statistical significance or a policy-quality claim",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--schedule-seed", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--actor-calls", type=int, default=100)
    parser.add_argument("--observations", type=int, default=123)
    parser.add_argument("--actions", type=int, default=37)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--target-min-wins", type=int, default=7)
    parser.add_argument("--target-median-ratio", type=float, default=0.97)
    args = parser.parse_args()
    if args.rounds < 4 or args.rounds % 4 or args.warmup < 2 or args.warmup % 2 or args.updates < 2 or args.updates % 2:
        parser.error("Use a positive multiple of four rounds and even warmup/update counts >=2.")
    if min(args.actor_calls, args.observations, args.actions, args.num_envs, args.batch_size) < 1:
        parser.error("Dimensions and action timing counts must be positive.")
    if (
        not 0 <= args.target_min_wins <= args.rounds
        or not math.isfinite(args.target_median_ratio)
        or args.target_median_ratio <= 0
    ):
        parser.error("Invalid engineering target.")
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("Use a fresh output path; existing results are not overwritten.")
    args.official_root = args.official_root.expanduser().resolve(strict=True)
    canonical = args.initial_weights.expanduser().resolve(strict=True) if args.initial_weights else None
    artifacts = args.output.parent / (args.output.stem + "-artifacts")
    artifacts.mkdir(exist_ok=False)
    orders, balance = schedule(args.rounds, args.schedule_seed)
    worker = Path(__file__).with_name("benchmark_official_flashsac.py").resolve()
    result = {
        "status": "RUNNING",
        "protocol": "fresh-process-williams-four-recipe-v1",
        "scope": "Synthetic complete learner updates; no physics or policy-quality measurement",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "orchestrator_source": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "worker_source": {"path": str(worker), "sha256": sha(worker)},
        "cache_policy": "Preconditioning workers exercise every recipe first. Disk compiler caches are shared; each fresh tuned worker selects its own algorithms during setup. No independent cold-start comparison.",
        "cache_environment": {
            key: os.environ.get(key)
            for key in ("TORCHINDUCTOR_CACHE_DIR", "WARP_CACHE_PATH", "ROBOLEARN_CUDA_CACHE_DIR")
        },
        "state_policy": "Every worker is a fresh process/learner with canonical weights, the same seed/fixture and zero optimizer/scaler/update history before equal warmup. Backend RNG implementations and trajectories may differ.",
        "aggregation": "All paired rounds retained; median and observed range, no confidence intervals. Rounds are not independent allocations or training seeds.",
        "engineering_target_definition": {
            "minimum_tuned_native_wins": args.target_min_wins,
            "maximum_median_tuned_native_over_official_time": args.target_median_ratio,
            "scope": "Predeclared descriptive target per allocation; failed targets remain reported",
        },
        "schedule": orders,
        "balance": balance,
        "workers": [],
        "started_unix": time.time(),
    }
    write_json(args.output, result)
    try:
        common = None
        for label in RECIPES:
            source, _ = run_worker(
                label, "preconditioning", artifacts / "preconditioning", args, result, common, canonical
            )
            if common is None:
                common = source
                if canonical is None:
                    canonical = artifacts / "preconditioning/official_amp-artifacts/common-initial-weights.npz"
                result["canonical_initial_weights"] = {
                    "path": str(canonical),
                    "sha256": sha(canonical),
                    "arrays_sha256": common["common_initial_weights"],
                }
                result["transition_fixture_arrays_sha256"] = common["transition_fixture"]
        for item in orders:
            for position, label in enumerate(item["order"]):
                run_worker(
                    label,
                    "measurement",
                    artifacts / f"round-{item['round']:02d}",
                    args,
                    result,
                    common,
                    canonical,
                    item["round"],
                    position,
                )
            result["rounds_completed"] = item["round"] + 1
            write_json(args.output, result)
        result["summary"] = summarize(result, args)
        result["status"] = "PASSED"
    except BaseException as error:
        result.update(
            status="FAILED",
            failure={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        )
        raise
    finally:
        result["finished_unix"] = time.time()
        result["protocol_wall_seconds"] = result["finished_unix"] - result["started_unix"]
        write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "engineering_target": result["summary"]["engineering_target"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
