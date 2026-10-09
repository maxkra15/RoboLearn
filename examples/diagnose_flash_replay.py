# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Compare Warp FlashSAC replay and reward scaling with the authors' Torch code.

This standalone numerical experiment uses fixed transitions, three environments
and a small nondivisible ring. CUDA is required for its capture checks. It writes
strict JSON and exits unsuccessfully if any comparison fails; no simulator or
training run is launched. Torch is an optional diagnostic dependency.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp

from robolearn.flashsac._utils.reward_normalization import RewardNormalizer
from robolearn.flashsac.config import FlashSACConfig
from robolearn.flashsac.replay import TorchUniformBuffer
from robolearn.warp._flash_replay import WarpFlashReplay, WarpRewardNormalizer


class Comparisons:
    """Accumulate named numerical results without stopping at the first mismatch."""

    def __init__(self, atol: float, rtol: float):
        self.atol, self.rtol = atol, rtol
        self.checks = {}

    def array(self, name, actual, expected, *, exact=False):
        actual, expected = np.asarray(actual), np.asarray(expected)
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        same_shape = actual.shape == expected.shape
        error = None
        passed = False
        if finite and same_shape:
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)), initial=0.0))
            passed = bool(
                np.array_equal(actual, expected)
                if exact
                else np.allclose(actual, expected, atol=self.atol, rtol=self.rtol)
            )
        previous = self.checks.get(name, {"passed": True, "finite": True, "comparisons": 0, "max_absolute_error": 0.0})
        self.checks[name] = {
            "passed": previous["passed"] and passed,
            "finite": previous["finite"] and finite,
            "comparisons": previous["comparisons"] + 1,
            "max_absolute_error": (
                max(previous["max_absolute_error"], error)
                if previous["max_absolute_error"] is not None and error is not None
                else None
            ),
            "exact": exact,
        }

    def condition(self, name, passed):
        previous = self.checks.get(name, {"passed": True, "comparisons": 0})
        self.checks[name] = {"passed": previous["passed"] and bool(passed), "comparisons": previous["comparisons"] + 1}


def fixture(step: int) -> dict[str, np.ndarray]:
    observation = (100.0 * step + 10.0 * np.arange(3)[:, None] + np.arange(5)[None, :]).astype(np.float32)
    terminated = np.zeros(3, dtype=np.int32)
    truncated = np.zeros(3, dtype=np.int32)
    if step in (0, 4, 8):
        terminated[0] = 1
    if step in (1, 5, 9):
        truncated[1] = 1
    if step in (2, 6, 10):
        terminated[2] = 1
        truncated[2] = 1
    next_observation = observation + np.float32(5.0)
    next_observation[(terminated | truncated).astype(bool)] += np.float32(10000.0)
    return {
        "observation": observation,
        "action": (np.array([[0.2, -0.3], [0.4, -0.1], [-0.2, 0.5]], dtype=np.float32) + step * 0.01).astype(
            np.float32
        ),
        "reward": (np.array([1.0, -2.0, 0.5], dtype=np.float32) + step * np.array([0.1, 0.05, -0.025])).astype(
            np.float32
        ),
        "terminated": terminated,
        "truncated": truncated,
        "next_observation": next_observation,
    }


def population(replay) -> dict[str, np.ndarray]:
    count = len(replay)
    return {key: value[:count].numpy().copy() for key, value in replay.storage.items()}


def reference_population(reference) -> dict[str, np.ndarray]:
    fields = {
        "observation": "_observations",
        "action": "_actions",
        "reward": "_rewards",
        "terminated": "_terminateds",
        "truncated": "_truncateds",
        "next_observation": "_next_observations",
        "discount": "_discounts",
    }
    return {key: getattr(reference, name)[: len(reference)].cpu().numpy().copy() for key, name in fields.items()}


def compare_population(checks, replay, reference, prefix="torch_replay"):
    checks.condition(f"{prefix}.length", len(replay) == len(reference))
    checks.condition(f"{prefix}.readiness", replay.can_sample() == reference.can_sample())
    for key, value in population(replay).items():
        checks.array(f"{prefix}.{key}", value, reference_population(reference)[key], exact=key != "reward")


def compare_normalizer(checks, normalizer, reference, prefix="reward_statistics"):
    state = normalizer.state_dict()
    checks.array(f"{prefix}.returns", state["returns"], reference.G_r.cpu().numpy())
    checks.array(f"{prefix}.maximum", state["maximum"], reference.G_r_max.cpu().numpy())
    checks.array(
        f"{prefix}.mean_variance_count",
        state["running"],
        np.array([reference.G_rms.mean.item(), reference.G_rms.var.item(), reference.G_rms.count.item()]),
    )


def disk_roundtrip(state):
    with tempfile.TemporaryDirectory(prefix="robolearn-replay-") as directory:
        path = Path(directory) / "replay.npz"
        np.savez(path, **state)
        with np.load(path, allow_pickle=False) as saved:
            return {key: value.copy() for key, value in saved.items()}


def run(args, checks):
    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise ValueError("This diagnostic requires a CUDA device for graph replay comparisons.")
    torch_device = torch.device(args.device)
    cfg = FlashSACConfig(
        device=args.device,
        seed=19,
        n_step=3,
        buffer_max_length=args.capacity,
        buffer_min_length=4,
        sample_batch_size=32,
    )
    replay = WarpFlashReplay(5, 2, 3, cfg, device)
    initial_rng = replay._rng.numpy().copy()
    reference = TorchUniformBuffer(
        gym.spaces.Box(-np.inf, np.inf, shape=(3, 5), dtype=np.float32),
        gym.spaces.Box(-1.0, 1.0, shape=(3, 2), dtype=np.float32),
        cfg.n_step,
        cfg.gamma,
        cfg.buffer_max_length,
        cfg.buffer_min_length,
        cfg.sample_batch_size,
        args.device,
    )
    normalizer = WarpRewardNormalizer(3, cfg, device)
    reference_normalizer = RewardNormalizer(cfg.gamma, cfg.normalized_G_max, True, torch_device)
    inputs = {
        key: wp.zeros(value.shape, dtype=wp.int32 if key in ("terminated", "truncated") else wp.float32, device=device)
        for key, value in fixture(0).items()
    }
    raw_rewards = np.linspace(-4.0, 4.0, cfg.sample_batch_size, dtype=np.float32)
    reward_batch = wp.array(raw_rewards, device=device)
    clones = {}
    checkpoint_positions = []
    for step in range(12):
        transition = fixture(step)
        for key, value in transition.items():
            inputs[key].assign(value)
        values = [inputs[key] for key in transition]
        replay.add(*values)
        reference.add({key: torch.tensor(value, device=torch_device) for key, value in transition.items()})
        normalizer.update(inputs["reward"], inputs["terminated"], inputs["truncated"])
        reference_normalizer.update_reward_stats(
            *[torch.tensor(transition[key], device=torch_device) for key in ("reward", "terminated", "truncated")]
        )
        compare_population(checks, replay, reference)
        compare_normalizer(checks, normalizer, reference_normalizer)
        checks.array(
            "normalized_rewards",
            normalizer.normalize_rewards(reward_batch).numpy(),
            reference_normalizer.normalize_rewards(torch.tensor(raw_rewards, device=torch_device)).cpu().numpy(),
        )
        for name, clone in clones.items():
            clone.add(*values)
            for key, value in population(replay).items():
                checks.array(f"{name}.continued_population.{key}", population(clone)[key], value, exact=True)
        if step == 2:
            expected_next = np.vstack([fixture(i)["next_observation"][i] for i in range(3)])
            checks.array(
                "early_done.pre_reset_next_observation",
                population(replay)["next_observation"],
                expected_next,
                exact=True,
            )
            checks.array("early_done.terminated", population(replay)["terminated"], [1.0, 0.0, 1.0], exact=True)
            checks.array("early_done.truncated", population(replay)["truncated"], [0.0, 1.0, 1.0], exact=True)
            checks.array(
                "early_done.actual_discount",
                population(replay)["discount"],
                np.array([cfg.gamma, cfg.gamma**2, cfg.gamma**3], dtype=np.float32),
            )
        if replay.can_sample():
            batch = replay.sample()
            expected = reference.sample(sample_idxs=replay._indices.numpy())
            for key, value in batch.items():
                checks.array(f"sample_gather.{key}", value.numpy(), expected[key].cpu().numpy(), exact=key != "reward")
            for name, clone in clones.items():
                cloned = clone.sample()
                for key, value in batch.items():
                    checks.array(f"{name}.continued_rng_sample.{key}", cloned[key].numpy(), value.numpy(), exact=True)
        if step in (1, 5):
            name = "partial_pending_checkpoint" if step == 1 else "wrapped_full_checkpoint"
            state = disk_roundtrip(replay.state_dict())
            clone = WarpFlashReplay(5, 2, 3, cfg, device)
            clone.load_state_dict(state)
            clones[name] = clone
            checkpoint_positions.append({"name": name, "step": step, "counters": state["counters"].tolist()})
            checks.array(f"{name}.counters", clone._counters.numpy(), state["counters"], exact=True)
            checks.array(f"{name}.rng", clone._rng.numpy(), state["rng"], exact=True)
    checks.condition("ring_wrap.nondivisible", cfg.buffer_max_length % 3 != 0)
    checks.condition("ring_wrap.full", len(replay) == cfg.buffer_max_length)

    state = normalizer.state_dict()
    restored_normalizer = WarpRewardNormalizer(3, cfg, device)
    restored_normalizer.load_state_dict(disk_roundtrip(state))
    for key, value in restored_normalizer.state_dict().items():
        checks.array(f"normalizer_checkpoint.{key}", value, state[key], exact=True)
    population_before = population(replay)
    snapshot = replay.state_dict(include_storage=False)
    addresses = {key: value.ptr for key, value in replay.storage.items()}
    replay.sample()
    checks.condition("metadata_snapshot.rng_changed", not np.array_equal(replay._rng.numpy(), snapshot["rng"]))
    replay.load_state_dict(snapshot, restore_storage=False)
    checks.array("metadata_snapshot.rng_restored", replay._rng.numpy(), snapshot["rng"], exact=True)
    for key, value in population(replay).items():
        checks.array(f"metadata_snapshot.population_unchanged.{key}", value, population_before[key], exact=True)
    checks.condition(
        "metadata_snapshot.storage_addresses_unchanged",
        addresses == {key: value.ptr for key, value in replay.storage.items()},
    )

    eager = WarpFlashReplay(5, 2, 3, cfg, device)
    eager.load_state_dict(replay.state_dict())
    captured_first = {key: wp.empty_like(value) for key, value in replay.batch.items()}
    with wp.ScopedCapture(device=device) as capture:
        first = replay.sample(check_ready=False)
        for key, value in first.items():
            wp.copy(captured_first[key], value)
        replay.sample(check_ready=False)
    replay.load_state_dict(snapshot, restore_storage=False)
    previous_rng = snapshot["rng"]
    for _ in range(3):
        expected_first = {key: value.numpy().copy() for key, value in eager.sample().items()}
        expected_last = {key: value.numpy().copy() for key, value in eager.sample().items()}
        wp.capture_launch(capture.graph)
        for key, value in replay.batch.items():
            checks.array(f"captured_pair.first.{key}", captured_first[key].numpy(), expected_first[key], exact=True)
            checks.array(f"captured_pair.second.{key}", value.numpy(), expected_last[key], exact=True)
        current_rng = replay._rng.numpy().copy()
        checks.array("captured_pair.eager_rng_parity", current_rng, eager._rng.numpy(), exact=True)
        checks.condition("captured_pair.rng_advances_between_replays", not np.array_equal(current_rng, previous_rng))
        previous_rng = current_rng

    with wp.ScopedCapture(device=device) as normalization_capture:
        restored_normalizer.update(inputs["reward"], inputs["terminated"], inputs["truncated"])
        restored_normalizer.normalize_rewards(reward_batch)
    restored_normalizer.load_state_dict(state)
    for _ in range(3):
        reference_normalizer.update_reward_stats(
            *[torch.tensor(transition[key], device=torch_device) for key in ("reward", "terminated", "truncated")]
        )
        wp.capture_launch(normalization_capture.graph)
        compare_normalizer(checks, restored_normalizer, reference_normalizer, "captured_reward_statistics")
        checks.array(
            "captured_normalized_rewards",
            restored_normalizer.normalized.numpy(),
            reference_normalizer.normalize_rewards(torch.tensor(raw_rewards, device=torch_device)).cpu().numpy(),
        )
    replay.reset()
    checks.condition("reset.readiness_and_length", len(replay) == 0 and not replay.can_sample())
    checks.array("reset.device_counters", replay._counters.numpy(), np.zeros(3, dtype=np.int64), exact=True)
    checks.array("reset.sampling_rng", replay._rng.numpy(), initial_rng, exact=True)
    checks.condition(
        "reset.storage_addresses_unchanged", addresses == {key: value.ptr for key, value in replay.storage.items()}
    )
    return {
        "config": {
            "num_envs": 3,
            "observation_dim": 5,
            "action_dim": 2,
            "capacity": cfg.buffer_max_length,
            "n_step": 3,
            "gamma": cfg.gamma,
            "sample_batch_size": cfg.sample_batch_size,
            "collection_steps": 12,
        },
        "checkpoint_positions": checkpoint_positions,
        "versions": {"torch": torch.__version__, "warp": wp.__version__, "numpy": np.__version__},
        "reference_source_sha256": {
            name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in [
                ("replay", __import__(TorchUniformBuffer.__module__, fromlist=["*"])),
                ("reward_normalization", __import__(RewardNormalizer.__module__, fromlist=["*"])),
            ]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, choices=(7, 11), default=11)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=2.0e-5)
    args = parser.parse_args()
    checks = Comparisons(args.atol, args.rtol)
    result = {
        "scope": "Fixed replay/reward numerical experiment; no simulator, learning-quality or throughput claim. Torch and Warp RNG streams intentionally differ; gathered rows share explicit sample indices.",
        "reference_revision": "Holiday-Robot/FlashSAC 87edc9061150ae9e962dd84e6544e27a1554b3ab (RoboLearn adapted Torch reference)",
        "device": args.device,
        "tolerances": {"absolute": args.atol, "relative": args.rtol, "checkpoint_and_capture": "exact"},
    }
    try:
        result.update(run(args, checks))
    except Exception as error:
        result["exception"] = {"type": type(error).__name__, "message": str(error)}
    result["checks"] = checks.checks
    result["failed_checks"] = [name for name, value in checks.checks.items() if not value["passed"]]
    result["passed"] = bool(checks.checks) and not result["failed_checks"] and "exception" not in result
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    print(serialized, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
