# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Reuse RSL-RL's observation normalization around a native PPO learner.

RSL-RL 5.5.1 updates both normalizers after all PPO minibatches. Statistics stay
fixed throughout collection and learning, so storing normalized observations in
the native rollout is equivalent to normalizing the raw rollout during learning.
This adapter calls the upstream implementation; it does not copy its formulas.
"""

import json
from pathlib import Path

import numpy as np


class NativePPOObservationNormalization:
    """Stock RSL normalization for symmetric, policy-only actor/critic inputs.

    Call ``record_raw(step, raw)`` before each collection step, pass
    ``normalize(raw)`` to native PPO, and call ``update_after_learning()`` once
    after the complete native PPO update. Normalize bootstrap observations with
    the same frozen statistics. The adapter owns its raw [T, N, O] storage.

    Include ``model_arrays()`` in policy identity checks and save a sidecar at
    every learner checkpoint. Loading restores statistics, not an unfinished
    rollout. These Torch operations stay outside the native learning graph.
    """

    _ROLES = ("actor", "critic")
    _STATE_KEYS = ("_mean", "_var", "_std", "count")

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_dim,
        device,
        *,
        evaluation=False,
        obs_groups=None,
    ):
        import torch
        from rsl_rl.modules import EmpiricalNormalization

        groups = {"actor": ["policy"], "critic": ["policy"]} if obs_groups is None else obs_groups
        if groups != {"actor": ["policy"], "critic": ["policy"]}:
            raise ValueError("Native normalization requires identical policy-only actor and critic observations")
        dimensions = (num_steps, num_envs, observation_dim)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
            raise ValueError("Rollout dimensions must be positive integers")
        self.num_steps, self.num_envs, self.observation_dim = dimensions
        self.device = torch.device(device)
        self.evaluation = bool(evaluation)
        self._normalizers = {
            role: EmpiricalNormalization(observation_dim).to(self.device).train(not self.evaluation)
            for role in self._ROLES
        }
        self._raw = None if self.evaluation else torch.empty(dimensions, dtype=torch.float32, device=self.device)
        self._next_step = 0
        # Initial statistics are the upstream defaults, including eps=0.01.
        self.model_arrays()

    def _check_observations(self, raw):
        import torch

        if not isinstance(raw, torch.Tensor) or raw.shape != (self.num_envs, self.observation_dim):
            raise ValueError("Expected raw observations with shape [num_envs, observation_dim]")
        if raw.dtype != torch.float32 or raw.device != self._normalizers["actor"]._mean.device:
            raise ValueError("Raw observations must be FP32 on the normalizers' device")

    def normalize(self, raw):
        """Apply frozen stock statistics, including the initial x / 1.01 scaling."""
        import torch

        self._check_observations(raw)
        with torch.no_grad():
            return self._normalizers["actor"](raw)

    def record_raw(self, step, raw):
        """Copy one raw observation batch into the next owned rollout slot."""
        import torch

        if self.evaluation:
            raise RuntimeError("Evaluation cannot collect normalizer updates")
        if isinstance(step, bool) or not isinstance(step, int) or step != self._next_step or step >= self.num_steps:
            raise ValueError("Raw rollout steps must be recorded exactly once in order")
        self._check_observations(raw)
        with torch.no_grad():
            self._raw[step].copy_(raw)
        self._next_step += 1

    def update_after_learning(self):
        """Update both upstream modules once, after all native PPO minibatches."""
        import torch

        if self.evaluation:
            raise RuntimeError("Evaluation cannot update normalization statistics")
        if self._next_step != self.num_steps:
            raise RuntimeError("Normalization requires one complete raw rollout")
        with torch.no_grad():
            raw = self._raw.flatten(0, 1)
            for normalizer in self._normalizers.values():
                normalizer.update(raw)
        self._next_step = 0

    def model_arrays(self):
        """Return independent CPU arrays and verify the shared-input invariant.

        This checkpoint/inspection operation synchronizes copies to the CPU;
        normalization and collection do not perform per-step equality checks.
        """
        arrays = {
            f"{role}.{name}": value.detach().cpu().numpy().copy()
            for role, normalizer in self._normalizers.items()
            for name, value in normalizer.state_dict().items()
        }
        expected = {f"{role}.{name}" for role in self._ROLES for name in self._STATE_KEYS}
        if set(arrays) != expected:
            raise RuntimeError("Unexpected upstream EmpiricalNormalization state schema")
        for name in self._STATE_KEYS:
            if not np.array_equal(arrays[f"actor.{name}"], arrays[f"critic.{name}"]):
                raise RuntimeError("Actor and critic observation normalization statistics differ")
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError("Nonfinite observation normalization state")
        return arrays

    def metadata(self):
        """Describe the sidecar format and the actual upstream configuration."""
        return {
            "schema_version": 1,
            "implementation": "rsl_rl.modules.EmpiricalNormalization",
            "observation_dim": self.observation_dim,
            "num_steps": self.num_steps,
            "num_envs": self.num_envs,
            "eps": {role: normalizer.eps for role, normalizer in self._normalizers.items()},
            "until": {role: normalizer.until for role, normalizer in self._normalizers.items()},
            "count": {role: int(normalizer.count.item()) for role, normalizer in self._normalizers.items()},
            "evaluation": self.evaluation,
            "update_timing": "once_after_all_PPO_minibatches_from_raw_rollout",
        }

    def state_dict(self):
        """Snapshot both normalizers; the unfinished raw rollout is excluded."""
        return {"metadata": self.metadata(), "arrays": self.model_arrays()}

    def load_state_dict(self, state):
        """Restore validated normalizers, preserving this adapter's eval mode."""
        import torch

        metadata, arrays = state["metadata"], state["arrays"]
        expected_metadata = self.metadata()
        # Evaluation can use a different batch size/horizon and loads training
        # statistics without restoring the checkpoint's collection mode.
        for name in ("schema_version", "implementation", "observation_dim", "eps", "until", "update_timing"):
            if metadata.get(name) != expected_metadata[name]:
                raise ValueError(f"Observation normalization metadata mismatch: {name}")
        expected = self.model_arrays()
        if set(arrays) != set(expected):
            raise ValueError("Observation normalization state inventory differs")
        for name, template in expected.items():
            value = arrays[name]
            if not isinstance(value, np.ndarray) or value.shape != template.shape or value.dtype != template.dtype:
                raise ValueError(f"Observation normalization state shape/dtype differs: {name}")
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite observation normalization state: {name}")
        for name in self._STATE_KEYS:
            if not np.array_equal(arrays[f"actor.{name}"], arrays[f"critic.{name}"]):
                raise ValueError("Loaded actor and critic normalization statistics differ")
        for role in self._ROLES:
            count = int(arrays[f"{role}.count"].item())
            if count < 0 or metadata.get("count", {}).get(role) != count:
                raise ValueError("Observation normalization count metadata differs")
            if (arrays[f"{role}._var"] < 0).any() or (arrays[f"{role}._std"] < 0).any():
                raise ValueError("Observation normalization variance/std must be nonnegative")
        for role, normalizer in self._normalizers.items():
            normalizer.load_state_dict(
                {name: torch.from_numpy(arrays[f"{role}.{name}"].copy()).to(self.device) for name in self._STATE_KEYS},
                strict=True,
            )
        self._next_step = 0

    def save_sidecar(self, path):
        """Write exact normalization buffers and JSON metadata to one NPZ file."""
        state = self.state_dict()
        with Path(path).open("wb") as stream:
            np.savez(stream, __metadata__=np.asarray(json.dumps(state["metadata"], sort_keys=True)), **state["arrays"])

    def load_sidecar(self, path):
        """Load a sidecar without pickle or modifying its bytes."""
        with np.load(path, allow_pickle=False) as data:
            if "__metadata__" not in data.files or data["__metadata__"].shape != ():
                raise ValueError("Missing scalar observation normalization metadata")
            state = {
                "metadata": json.loads(str(data["__metadata__"].item())),
                "arrays": {name: data[name].copy() for name in data.files if name != "__metadata__"},
            }
        self.load_state_dict(state)
