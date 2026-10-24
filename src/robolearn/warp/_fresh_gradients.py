# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""Bounded ownership checks for private residual first-writer adjoints.

Only an internal optimizer phase may activate overwrite. Ordinary tapes keep
accumulation, and callers clear the managed raw caches at their boundary.
Prepared CUDA graphs retain their original array owners and policy choices.
"""

import warp as wp


def _span(array):
    return (int(array.ptr or 0), array.size * 4)


def _inventory(tape):
    # This is the same registration backward performs for ordinary launches.
    # Custom callbacks already register their arrays in record_func(). No
    # adjoint storage is allocated by get_adjoint for existing Warp arrays.
    for launch in tape.launches:
        if not callable(launch):
            for value in (*launch[3], *launch[4]):
                tape.get_adjoint(value)
    return tuple(tape.gradients.values())


class FreshPhase:
    """Retain one immutable full inventory and reduced clear plan per phase."""

    def __init__(self, device):
        self.device = device
        self.plan = None
        self.signature = None
        self.owners = ()
        self.primal_gradients = ()
        self.disabled_reason = None

    def begin(self):
        return FreshCall(self)

    def matches_raw_owners(self):
        return all(
            layer._cache.get((raw.shape, wp.float32)) is raw and raw.grad is gradient and _span(gradient) == span
            for layer, raw, gradient, span in self.owners
        )

    def matches_owners(self):
        return self.matches_raw_owners() and all(primal.grad is gradient for primal, gradient in self.primal_gradients)

    def clear(self):
        """Clear every omitted owner, including storage retained by old graphs."""
        seen = set()
        for _, _, gradient, _ in self.owners:
            if id(gradient) not in seen:
                gradient.zero_()
                seen.add(id(gradient))


class FreshCall:
    """A forward's deferred decision, finalized before any backward callback."""

    def __init__(self, phase):
        self.phase = phase
        self.enabled = False
        self.owners = []
        self.remaining = ()
        self.reason = None

    def record(self, layer, raw):
        gradient = raw.grad
        if gradient is None:
            self.reason = "missing_raw_adjoint"
            return
        self.owners.append((layer, raw, gradient, _span(gradient)))

    def select(self, tape):
        """Choose overwrite and reduced clear together, or clear fully first."""
        from robolearn.cuda.zero import CudaZero

        phase = self.phase
        gradients = _inventory(tape)
        reason = self.reason or phase.disabled_reason
        if not gradients or not CudaZero.supports(gradients, phase.device):
            reason = reason or "unsupported_gradient_inventory"
        spans = [entry[3] for entry in self.owners]
        if not spans or len(set(spans)) != len(spans):
            reason = "missing_or_shared_raw_adjoint"
        if not all(
            layer._cache.get((raw.shape, wp.float32)) is raw
            and raw.grad is gradient
            and gradient.dtype == wp.float32
            and gradient.device == phase.device
            and gradient.is_contiguous
            and gradient.shape == raw.shape
            and gradient.ptr
            and tape.gradients.get(raw) is gradient
            for layer, raw, gradient, _ in self.owners
        ):
            reason = "raw_owner_changed"
        for layer, raw, gradient, (begin, size) in self.owners:
            end = begin + size
            for primal, value in tape.gradients.items():
                if not isinstance(value, wp.array):
                    reason = reason or "unsupported_gradient_inventory"
                    continue
                address, length = _span(value)
                if begin < address + length and address < end and (primal is not raw or value is not gradient):
                    reason = "raw_adjoint_alias"
        signature = CudaZero._signature(gradients) if reason is None else None
        owner_signature = tuple((id(layer), id(raw), id(gradient), span) for layer, raw, gradient, span in self.owners)
        if reason is None and phase.signature is not None:
            expected = tuple((id(layer), id(raw), id(gradient), span) for layer, raw, gradient, span in phase.owners)
            if signature != phase.signature or owner_signature != expected or not phase.matches_owners():
                reason = "gradient_pointer_signature_changed"
        if reason is None:
            excluded = set(spans)
            self.remaining = tuple(value for value in gradients if _span(value) not in excluded)
            if phase.plan is None:
                if phase.device.is_capturing:
                    reason = "fresh_inventory_not_prepared_before_capture"
                else:
                    phase.plan = CudaZero(self.remaining).prepare()
                    phase.signature = signature
                    phase.owners = tuple(self.owners)
                    phase.primal_gradients = tuple(tape.gradients.items())
                    for layer, raw, gradient, span in self.owners:
                        if not hasattr(layer, "_fresh_raw_owners"):
                            layer._fresh_raw_owners = {}
                        layer._fresh_raw_owners[raw.shape] = (raw, gradient, span)
            elif not phase.plan.matches(self.remaining):
                reason = "reduced_gradient_signature_changed"
        if reason is not None:
            self.reason = phase.disabled_reason = reason
            if reason in (
                "raw_adjoint_alias",
                "raw_owner_changed",
                "missing_raw_adjoint",
                "missing_or_shared_raw_adjoint",
            ):
                raise RuntimeError(f"Invalid fresh raw-adjoint storage ({reason}); backward aborted")
            # Inventory registration above makes this a complete clear, before
            # any callback can choose ordinary accumulation. Never filter a
            # post-backward inventory whose ownership was not proven first.
            phase.clear()
            tape.zero()
            return
        self.enabled = True

    def finish(self, tape):
        """Validate again before optimizer mutation or a reduced post-clear."""
        from robolearn.cuda.zero import CudaZero

        if self.enabled and (
            CudaZero._signature(tuple(tape.gradients.values())) != self.phase.signature
            or not self.phase.matches_owners()
        ):
            raise RuntimeError("Fresh-gradient inventory changed during backward; optimizer update aborted")

    def clear(self, tape):
        self.finish(tape)
        self.phase.plan.step(self.remaining)
        return {
            "mode": "native",
            "arrays": self.phase.plan.array_count,
            "ranges": self.phase.plan.range_count,
            "bytes": self.phase.plan.total_bytes,
            "fresh_raw_arrays": len(self.owners),
            "fresh_raw_bytes": sum(span[1] for *_, span in self.owners),
            "full_arrays": len(tape.gradients),
        }
