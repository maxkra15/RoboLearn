# SPDX-FileCopyrightText: Copyright (c) 2026 Holiday Robotics
# SPDX-FileCopyrightText: Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT
"""FP32 Warp-NN networks with the authors' FlashSAC architecture and projections.

Adapted from Holiday-Robot/FlashSAC, revision
87edc9061150ae9e962dd84e6544e27a1554b3ab. Warp-NN supplies the layers;
the residual architecture, normalization constraints and categorical critic
follow the MIT-licensed authors' implementation. Returned arrays are reusable
module outputs: consume them before another call with the same batch shape.
"""

from collections.abc import Mapping
from contextlib import contextmanager

import numpy as np
import warp as wp
from warp_nn import nn

from ._flash_normalization import BatchNorm, LogSoftmax, RMSNorm
from ._linear import TiledLinear


@wp.kernel(enable_backward=False)
def _normalize_rows(weight: wp.array2d(dtype=wp.float32)):
    i = wp.tid()
    total = float(0.0)
    for j in range(weight.shape[1]):
        value = weight[i, j]
        total = total + value * value
    denominator = wp.max(wp.sqrt(total), 1.0e-8)
    for j in range(weight.shape[1]):
        weight[i, j] = weight[i, j] / denominator


@wp.kernel(enable_backward=False)
def _normalize_affine(
    weight: wp.array(dtype=wp.float32),
    bias: wp.array(dtype=wp.float32),
):
    total = float(0.0)
    for j in range(weight.shape[0]):
        total = total + weight[j] * weight[j] + bias[j] * bias[j]
    factor = wp.sqrt(float(weight.shape[0])) / wp.sqrt(total + 1.0e-8)
    for j in range(weight.shape[0]):
        weight[j] = weight[j] * factor
        bias[j] = bias[j] * factor


@wp.kernel(enable_backward=False)
def _normalize_scale(weight: wp.array(dtype=wp.float32)):
    total = float(0.0)
    for j in range(weight.shape[0]):
        total = total + weight[j] * weight[j]
    factor = wp.sqrt(float(weight.shape[0])) / wp.sqrt(total + 1.0e-8)
    for j in range(weight.shape[0]):
        weight[j] = weight[j] * factor


@wp.kernel
def _actor_heads(
    raw_mean: wp.array2d(dtype=wp.float32),
    raw_log_std: wp.array2d(dtype=wp.float32),
    mean_bias: wp.array(dtype=wp.float32),
    std_bias: wp.array(dtype=wp.float32),
    log_std_min: float,
    log_std_max: float,
    mean: wp.array2d(dtype=wp.float32),
    log_std: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    mean[i, j] = raw_mean[i, j] + mean_bias[j]
    log_std[i, j] = log_std_min + (log_std_max - log_std_min) * 0.5 * (1.0 + wp.tanh(raw_log_std[i, j] + std_bias[j]))


@wp.kernel
def _join_observations_actions(
    observations: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    if j < observations.shape[1]:
        output[i, j] = observations[i, j]
    else:
        output[i, j] = actions[i, j - observations.shape[1]]


@wp.kernel
def _categorical_head(
    raw_logits: wp.array2d(dtype=wp.float32),
    bias: wp.array(dtype=wp.float32),
    logits: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    logits[i, j] = raw_logits[i, j] + bias[j]


@wp.kernel
def _expected_value(
    log_probs: wp.array2d(dtype=wp.float32),
    support: wp.array(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    value = float(0.0)
    for j in range(log_probs.shape[1]):
        value = value + wp.exp(log_probs[i, j]) * support[j]
    values[i] = value


@wp.kernel
def _stack_critics(
    q0: wp.array(dtype=wp.float32),
    q1: wp.array(dtype=wp.float32),
    log_probs0: wp.array2d(dtype=wp.float32),
    log_probs1: wp.array2d(dtype=wp.float32),
    values: wp.array2d(dtype=wp.float32),
    log_probs: wp.array3d(dtype=wp.float32),
):
    i, j = wp.tid()
    if j == 0:
        values[0, i] = q0[i]
        values[1, i] = q1[i]
    log_probs[0, i, j] = log_probs0[i, j]
    log_probs[1, i, j] = log_probs1[i, j]


def _orthogonal(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Initialize like the authors' orthogonal Linear, using a CPU NumPy RNG."""
    rows, columns = shape
    matrix = rng.standard_normal((max(rows, columns), min(rows, columns))).astype(np.float32)
    q, r = np.linalg.qr(matrix, mode="reduced")
    q *= np.sign(np.diag(r))
    return np.ascontiguousarray(q.T if rows < columns else q)


class _UnitLinear(nn.Module):
    def __init__(self, input_dim, output_dim, rng, optimized_linear_backward, requires_grad):
        super().__init__(requires_grad=requires_grad)
        linear = TiledLinear if optimized_linear_backward else nn.Linear
        self.w = linear(input_dim, output_dim, bias=False, initialize_parameters=False, requires_grad=requires_grad)
        wp.copy(self.w.weight.data, wp.array(_orthogonal((output_dim, input_dim), rng), device=self.device))
        super().__post_init__()

    def __call__(self, input: wp.array) -> wp.array:
        return self.w(input)

    def normalize_parameters(self) -> None:
        wp.launch(_normalize_rows, dim=self.w.out_features, inputs=[self.w.weight.data], device=self.device)


class _UnitBatchNorm(BatchNorm):
    def __init__(self, dimension, requires_grad):
        super().__init__(dimension, eps=1.0e-5, momentum=0.01, requires_grad=requires_grad)

    def normalize_parameters(self) -> None:
        wp.launch(_normalize_affine, dim=1, inputs=[self.weight.data, self.bias.data], device=self.device)


class _UnitRMSNorm(RMSNorm):
    def __init__(self, dimension, requires_grad):
        super().__init__(dimension, eps=1.0e-6, requires_grad=requires_grad)

    def normalize_parameters(self) -> None:
        wp.launch(_normalize_scale, dim=1, inputs=[self.weight.data], device=self.device)


class _Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, rng, optimized_linear_backward, requires_grad):
        super().__init__(requires_grad=requires_grad)
        self.norm = _UnitBatchNorm(input_dim, requires_grad)
        self.w = _UnitLinear(input_dim, hidden_dim, rng, optimized_linear_backward, requires_grad)
        super().__post_init__()

    def __call__(self, input: wp.array) -> wp.array:
        return self.w(self.norm(input))


class _ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, rng, optimized_linear_backward, requires_grad):
        super().__init__(requires_grad=requires_grad)
        self.w1 = _UnitLinear(hidden_dim, hidden_dim * 4, rng, optimized_linear_backward, requires_grad)
        self.w2 = _UnitLinear(hidden_dim * 4, hidden_dim, rng, optimized_linear_backward, requires_grad)
        self.norm1 = _UnitBatchNorm(hidden_dim * 4, requires_grad)
        self.norm2 = _UnitBatchNorm(hidden_dim, requires_grad)
        self.relu1 = nn.ReLU(requires_grad=requires_grad)
        self.relu2 = nn.ReLU(requires_grad=requires_grad)
        self.add = nn.Add(requires_grad=requires_grad)
        super().__post_init__()

    def __call__(self, input: wp.array) -> wp.array:
        output = self.relu1(self.norm1(self.w1(input)))
        output = self.relu2(self.norm2(self.w2(output)))
        return self.add(output, input)


def _named_parameters(module: nn.Module, prefix: str = "") -> list[tuple[str, nn.Parameter]]:
    result = [(prefix + name, value) for name, value in nn.Module.named_parameters(module)]
    for name, child in nn.Module.named_modules(module):
        result.extend(_named_parameters(child, prefix + name + "."))
    return result


def _load_author_arrays(module: nn.Module, mapping: Mapping[str, np.ndarray]) -> None:
    destination = module.state_dict()
    if set(mapping) != set(destination):
        raise ValueError("Authors' checkpoint keys do not match this network.")
    for name, array in destination.items():
        value = np.asarray(mapping[name], dtype=np.float32)
        if value.shape != array.shape:
            raise ValueError(f"Checkpoint {name!r} has shape {value.shape}, expected {array.shape}.")
        wp.copy(array, wp.array(np.ascontiguousarray(value), dtype=wp.float32, device=array.device))


class _FlashNetwork(nn.Module):
    def named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Return all parameter names, including registered nested modules."""
        return _named_parameters(self)

    def normalize_parameters(self) -> None:
        """Apply the authors' projections after an Adam update, outside its tape."""
        pending = list(self.modules())
        while pending:
            module = pending.pop()
            if isinstance(module, (_UnitLinear, _UnitBatchNorm, _UnitRMSNorm)):
                module.normalize_parameters()
            pending.extend(module.modules())

    @contextmanager
    def freeze_parameters(self):
        """Keep input gradients while omitting parameter gradients in a tape.

        Keep this context active through ``tape.backward``. Existing gradient
        arrays are restored without allocating new storage, so optimizers and
        captured critic updates retain their original pointers. Cached outputs
        continue to require gradients. This context is not thread-safe.
        """
        parameters = self.parameters()
        gradients = [parameter.grad for parameter in parameters]
        try:
            for parameter in parameters:
                parameter.grad = None
            yield
        finally:
            for parameter, gradient in zip(parameters, gradients, strict=True):
                parameter.grad = gradient


class _Backbone(_FlashNetwork):
    def __init__(self, input_dim, hidden_dim, num_blocks, rng, optimized_linear_backward, requires_grad):
        super().__init__(requires_grad=requires_grad)
        self.embedder = self.register_module(
            "embedder", _Embedder(input_dim, hidden_dim, rng, optimized_linear_backward, requires_grad)
        )
        self.encoder = tuple(
            self.register_module(
                f"encoder.{i}", _ResidualBlock(hidden_dim, rng, optimized_linear_backward, requires_grad)
            )
            for i in range(num_blocks)
        )
        self.post_norm = self.register_module("post_norm", _UnitRMSNorm(hidden_dim, requires_grad))

    def features(self, input: wp.array) -> wp.array:
        output = self.embedder(input)
        for block in self.encoder:
            output = block(output)
        return self.post_norm(output)


class _ActorHead(nn.Module):
    def __init__(self, hidden_dim, action_dim, rng, optimized_linear_backward, requires_grad):
        super().__init__(requires_grad=requires_grad)
        self.mean_w = _UnitLinear(hidden_dim, action_dim, rng, optimized_linear_backward, requires_grad)
        self.std_w = _UnitLinear(hidden_dim, action_dim, rng, optimized_linear_backward, requires_grad)
        self.mean_bias = nn.Parameter(wp.zeros(action_dim, device=self.device), requires_grad=requires_grad)
        self.std_bias = nn.Parameter(wp.zeros(action_dim, device=self.device), requires_grad=requires_grad)
        super().__post_init__()

    def __call__(self, input: wp.array) -> tuple[wp.array, wp.array]:
        shape = (input.shape[0], self.mean_bias.shape[0])
        if shape not in self._cache:
            self._cache[shape] = tuple(
                wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=self.requires_grad)
                for _ in range(2)
            )
        mean, log_std = self._cache[shape]
        wp.launch(
            _actor_heads,
            dim=shape,
            inputs=[self.mean_w(input), self.std_w(input), self.mean_bias.data, self.std_bias.data, -10.0, 2.0],
            outputs=[mean, log_std],
            device=self.device,
        )
        return mean, log_std


class FlashActor(_Backbone):
    """Authors' residual actor; return mean and bounded log standard deviation.

    Construct on the final device. NumPy supplies orthogonal initialization;
    ``load_author_state_dict`` accepts converted Torch weights for exact common
    initialization without a Torch dependency in this module.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        *,
        seed: int = 0,
        device: str | wp.Device | None = None,
        optimized_linear_backward: bool = True,
        requires_grad: bool = True,
    ):
        if min(observation_dim, action_dim, hidden_dim) <= 0 or num_blocks < 0:
            raise ValueError("Network dimensions must be positive and block count nonnegative.")
        with wp.ScopedDevice(device):
            rng = np.random.default_rng(seed)
            super().__init__(observation_dim, hidden_dim, num_blocks, rng, optimized_linear_backward, requires_grad)
            self.predictor = self.register_module(
                "predictor", _ActorHead(hidden_dim, action_dim, rng, optimized_linear_backward, requires_grad)
            )
            self.observation_dim = observation_dim
            self.action_dim = action_dim
            self.normalize_parameters()

    def __call__(self, observations: wp.array, training: bool = True) -> tuple[wp.array, wp.array]:
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim or observations.dtype != wp.float32:
            raise ValueError(f"Expected FP32 observations shaped (batch, {self.observation_dim}).")
        self.train(training)
        return self.predictor(self.features(observations))

    def load_author_state_dict(self, state: Mapping[str, np.ndarray]) -> None:
        """Load an authors' raw actor state dictionary converted to NumPy arrays."""
        _load_author_arrays(self, {key.removeprefix("_orig_mod."): value for key, value in state.items()})


class _ValueHead(nn.Module):
    def __init__(self, hidden_dim, num_bins, min_v, max_v, rng, optimized_linear_backward, requires_grad):
        super().__init__(requires_grad=requires_grad)
        self.w = _UnitLinear(hidden_dim, num_bins, rng, optimized_linear_backward, requires_grad)
        self.bias = nn.Parameter(wp.zeros(num_bins, device=self.device), requires_grad=requires_grad)
        self.bin_values = nn.Buffer(wp.array(np.linspace(min_v, max_v, num_bins, dtype=np.float32), device=self.device))
        self.log_softmax = LogSoftmax(dim=-1, requires_grad=requires_grad)
        super().__post_init__()

    def __call__(self, input: wp.array) -> tuple[wp.array, wp.array]:
        shape = (input.shape[0], self.bias.shape[0])
        if shape not in self._cache:
            self._cache[shape] = (
                wp.empty(shape, device=self.device, requires_grad=self.requires_grad),
                wp.empty(shape[0], device=self.device, requires_grad=self.requires_grad),
            )
        logits, values = self._cache[shape]
        wp.launch(
            _categorical_head, dim=shape, inputs=[self.w(input), self.bias.data], outputs=[logits], device=self.device
        )
        log_probs = self.log_softmax(logits)
        wp.launch(
            _expected_value,
            dim=shape[0],
            inputs=[log_probs, self.bin_values.data],
            outputs=[values],
            device=self.device,
        )
        return values, log_probs


class _ValueNetwork(_Backbone):
    def __init__(
        self, input_dim, hidden_dim, num_blocks, num_bins, min_v, max_v, rng, optimized_linear_backward, requires_grad
    ):
        super().__init__(input_dim, hidden_dim, num_blocks, rng, optimized_linear_backward, requires_grad)
        self.predictor = self.register_module(
            "predictor", _ValueHead(hidden_dim, num_bins, min_v, max_v, rng, optimized_linear_backward, requires_grad)
        )

    def __call__(self, input: wp.array) -> tuple[wp.array, wp.array]:
        return self.predictor(self.features(input))


class FlashDoubleCritic(_FlashNetwork):
    """Independent twin residual critics, returning ``(2,B)`` Q and ``(2,B,K)`` log probabilities.

    Both critics use independent parameters and batch statistics. This retains
    the authors' ensemble mathematics while using Warp-NN's 2D layer APIs.
    Target networks may use ``requires_grad=False``; an online critic used for
    actor gradients must retain differentiable outputs.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        num_bins: int = 101,
        min_v: float = -5.0,
        max_v: float = 5.0,
        *,
        seed: int = 0,
        device: str | wp.Device | None = None,
        optimized_linear_backward: bool = True,
        requires_grad: bool = True,
    ):
        if min(observation_dim, action_dim, hidden_dim) <= 0 or num_blocks < 0 or num_bins < 2 or min_v >= max_v:
            raise ValueError("Invalid categorical critic dimensions or support.")
        with wp.ScopedDevice(device):
            super().__init__(requires_grad=requires_grad)
            rng = np.random.default_rng(seed)
            self.critics = tuple(
                self.register_module(
                    f"critics.{i}",
                    _ValueNetwork(
                        observation_dim + action_dim,
                        hidden_dim,
                        num_blocks,
                        num_bins,
                        min_v,
                        max_v,
                        rng,
                        optimized_linear_backward,
                        requires_grad,
                    ),
                )
                for i in range(2)
            )
            self.observation_dim = observation_dim
            self.action_dim = action_dim
            self.num_bins = num_bins
            self.normalize_parameters()

    def __call__(self, observations: wp.array, actions: wp.array, training: bool = True) -> tuple[wp.array, wp.array]:
        if (
            observations.ndim != 2
            or actions.ndim != 2
            or observations.shape[0] != actions.shape[0]
            or observations.shape[1] != self.observation_dim
            or actions.shape[1] != self.action_dim
            or observations.dtype != wp.float32
            or actions.dtype != wp.float32
        ):
            raise ValueError(
                "Critic expects FP32 observations and actions with matching batches and configured widths."
            )
        batch = observations.shape[0]
        if batch not in self._cache:
            self._cache[batch] = (
                wp.empty(
                    (batch, self.observation_dim + self.action_dim),
                    device=self.device,
                    requires_grad=self.requires_grad,
                ),
                wp.empty((2, batch), device=self.device, requires_grad=self.requires_grad),
                wp.empty((2, batch, self.num_bins), device=self.device, requires_grad=self.requires_grad),
            )
        input, values, log_probs = self._cache[batch]
        self.train(training)
        wp.launch(
            _join_observations_actions,
            dim=input.shape,
            inputs=[observations, actions],
            outputs=[input],
            device=self.device,
        )
        q0, lp0 = self.critics[0](input)
        q1, lp1 = self.critics[1](input)
        wp.launch(
            _stack_critics,
            dim=(batch, self.num_bins),
            inputs=[q0, q1, lp0, lp1],
            outputs=[values, log_probs],
            device=self.device,
        )
        return values, log_probs

    def load_author_state_dict(self, state: Mapping[str, np.ndarray]) -> None:
        """Load NumPy arrays from the authors' stacked twin-critic checkpoint."""
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
        mapped = {}
        used = set()
        for i, critic in enumerate(self.critics):
            for name in critic.state_dict():
                source = (
                    name.replace(".w.w.weight", ".w.weight")
                    .replace(".w1.w.weight", ".w1.weight")
                    .replace(".w2.w.weight", ".w2.weight")
                )
                value = np.asarray(state[source], dtype=np.float32)
                if name == "predictor.bin_values":
                    value = value.reshape(-1)
                else:
                    if value.shape[0] != 2:
                        raise ValueError(f"Checkpoint {source!r} must contain two critics.")
                    value = value[i]
                mapped[f"critics.{i}.{name}"] = value
                used.add(source)
        if set(state) != used:
            raise ValueError("Authors' checkpoint has unexpected twin-critic keys.")
        _load_author_arrays(self, mapped)
