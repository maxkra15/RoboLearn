# Copyright (c) 2026 RoboLearn contributors
# SPDX-License-Identifier: MIT

"""Types for the PyTorch implementation, without a JAX dependency."""

from typing import Any, Union

import numpy.typing as npt
import torch

NDArray = npt.NDArray[Any]
Tensor = Union[NDArray, torch.Tensor]
