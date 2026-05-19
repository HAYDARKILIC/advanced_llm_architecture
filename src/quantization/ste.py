"""
Straight-Through Estimator (STE).

The classical PyTorch idiom for back-propagating through a non-differentiable
operator. The forward pass passes ``f(x)``; the backward pass behaves as
if the operator were the identity.
"""

from __future__ import annotations

import torch
from torch import Tensor


def ste_round(x: Tensor) -> Tensor:
    """Round with a straight-through gradient."""
    return (x.round() - x).detach() + x


def ste_sign(x: Tensor) -> Tensor:
    """Sign function with a straight-through gradient."""
    return (x.sign() - x).detach() + x
