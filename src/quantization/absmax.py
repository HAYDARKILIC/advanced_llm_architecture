"""
Absolute-max (symmetric) quantization.

For a tensor x of bit-width b:
    s = (2^(b-1) - 1) / max(|x|)
    q = round(s * x)
    dequant: q / s
"""

from __future__ import annotations

import torch
from torch import Tensor

from .ste import ste_round


def absmax_quantize_activation(x: Tensor, bits: int = 8, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    """Per-token absmax quantizer for activations.

    Returns
    -------
    x_dequant : Tensor
        Tensor of the same shape as ``x`` containing dequantized values.
        Gradients flow through via STE.
    scale : Tensor
        The per-token scale (last dim reduced), shape ``x.shape[:-1] + (1,)``.
    """
    qmax = 2 ** (bits - 1) - 1
    x_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = qmax / x_max
    x_q = ste_round((x * scale).clamp(-qmax, qmax))
    x_dequant = x_q / scale
    return x_dequant, scale


def absmax_quantize_weight(w: Tensor, bits: int = 8, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    """Per-tensor absmax quantizer for weight tensors."""
    qmax = 2 ** (bits - 1) - 1
    w_max = w.abs().max().clamp(min=eps)
    scale = qmax / w_max
    w_q = ste_round((w * scale).clamp(-qmax, qmax))
    return w_q / scale, scale
