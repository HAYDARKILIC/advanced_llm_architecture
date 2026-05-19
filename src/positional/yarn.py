"""
YaRN — Yet another RoPE extensioN — Peng et al., 2023.

NTK-aware interpolation of RoPE frequencies. Dimensions with short wavelengths
(high frequency) are left untouched; long-wavelength (low frequency) dimensions
are interpolated linearly; a smooth ramp blends the two regimes.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .rope import RotaryPositionalEmbedding


def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    """Compute the dimension that achieves a given number of rotations in the training window."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    # There are ``dim // 2`` frequency components — clamp accordingly.
    return max(low, 0), min(high, dim // 2 - 1)


def _yarn_linear_ramp_mask(low: float, high: float, dim: int, dtype: torch.dtype) -> torch.Tensor:
    if low == high:
        high += 1e-3
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    return linear_func.clamp(0, 1)


class YaRNRotaryPositionalEmbedding(RotaryPositionalEmbedding):
    """RoPE with YaRN NTK-aware interpolation for context-window extension.

    Parameters
    ----------
    dim, max_seq_len, base : as in ``RotaryPositionalEmbedding``.
    original_max_seq_len : int
        The context length the parent model was trained on.
    scale_factor : float
        Ratio of target / original context length.
    beta_fast, beta_slow : float
        Number of rotations defining the start/end of the smooth ramp.
        Defaults follow the original paper.
    """

    def __init__(
        self,
        dim: int,
        *,
        max_seq_len: int,
        original_max_seq_len: int,
        scale_factor: float,
        base: float = 10_000.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ) -> None:
        # ``nn.Module`` requires its own ``__init__`` to run before any
        # attribute is assigned that might be a Parameter / Module / Buffer.
        # We initialise the base ``nn.Module`` machinery explicitly *first*,
        # then set our YaRN-specific scalars, *then* hand off to the parent
        # ``RotaryPositionalEmbedding.__init__`` — which will dispatch to the
        # overridden ``_build_cache`` and rely on those scalars.
        nn.Module.__init__(self)
        self.original_max_seq_len = original_max_seq_len
        self.scale_factor = scale_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # Re-enter the RoPE constructor by name; we cannot use ``super().__init__``
        # because ``nn.Module.__init__`` has already been called.
        RotaryPositionalEmbedding.__init__(
            self, dim=dim, max_seq_len=max_seq_len, base=base
        )

    def _build_cache(self, seq_len: int) -> None:
        # Recompute inv_freq with YaRN scaling
        dim = self.dim
        inv_freq_base = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        inv_freq_scaled = inv_freq_base / self.scale_factor

        low, high = _yarn_find_correction_range(
            self.beta_fast, self.beta_slow, dim, self.base, self.original_max_seq_len
        )
        ramp_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2, torch.float32)
        inv_freq = inv_freq_scaled * (1 - ramp_mask) + inv_freq_base * ramp_mask

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        super()._build_cache(seq_len)
