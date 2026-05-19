"""
Rotary Position Embedding (RoPE) — Su et al., 2021.

Implements the **halves-style** (a.k.a. GPT-NeoX / Llama) variant of RoPE.
The hidden dimension is split into two equal halves, so that
``x = [x_first_half, x_second_half]`` with each half of size ``dim/2``.
The pair ``(x_first_half[i], x_second_half[i])`` is then rotated by
``m * theta_i`` in the 2D plane:

    x'_first_half[i]   =  cos(m theta_i) * x_first_half[i]
                       -  sin(m theta_i) * x_second_half[i]
    x'_second_half[i]  =  sin(m theta_i) * x_first_half[i]
                       +  cos(m theta_i) * x_second_half[i]

with ``theta_i = base^{-2i/d}``. This is mathematically equivalent to the
"interleaved pairs" convention used in the original RoPE paper but matches
the layout adopted by the dominant open-weights model families (Llama,
Mistral, Falcon, …), making the weights interoperable.

The (cos, sin) tables are precomputed at construction and grown lazily
when a longer sequence is encountered.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryPositionalEmbedding(nn.Module):
    """Standard RoPE (Su et al. 2021).

    Parameters
    ----------
    dim : int
        The per-head dimension. Must be even.
    max_seq_len : int
        Largest position to precompute (sin/cos tables sized to this length).
    base : float, default 10000.0
        The base of the geometric frequency series (``theta_i = base ** (-2i/dim)``).
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10_000.0) -> None:
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even."
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # Inverse frequencies, shape (dim/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos / sin tables.
        self._build_cache(max_seq_len)

    # ------------------------------------------------------------------ #
    # Cache management                                                   #
    # ------------------------------------------------------------------ #
    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)              # (seq_len, dim/2)
        cos = freqs.cos()                                  # (seq_len, dim/2)
        sin = freqs.sin()                                  # (seq_len, dim/2)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.max_seq_len = seq_len

    def _maybe_grow_cache(self, seq_len: int) -> None:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

    # ------------------------------------------------------------------ #
    # Rotation                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """Map [x1, x2, x3, x4, ...] to [-x2, x1, -x4, x3, ...]."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: Tensor, position_ids: Tensor | None = None) -> Tensor:
        """Apply RoPE to a tensor of shape ``(..., seq_len, dim)``."""
        seq_len = x.shape[-2]
        self._maybe_grow_cache(seq_len)

        if position_ids is None:
            cos = self.cos_cached[:seq_len]                  # (seq_len, dim/2)
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]

        # Duplicate to full dim: (seq_len, dim)
        cos = torch.cat((cos, cos), dim=-1).to(x.dtype)
        sin = torch.cat((sin, sin), dim=-1).to(x.dtype)

        return x * cos + self._rotate_half(x) * sin
