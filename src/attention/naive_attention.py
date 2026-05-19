"""
Textbook scaled dot-product attention. Materializes the full N x N matrix in HBM —
used only as a reference baseline against which the tiled / linear variants are
benchmarked.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class NaiveAttention(nn.Module):
    """Vanilla multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, causal: bool = True) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        qkv = self.qkv_proj(x).reshape(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.einsum("b h n d, b h m d -> b h n m", q, k) * self.scale

        if self.causal:
            mask = torch.ones(n, n, device=x.device, dtype=torch.bool).triu(diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("b h n m, b h m d -> b h n d", attn, v)
        out = out.transpose(1, 2).reshape(b, n, self.d_model)
        return self.o_proj(out)
