"""
Linear attention via the kernel-trick reordering of the matmul.

    Attn(Q, K, V) ~= phi(Q) (phi(K)^T V) / phi(Q) (phi(K)^T 1)

For the feature map  phi(x) = ELU(x) + 1 this is the original Katharopoulos
formulation. Time and memory are O(N d^2), not O(N^2 d).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def elu_plus_one(x: Tensor) -> Tensor:
    return torch.nn.functional.elu(x) + 1.0


class LinearAttention(nn.Module):
    """Causal linear attention with ELU+1 feature map."""

    def __init__(self, d_model: int, n_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        qkv = self.qkv_proj(x).reshape(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)                      # (b, h, n, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = elu_plus_one(q)
        k = elu_plus_one(k)

        # Causal recurrence: implemented as a cumulative outer product.
        # kv has shape (b, h, n, d, d); z has shape (b, h, n, d).
        kv = torch.einsum("b h n d, b h n e -> b h n d e", k, v).cumsum(dim=2)
        z = k.cumsum(dim=2)

        out = torch.einsum("b h n d, b h n d e -> b h n e", q, kv)
        denom = torch.einsum("b h n d, b h n d -> b h n", q, z).clamp(min=self.eps).unsqueeze(-1)
        out = out / denom

        out = out.transpose(1, 2).reshape(b, n, self.d_model)
        return self.o_proj(out)
