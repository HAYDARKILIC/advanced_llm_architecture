"""
Tiled (FlashAttention-style) attention implemented in pure PyTorch.

This is an *algorithmic* re-expression of FlashAttention — it does not call
custom CUDA kernels. The point is to demonstrate the online-softmax recurrence
and the block-iteration structure that makes the real kernel memory-efficient.

For wall-clock speed in production, use ``torch.nn.functional.scaled_dot_product_attention``
(which dispatches to FlashAttention-2 when available).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TiledAttention(nn.Module):
    """FlashAttention-style tiled multi-head self-attention.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    n_heads : int
        Number of attention heads. Must divide ``d_model``.
    block_size_q : int
        Tile size along the query axis (B_r in the paper).
    block_size_kv : int
        Tile size along the key/value axis (B_c in the paper).
    causal : bool
        Whether to apply a causal mask.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size_q: int = 64,
        block_size_kv: int = 64,
        causal: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv
        self.causal = causal
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        qkv = self.qkv_proj(x).reshape(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                       # (b, n, h, d_head)
        q = q.transpose(1, 2)                              # (b, h, n, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = self._tiled_attention(q, k, v)               # (b, h, n, d_head)
        out = out.transpose(1, 2).reshape(b, n, self.d_model)
        return self.o_proj(out)

    # ------------------------------------------------------------------ #
    # Core algorithm                                                     #
    # ------------------------------------------------------------------ #
    def _tiled_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Algorithmic reproduction of FlashAttention v1 online softmax."""
        b, h, n, d = q.shape
        br, bc = self.block_size_q, self.block_size_kv

        o = torch.zeros_like(q)
        # Running max and softmax denominator, per (b, h, query-row).
        m_state = torch.full((b, h, n), float("-inf"), device=q.device, dtype=q.dtype)
        l_state = torch.zeros((b, h, n), device=q.device, dtype=q.dtype)

        for i in range(0, n, br):
            qi = q[:, :, i : i + br, :]                    # (b, h, br, d)
            mi = m_state[:, :, i : i + br].clone()
            li = l_state[:, :, i : i + br].clone()
            oi = o[:, :, i : i + br, :].clone()
            i_end = min(i + br, n)

            for j in range(0, n, bc):
                # Causal early-exit: every key position in this block is
                # strictly greater than every query position in the current
                # Q-block, so the entire score-tile would be ``-inf``.
                if self.causal and j >= i_end:
                    break

                kj = k[:, :, j : j + bc, :]
                vj = v[:, :, j : j + bc, :]

                sij = torch.einsum("b h r d, b h c d -> b h r c", qi, kj) * self.scale

                if self.causal:
                    # mask out positions where key index > query index
                    rs = torch.arange(qi.shape[2], device=q.device).unsqueeze(1) + i
                    cs = torch.arange(kj.shape[2], device=q.device).unsqueeze(0) + j
                    sij = sij.masked_fill(cs > rs, float("-inf"))

                mij = sij.amax(dim=-1)                     # (b, h, br)
                m_new = torch.maximum(mi, mij)
                # If a query row has *no* valid keys yet (still happens for the
                # first row of the very first Q-block in some edge cases),
                # ``m_new`` is ``-inf`` and ``exp(s - m_new)`` would be NaN.
                # Replace with a sentinel that yields zero contribution.
                safe_m = torch.where(
                    torch.isfinite(m_new), m_new, torch.zeros_like(m_new)
                )
                pij = torch.exp(sij - safe_m.unsqueeze(-1))
                pij = torch.where(
                    torch.isfinite(sij), pij, torch.zeros_like(pij)
                )
                lij = pij.sum(dim=-1)

                # alpha rescales the running output to the new max.
                alpha = torch.exp(mi - safe_m)
                alpha = torch.where(
                    torch.isfinite(mi), alpha, torch.zeros_like(alpha)
                )
                li = alpha * li + lij
                oi = oi * alpha.unsqueeze(-1) + torch.einsum(
                    "b h r c, b h c d -> b h r d", pij, vj
                )
                mi = m_new

            # Final normalisation. ``li`` can be 0 for rows whose attention
            # window is empty (impossible under standard causal masks but
            # guarded here for robustness).
            li_safe = li.clamp(min=torch.finfo(li.dtype).tiny)
            o[:, :, i : i + br, :] = oi / li_safe.unsqueeze(-1)
            m_state[:, :, i : i + br] = mi
            l_state[:, :, i : i + br] = li

        return o
