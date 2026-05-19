"""Tests for tiled attention — must agree with naive attention numerically."""

from __future__ import annotations

import torch

from src.attention import NaiveAttention, TiledAttention


def _seed_modules(m1: torch.nn.Module, m2: torch.nn.Module) -> None:
    """Copy the qkv / o projection weights from m1 to m2 so they compute the same function."""
    m2.qkv_proj.load_state_dict(m1.qkv_proj.state_dict())
    m2.o_proj.load_state_dict(m1.o_proj.state_dict())


def test_tiled_matches_naive() -> None:
    torch.manual_seed(0)
    d_model, n_heads, seq_len = 64, 4, 32
    a = NaiveAttention(d_model, n_heads, causal=True).eval()
    b = TiledAttention(d_model, n_heads, block_size_q=8, block_size_kv=8, causal=True).eval()
    _seed_modules(a, b)

    x = torch.randn(2, seq_len, d_model)
    ya = a(x)
    yb = b(x)
    assert torch.allclose(ya, yb, atol=1e-4), f"max diff = {(ya - yb).abs().max()}"


def test_different_block_sizes() -> None:
    torch.manual_seed(0)
    d_model, n_heads, seq_len = 32, 2, 24
    a = NaiveAttention(d_model, n_heads).eval()
    for br in (4, 8, 12):
        for bc in (4, 8, 12):
            b = TiledAttention(d_model, n_heads, block_size_q=br, block_size_kv=bc).eval()
            _seed_modules(a, b)
            x = torch.randn(1, seq_len, d_model)
            assert torch.allclose(a(x), b(x), atol=1e-4)
