"""Tests for the RoPE module."""

from __future__ import annotations

import torch

from src.positional import RotaryPositionalEmbedding


def test_shape_preservation() -> None:
    rope = RotaryPositionalEmbedding(dim=64, max_seq_len=512)
    x = torch.randn(2, 8, 128, 64)               # (B, H, N, D)
    y = rope(x)
    assert y.shape == x.shape


def test_relative_property() -> None:
    """Inner product of two RoPE'd vectors depends only on positional offset."""
    rope = RotaryPositionalEmbedding(dim=8, max_seq_len=128)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)

    def rope_at(x: torch.Tensor, pos: int) -> torch.Tensor:
        pos_ids = torch.tensor([pos])
        return rope(x, position_ids=pos_ids.expand(1))

    # Same offset, two different absolute positions → inner products equal.
    q1 = rope_at(q, 10)
    k1 = rope_at(k, 12)
    q2 = rope_at(q, 50)
    k2 = rope_at(k, 52)

    s1 = (q1 * k1).sum().item()
    s2 = (q2 * k2).sum().item()
    assert abs(s1 - s2) < 1e-4


def test_cache_growth() -> None:
    rope = RotaryPositionalEmbedding(dim=16, max_seq_len=64)
    assert rope.max_seq_len == 64
    x = torch.randn(1, 1, 200, 16)
    _ = rope(x)
    assert rope.max_seq_len >= 200
