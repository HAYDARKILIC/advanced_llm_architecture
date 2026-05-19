"""Tests for the TTT-Linear layer."""

from __future__ import annotations

import torch

from src.ttt import TTTLinear


def test_forward_shape() -> None:
    layer = TTTLinear(d_model=32)
    x = torch.randn(2, 16, 32)
    y = layer(x)
    assert y.shape == (2, 16, 32)


def test_gradient_flow() -> None:
    layer = TTTLinear(d_model=16)
    x = torch.randn(1, 8, 16)
    y = layer(x).sum()
    y.backward()
    for p in layer.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_zero_input_zero_output() -> None:
    """Zero token stream → zero output (W stays zero, q stays zero)."""
    layer = TTTLinear(d_model=8)
    x = torch.zeros(1, 4, 8)
    y = layer(x)
    assert torch.allclose(y, torch.zeros_like(y))
