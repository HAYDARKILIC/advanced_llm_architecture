"""Tests for the BitLinear ternary layer."""

from __future__ import annotations

import pytest
import torch

from src.quantization import BitLinear


def test_forward_shape() -> None:
    layer = BitLinear(64, 128)
    x = torch.randn(4, 16, 64)
    y = layer(x)
    assert y.shape == (4, 16, 128)


def test_weight_ternarization() -> None:
    layer = BitLinear(64, 128)
    w_tern, gamma = layer.quantize_weights()
    unique = w_tern.unique()
    # After STE the rounded forward values must be in {-1, 0, +1}.
    assert set(unique.tolist()).issubset({-1.0, 0.0, 1.0})
    assert gamma > 0


def test_gradient_flows() -> None:
    layer = BitLinear(32, 32)
    x = torch.randn(2, 8, 32, requires_grad=True)
    y = layer(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_different_activation_bits(bits: int) -> None:
    layer = BitLinear(16, 16, activation_bits=bits)
    x = torch.randn(2, 4, 16)
    y = layer(x)
    assert y.shape == (2, 4, 16)
