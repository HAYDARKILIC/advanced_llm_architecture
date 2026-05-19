"""Tests for the differentiable-logic layer."""

from __future__ import annotations

import torch

from src.difflogic import DiffLogicLayer, NUM_GATES, apply_all_gates


def test_all_gates_shape() -> None:
    a = torch.rand(5, 7)
    b = torch.rand(5, 7)
    g = apply_all_gates(a, b)
    assert g.shape == (5, 7, NUM_GATES)


def test_layer_forward() -> None:
    layer = DiffLogicLayer(in_features=10, out_features=20)
    x = torch.rand(4, 10)
    y = layer(x)
    assert y.shape == (4, 20)


def test_discretization() -> None:
    layer = DiffLogicLayer(in_features=10, out_features=5)
    idx = layer.discretize()
    assert idx.shape == (5,)
    assert idx.min() >= 0 and idx.max() < NUM_GATES
