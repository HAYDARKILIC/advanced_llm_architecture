"""
DiffLogicLayer — a differentiable replacement for ``nn.Linear`` + nonlinearity.

Each output neuron j:
  1. picks two input indices (a_j, b_j) at random (fixed at init);
  2. learns a categorical distribution over the 16 binary gates;
  3. outputs softmax(alpha_j) . [g_k(x_{a_j}, x_{b_j}) for k in 0..15].
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .gates import NUM_GATES, apply_all_gates


class DiffLogicLayer(nn.Module):
    """Differentiable logic-gate layer."""

    def __init__(self, in_features: int, out_features: int, connections: int = 2) -> None:
        super().__init__()
        if connections != 2:
            raise NotImplementedError("Only binary gates are implemented in this reference.")

        self.in_features = in_features
        self.out_features = out_features

        # Fixed random wiring -- registered as a non-trainable buffer.
        idx_a = torch.randint(0, in_features, (out_features,))
        idx_b = torch.randint(0, in_features, (out_features,))
        self.register_buffer("idx_a", idx_a, persistent=True)
        self.register_buffer("idx_b", idx_b, persistent=True)

        # Per-neuron logits over the 16 gates.
        self.gate_logits = nn.Parameter(torch.zeros(out_features, NUM_GATES))
        nn.init.normal_(self.gate_logits, std=0.1)

    def forward(self, x: Tensor) -> Tensor:
        """``x`` of shape ``(B, in_features)`` -> ``(B, out_features)``."""
        a = x[..., self.idx_a]                          # (B, out_features)
        b = x[..., self.idx_b]                          # (B, out_features)
        all_gate_outputs = apply_all_gates(a, b)        # (B, out_features, 16)
        weights = torch.softmax(self.gate_logits, dim=-1)  # (out_features, 16)
        return (all_gate_outputs * weights).sum(dim=-1)

    @torch.no_grad()
    def discretize(self) -> torch.Tensor:
        """Return the hard gate index per output neuron."""
        return self.gate_logits.argmax(dim=-1)
