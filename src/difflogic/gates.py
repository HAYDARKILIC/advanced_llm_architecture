"""
The 16 binary boolean gates, expressed as differentiable polynomials in (a, b)
where a, b in [0, 1].
"""

from __future__ import annotations

import torch
from torch import Tensor

# Each function is a closure over (a, b). They return tensors of the same shape as a.

GATE_FUNCTIONS = [
    lambda a, b: torch.zeros_like(a),                # 0: FALSE
    lambda a, b: a * b,                              # 1: AND
    lambda a, b: a * (1 - b),                        # 2: A AND NOT B
    lambda a, b: a,                                  # 3: A
    lambda a, b: (1 - a) * b,                        # 4: NOT A AND B
    lambda a, b: b,                                  # 5: B
    lambda a, b: a + b - 2 * a * b,                  # 6: XOR
    lambda a, b: a + b - a * b,                      # 7: OR
    lambda a, b: 1 - (a + b - a * b),                # 8: NOR
    lambda a, b: 1 - (a + b - 2 * a * b),            # 9: XNOR
    lambda a, b: 1 - b,                              # 10: NOT B
    lambda a, b: 1 - b + a * b,                      # 11: A OR NOT B
    lambda a, b: 1 - a,                              # 12: NOT A
    lambda a, b: 1 - a + a * b,                      # 13: NOT A OR B
    lambda a, b: 1 - a * b,                          # 14: NAND
    lambda a, b: torch.ones_like(a),                 # 15: TRUE
]

NUM_GATES = len(GATE_FUNCTIONS)


def apply_all_gates(a: Tensor, b: Tensor) -> Tensor:
    """Stack the outputs of all 16 gates along a new last dimension."""
    return torch.stack([g(a, b) for g in GATE_FUNCTIONS], dim=-1)
