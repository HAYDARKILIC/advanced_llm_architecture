"""Differentiable logic layers — week 3."""

from .gates import GATE_FUNCTIONS, NUM_GATES, apply_all_gates
from .layer import DiffLogicLayer

__all__ = ["DiffLogicLayer", "GATE_FUNCTIONS", "NUM_GATES", "apply_all_gates"]
