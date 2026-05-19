"""
BitLinear: ternary-weight (BitNet b1.58) linear layer with 8-bit activation quantization.

References
----------
Ma, S., et al. "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits."
    arXiv:2402.17764, 2024.

Mathematical formulation
------------------------
Per-tensor ternarization rule
    gamma   = mean(|W|)
    W_tern  = round(clip(W / (gamma + eps), -1, +1)) in {-1, 0, +1}
    Forward: y = (LayerNorm(x))_q8 @ W_tern * gamma

The Straight-Through Estimator (STE) is used so that gradients flow through the
otherwise non-differentiable rounding operator.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .absmax import absmax_quantize_activation
from .ste import ste_round


class BitLinear(nn.Module):
    """Ternary BitNet b1.58 replacement for ``nn.Linear``.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, default False
        BitNet variants typically drop the bias term. Kept for API parity.
    activation_bits : int, default 8
        Bit-width used for the activation quantizer.
    eps : float, default 1e-5
        Numerical stabilizer for the ternarization scale.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        activation_bits: int = 8,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.norm = nn.LayerNorm(in_features)
        self.reset_parameters()

    # ------------------------------------------------------------------ #
    # Initialization                                                     #
    # ------------------------------------------------------------------ #
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    # ------------------------------------------------------------------ #
    # Weight quantization                                                #
    # ------------------------------------------------------------------ #
    def quantize_weights(self) -> tuple[Tensor, Tensor]:
        """Return the ternary weight matrix and its scalar scale ``gamma``."""
        gamma = self.weight.abs().mean()
        w_scaled = self.weight / (gamma + self.eps)
        w_tern = ste_round(w_scaled.clamp(-1.0, 1.0))
        return w_tern, gamma

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.norm(x)
        # ``absmax_quantize_activation`` returns the *dequantized* activation
        # together with its scale; gradients flow through via STE. We can use
        # the dequantized tensor directly — no further rescaling required.
        x_dq, _ = absmax_quantize_activation(x_norm, bits=self.activation_bits)
        w_tern, gamma = self.quantize_weights()

        # Logically this matmul is integer-only on dedicated hardware; we
        # compute it in fp32 here purely for portability.
        out = torch.nn.functional.linear(x_dq, w_tern) * gamma
        if self.bias is not None:
            out = out + self.bias
        return out

    # ------------------------------------------------------------------ #
    # Representation                                                     #
    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"activation_bits={self.activation_bits}, "
            f"bias={self.bias is not None}"
        )
