"""
TTT-Linear: a Test-Time-Training layer with a linear inner model.

Implements the recurrence
    W_t = W_{t-1} - eta (W_{t-1} k_t - v_t) k_t^T
    y_t = W_t q_t
where (q_t, k_t, v_t) are linear projections of the input token x_t.

This is the *recurrent* form. A chunked / parallel form is left as an
optimization exercise in notebook 02.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TTTLinear(nn.Module):
    """Test-Time-Training linear layer (Sun et al. 2024)."""

    def __init__(self, d_model: int, inner_lr: float = 1.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.inner_lr = inner_lr

        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.proj_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the recurrence to a batch of sequences.

        Parameters
        ----------
        x : Tensor of shape ``(B, N, d_model)``
        """
        b, n, d = x.shape
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        W = torch.zeros(b, d, d, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(n):
            kt = k[:, t, :].unsqueeze(-1)              # (B, d, 1)
            vt = v[:, t, :].unsqueeze(-1)              # (B, d, 1)
            qt = q[:, t, :].unsqueeze(-1)              # (B, d, 1)

            # Inner-loop SGD step on the squared inner loss.
            residual = torch.bmm(W, kt) - vt           # (B, d, 1)
            W = W - self.inner_lr * torch.bmm(residual, kt.transpose(-1, -2))

            yt = torch.bmm(W, qt).squeeze(-1)          # (B, d)
            outs.append(yt)

        y = torch.stack(outs, dim=1)
        return self.proj_o(y)
