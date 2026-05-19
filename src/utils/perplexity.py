"""Token-level perplexity utilities."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    token_stream: Tensor,
    context_length: int,
    stride: int | None = None,
    device: torch.device | None = None,
) -> float:
    """Compute perplexity of ``model`` over a long token stream using a sliding window.

    Parameters
    ----------
    token_stream : 1-D LongTensor of token ids.
    context_length : window size to feed the model.
    stride : step between windows (defaults to ``context_length // 2``).
    """
    model.eval()
    if device is not None:
        model = model.to(device)
        token_stream = token_stream.to(device)
    stride = stride or (context_length // 2)

    total_nll = 0.0
    total_tokens = 0
    n = token_stream.size(0)
    for start in range(0, n - context_length, stride):
        end = start + context_length + 1
        chunk = token_stream[start:end].unsqueeze(0)
        logits = model(chunk[:, :-1])
        targets = chunk[:, 1:]
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        total_nll += float(nll.item())
        total_tokens += targets.numel()

    return math.exp(total_nll / max(total_tokens, 1))
