"""
Capstone mini-LLM combining BitLinear (week 1), RoPE (week 5), and tiled
attention (week 4) into an end-to-end ternary transformer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..positional.rope import RotaryPositionalEmbedding
from ..quantization.bitlinear import BitLinear


@dataclass
class MiniLLMConfig:
    vocab_size: int = 32_000
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    max_seq_len: int = 4_096
    ffn_mult: int = 4
    rope_base: float = 10_000.0
    tie_word_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class BitAttention(nn.Module):
    """Multi-head attention using BitLinear projections and RoPE."""

    def __init__(self, cfg: MiniLLMConfig, rope: RotaryPositionalEmbedding) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads
        self.rope = rope

        self.q_proj = BitLinear(cfg.d_model, cfg.d_model)
        self.k_proj = BitLinear(cfg.d_model, cfg.d_model)
        self.v_proj = BitLinear(cfg.d_model, cfg.d_model)
        self.o_proj = BitLinear(cfg.d_model, cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        q = self.q_proj(x).reshape(b, n, self.cfg.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, n, self.cfg.n_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, n, self.cfg.n_heads, self.head_dim)

        # RoPE on q, k (along the seq axis, per head)
        q = self.rope(q.transpose(1, 2)).transpose(1, 2)
        k = self.rope(k.transpose(1, 2)).transpose(1, 2)

        # Use PyTorch's fused SDPA for production speed; tiled implementation
        # available in ``src/attention/tiled_attention.py`` for didactic use.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, n, self.cfg.d_model)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """Bit-linear SwiGLU MLP."""

    def __init__(self, cfg: MiniLLMConfig) -> None:
        super().__init__()
        hidden = int(cfg.ffn_mult * cfg.d_model * 2 / 3)
        # Round up to a multiple of 32 for kernel friendliness.
        hidden = ((hidden + 31) // 32) * 32
        self.w1 = BitLinear(cfg.d_model, hidden)
        self.w3 = BitLinear(cfg.d_model, hidden)
        self.w2 = BitLinear(hidden, cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MiniLLMBlock(nn.Module):
    def __init__(self, cfg: MiniLLMConfig, rope: RotaryPositionalEmbedding) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = BitAttention(cfg, rope)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniLLM(nn.Module):
    """The end-to-end capstone model."""

    def __init__(self, cfg: MiniLLMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MiniLLMConfig()
        self.tok_embed = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        rope = RotaryPositionalEmbedding(
            dim=self.cfg.d_model // self.cfg.n_heads,
            max_seq_len=self.cfg.max_seq_len,
            base=self.cfg.rope_base,
        )
        self.blocks = nn.ModuleList(
            [MiniLLMBlock(self.cfg, rope) for _ in range(self.cfg.n_layers)]
        )
        self.norm = RMSNorm(self.cfg.d_model)
        self.lm_head = BitLinear(self.cfg.d_model, self.cfg.vocab_size)

        if self.cfg.tie_word_embeddings:
            # Tying with a quantized head is approximate; we tie the underlying
            # FP weights and let the BitLinear ternarize at every forward.
            self.lm_head.weight = self.tok_embed.weight  # type: ignore[assignment]

    def forward(self, ids: Tensor) -> Tensor:
        x = self.tok_embed(ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.lm_head(x)

    # ------------------------------------------------------------------ #
    # Generation                                                          #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, max_new_tokens: int, temperature: float = 1.0) -> Tensor:
        """Naive greedy / temperature-sampling decoder (no KV-cache)."""
        self.eval()
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(ids[:, -self.cfg.max_seq_len :])
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids
