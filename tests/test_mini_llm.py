"""Smoke tests for the end-to-end MiniLLM."""

from __future__ import annotations

import torch

from src.model import MiniLLM, MiniLLMConfig


def test_forward_pass() -> None:
    cfg = MiniLLMConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, max_seq_len=32)
    model = MiniLLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_generation_runs() -> None:
    cfg = MiniLLMConfig(vocab_size=128, d_model=32, n_layers=1, n_heads=2, max_seq_len=16)
    model = MiniLLM(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(prompt, max_new_tokens=5)
    assert out.shape == (1, 9)
