"""
Evaluate perplexity of a trained MiniLLM checkpoint at one or more context lengths.

Example
-------
    python scripts/evaluate_perplexity.py \\
        --checkpoint ./checkpoints/mini_llm_final.pt \\
        --rope-scaling yarn \\
        --max-context 32768
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.model import MiniLLM, MiniLLMConfig
from src.utils.perplexity import evaluate_perplexity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rope-scaling", choices=("none", "pi", "yarn"), default="none")
    p.add_argument("--max-context", type=int, default=32768)
    p.add_argument("--out", type=Path, default=Path("benchmarks/results/perplexity.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = MiniLLMConfig(**ckpt["config"])
    model = MiniLLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # TODO: replace with the real validation token stream.
    token_stream = torch.randint(0, cfg.vocab_size, (200_000,))

    context_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    context_lengths = [c for c in context_lengths if c <= args.max_context]

    results: dict[int, float] = {}
    for ctx in context_lengths:
        ppl = evaluate_perplexity(model, token_stream, context_length=ctx, stride=ctx // 2)
        print(f"context={ctx:>6d}  ppl={ppl:.3f}")
        results[ctx] = ppl

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
