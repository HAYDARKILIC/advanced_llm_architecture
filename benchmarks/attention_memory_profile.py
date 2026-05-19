"""Benchmark the memory footprint of naive vs. tiled attention.

Compares peak GPU memory and wall-clock time across a sweep of
sequence lengths to empirically reproduce the Θ(N²d²/M) I/O
complexity argument of FlashAttention (Dao et al., 2023).

Usage
-----
    python benchmarks/attention_memory_profile.py \\
        --seq-lengths 512 1024 2048 4096 8192 \\
        --d-model 384 --n-heads 6 \\
        --output benchmarks/results/attention_memory.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.attention.naive_attention import NaiveAttention
from src.attention.tiled_attention import TiledAttention
from src.utils.benchmarks import cuda_memory_profile, timed
from src.utils.seeding import set_seed


def _profile(
    module: torch.nn.Module,
    *,
    batch: int,
    seq_len: int,
    d_model: int,
    device: torch.device,
) -> dict[str, float]:
    x = torch.randn(batch, seq_len, d_model, device=device)
    with cuda_memory_profile() as mem:
        latency_ms = timed(lambda: module(x), warmup=3, iters=10)
    return {
        "peak_mem_mb": float(mem.get("peak_alloc_mb", float("nan"))),
        "latency_ms": latency_ms,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lengths", type=int, nargs="+",
                   default=[512, 1024, 2048, 4096])
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--output", type=Path,
                   default=Path("benchmarks/results/attention_memory.json"))
    args = p.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not available — memory numbers will be CPU RSS proxies.")

    naive = NaiveAttention(args.d_model, args.n_heads).to(device).eval()
    tiled = TiledAttention(
        args.d_model, args.n_heads,
        block_q=args.block_size, block_kv=args.block_size,
    ).to(device).eval()

    results: list[dict[str, Any]] = []
    for n in args.seq_lengths:
        row: dict[str, Any] = {"seq_len": n}
        try:
            row["naive"] = _profile(
                naive, batch=args.batch, seq_len=n,
                d_model=args.d_model, device=device,
            )
        except torch.cuda.OutOfMemoryError:
            row["naive"] = {"peak_mem_mb": float("inf"), "latency_ms": float("nan")}
            torch.cuda.empty_cache()
        row["tiled"] = _profile(
            tiled, batch=args.batch, seq_len=n,
            d_model=args.d_model, device=device,
        )
        print(f"N={n:>6} | "
              f"naive {row['naive']['peak_mem_mb']:.1f} MB / "
              f"{row['naive']['latency_ms']:.2f} ms | "
              f"tiled {row['tiled']['peak_mem_mb']:.1f} MB / "
              f"{row['tiled']['latency_ms']:.2f} ms")
        results.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
