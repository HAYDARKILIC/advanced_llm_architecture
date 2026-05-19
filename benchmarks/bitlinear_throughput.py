"""Throughput benchmark: full-precision nn.Linear vs. BitLinear.

Reports forward / backward latency and approximate memory savings
when weights are stored as ternary {-1, 0, +1} values.

Usage
-----
    python benchmarks/bitlinear_throughput.py \\
        --hidden-sizes 256 512 1024 2048 4096 \\
        --batch 32 --seq-len 1024 \\
        --output benchmarks/results/bitlinear_throughput.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.quantization.bitlinear import BitLinear
from src.utils.benchmarks import timed
from src.utils.seeding import set_seed


def _fwd_bwd(layer: nn.Module, x: torch.Tensor) -> None:
    y = layer(x)
    loss = y.float().pow(2).mean()
    loss.backward()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-sizes", type=int, nargs="+",
                   default=[256, 512, 1024, 2048])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--output", type=Path,
                   default=Path("benchmarks/results/bitlinear_throughput.json"))
    args = p.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for d in args.hidden_sizes:
        x = torch.randn(args.batch, args.seq_len, d, device=device, requires_grad=True)
        fp = nn.Linear(d, d, bias=False).to(device)
        bn = BitLinear(d, d, bias=False).to(device)

        # Forward-only latency
        fp_fwd = timed(lambda: fp(x), warmup=3, iters=20)
        bn_fwd = timed(lambda: bn(x), warmup=3, iters=20)

        # Forward+backward latency
        fp_bw = timed(lambda: _fwd_bwd(fp, x.detach().clone().requires_grad_()),
                      warmup=3, iters=10)
        bn_bw = timed(lambda: _fwd_bwd(bn, x.detach().clone().requires_grad_()),
                      warmup=3, iters=10)

        # Storage footprint (theoretical):
        #   fp32 weights = 4 * d * d  bytes
        #   ternary     ≈ (d * d) * log2(3) / 8  bytes  ≈ 0.198 * d * d
        fp_bytes = 4 * d * d
        bn_bytes = int(d * d * 1.585 / 8)        # log2(3) ≈ 1.585 bits / weight
        ratio = fp_bytes / max(bn_bytes, 1)

        row = {
            "d_model": d,
            "fp32_fwd_ms": fp_fwd,
            "bitlinear_fwd_ms": bn_fwd,
            "fp32_fwd_bwd_ms": fp_bw,
            "bitlinear_fwd_bwd_ms": bn_bw,
            "fp32_weight_bytes": fp_bytes,
            "ternary_weight_bytes": bn_bytes,
            "compression_ratio": ratio,
        }
        rows.append(row)
        print(f"d={d:>5} | fp32 fwd {fp_fwd:.2f} ms / bit fwd {bn_fwd:.2f} ms | "
              f"compression ≈ {ratio:5.1f}×")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
