"""
Pack the ternary weights of a trained MiniLLM into a compact binary file.

Format (BWF v0):
    magic  : b"BWF0"
    n_keys : uint32
    repeat n_keys:
        key_len  : uint16
        key      : utf-8 bytes
        rows     : uint32
        cols     : uint32
        gamma    : float32
        packed   : ceil(rows*cols/5) bytes  (5 ternary values per byte, base-3)
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from src.model import MiniLLM, MiniLLMConfig


def _pack_ternary(t: torch.Tensor) -> tuple[np.ndarray, float]:
    """Pack a {-1, 0, +1} tensor into base-3 bytes."""
    flat = t.flatten().to(torch.int8).cpu().numpy() + 1   # shift to {0, 1, 2}
    n = flat.size
    # Pad up to a multiple of 5 trits per byte (since 3**5 = 243 < 256).
    pad = (5 - n % 5) % 5
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.int8)])
    packed = np.zeros(flat.size // 5, dtype=np.uint8)
    for k in range(5):
        packed += flat[k::5].astype(np.uint8) * (3 ** k)
    return packed, pad


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = MiniLLMConfig(**ckpt["config"])
    model = MiniLLM(cfg)
    model.load_state_dict(ckpt["model"])

    entries: list[tuple[str, torch.Tensor, float]] = []
    for name, module in model.named_modules():
        if hasattr(module, "quantize_weights"):
            w_tern, gamma = module.quantize_weights()
            entries.append((name + ".weight", w_tern.detach(), float(gamma.item())))

    with open(args.out, "wb") as f:
        f.write(b"BWF0")
        f.write(struct.pack("<I", len(entries)))
        for key, w, gamma in entries:
            kb = key.encode("utf-8")
            f.write(struct.pack("<H", len(kb)))
            f.write(kb)
            f.write(struct.pack("<II", w.shape[0], w.shape[1]))
            f.write(struct.pack("<f", gamma))
            packed, _pad = _pack_ternary(w)
            f.write(packed.tobytes())

    size_mb = args.out.stat().st_size / (1024 ** 2)
    print(f"Wrote {args.out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
