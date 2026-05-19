"""Lightweight memory and latency profiling helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def cuda_memory_profile() -> Iterator[dict[str, float]]:
    """Context manager that records peak CUDA memory inside the block."""
    stats: dict[str, float] = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    yield stats
    if torch.cuda.is_available():
        stats["peak_alloc_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        stats["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 ** 2)


def timed(fn, *args, warmup: int = 3, iters: int = 50, **kwargs) -> float:
    """Median wall-time of ``fn(*args, **kwargs)`` in milliseconds."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return times[len(times) // 2]
