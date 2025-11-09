from __future__ import annotations

import os
from typing import Dict

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


def process_rss_mb() -> float:
    """Return process resident set size (MB). Falls back if psutil missing."""
    # Prefer psutil if available
    if psutil is not None:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        except Exception:
            pass
    # Fallback: parse /proc/self/status (Linux)
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    # e.g., VmRSS:  123456 kB
                    return float(parts[1]) / 1024.0
    except Exception:
        pass
    return -1.0


def gpu_memory_stats() -> Dict[str, float]:
    """Return basic CUDA memory stats in MB (or -1 when unavailable)."""
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return {
                "gpu/alloc_mb": -1.0,
                "gpu/reserved_mb": -1.0,
                "gpu/max_alloc_mb": -1.0,
                "gpu/max_reserved_mb": -1.0,
            }
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        return {
            "gpu/alloc_mb": float(alloc),
            "gpu/reserved_mb": float(reserved),
            "gpu/max_alloc_mb": float(max_alloc),
            "gpu/max_reserved_mb": float(max_reserved),
        }
    except Exception:
        return {
            "gpu/alloc_mb": -1.0,
            "gpu/reserved_mb": -1.0,
            "gpu/max_alloc_mb": -1.0,
            "gpu/max_reserved_mb": -1.0,
        }

