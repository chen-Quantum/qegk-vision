"""Bootstrap statistics and the 'quantum advantage signal' metric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class BootstrapCI:
    mean: float
    std: float
    lo: float
    hi: float
    n: int


def bootstrap_ci(samples: Sequence[float], alpha: float = 0.05,
                 n_boot: int = 2000, seed: int = 0) -> BootstrapCI:
    """Empirical bootstrap confidence interval for the mean."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return BootstrapCI(mean=float("nan"), std=float("nan"),
                            lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        means[b] = arr[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return BootstrapCI(
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        lo=lo, hi=hi, n=int(arr.size),
    )


def advantage_score(quantum: Sequence[float],
                     classical_by_method: dict[str, Sequence[float]]) -> dict:
    """For paired runs (same seeds), compute
        advantage = mean(quantum) - mean(best_classical)
    where best_classical is selected by mean per method, NOT by per-seed
    cheating. Returns the chosen baseline and a bootstrap CI on the paired
    differences."""
    quantum = list(quantum)
    n = len(quantum)
    method_means = {m: float(np.mean(v)) for m, v in classical_by_method.items()}
    best_method = max(method_means, key=method_means.get) if method_means else None
    if best_method is None:
        return {
            "advantage_mean": float("nan"),
            "advantage_std": float("nan"),
            "advantage_lo": float("nan"),
            "advantage_hi": float("nan"),
            "best_classical": None,
            "best_classical_mean": float("nan"),
            "quantum_mean": float(np.mean(quantum)) if n > 0 else float("nan"),
            "n": n,
        }
    best = list(classical_by_method[best_method])
    if len(best) != n:
        # Cannot do paired - fall back to mean difference + parametric CI.
        diff = float(np.mean(quantum) - np.mean(best))
        return {
            "advantage_mean": diff,
            "advantage_std": float("nan"),
            "advantage_lo": float("nan"),
            "advantage_hi": float("nan"),
            "best_classical": best_method,
            "best_classical_mean": float(np.mean(best)),
            "quantum_mean": float(np.mean(quantum)),
            "n": n,
        }
    diffs = np.asarray(quantum, dtype=np.float64) - np.asarray(best, dtype=np.float64)
    ci = bootstrap_ci(diffs)
    return {
        "advantage_mean": ci.mean,
        "advantage_std": ci.std,
        "advantage_lo": ci.lo,
        "advantage_hi": ci.hi,
        "best_classical": best_method,
        "best_classical_mean": float(np.mean(best)),
        "quantum_mean": float(np.mean(quantum)),
        "n": ci.n,
    }
