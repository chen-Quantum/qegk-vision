"""Run the full QEGK experiment suite and write metrics + plots.

    python scripts/run_all_experiments.py            # 5 seeds, full suite
    python scripts/run_all_experiments.py --quick    # 2 seeds, faster
    python scripts/run_all_experiments.py --include-optional  # add real-data exps
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "outputs" / "experiments"


def _write_metrics(summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # raw nested
    with (OUT_DIR / "all_metrics.json").open("w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    # per-seed long table
    flat_rows: List[Dict[str, Any]] = []
    for r in summary["results"]:
        if r.get("error"):
            flat_rows.append({"experiment": r["name"], "method": "ERROR",
                               "value": r["error"]})
            continue
        for k, s in enumerate(r["per_seed"]):
            seed = r["seeds"][k]
            for method, value in s.items():
                flat_rows.append({
                    "experiment": r["name"], "family": r["family"],
                    "n_qubits": r["n_qubits"], "group": r["group"],
                    "depth": r["entanglement_depth"],
                    "seed": seed, "method": method, "value": value,
                })
    if flat_rows:
        keys = list(flat_rows[0].keys())
        with (OUT_DIR / "all_metrics.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for row in flat_rows:
                w.writerow(row)
    # bootstrap summary
    keys = ["experiment", "family", "method", "best_quantum",
            "best_classical", "mean", "std", "lo", "hi", "n"]
    with (OUT_DIR / "bootstrap_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _per_experiment_plots(summary: Dict[str, Any]) -> None:
    """Write a small bar plot per experiment in outputs/experiments/per_experiment_plots/."""
    import matplotlib.pyplot as plt
    out = OUT_DIR / "per_experiment_plots"
    out.mkdir(parents=True, exist_ok=True)
    for exp in summary["results"]:
        if exp.get("error"):
            continue
        per_seed = exp["per_seed"]
        if not per_seed:
            continue
        methods = sorted(per_seed[0].keys())
        means = [np.mean([s.get(m, np.nan) for s in per_seed]) for m in methods]
        stds = [np.std([s.get(m, np.nan) for s in per_seed]) for m in methods]
        colors = ["tab:blue" if m.startswith("quantum") else "tab:orange"
                  for m in methods]
        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(methods))))
        y = np.arange(len(methods))
        ax.barh(y, means, xerr=stds, color=colors, alpha=0.85, capsize=3)
        ax.set_yticks(y); ax.set_yticklabels(methods, fontsize=8)
        ax.set_xlabel("metric (accuracy or AUC)")
        ax.set_title(exp["name"], fontsize=10)
        ax.set_xlim(0, 1.05)
        fig.tight_layout()
        fig.savefig(out / f"{exp['name']}.png", dpi=140)
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                    help="2 seeds, fewer samples per dataset.")
    p.add_argument("--include-optional", action="store_true",
                    help="Append optional real-data experiments (no-op if deps missing).")
    args = p.parse_args(argv)

    from src.experiments import (
        FULL_SEEDS, QUICK_SEEDS, all_experiments,
        long_table, optional_real_data_experiments, run_suite,
    )

    seeds = QUICK_SEEDS if args.quick else FULL_SEEDS
    exps = all_experiments()
    if args.include_optional:
        exps = exps + optional_real_data_experiments()

    if args.quick:
        # Trim sample sizes for the quick path.
        for e in exps:
            e.n_train = min(e.n_train, 24)
            e.n_test = min(e.n_test, 24)
            if "n_clean" in e.dataset_kwargs:
                e.dataset_kwargs["n_clean"] = min(e.dataset_kwargs["n_clean"], 60)
                e.dataset_kwargs["n_defect"] = min(e.dataset_kwargs.get("n_defect", 0), 15)
            if "n" in e.dataset_kwargs:
                e.dataset_kwargs["n"] = min(e.dataset_kwargs["n"], 80)

    print(f"[QEGK] Running {len(exps)} experiments x {len(seeds)} seeds "
          f"({'quick' if args.quick else 'full'} mode)")
    t0 = time.time()

    def progress(k: int, total: int, name: str) -> None:
        dt = time.time() - t0
        print(f"  [{k + 1:>2}/{total}]  {name:40s}   elapsed {dt:5.1f}s")

    summary = run_suite(exps, seeds, OUT_DIR, on_progress=progress)
    rows = long_table(summary)
    _write_metrics(summary, rows)
    _per_experiment_plots(summary)

    print(f"\n[QEGK] suite finished in {summary['elapsed_seconds']:.1f}s")
    print(f"[QEGK] outputs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
