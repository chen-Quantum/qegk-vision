"""Run only the synthetic experiments (parity / D4 shape / anomaly) - a fast
subset for advantage exploration without the ablation sweeps."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_all_experiments import _per_experiment_plots, _write_metrics
from src.experiments import QUICK_SEEDS, all_experiments, long_table, run_suite

OUT_DIR = ROOT / "outputs" / "experiments"


def main() -> int:
    exps = all_experiments()
    synthetic = [e for e in exps if e.name.startswith(("01_", "02_", "03_",
                                                        "06_", "07_", "10_"))]
    print(f"[QEGK synthetic] {len(synthetic)} experiments")
    summary = run_suite(synthetic, QUICK_SEEDS, OUT_DIR,
                         on_progress=lambda k, t, n: print(f"  [{k + 1}/{t}] {n}"))
    rows = long_table(summary)
    _write_metrics(summary, rows)
    _per_experiment_plots(summary)
    print(f"[QEGK synthetic] outputs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
