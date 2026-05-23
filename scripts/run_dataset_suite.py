"""Run all experiments tied to a single dataset (`--dataset parity_patch`,
`--dataset d4_shapes`, or `--dataset defect_patches`)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_all_experiments import _per_experiment_plots, _write_metrics
from src.experiments import QUICK_SEEDS, FULL_SEEDS, all_experiments, long_table, run_suite

OUT_DIR = ROOT / "outputs" / "experiments"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                    choices=["parity_patch", "d4_shapes", "defect_patches"])
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    seeds = QUICK_SEEDS if args.quick else FULL_SEEDS
    exps = [e for e in all_experiments() if e.dataset.__name__ == args.dataset]
    print(f"[QEGK {args.dataset}] {len(exps)} experiments x {len(seeds)} seeds")
    summary = run_suite(exps, seeds, OUT_DIR,
                         on_progress=lambda k, t, n: print(f"  [{k + 1}/{t}] {n}"))
    rows = long_table(summary)
    _write_metrics(summary, rows)
    _per_experiment_plots(summary)
    print(f"[QEGK {args.dataset}] outputs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
