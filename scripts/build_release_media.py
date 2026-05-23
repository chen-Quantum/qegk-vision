"""Produce polished release media under outputs/release_media/.

Uses cached experiment metrics in outputs/experiments/ if available; runs a
quick suite otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_EXP = ROOT / "outputs" / "experiments"
OUT_REL = ROOT / "outputs" / "release_media"

from src.datasets import d4_shapes, parity_patch
from src.experiments import long_table
from src.features import images_to_angles
from src.kernels import quantum_kernel, cosine_kernel
from src.quantum_states import states
from src.visualize import (
    advantage_summary,
    anomaly_heatmap,
    cinematic_gif,
    d4_orbit_grid,
    entangled_vs_separable_kernel,
    entanglement_diagnostics_plot,
    experiment_grid,
    pipeline_overview,
    readme_hero,
)


def _ensure_metrics_exist() -> dict:
    p = OUT_EXP / "all_metrics.json"
    if not p.exists():
        from scripts.run_all_experiments import main as run_main
        run_main(["--quick"])
    with p.open() as fh:
        return json.load(fh)


def main() -> int:
    OUT_REL.mkdir(parents=True, exist_ok=True)
    summary = _ensure_metrics_exist()
    rows = long_table(summary)

    print("[QEGK media] 1/8  pipeline overview ...")
    pipeline_overview(OUT_REL / "pipeline_overview.png")

    print("[QEGK media] 2/8  D4 orbit example ...")
    ds = d4_shapes(n=10, side=12, seed=0, add_random_d4=False)
    d4_orbit_grid(ds.X[0], OUT_REL / "d4_orbit_example.png",
                  title="D4 orbit of an L-shape (canonical pose, no rotation)")

    print("[QEGK media] 3/8  entangled vs separable kernel ...")
    parity = parity_patch(n=30, patch=2, seed=0)
    _, angles = images_to_angles(parity.X, n_qubits=4, seed=0)
    K_sep = quantum_kernel(states(angles, feature_map="separable_ry"))
    K_ent = quantum_kernel(states(angles, feature_map="entangled_cx"))
    entangled_vs_separable_kernel(K_sep, K_ent,
                                    OUT_REL / "entangled_vs_separable_kernel.png")

    print("[QEGK media] 4/8  anomaly demo heatmap ...")
    from src.datasets import defect_patches
    from src.models import kernel_anomaly_scores
    anom = defect_patches(n_clean=40, n_defect=10, side=8, seed=0,
                          defect_type="local")
    _, ang = images_to_angles(anom.X, n_qubits=4, seed=0)
    clean_idx = np.flatnonzero(anom.y == 0)[:25]
    state_clean = states(ang[clean_idx], feature_map="entangled_cx")
    state_all = states(ang, feature_map="entangled_cx")
    K_test_train = quantum_kernel(state_all, state_clean)
    sc = kernel_anomaly_scores(K_test_train)
    anomaly_heatmap(sc, anom.y, anom.X,
                    OUT_REL / "anomaly_heatmap_demo.png")

    print("[QEGK media] 5/8  advantage summary plot ...")
    advantage_summary(rows, OUT_REL / "advantage_summary.png")

    print("[QEGK media] 6/8  experiment grid tile ...")
    experiment_grid(rows, OUT_REL / "experiment_grid_20.png")

    print("[QEGK media] 7/8  entanglement diagnostics ...")
    from src.entanglement import batch_diagnostics
    diagnostics = {}
    parity_more = parity_patch(n=40, patch=2, seed=0)
    _, ang = images_to_angles(parity_more.X, n_qubits=4, seed=0)
    for fm in ("separable_ry", "entangled_cx", "rzz_ring",
                "hardware_efficient", "pauli_z_features"):
        diagnostics[fm] = batch_diagnostics(states(ang, feature_map=fm))
    entanglement_diagnostics_plot(diagnostics,
                                    OUT_REL / "entanglement_diagnostics.png")

    print("[QEGK media] 8/8  README hero ...")
    K_sep_big = quantum_kernel(states(ang, feature_map="separable_ry"))
    K_ent_big = quantum_kernel(states(ang, feature_map="entangled_cx"))
    readme_hero(parity_more.X[:20], K_ent_big, K_sep_big, rows,
                 OUT_REL / "readme_hero.png")

    print("[QEGK media] cinematic GIF + MP4 ...")
    cinematic_gif(parity_more.X, K_ent_big, K_sep_big, rows, OUT_REL)

    print("\n[QEGK media] Done.  Outputs:")
    for p in sorted(OUT_REL.iterdir()):
        if p.is_file():
            print(f"  {p.name:42s}  {p.stat().st_size:>10d} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
