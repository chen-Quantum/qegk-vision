"""Unified experiment runner.

Each registered experiment is a dict with:
    name        : human-readable identifier
    family      : "classification" or "anomaly"
    dataset     : factory + kwargs producing a Dataset
    n_qubits    : circuit width
    feature_maps: which quantum feature maps to evaluate
    group       : "D4" | "C4" | "Identity"
    classical   : list of classical baselines to evaluate
    n_train     : number of training samples
    n_test      : number of test samples
    seeds       : list of integer seeds (one run per seed)
    notes       : free-text description shown in the summary
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from . import SEED
from .datasets import (
    Dataset, d4_shapes, defect_patches, fashion_mnist_pair, medmnist_pair,
    parity_patch,
)
from .features import images_to_angles
from .groups import make_group
from .kernels import (
    CLASSICAL_KERNELS,
    cosine_kernel,
    group_kernel_matrix,
    poly_kernel,
    quantum_kernel,
    rbf_gaussian,
    rff_kernel,
)
from .models import kernel_anomaly, svm_with_kernel
from .quantum_states import states
from .stats import advantage_score, bootstrap_ci


# -----------------------------------------------------------------------
# Quick configuration helpers
# -----------------------------------------------------------------------

DEFAULT_FEATURE_MAPS = ["separable_ry", "entangled_cx", "rzz_ring",
                        "hardware_efficient", "pauli_z_features"]
DEFAULT_CLASSICAL = ["cosine", "rbf", "polynomial", "rff"]
QUICK_SEEDS = [0, 1]
FULL_SEEDS = [0, 1, 2, 3, 4]


def _angles_from(images: np.ndarray, n_qubits: int, seed: int
                  ) -> np.ndarray:
    _, angles = images_to_angles(images, n_qubits=n_qubits, seed=seed)
    return angles


# -----------------------------------------------------------------------
# Per-method classification accuracy on one (train, test) split
# -----------------------------------------------------------------------

def _run_classification_methods(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_qubits: int,
    feature_maps: Sequence[str],
    group_name: str,
    classical: Sequence[str],
    seed: int,
    fm_depth: int = 1,
) -> Dict[str, float]:
    """Returns {method_name: accuracy}."""
    out: Dict[str, float] = {}
    angles_train = _angles_from(X_train, n_qubits, seed=seed)
    angles_test = _angles_from(X_test, n_qubits, seed=seed)

    # Plain (no-group) quantum kernels.
    for fm in feature_maps:
        s_train = states(angles_train, feature_map=fm)
        s_test = states(angles_test, feature_map=fm)
        K_tr = quantum_kernel(s_train)
        K_te = quantum_kernel(s_test, s_train)
        r = svm_with_kernel(K_tr, y_train, K_te, y_test, seed=seed)
        out[f"quantum_{fm}"] = r.accuracy

    # Group-averaged quantum kernel using the *entangled_cx* map only -
    # this isolates the contribution of group averaging from the choice of map.
    G = make_group(group_name)
    if G.size > 1:
        orbit_train = _orbit_states(X_train, G, n_qubits, seed=seed,
                                    feature_map="entangled_cx")
        orbit_test = _orbit_states(X_test, G, n_qubits, seed=seed,
                                    feature_map="entangled_cx")
        K_tr = group_kernel_matrix(orbit_train)
        K_te = group_kernel_matrix(orbit_test, orbit_train)
        r = svm_with_kernel(K_tr, y_train, K_te, y_test, seed=seed)
        out[f"quantum_{group_name.lower()}_entangled"] = r.accuracy

    # Classical baselines.
    for c in classical:
        K_tr = CLASSICAL_KERNELS[c](X_train.reshape(len(X_train), -1))
        K_te = CLASSICAL_KERNELS[c](X_test.reshape(len(X_test), -1),
                                     X_train.reshape(len(X_train), -1))
        if c == "rff":
            K_tr = rff_kernel(X_train.reshape(len(X_train), -1), seed=seed)
            K_te = rff_kernel(X_test.reshape(len(X_test), -1),
                              X_train.reshape(len(X_train), -1), seed=seed)
        r = svm_with_kernel(K_tr, y_train, K_te, y_test, seed=seed)
        out[f"classical_{c}"] = r.accuracy
    return out


def _orbit_states(images: np.ndarray, group, n_qubits: int, seed: int,
                  feature_map: str = "entangled_cx") -> List[List]:
    """For each image, build a list of |G| Statevectors over its orbit."""
    out = []
    for img in images:
        orbit_imgs = group.orbit(img)
        _, angles = images_to_angles(np.stack(orbit_imgs, axis=0),
                                      n_qubits=n_qubits, seed=seed)
        out.append(states(angles, feature_map=feature_map))
    return out


# -----------------------------------------------------------------------
# Anomaly path
# -----------------------------------------------------------------------

def _run_anomaly_methods(
    X_train_clean: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    n_qubits: int, feature_maps: Sequence[str], classical: Sequence[str],
    seed: int,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    angles_train = _angles_from(X_train_clean, n_qubits, seed=seed)
    angles_test = _angles_from(X_test, n_qubits, seed=seed)
    for fm in feature_maps:
        s_train = states(angles_train, feature_map=fm)
        s_test = states(angles_test, feature_map=fm)
        K_tr = quantum_kernel(s_train)
        K_te = quantum_kernel(s_test, s_train)
        r = kernel_anomaly(K_tr, K_te, y_test)
        out[f"quantum_{fm}"] = r.auc
    for c in classical:
        K_tr = CLASSICAL_KERNELS[c](X_train_clean.reshape(len(X_train_clean), -1))
        K_te = CLASSICAL_KERNELS[c](X_test.reshape(len(X_test), -1),
                                     X_train_clean.reshape(len(X_train_clean), -1))
        if c == "rff":
            K_tr = rff_kernel(X_train_clean.reshape(len(X_train_clean), -1), seed=seed)
            K_te = rff_kernel(X_test.reshape(len(X_test), -1),
                              X_train_clean.reshape(len(X_train_clean), -1), seed=seed)
        r = kernel_anomaly(K_tr, K_te, y_test)
        out[f"classical_{c}"] = r.auc
    return out


# -----------------------------------------------------------------------
# Experiment registry
# -----------------------------------------------------------------------

@dataclass
class Experiment:
    name: str
    family: str                 # "classification" or "anomaly"
    dataset: Callable[..., Dataset]
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    n_qubits: int = 4
    n_train: int = 40
    n_test: int = 40
    feature_maps: Sequence[str] = field(default_factory=lambda: DEFAULT_FEATURE_MAPS)
    classical: Sequence[str] = field(default_factory=lambda: DEFAULT_CLASSICAL)
    group: str = "Identity"
    entanglement_depth: int = 1  # only consulted for the 'depth' ablations
    notes: str = ""


def _split(ds: Dataset, n_train: int, n_test: int, seed: int
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))
    n_train = min(n_train, len(ds) - 1)
    n_test = min(n_test, len(ds) - n_train)
    tr = idx[:n_train]; te = idx[n_train:n_train + n_test]
    return ds.X[tr], ds.y[tr], ds.X[te], ds.y[te]


def _split_anomaly(ds: Dataset, n_train_clean: int, n_test: int, seed: int
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train on a subset of class-0 (clean) samples; test on the remainder
    of the dataset (mix of clean and anomaly)."""
    rng = np.random.default_rng(seed)
    clean_idx = np.flatnonzero(ds.y == 0); rng.shuffle(clean_idx)
    n_train_clean = min(n_train_clean, max(1, len(clean_idx) - 1))
    train_clean = clean_idx[:n_train_clean]
    used = set(train_clean.tolist())
    remaining = np.asarray([i for i in range(len(ds)) if i not in used])
    rng.shuffle(remaining)
    test = remaining[:n_test]
    return ds.X[train_clean], ds.X[test], ds.y[test]


def run_experiment(exp: Experiment, seeds: Sequence[int]) -> Dict[str, Any]:
    """Run an experiment across the seeds and return per-seed metric dicts."""
    per_seed: List[Dict[str, float]] = []
    for s in seeds:
        ds = exp.dataset(seed=int(s), **exp.dataset_kwargs)
        if exp.family == "classification":
            X_tr, y_tr, X_te, y_te = _split(ds, exp.n_train, exp.n_test, seed=int(s))
            metrics = _run_classification_methods(
                X_tr, y_tr, X_te, y_te,
                n_qubits=exp.n_qubits, feature_maps=exp.feature_maps,
                group_name=exp.group, classical=exp.classical,
                seed=int(s), fm_depth=exp.entanglement_depth,
            )
        elif exp.family == "anomaly":
            X_tr, X_te, y_te = _split_anomaly(ds, exp.n_train, exp.n_test, seed=int(s))
            metrics = _run_anomaly_methods(
                X_tr, X_te, y_te,
                n_qubits=exp.n_qubits, feature_maps=exp.feature_maps,
                classical=exp.classical, seed=int(s),
            )
        else:
            raise ValueError(f"Unknown family {exp.family!r}")
        per_seed.append(metrics)
    return {
        "name": exp.name,
        "family": exp.family,
        "dataset": exp.dataset.__name__,
        "n_qubits": exp.n_qubits,
        "group": exp.group,
        "entanglement_depth": exp.entanglement_depth,
        "notes": exp.notes,
        "seeds": list(seeds),
        "per_seed": per_seed,
    }


# -----------------------------------------------------------------------
# The 20 experiments
# -----------------------------------------------------------------------

def all_experiments() -> List[Experiment]:
    exps: List[Experiment] = []

    # 1) parity_patch: separable quantum vs entangled quantum
    exps.append(Experiment("01_parity_separable_vs_entangled", "classification",
                            parity_patch, dict(n=200, patch=2),
                            n_qubits=4, n_train=40, n_test=40,
                            feature_maps=["separable_ry", "entangled_cx", "rzz_ring"],
                            classical=[], notes="Parity: separable Ry vs entangled CX vs RZZ ring."))
    # 2) parity_patch: entangled quantum vs RBF
    exps.append(Experiment("02_parity_entangled_vs_rbf", "classification",
                            parity_patch, dict(n=200, patch=2),
                            n_qubits=4, n_train=40, n_test=40,
                            feature_maps=["entangled_cx", "rzz_ring"],
                            classical=["rbf"], notes="Parity: entangled quantum vs RBF baseline."))
    # 3) parity_patch: entangled vs polynomial
    exps.append(Experiment("03_parity_entangled_vs_polynomial", "classification",
                            parity_patch, dict(n=200, patch=2),
                            n_qubits=4, n_train=40, n_test=40,
                            feature_maps=["entangled_cx", "rzz_ring"],
                            classical=["polynomial"], notes="Parity: vs polynomial kernel."))
    # 4) parity_patch under noise 0.05
    exps.append(Experiment("04_parity_noise05", "classification",
                            parity_patch, dict(n=200, patch=2, noise=0.05),
                            n_qubits=4, n_train=40, n_test=40,
                            feature_maps=["entangled_cx", "rzz_ring"],
                            classical=["rbf", "polynomial"], notes="Parity with low noise."))
    # 5) parity_patch under noise 0.15
    exps.append(Experiment("05_parity_noise15", "classification",
                            parity_patch, dict(n=200, patch=2, noise=0.15),
                            n_qubits=4, n_train=40, n_test=40,
                            feature_maps=["entangled_cx", "rzz_ring"],
                            classical=["rbf", "polynomial"], notes="Parity with higher noise."))
    # 6) D4-shape: no group pooling vs D4 group quantum kernel
    exps.append(Experiment("06_d4shape_groupavg", "classification",
                            d4_shapes, dict(n=120, side=8, add_random_d4=True),
                            n_qubits=4, n_train=40, n_test=40,
                            feature_maps=["entangled_cx"],
                            classical=[], group="D4",
                            notes="D4-shape: identity-group vs D4 group-averaged quantum kernel."))
    # 7) D4-shape: D4 quantum kernel vs RBF on raw features
    exps.append(Experiment("07_d4shape_d4_vs_rbf", "classification",
                            d4_shapes, dict(n=120, side=8, add_random_d4=True),
                            n_qubits=4, n_train=40, n_test=40,
                            feature_maps=["entangled_cx"],
                            classical=["rbf"], group="D4",
                            notes="D4 group quantum kernel vs classical RBF."))
    # 8) D4-shape under random rotations only (C4)
    exps.append(Experiment("08_d4shape_c4_only", "classification",
                            d4_shapes, dict(n=120, side=8, add_random_d4=True),
                            n_qubits=4, n_train=40, n_test=40,
                            feature_maps=["entangled_cx"],
                            classical=["rbf"], group="C4",
                            notes="D4 dataset, only C4 averaging used; tests if reflections help."))
    # 9) D4-shape under random reflections + rotations
    exps.append(Experiment("09_d4shape_full_d4", "classification",
                            d4_shapes, dict(n=120, side=8, add_random_d4=True),
                            n_qubits=4, n_train=40, n_test=40,
                            feature_maps=["entangled_cx"],
                            classical=["rbf", "polynomial"], group="D4",
                            notes="D4 dataset, full D4 averaging vs classical."))
    # 10) defect detection (local anomaly)
    exps.append(Experiment("10_anomaly_local", "anomaly",
                            defect_patches, dict(n_clean=80, n_defect=20, side=8,
                                                 defect_type="local"),
                            n_qubits=4, n_train=40, n_test=40,
                            feature_maps=["entangled_cx", "rzz_ring"],
                            classical=["rbf", "polynomial"],
                            notes="One-class anomaly: local bright squares."))
    # 11) defect detection with color confounder
    exps.append(Experiment("11_anomaly_color_confound", "anomaly",
                            defect_patches, dict(n_clean=80, n_defect=20, side=8,
                                                 defect_type="local",
                                                 color_confounder=True),
                            n_qubits=4, n_train=40, n_test=40,
                            feature_maps=["entangled_cx", "rzz_ring"],
                            classical=["rbf", "polynomial"],
                            notes="Anomaly with global brightness confounder."))
    # 12) defect detection with rotated defects
    exps.append(Experiment("12_anomaly_rotated", "anomaly",
                            defect_patches, dict(n_clean=80, n_defect=20, side=8,
                                                 defect_type="local",
                                                 rotate_defect=True),
                            n_qubits=4, n_train=40, n_test=40,
                            feature_maps=["entangled_cx", "rzz_ring"],
                            classical=["rbf", "polynomial"],
                            notes="Anomaly with randomly placed defects."))
    # 13-15) n_qubits ablations
    for nq in (3, 4, 5):
        exps.append(Experiment(f"{12 + nq - 2:02d}_ablation_nq_{nq}", "classification",
                                d4_shapes, dict(n=120, side=8, add_random_d4=True),
                                n_qubits=nq, n_train=40, n_test=40,
                                feature_maps=["entangled_cx"],
                                classical=["rbf"], group="D4",
                                notes=f"Ablation: n_qubits = {nq}."))
    # 16-18) entanglement-depth ablations
    for d in (0, 1, 2):
        exps.append(Experiment(f"{16 + d:02d}_ablation_depth_{d}", "classification",
                                parity_patch, dict(n=200, patch=2),
                                n_qubits=4, n_train=40, n_test=40,
                                feature_maps=["entangled_cx"],
                                classical=["rbf"], group="Identity",
                                entanglement_depth=d,
                                notes=f"Ablation: entanglement depth = {d}."))
    # 19) group = identity only
    exps.append(Experiment("19_ablation_group_identity", "classification",
                            d4_shapes, dict(n=120, side=8, add_random_d4=True),
                            n_qubits=4, n_train=40, n_test=40,
                            feature_maps=["entangled_cx"],
                            classical=["rbf"], group="Identity",
                            notes="Ablation: no group averaging."))
    # 20) group = full D4
    exps.append(Experiment("20_ablation_group_d4_full", "classification",
                            d4_shapes, dict(n=120, side=8, add_random_d4=True),
                            n_qubits=4, n_train=40, n_test=40,
                            feature_maps=["entangled_cx"],
                            classical=["rbf"], group="D4",
                            notes="Ablation: full D4 group averaging (paired with 19)."))

    return exps


# Optional real-data add-ons (graceful no-op if dependencies are missing).
def optional_real_data_experiments() -> List[Experiment]:
    out: List[Experiment] = []
    for ds_name, factory in (("fashion_mnist", fashion_mnist_pair),
                              ("medmnist", medmnist_pair)):
        out.append(Experiment(f"opt_{ds_name}_low_data", "classification",
                               factory, dict(n_per_class=20),
                               n_qubits=4, n_train=20, n_test=20,
                               feature_maps=["entangled_cx"],
                               classical=["rbf"], group="Identity",
                               notes=f"Optional real-data: {ds_name} low-data 2-class."))
    return out


# -----------------------------------------------------------------------
# Driver entrypoint
# -----------------------------------------------------------------------

def run_suite(experiments: Sequence[Experiment], seeds: Sequence[int],
               out_dir: Path, on_progress: Callable[[int, int, str], None] | None = None
               ) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []
    t0 = time.time()
    for k, exp in enumerate(experiments):
        if on_progress:
            on_progress(k, len(experiments), exp.name)
        try:
            r = run_experiment(exp, seeds)
        except Exception as e:  # pragma: no cover - reported in summary
            r = {"name": exp.name, "family": exp.family, "error": str(e),
                 "seeds": list(seeds), "per_seed": []}
        all_results.append(r)
    elapsed = time.time() - t0
    summary = {"elapsed_seconds": elapsed,
                "n_experiments": len(experiments),
                "seeds": list(seeds),
                "results": all_results}
    return summary


# -----------------------------------------------------------------------
# Summary-table helpers
# -----------------------------------------------------------------------

def long_table(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the nested summary into per-(experiment, method) rows with
    bootstrap CIs and the quantum-advantage signal."""
    rows: List[Dict[str, Any]] = []
    for exp in summary["results"]:
        if exp.get("error"):
            rows.append({"experiment": exp["name"], "error": exp["error"]})
            continue
        per_seed: List[Dict[str, float]] = exp["per_seed"]
        if not per_seed:
            continue
        methods = sorted(per_seed[0].keys())
        means: Dict[str, float] = {}
        for m in methods:
            vals = [d.get(m, float("nan")) for d in per_seed]
            ci = bootstrap_ci(vals)
            rows.append({
                "experiment": exp["name"], "family": exp["family"],
                "method": m, "mean": ci.mean, "std": ci.std,
                "lo": ci.lo, "hi": ci.hi, "n": ci.n,
            })
            means[m] = ci.mean
        # Quantum-advantage signal: best quantum mean - best classical mean
        q_means = {m: v for m, v in means.items() if m.startswith("quantum")}
        c_means = {m: v for m, v in means.items() if m.startswith("classical")}
        if q_means and c_means:
            best_q = max(q_means, key=q_means.get)
            best_c = max(c_means, key=c_means.get)
            advantage = q_means[best_q] - c_means[best_c]
            paired_q = [d.get(best_q, float("nan")) for d in per_seed]
            paired_c = [d.get(best_c, float("nan")) for d in per_seed]
            adv_ci = bootstrap_ci([a - b for a, b in zip(paired_q, paired_c)])
            rows.append({
                "experiment": exp["name"], "family": exp["family"],
                "method": "ADVANTAGE_SIGNAL",
                "best_quantum": best_q, "best_classical": best_c,
                "mean": advantage, "lo": adv_ci.lo, "hi": adv_ci.hi,
                "n": adv_ci.n,
            })
    return rows
