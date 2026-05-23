"""Smoke test: a tiny experiment runs end-to-end and produces sensible values."""

from __future__ import annotations

import numpy as np

from src.experiments import Experiment, run_experiment
from src.datasets import parity_patch


def test_tiny_experiment_end_to_end() -> None:
    exp = Experiment(
        name="t_parity",
        family="classification",
        dataset=parity_patch,
        dataset_kwargs=dict(n=40, patch=2),
        n_qubits=3,
        n_train=12, n_test=12,
        feature_maps=["entangled_cx"],
        classical=["rbf"],
        group="Identity",
        notes="tiny smoke test",
    )
    result = run_experiment(exp, seeds=[0])
    assert result["name"] == "t_parity"
    per_seed = result["per_seed"]
    assert len(per_seed) == 1
    metrics = per_seed[0]
    assert set(metrics.keys()) == {"quantum_entangled_cx", "classical_rbf"}
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
