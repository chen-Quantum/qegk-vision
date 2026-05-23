"""Thin wrapper that builds Qiskit Statevectors from feature angles via a
named feature map. The actual circuit definitions live in `feature_maps.py`.
"""

from __future__ import annotations

from typing import List

import numpy as np
from qiskit.quantum_info import Statevector

from .feature_maps import build_circuit, available_feature_maps


def state(theta: np.ndarray, feature_map: str = "entangled_cx") -> Statevector:
    qc = build_circuit(theta, feature_map=feature_map)
    return Statevector.from_instruction(qc)


def states(thetas: np.ndarray, feature_map: str = "entangled_cx") -> List[Statevector]:
    thetas = np.asarray(thetas, dtype=float)
    if thetas.ndim != 2:
        raise ValueError("thetas must be (N, n_qubits)")
    return [state(thetas[i], feature_map=feature_map) for i in range(thetas.shape[0])]


__all__ = ["state", "states", "available_feature_maps"]
