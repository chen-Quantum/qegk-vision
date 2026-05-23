"""Five Qiskit feature maps used in the experimental suite:

    separable_ry            - Ry(theta_i) only, no entanglement
    entangled_cx            - Ry then linear CX chain
    rzz_ring                - Ry then RZZ on a ring (i, i+1 mod n_q)
    hardware_efficient      - Ry / Rz alternating with CX chain
    pauli_z_features        - separable Rz(theta_i) Hadamarded into Z-basis

`entanglement_depth` parameter on `entangled_cx` controls how many CX-chain
layers are stacked, allowing ablations 0 (no entanglement) / 1 (single
chain) / 2 (two chains with a Ry refresh between).
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
from qiskit import QuantumCircuit

_INV_PI = 1.0 / np.pi


def separable_ry(theta: np.ndarray, **_: object) -> QuantumCircuit:
    theta = np.asarray(theta, dtype=float).flatten()
    n = theta.size
    qc = QuantumCircuit(n, name="separable_ry")
    for i, t in enumerate(theta):
        qc.ry(float(t), i)
    return qc


def entangled_cx(theta: np.ndarray, entanglement_depth: int = 1,
                 **_: object) -> QuantumCircuit:
    """Ry per qubit followed by `entanglement_depth` linear CX chains, each
    chain preceded by a fresh Ry layer (parameter-free; constant pi/4 to
    keep the depth-effect visible at depth >= 2).
    """
    theta = np.asarray(theta, dtype=float).flatten()
    n = theta.size
    qc = QuantumCircuit(n, name=f"entangled_cx_d{entanglement_depth}")
    for i, t in enumerate(theta):
        qc.ry(float(t), i)
    for d in range(int(entanglement_depth)):
        for i in range(n - 1):
            qc.cx(i, i + 1)
        if d < entanglement_depth - 1:
            # Refresh layer between chains.
            for i in range(n):
                qc.ry(np.pi / 4, i)
    return qc


def rzz_ring(theta: np.ndarray, **_: object) -> QuantumCircuit:
    theta = np.asarray(theta, dtype=float).flatten()
    n = theta.size
    qc = QuantumCircuit(n, name="rzz_ring")
    for i, t in enumerate(theta):
        qc.ry(float(t), i)
    for i, t in enumerate(theta):
        qc.rz(float(t * t * _INV_PI), i)
    # Adjacent pairs.
    for i in range(n - 1):
        qc.rzz(float(theta[i] * theta[i + 1] * _INV_PI), i, i + 1)
    # Wrap-around edge to make it a ring (only when n_q >= 3 to avoid
    # double-counting the only edge for n=2).
    if n >= 3:
        qc.rzz(float(theta[-1] * theta[0] * _INV_PI), n - 1, 0)
    return qc


def hardware_efficient(theta: np.ndarray, **_: object) -> QuantumCircuit:
    """A two-layer hardware-efficient ansatz, brick-wall entanglement."""
    theta = np.asarray(theta, dtype=float).flatten()
    n = theta.size
    qc = QuantumCircuit(n, name="hardware_efficient")
    for i, t in enumerate(theta):
        qc.ry(float(t), i)
        qc.rz(float(t * 0.5), i)
    # Even pairs
    for i in range(0, n - 1, 2):
        qc.cx(i, i + 1)
    # Refresh
    for i, t in enumerate(theta):
        qc.ry(float(t * 0.7 + np.pi / 6), i)
    # Odd pairs (brick-wall second layer)
    for i in range(1, n - 1, 2):
        qc.cx(i, i + 1)
    return qc


def pauli_z_features(theta: np.ndarray, **_: object) -> QuantumCircuit:
    """Hadamard the qubits then apply diagonal Rz rotations.  Separable but
    represents a different geometry from `separable_ry`.  Used for the
    'separable but different' control in ablations.
    """
    theta = np.asarray(theta, dtype=float).flatten()
    n = theta.size
    qc = QuantumCircuit(n, name="pauli_z_features")
    for i in range(n):
        qc.h(i)
    for i, t in enumerate(theta):
        qc.rz(float(t), i)
    return qc


_REGISTRY = {
    "separable_ry": separable_ry,
    "entangled_cx": entangled_cx,
    "rzz_ring": rzz_ring,
    "hardware_efficient": hardware_efficient,
    "pauli_z_features": pauli_z_features,
}


def available_feature_maps() -> List[str]:
    return list(_REGISTRY.keys())


def build_circuit(theta: np.ndarray, feature_map: str = "entangled_cx",
                   **kw) -> QuantumCircuit:
    if feature_map not in _REGISTRY:
        raise KeyError(
            f"Unknown feature_map {feature_map!r}. "
            f"Choices: {available_feature_maps()}"
        )
    return _REGISTRY[feature_map](theta, **kw)
