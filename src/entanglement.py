"""Entanglement diagnostics: reduced density matrices, purity, linear and
von Neumann entropies for a bipartition of `n_qubits` into the first
`n_left` qubits versus the rest.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from qiskit.quantum_info import Statevector, partial_trace


def reduced_rho(psi: Statevector, n_left: int) -> np.ndarray:
    """Trace out the right block of size n_q - n_left, returning the left
    block reduced density matrix as a numpy array of shape (2^n_left, 2^n_left).
    """
    n_q = int(np.log2(psi.dim))
    if n_left < 1 or n_left >= n_q:
        raise ValueError(f"n_left must be in [1, n_q-1]; got {n_left} for n_q={n_q}")
    # Qiskit uses little-endian qubit ordering: qubit 0 is the right-most
    # tensor factor.  We trace out qubits [n_left, n_q) for the left block.
    right_qubits = list(range(n_left, n_q))
    rho = partial_trace(psi, right_qubits).data
    return np.asarray(rho)


def purity(rho: np.ndarray) -> float:
    """Tr(rho^2) - 1 means pure (separable), < 1 means mixed (entangled)."""
    p = float(np.real(np.trace(rho @ rho)))
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


def linear_entropy(rho: np.ndarray) -> float:
    """1 - Tr(rho^2)."""
    return 1.0 - purity(rho)


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    """-Tr(rho log rho), in nats. Eigenvalues are clipped at `eps`."""
    w = np.linalg.eigvalsh(rho)
    w = np.clip(np.real(w), eps, 1.0)
    return float(-np.sum(w * np.log(w)))


def state_diagnostics(psi: Statevector, n_left: int | None = None) -> dict:
    """Compute purity, linear entropy, and von Neumann entropy for the
    bipartition with `n_left` qubits in the left block. Defaults to a
    balanced split.
    """
    n_q = int(np.log2(psi.dim))
    if n_left is None:
        n_left = max(1, n_q // 2)
    rho = reduced_rho(psi, n_left=n_left)
    return {
        "n_qubits": n_q,
        "n_left": n_left,
        "purity": purity(rho),
        "linear_entropy": linear_entropy(rho),
        "von_neumann_entropy": von_neumann_entropy(rho),
    }


def batch_diagnostics(states: Iterable[Statevector], n_left: int | None = None) -> dict:
    """Aggregate purity / entropy stats across a batch of states."""
    purities = []
    linears = []
    vns = []
    for psi in states:
        d = state_diagnostics(psi, n_left=n_left)
        purities.append(d["purity"])
        linears.append(d["linear_entropy"])
        vns.append(d["von_neumann_entropy"])
    return {
        "n": len(purities),
        "purity_mean": float(np.mean(purities)),
        "purity_std": float(np.std(purities)),
        "linear_entropy_mean": float(np.mean(linears)),
        "linear_entropy_std": float(np.std(linears)),
        "von_neumann_entropy_mean": float(np.mean(vns)),
        "von_neumann_entropy_std": float(np.std(vns)),
    }
