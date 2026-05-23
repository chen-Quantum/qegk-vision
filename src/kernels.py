"""Quantum and classical kernel matrices.

Definitions
-----------
Fidelity kernel:        K(x, y) = |<psi(x) | psi(y)>|^2
Quantum distance:       d(x, y) = arccos(sqrt(K(x, y)))
Group-averaged kernel:  K_G(x, y) = (1/|G|^2) sum_{g, h in G}
                                    |<psi(g.x) | psi(h.y)>|^2

For classical baselines we follow scikit-learn conventions:
    cosine, rbf (Gaussian), polynomial, random Fourier features (linear after
    RFF projection).
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

import numpy as np
from qiskit.quantum_info import Statevector
from sklearn.metrics.pairwise import (
    cosine_similarity, polynomial_kernel, rbf_kernel,
)


# -----------------------------------------------------------------------
# Quantum kernels
# -----------------------------------------------------------------------

def _fidelity(psi: Statevector, phi: Statevector) -> float:
    amp = complex(psi.inner(phi))
    f = float(abs(amp) ** 2)
    return min(max(f, 0.0), 1.0)


def quantum_kernel(states_a: Sequence[Statevector],
                    states_b: Sequence[Statevector] | None = None) -> np.ndarray:
    """Pairwise fidelity matrix between two sets of states. If `states_b` is
    None, returns the symmetric NxN matrix on a single set."""
    sa = list(states_a)
    if states_b is None:
        n = len(sa)
        K = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            K[i, i] = 1.0
            for j in range(i + 1, n):
                v = _fidelity(sa[i], sa[j])
                K[i, j] = v; K[j, i] = v
        return K
    sb = list(states_b)
    K = np.empty((len(sa), len(sb)), dtype=np.float64)
    for i, p in enumerate(sa):
        for j, q in enumerate(sb):
            K[i, j] = _fidelity(p, q)
    return K


def quantum_distance(K: np.ndarray) -> np.ndarray:
    """arccos(sqrt(K)) elementwise. Clips for numerical safety."""
    K = np.clip(np.asarray(K, dtype=np.float64), 0.0, 1.0)
    return np.arccos(np.sqrt(K))


def group_kernel_matrix(orbit_states_a: List[List[Statevector]],
                         orbit_states_b: List[List[Statevector]] | None = None
                         ) -> np.ndarray:
    """Group-averaged kernel.

    Parameters
    ----------
    orbit_states_a : list of length N, each element is a list of |G| states
                     representing the orbit of x_i under G.
    orbit_states_b : same for the second set, or None for the symmetric case.

    Returns
    -------
    K_G[i, j] = (1/|G|^2) * sum_{g, h} |<psi(g x_i) | psi(h x_j)>|^2.
    """
    a_orbits = orbit_states_a
    if orbit_states_b is None:
        b_orbits = a_orbits
        symmetric = True
    else:
        b_orbits = orbit_states_b
        symmetric = False

    G = len(a_orbits[0])
    N = len(a_orbits); M = len(b_orbits)
    K = np.empty((N, M), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            if symmetric and j < i:
                K[i, j] = K[j, i]
                continue
            acc = 0.0
            for g in range(G):
                pa = a_orbits[i][g]
                for h in range(G):
                    acc += _fidelity(pa, b_orbits[j][h])
            K[i, j] = acc / (G * G)
    if symmetric:
        # Force symmetry to numerical precision.
        K = 0.5 * (K + K.T)
    return K


# -----------------------------------------------------------------------
# Classical baselines
# -----------------------------------------------------------------------

def cosine_kernel(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Cosine similarity, NOT distance. Maps into [-1, 1]."""
    if Y is None:
        Y = X
    K = cosine_similarity(X, Y)
    return np.clip(K, -1.0, 1.0)


def rbf_gaussian(X: np.ndarray, Y: np.ndarray | None = None,
                  gamma: float | None = None) -> np.ndarray:
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / max(X.shape[1], 1)
    return rbf_kernel(X, Y, gamma=gamma)


def poly_kernel(X: np.ndarray, Y: np.ndarray | None = None,
                 degree: int = 3, coef0: float = 1.0,
                 gamma: float | None = None) -> np.ndarray:
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / max(X.shape[1], 1)
    return polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)


def random_fourier_features(X: np.ndarray, n_components: int = 64,
                              gamma: float = 1.0, seed: int = 0
                              ) -> np.ndarray:
    """Approximate RBF kernel via Rahimi-Recht random Fourier features."""
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=np.sqrt(2 * gamma), size=(X.shape[1], n_components))
    b = rng.uniform(0, 2 * np.pi, size=n_components)
    Z = np.sqrt(2.0 / n_components) * np.cos(X @ W + b)
    return Z


def rff_kernel(X: np.ndarray, Y: np.ndarray | None = None,
                n_components: int = 64, gamma: float | None = None,
                seed: int = 0) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / max(X.shape[1], 1)
    Zx = random_fourier_features(X, n_components=n_components, gamma=gamma, seed=seed)
    if Y is None:
        Zy = Zx
    else:
        Zy = random_fourier_features(Y, n_components=n_components, gamma=gamma, seed=seed)
    return Zx @ Zy.T


# -----------------------------------------------------------------------
# Registry helper
# -----------------------------------------------------------------------

CLASSICAL_KERNELS: dict[str, Callable[..., np.ndarray]] = {
    "cosine": cosine_kernel,
    "rbf": rbf_gaussian,
    "polynomial": poly_kernel,
    "rff": rff_kernel,
}
