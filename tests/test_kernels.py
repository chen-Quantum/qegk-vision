"""Tests for fidelity range, kernel symmetry, and group-averaged kernel."""

from __future__ import annotations

import numpy as np
import pytest

from src.feature_maps import available_feature_maps
from src.groups import make_group
from src.kernels import group_kernel_matrix, quantum_distance, quantum_kernel
from src.quantum_states import state, states


@pytest.mark.parametrize("fm", available_feature_maps())
def test_self_fidelity_is_one(fm: str) -> None:
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, np.pi, size=4)
    psi = state(theta, feature_map=fm)
    K = quantum_kernel([psi, psi])
    assert pytest.approx(1.0, abs=1e-9) == K[0, 1]


@pytest.mark.parametrize("fm", available_feature_maps())
def test_kernel_range_and_symmetry(fm: str) -> None:
    rng = np.random.default_rng(1)
    angles = rng.uniform(0.0, np.pi, size=(6, 4))
    K = quantum_kernel(states(angles, feature_map=fm))
    assert K.shape == (6, 6)
    assert (K >= -1e-12).all()
    assert (K <= 1.0 + 1e-9).all()
    assert np.allclose(K, K.T, atol=1e-12)
    assert np.allclose(np.diag(K), 1.0, atol=1e-9)


def test_quantum_distance_range() -> None:
    rng = np.random.default_rng(2)
    K = rng.uniform(0.0, 1.0, size=(8, 8))
    K = (K + K.T) / 2
    np.fill_diagonal(K, 1.0)
    D = quantum_distance(K)
    assert D.shape == K.shape
    assert (D >= 0).all() and (D <= np.pi / 2 + 1e-9).all()


def _to_orbit_states(images, n_qubits=4, seed=0, feature_map="entangled_cx"):
    from src.features import images_to_angles
    from src.groups import d4_actions
    out = []
    actions = list(d4_actions().values())
    for img in images:
        orbit = np.stack([g(img) for g in actions], axis=0)
        _, angles = images_to_angles(orbit, n_qubits=n_qubits, seed=seed)
        out.append(states(angles, feature_map=feature_map))
    return out


def test_group_averaged_kernel_symmetry() -> None:
    rng = np.random.default_rng(3)
    imgs = rng.standard_normal((4, 4, 4)).astype(np.float32)
    orbits = _to_orbit_states(imgs)
    K_G = group_kernel_matrix(orbits)
    assert K_G.shape == (4, 4)
    assert np.allclose(K_G, K_G.T, atol=1e-12)
    assert (K_G >= -1e-12).all()
    assert (K_G <= 1.0 + 1e-9).all()
