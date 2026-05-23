"""Tests for entanglement diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from src.entanglement import (
    linear_entropy, purity, reduced_rho, state_diagnostics, von_neumann_entropy,
)


def _bell() -> Statevector:
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return Statevector(psi)


def _product() -> Statevector:
    # |00>
    return Statevector(np.array([1, 0, 0, 0], dtype=complex))


def test_bell_state_purity_is_half() -> None:
    rho_A = reduced_rho(_bell(), n_left=1)
    assert pytest.approx(0.5, abs=1e-9) == purity(rho_A)
    assert pytest.approx(0.5, abs=1e-9) == linear_entropy(rho_A)


def test_product_state_purity_is_one() -> None:
    rho_A = reduced_rho(_product(), n_left=1)
    assert pytest.approx(1.0, abs=1e-9) == purity(rho_A)
    assert pytest.approx(0.0, abs=1e-9) == linear_entropy(rho_A)


def test_von_neumann_entropy_nonneg() -> None:
    for psi in (_bell(), _product()):
        rho_A = reduced_rho(psi, n_left=1)
        assert von_neumann_entropy(rho_A) >= -1e-12


def test_diagnostics_dict_keys() -> None:
    d = state_diagnostics(_bell(), n_left=1)
    for k in ("n_qubits", "n_left", "purity", "linear_entropy", "von_neumann_entropy"):
        assert k in d
    assert d["n_qubits"] == 2
