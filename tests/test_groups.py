"""Tests for D4 actions, orbits, and the Cayley structure."""

from __future__ import annotations

import numpy as np
import pytest

from src.groups import (
    D4_LABELS, d4_actions, d4_cayley_table, d4_orbit, identity_group, make_group,
)


def _rand_img(side: int = 6, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((side, side)).astype(np.float32)


def test_d4_has_eight_elements() -> None:
    assert len(D4_LABELS) == 8
    assert set(D4_LABELS) == {"e", "r", "r2", "r3", "s", "sr", "sr2", "sr3"}


def test_d4_orbit_has_eight_images() -> None:
    img = _rand_img()
    orbit = d4_orbit(img)
    assert len(orbit) == 8
    # All images have the same shape as the input.
    for o in orbit:
        assert o.shape == img.shape


def test_identity_acts_trivially() -> None:
    img = _rand_img()
    assert np.array_equal(d4_actions()["e"](img), img)


def test_d4_orbit_contains_unique_images_for_generic_input() -> None:
    img = _rand_img(side=6, seed=42)
    orbit = d4_orbit(img)
    # For a random asymmetric image, the eight orbit elements should all
    # differ pairwise.
    for i in range(8):
        for j in range(i + 1, 8):
            assert not np.array_equal(orbit[i], orbit[j])


def test_r_has_order_four() -> None:
    img = _rand_img()
    act = d4_actions()
    rotated = img
    for _ in range(4):
        rotated = act["r"](rotated)
    assert np.allclose(rotated, img, atol=1e-7)


def test_s_has_order_two() -> None:
    img = _rand_img()
    act = d4_actions()
    assert np.allclose(act["s"](act["s"](img)), img, atol=1e-7)


def test_cayley_table_is_closed_in_d4() -> None:
    table = d4_cayley_table()
    for g in D4_LABELS:
        for h in D4_LABELS:
            assert table[g][h] in D4_LABELS
    # Every element should have a unique inverse: g*h = e for exactly one h per g.
    for g in D4_LABELS:
        inverses = [h for h in D4_LABELS if table[g][h] == "e"]
        assert len(inverses) == 1


def test_identity_group_is_trivial() -> None:
    G = make_group("identity")
    assert G.size == 1
    img = _rand_img()
    orbit = G.orbit(img)
    assert len(orbit) == 1
    assert np.array_equal(orbit[0], img)
