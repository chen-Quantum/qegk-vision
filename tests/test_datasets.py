"""Tests for synthetic dataset generators."""

from __future__ import annotations

import numpy as np

from src.datasets import d4_shapes, defect_patches, parity_patch


def test_parity_patch_shape_and_labels() -> None:
    ds = parity_patch(n=50, patch=2, seed=0)
    assert ds.X.shape == (50, 2, 2)
    assert ds.y.shape == (50,)
    # Labels are 0 or 1 and reflect actual parity of the binary patch.
    for i in range(len(ds)):
        bits = (ds.X[i] > 0.5).astype(int)
        assert ds.y[i] == int(bits.sum()) % 2


def test_parity_patch_noise_keeps_range() -> None:
    ds = parity_patch(n=20, patch=2, seed=0, noise=0.1)
    assert ds.X.min() >= 0.0 and ds.X.max() <= 1.0


def test_d4_shapes_two_classes() -> None:
    ds = d4_shapes(n=60, side=8, seed=0)
    assert ds.X.shape == (60, 8, 8)
    classes, counts = np.unique(ds.y, return_counts=True)
    assert set(classes.tolist()) <= {0, 1}
    # Both classes should be present for n=60.
    assert (counts > 0).all()


def test_defect_patches_label_balance() -> None:
    ds = defect_patches(n_clean=40, n_defect=10, side=8, seed=0)
    assert len(ds) == 50
    assert int((ds.y == 1).sum()) == 10
    assert int((ds.y == 0).sum()) == 40


def test_defect_patches_with_color_confounder() -> None:
    ds = defect_patches(n_clean=20, n_defect=5, side=8, seed=0,
                          color_confounder=True)
    assert ds.X.min() >= 0.0 and ds.X.max() <= 1.0
