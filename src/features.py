"""Compact feature extractors. Goal is to compress a small image (typically
4x4 to 16x16) into an `n_qubits`-dimensional real vector that can be passed
to either a quantum feature map or a classical kernel.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from scipy import ndimage

from . import SEED


# -----------------------------------------------------------------------
# Individual descriptors
# -----------------------------------------------------------------------

def lowres_gray(img: np.ndarray, size: int = 4) -> np.ndarray:
    """Average-pool an image to `size x size`, flatten."""
    if img.ndim == 3:
        img = img.mean(axis=-1)
    H, W = img.shape
    if size >= min(H, W):
        return img.astype(np.float32).flatten()
    bh, bw = H // size, W // size
    if bh == 0 or bw == 0:
        ys = np.linspace(0, H - 1, size).astype(int)
        xs = np.linspace(0, W - 1, size).astype(int)
        return img[np.ix_(ys, xs)].astype(np.float32).flatten()
    trimmed = img[: bh * size, : bw * size]
    return trimmed.reshape(size, bh, size, bw).mean(axis=(1, 3)).astype(np.float32).flatten()


def sobel_features(img: np.ndarray, bins: int = 4) -> np.ndarray:
    """Histogram of Sobel-magnitude responses."""
    if img.ndim == 3:
        img = img.mean(axis=-1)
    gx = ndimage.sobel(img.astype(np.float64), axis=1)
    gy = ndimage.sobel(img.astype(np.float64), axis=0)
    mag = np.sqrt(gx * gx + gy * gy)
    hi = max(mag.max(), 1e-6)
    h, _ = np.histogram(mag, bins=bins, range=(0, hi))
    h = h.astype(np.float32)
    s = h.sum()
    return h / s if s > 0 else h


def quadrant_stats(img: np.ndarray) -> np.ndarray:
    """Per-quadrant mean. Returns 4 numbers (TL, TR, BL, BR)."""
    if img.ndim == 3:
        img = img.mean(axis=-1)
    H, W = img.shape
    mh, mw = H // 2, W // 2
    return np.asarray([
        img[:mh, :mw].mean(),
        img[:mh, mw:].mean(),
        img[mh:, :mw].mean(),
        img[mh:, mw:].mean(),
    ], dtype=np.float32)


def patch_correlations(img: np.ndarray) -> np.ndarray:
    """Quick texture descriptor: correlation of adjacent rows and columns."""
    if img.ndim == 3:
        img = img.mean(axis=-1)
    a = img.astype(np.float64)
    # np.corrcoef divides by zero on constant inputs - silence that.
    with np.errstate(divide="ignore", invalid="ignore"):
        rows = np.corrcoef(a[:-1].flatten(), a[1:].flatten())[0, 1]
        cols = np.corrcoef(a[:, :-1].flatten(), a[:, 1:].flatten())[0, 1]
    rows = 0.0 if np.isnan(rows) else float(rows)
    cols = 0.0 if np.isnan(cols) else float(cols)
    return np.asarray([rows, cols], dtype=np.float32)


# -----------------------------------------------------------------------
# Combined extractor
# -----------------------------------------------------------------------

def raw_features(img: np.ndarray, lowres_size: int = 4, edge_bins: int = 4) -> np.ndarray:
    """Concatenate all descriptors into a single feature vector."""
    return np.concatenate([
        lowres_gray(img, size=lowres_size),
        sobel_features(img, bins=edge_bins),
        quadrant_stats(img),
        patch_correlations(img),
    ])


def stack_features(images: Iterable[np.ndarray], **kw) -> np.ndarray:
    return np.stack([raw_features(img, **kw) for img in images], axis=0)


# -----------------------------------------------------------------------
# Reduction to qubit dimension
# -----------------------------------------------------------------------

def _orthonormal(rng: np.random.Generator, d_in: int, d_out: int) -> np.ndarray:
    a = rng.standard_normal((d_in, d_out))
    q, _ = np.linalg.qr(a)
    return q[:, :d_out]


def reduce_to_qubits(raw: np.ndarray, n_qubits: int,
                      seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Standardise then project to `n_qubits` real dimensions, finally
    min-max scale to angles in [0, pi].

    Returns
    -------
    reduced : (N, n_qubits) - standardized then orthonormal-projected
    angles  : (N, n_qubits) - scaled to [0, pi] per dimension
    """
    rng = np.random.default_rng(seed)
    mu = raw.mean(axis=0, keepdims=True)
    sigma = raw.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    Z = (raw - mu) / sigma
    P = _orthonormal(rng, raw.shape[1], n_qubits)
    reduced = Z @ P
    rmin = reduced.min(axis=0, keepdims=True)
    rmax = reduced.max(axis=0, keepdims=True)
    span = np.where(rmax - rmin < 1e-12, 1.0, rmax - rmin)
    angles = (reduced - rmin) / span * np.pi
    return reduced.astype(np.float32), angles.astype(np.float32)


def images_to_angles(images: np.ndarray, n_qubits: int,
                      seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    raw = stack_features(images)
    return reduce_to_qubits(raw, n_qubits=n_qubits, seed=seed)
