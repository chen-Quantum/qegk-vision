"""Synthetic datasets used by the experiment suite.

All datasets are reproducible from the project seed. Optional real-data
loaders (Fashion-MNIST, MedMNIST) degrade gracefully when their dependency
or data is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import SEED
from .groups import d4_actions


# -----------------------------------------------------------------------
# Common dataset container
# -----------------------------------------------------------------------

@dataclass
class Dataset:
    name: str
    X: np.ndarray            # (N, H, W) grayscale patches in [0, 1]
    y: np.ndarray            # (N,) labels
    classes: Sequence[str]   # names per integer label

    def __len__(self) -> int:
        return self.X.shape[0]

    def subset(self, idx: np.ndarray) -> "Dataset":
        return Dataset(self.name, self.X[idx], self.y[idx], self.classes)


# -----------------------------------------------------------------------
# 1) Synthetic parity-patch dataset
# -----------------------------------------------------------------------

def parity_patch(n: int = 200, patch: int = 2, seed: int = SEED,
                  noise: float = 0.0) -> Dataset:
    """Binary classification where the label is the parity of the number of
    "on" cells in a small {0,1} patch.

    Parity is a textbook 'hard for kernels without products' target: a plain
    RBF on the raw pixels cannot separate the two classes, while a feature
    map that introduces multiplicative interactions (ZZ entanglement, or a
    polynomial kernel of sufficient degree) can.

    Parameters
    ----------
    n : number of samples.
    patch : side length. Total cells = patch * patch.
    noise : additive gaussian noise (std) applied after the binary draw.
    """
    rng = np.random.default_rng(seed)
    cells = patch * patch
    bits = rng.integers(0, 2, size=(n, cells)).astype(np.float32)
    y = bits.sum(axis=1).astype(int) & 1  # parity
    X = bits.reshape(n, patch, patch)
    if noise > 0:
        X = X + noise * rng.standard_normal(X.shape).astype(np.float32)
        X = np.clip(X, 0.0, 1.0)
    return Dataset("parity_patch", X, y, ["even", "odd"])


# -----------------------------------------------------------------------
# 2) Synthetic D4-shape dataset
# -----------------------------------------------------------------------

def d4_shapes(n: int = 200, side: int = 8, seed: int = SEED,
              add_random_d4: bool = True, noise: float = 0.05) -> Dataset:
    """Two-class shape classification. Class 0 is an L-shape; class 1 is a
    T-shape. Each instance is randomly rotated/reflected by a D4 element,
    making the task group-equivariant.

    With `add_random_d4=True`, the rotation is sampled uniformly from D4.
    With it False, the canonical orientation is used.
    """
    rng = np.random.default_rng(seed)
    actions = list(d4_actions().values())

    def _canonical_L(side: int) -> np.ndarray:
        img = np.zeros((side, side), dtype=np.float32)
        h = side // 2
        img[1:side-1, h] = 1.0           # vertical bar
        img[side-2, h:side-1] = 1.0      # horizontal foot
        return img

    def _canonical_T(side: int) -> np.ndarray:
        img = np.zeros((side, side), dtype=np.float32)
        h = side // 2
        img[1, 1:side-1] = 1.0           # top bar
        img[1:side-1, h] = 1.0           # vertical stem
        return img

    X = np.zeros((n, side, side), dtype=np.float32)
    y = np.zeros(n, dtype=int)
    canonical = [_canonical_L(side), _canonical_T(side)]
    for i in range(n):
        cls = int(rng.integers(0, 2))
        y[i] = cls
        img = canonical[cls].copy()
        if add_random_d4:
            g = actions[int(rng.integers(0, len(actions)))]
            img = g(img)
        if noise > 0:
            img = img + noise * rng.standard_normal(img.shape).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)
        X[i] = img
    return Dataset("d4_shapes", X, y, ["L", "T"])


# -----------------------------------------------------------------------
# 3) Synthetic defect / anomaly patch dataset
# -----------------------------------------------------------------------

def defect_patches(n_clean: int = 100, n_defect: int = 30, side: int = 8,
                    seed: int = SEED, defect_type: str = "local",
                    color_confounder: bool = False,
                    rotate_defect: bool = False) -> Dataset:
    """A simple anomaly dataset.

    Clean patches are smoothly varying textures; defective patches contain a
    small contrasting square. The label is 0 (clean) or 1 (defect).

    defect_type: "local" (small bright square), "noise" (high-noise patch).
    color_confounder: shift mean brightness across the clean class so the
                      simple "global mean" heuristic does not work.
    rotate_defect: place the defect in a random D4-rotated position.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    yy /= max(side - 1, 1); xx /= max(side - 1, 1)

    def _clean(brightness: float = 0.5) -> np.ndarray:
        a = rng.uniform(-0.4, 0.4)
        b = rng.uniform(-0.4, 0.4)
        img = brightness + 0.20 * (np.sin(2 * np.pi * (xx + a)) * np.cos(2 * np.pi * (yy + b)))
        img = img + 0.03 * rng.standard_normal(img.shape).astype(np.float32)
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def _defect(brightness: float = 0.5) -> np.ndarray:
        img = _clean(brightness)
        d_side = max(2, side // 3)
        if rotate_defect:
            i0 = int(rng.integers(0, side - d_side + 1))
            j0 = int(rng.integers(0, side - d_side + 1))
        else:
            i0 = side // 2 - d_side // 2
            j0 = side // 2 - d_side // 2
        if defect_type == "local":
            img[i0:i0 + d_side, j0:j0 + d_side] = 1.0
        elif defect_type == "noise":
            img[i0:i0 + d_side, j0:j0 + d_side] = rng.uniform(
                0.0, 1.0, (d_side, d_side)
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown defect_type {defect_type!r}")
        return img

    X = np.zeros((n_clean + n_defect, side, side), dtype=np.float32)
    y = np.zeros(n_clean + n_defect, dtype=int)
    for i in range(n_clean):
        b = float(rng.uniform(0.3, 0.7)) if color_confounder else 0.5
        X[i] = _clean(b)
        y[i] = 0
    for i in range(n_defect):
        b = float(rng.uniform(0.3, 0.7)) if color_confounder else 0.5
        X[n_clean + i] = _defect(b)
        y[n_clean + i] = 1
    # Shuffle once at the end so train/test splits are not class-blocked.
    perm = rng.permutation(len(y))
    return Dataset("defect_patches", X[perm], y[perm], ["clean", "defect"])


# -----------------------------------------------------------------------
# 4) Optional Fashion-MNIST loader
# -----------------------------------------------------------------------

def fashion_mnist_pair(class_a: int = 0, class_b: int = 9,
                       n_per_class: int = 50, side: int = 8,
                       seed: int = SEED) -> Optional[Dataset]:
    """Return a balanced 2-class subset of Fashion-MNIST if torchvision can
    load it locally. Returns None otherwise (no internet downloads).
    """
    try:
        from torchvision import datasets, transforms  # type: ignore
    except Exception:
        return None
    try:
        root = "/tmp/fashion_mnist_qegk"
        ds = datasets.FashionMNIST(root=root, train=True, download=False,
                                    transform=transforms.ToTensor())
    except Exception:
        return None
    rng = np.random.default_rng(seed)
    X_list = []
    y_list = []
    for cls, new_label in ((class_a, 0), (class_b, 1)):
        idx = [i for i in range(len(ds)) if int(ds.targets[i]) == cls]
        rng.shuffle(idx)
        for i in idx[:n_per_class]:
            img, _ = ds[i]
            arr = img[0].numpy().astype(np.float32)
            if arr.shape[0] != side:
                from PIL import Image
                arr = np.asarray(Image.fromarray((arr * 255).astype(np.uint8))
                                  .resize((side, side))) / 255.0
            X_list.append(arr.astype(np.float32))
            y_list.append(new_label)
    if not X_list:
        return None
    X = np.stack(X_list, axis=0); y = np.asarray(y_list, dtype=int)
    return Dataset(f"fashion_{class_a}_{class_b}", X, y,
                    [f"cls_{class_a}", f"cls_{class_b}"])


# -----------------------------------------------------------------------
# 5) Optional MedMNIST loader
# -----------------------------------------------------------------------

def medmnist_pair(name: str = "pneumoniamnist", n_per_class: int = 50,
                   side: int = 8, seed: int = SEED) -> Optional[Dataset]:
    """Return a small MedMNIST-style dataset if the medmnist package can be
    imported and the npz file is already cached. No network access.
    """
    try:
        import medmnist  # type: ignore
        from medmnist import INFO
    except Exception:
        return None
    if name not in INFO:
        return None
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    try:
        ds = DataClass(split="train", download=False, size=28)
    except Exception:
        return None
    rng = np.random.default_rng(seed)
    X = ds.imgs.astype(np.float32) / 255.0
    if X.ndim == 4:
        X = X.mean(axis=-1)
    y = np.asarray(ds.labels).flatten()
    classes = sorted(set(y.tolist()))[:2]
    if len(classes) < 2:
        return None
    X_list, y_list = [], []
    for new_label, cls in enumerate(classes):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for i in idx[:n_per_class]:
            arr = X[i]
            if arr.shape[0] != side:
                from PIL import Image
                arr = np.asarray(Image.fromarray((arr * 255).astype(np.uint8))
                                  .resize((side, side))) / 255.0
            X_list.append(arr.astype(np.float32))
            y_list.append(new_label)
    if not X_list:
        return None
    return Dataset(f"medmnist_{name}", np.stack(X_list), np.asarray(y_list, dtype=int),
                    [f"med_{c}" for c in classes])


# -----------------------------------------------------------------------
# Quick reference
# -----------------------------------------------------------------------

SYNTHETIC_REGISTRY = {
    "parity_patch": parity_patch,
    "d4_shapes": d4_shapes,
    "defect_patches": defect_patches,
}
