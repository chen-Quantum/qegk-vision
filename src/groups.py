"""Finite-group actions on small images, with full support for the dihedral
group D4 of order 8 (square symmetries).

D4 elements (label / action on a 2-D array):

    e       identity
    r       90 degree rotation       (np.rot90, k=1)
    r2      180 degree rotation      (np.rot90, k=2)
    r3      270 degree rotation      (np.rot90, k=3)
    s       horizontal flip          (np.fliplr)
    sr      flip then rotate 90      (== diagonal flip)
    sr2     vertical flip            (np.flipud)
    sr3     anti-diagonal flip

The action is on numpy arrays of shape (H, W) or (H, W, C). All actions
return a new array (no in-place mutation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

GroupAction = Callable[[np.ndarray], np.ndarray]


def _rot(k: int) -> GroupAction:
    def _f(x: np.ndarray) -> np.ndarray:
        return np.rot90(x, k=k).copy()
    return _f


def _flip(axis: int) -> GroupAction:
    def _f(x: np.ndarray) -> np.ndarray:
        return np.flip(x, axis=axis).copy()
    return _f


def _flip_diag() -> GroupAction:
    def _f(x: np.ndarray) -> np.ndarray:
        # Diagonal flip = transpose on the spatial axes.
        if x.ndim == 2:
            return x.T.copy()
        return np.swapaxes(x, 0, 1).copy()
    return _f


def _flip_antidiag() -> GroupAction:
    def _f(x: np.ndarray) -> np.ndarray:
        # Anti-diagonal flip = transpose then rotate 180.
        if x.ndim == 2:
            return np.rot90(x.T, k=2).copy()
        return np.rot90(np.swapaxes(x, 0, 1), k=2).copy()
    return _f


_D4_ACTIONS: Dict[str, GroupAction] = {
    "e": _rot(0),
    "r": _rot(1),
    "r2": _rot(2),
    "r3": _rot(3),
    "s": _flip(axis=1),       # horizontal flip (left/right)
    "sr": _flip_diag(),       # diagonal
    "sr2": _flip(axis=0),     # vertical flip (up/down)
    "sr3": _flip_antidiag(),  # anti-diagonal
}

D4_LABELS: List[str] = list(_D4_ACTIONS.keys())


def d4_actions() -> Dict[str, GroupAction]:
    """Return a fresh dict of the eight D4 actions, keyed by label."""
    return dict(_D4_ACTIONS)


def d4_orbit(x: np.ndarray) -> List[np.ndarray]:
    """Return the eight images in the D4 orbit of x, in the canonical order
    [e, r, r2, r3, s, sr, sr2, sr3]."""
    return [act(x) for act in _D4_ACTIONS.values()]


def identity_group() -> Dict[str, GroupAction]:
    """A single-element group {e}, used for ablations against D4."""
    return {"e": _rot(0)}


# -----------------------------------------------------------------------
# Cayley table for D4 (label-level multiplication).
# Useful for tests and for the course-notes section on group structure.
# -----------------------------------------------------------------------

def _compose(g_label: str, h_label: str) -> str:
    """Return the D4 label of g * h, computed by acting on a test pattern."""
    base = np.arange(16, dtype=int).reshape(4, 4)
    g_then_h = _D4_ACTIONS[h_label](_D4_ACTIONS[g_label](base))
    for k, act in _D4_ACTIONS.items():
        if np.array_equal(act(base), g_then_h):
            return k
    raise RuntimeError(f"Composition {g_label} * {h_label} not in D4")


def d4_cayley_table() -> Dict[str, Dict[str, str]]:
    """Full Cayley table for D4 as a nested dict: table[g][h] = label of g*h."""
    table: Dict[str, Dict[str, str]] = {}
    for g in D4_LABELS:
        table[g] = {h: _compose(g, h) for h in D4_LABELS}
    return table


@dataclass
class FiniteGroup:
    """A finite group of image transformations."""
    name: str
    actions: Dict[str, GroupAction]

    @property
    def size(self) -> int:
        return len(self.actions)

    def labels(self) -> List[str]:
        return list(self.actions.keys())

    def orbit(self, x: np.ndarray) -> List[np.ndarray]:
        return [act(x) for act in self.actions.values()]


def make_group(name: str) -> FiniteGroup:
    name = name.lower()
    if name in ("d4", "dihedral4"):
        return FiniteGroup("D4", d4_actions())
    if name in ("identity", "trivial", "e", "{e}"):
        return FiniteGroup("Identity", identity_group())
    if name in ("c4", "cyclic4"):
        return FiniteGroup("C4", {k: _D4_ACTIONS[k] for k in ("e", "r", "r2", "r3")})
    raise KeyError(f"Unknown group {name!r}. Use 'D4', 'C4', or 'Identity'.")
