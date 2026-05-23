"""Kernel-based classifiers and anomaly scorers.

We use scikit-learn SVMs with precomputed kernels for the classification
pipelines, and a simple kernel nearest-neighbour anomaly score for the
anomaly experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC

from . import SEED


@dataclass
class ClassificationResult:
    accuracy: float
    f1: float
    n_train: int
    n_test: int


def svm_with_kernel(K_train: np.ndarray, y_train: np.ndarray,
                     K_test: np.ndarray, y_test: np.ndarray,
                     C: float = 1.0, seed: int = SEED) -> ClassificationResult:
    """SVM with a precomputed kernel.  K_test is (n_test, n_train)."""
    clf = SVC(kernel="precomputed", C=C, random_state=int(seed))
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    return ClassificationResult(
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        n_train=int(y_train.size),
        n_test=int(y_test.size),
    )


# -----------------------------------------------------------------------
# Anomaly scoring via kernel distance to training clean samples
# -----------------------------------------------------------------------

@dataclass
class AnomalyResult:
    auc: float
    f1: float
    threshold: float
    n_train_clean: int
    n_test: int


def kernel_anomaly_scores(K_test_train: np.ndarray) -> np.ndarray:
    """Convert a (test, train) kernel matrix into an anomaly score.
    Higher value == more anomalous.  We use 1 - max(kernel) per row, which
    is the kernelised analogue of nearest-neighbour distance."""
    K = np.asarray(K_test_train, dtype=np.float64)
    # Normalise rows by their own self-similarity if available; otherwise
    # just take max similarity to training clean samples.
    nearest = K.max(axis=1)
    return 1.0 - np.clip(nearest, 0.0, 1.0)


def kernel_anomaly(K_train_clean: np.ndarray,
                    K_test_train: np.ndarray,
                    y_test: np.ndarray) -> AnomalyResult:
    """One-class anomaly using nearest-neighbour kernel similarity.

    Parameters
    ----------
    K_train_clean : NxN kernel between the clean training samples (unused
                    in scoring but kept for future extensions).
    K_test_train  : (n_test, n_train_clean) kernel from test to clean train.
    y_test        : 0 (clean) / 1 (anomaly) labels.
    """
    scores = kernel_anomaly_scores(K_test_train)
    # ROC-AUC is robust to threshold; F1 picks the score median as threshold.
    try:
        auc = float(roc_auc_score(y_test, scores))
    except ValueError:
        auc = 0.5
    thr = float(np.median(scores))
    pred = (scores >= thr).astype(int)
    return AnomalyResult(
        auc=auc,
        f1=float(f1_score(y_test, pred, average="binary", zero_division=0)),
        threshold=thr,
        n_train_clean=int(K_train_clean.shape[0]),
        n_test=int(y_test.size),
    )
