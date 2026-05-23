"""QEGK: Entangled Group-Equivariant Quantum Kernels for Visual Anomaly
and Symmetry Detection.

Simulator-based research prototype. No claim of true quantum advantage; we
instead measure a "quantum advantage signal" - cases where the entangled
quantum kernel beats matched classical baselines under the same low-data,
low-feature, or symmetry-heavy constraints.
"""

SEED = 0xBEEF1234  # 0xbeef1234 == 3203386420

__all__ = [
    "groups",
    "datasets",
    "features",
    "quantum_states",
    "feature_maps",
    "entanglement",
    "kernels",
    "models",
    "experiments",
    "visualize",
    "stats",
]
