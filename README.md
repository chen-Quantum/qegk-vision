# QEGK: Entangled Group-Equivariant Quantum Kernels

A simulator-based research prototype for **symmetry-aware image kernels**. Small
image patches are encoded into quantum-inspired feature states, compared with a
fidelity-style kernel, and that kernel is averaged over a finite symmetry group
(rotations / reflections) so the similarity score is invariant to those
transformations. The group-averaged kernel is benchmarked against matched
classical kernels on low-data, symmetry-heavy tasks.

<p align="center">
  <img src="outputs/public_readme/hero.png" width="760">
</p>

> Simulator-based research prototype. No quantum-advantage claim.

## Core idea

- Each input is mapped to a normalized feature state via a parameterized encoding.
- Two inputs are compared by a **fidelity kernel** — the squared overlap of their states.
- To build in symmetry, the kernel is **averaged over a finite group** $G$ of input
  transformations (for example the eight rotations and reflections of a square).
- The resulting kernel feeds a standard kernel SVM (classification) or a
  nearest-neighbour anomaly score, and is compared against classical kernels.

## Mathematical sketch

Encode an input $x$ into a unit feature state and compare two inputs by their
squared overlap (a fidelity kernel):

```math
|\phi(x)\rangle = U(x)\,|0\rangle,
\qquad
k(x,y) = \bigl|\langle \phi(x)\,|\,\phi(y)\rangle\bigr|^{2}.
```

Make the kernel invariant to a finite symmetry group $G$ by averaging over the
group orbit of both inputs:

```math
k_{G}(x,y) \;=\; \frac{1}{|G|^{2}} \sum_{g,h \in G}
\bigl|\langle \phi(g\!\cdot\! x)\,|\,\phi(h\!\cdot\! y)\rangle\bigr|^{2}.
```

By construction $k_G$ is two-sided $G$-invariant, i.e. $k_G(g\!\cdot\! x,\,y)=k_G(x,y)$
for every $g \in G$.

## Selected visuals and results

<p align="center">
  <img src="outputs/public_readme/selected_result.png" width="520">
</p>

*Exploratory matched-baseline comparison on small simulator tasks. Each point is
one task; the dashed line is parity. Results are mixed.*

<p align="center">
  <img src="outputs/public_readme/gallery.png" width="760">
</p>

*Selected qualitative example inputs and their transformations.*

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_all_experiments.py --quick
pytest -q
```

## Results status

- Exploratory, simulator-based, small scale.
- Results are **mixed**: the group-averaged quantum kernel matches or exceeds
  matched classical kernels on some symmetry-heavy, low-data tasks, and not on others.
- No quantum advantage is claimed. No state-of-the-art claim is made.

## Limitations

- Statevector simulator only, few qubits — classically tractable by design.
- Synthetic / small datasets chosen to probe symmetry and low-data regimes, not
  for external validity.
- No hardware noise model; finite-shot estimation would degrade the kernel.
- Classical kernel methods are the only baselines; deep models are out of scope.
