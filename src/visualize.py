"""Figures, GIFs, and release-quality media for QEGK."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from .groups import d4_actions

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR = {
    "deep_blue": "#1F4E79",
    "blue": "#2E75B6",
    "orange": "#C65911",
    "green": "#548235",
    "purple": "#8064A2",
    "grey": "#595959",
    "panel": "#0E1B2C",
    "panel_light": "#1F2D44",
    "ink": "#F5F7FA",
    "amber": "#F2BD46",
    "red": "#C00000",
}


# -----------------------------------------------------------------------
# Group orbit grid
# -----------------------------------------------------------------------

def d4_orbit_grid(img: np.ndarray, out_path: Path | str,
                   title: str = "D4 orbit of a single image") -> None:
    actions = d4_actions()
    labels = list(actions.keys())
    fig, axes = plt.subplots(2, 4, figsize=(8, 4.4))
    for k, (lab, act) in enumerate(actions.items()):
        ax = axes.flat[k]
        ax.imshow(act(img), cmap="gray")
        ax.set_axis_off()
        ax.set_title(f"${lab}$", fontsize=12, color=COLOR["deep_blue"])
    fig.suptitle(title, color=COLOR["deep_blue"], y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# Kernel matrices
# -----------------------------------------------------------------------

def kernel_matrix_image(K: np.ndarray, out_path: Path | str, title: str,
                         cbar_label: str = "K(x, y)") -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(K, cmap="viridis", origin="lower")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    ax.set_title(title, color=COLOR["deep_blue"])
    ax.set_xlabel("sample j"); ax.set_ylabel("sample i")
    fig.tight_layout()
    fig.savefig(out_path); plt.close(fig)


def entangled_vs_separable_kernel(K_sep: np.ndarray, K_ent: np.ndarray,
                                   out_path: Path | str,
                                   title: str = "Separable vs entangled quantum kernel"
                                   ) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, K, ttl in zip(axes, [K_sep, K_ent], ["separable Ry", "entangled CX"]):
        im = ax.imshow(K, cmap="viridis", origin="lower")
        ax.set_title(ttl, color=COLOR["deep_blue"])
        ax.set_xlabel("sample j"); ax.set_ylabel("sample i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, y=1.02, fontsize=13, color=COLOR["deep_blue"])
    fig.tight_layout()
    fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# Anomaly heatmap
# -----------------------------------------------------------------------

def anomaly_heatmap(scores: np.ndarray, labels: np.ndarray,
                     images: np.ndarray, out_path: Path | str,
                     title: str = "Anomaly scores over test patches") -> None:
    order = np.argsort(scores)[::-1]
    n = min(20, len(scores))
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 0.8, 3.4))
    for k in range(n):
        ax = axes.flat[k]
        ax.imshow(images[order[k]], cmap="gray")
        ax.set_axis_off()
        c = COLOR["red"] if labels[order[k]] == 1 else COLOR["green"]
        ax.set_title(f"{scores[order[k]]:.2f}", color=c, fontsize=9)
    fig.suptitle(title + "\n(red = defect, green = clean; sorted by score, descending)",
                  fontsize=11, color=COLOR["deep_blue"])
    fig.tight_layout()
    fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# Advantage summary
# -----------------------------------------------------------------------

def advantage_summary(rows: List[Dict[str, Any]], out_path: Path | str) -> None:
    """Plot quantum advantage signal per experiment."""
    sig_rows = [r for r in rows if r.get("method") == "ADVANTAGE_SIGNAL"]
    if not sig_rows:
        # nothing to plot
        fig = plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No advantage rows yet (no classical baselines configured).",
                 ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path); plt.close(fig)
        return
    names = [r["experiment"] for r in sig_rows]
    means = [r["mean"] for r in sig_rows]
    los = [r.get("lo", r["mean"]) for r in sig_rows]
    his = [r.get("hi", r["mean"]) for r in sig_rows]
    colors = [COLOR["green"] if m > 0 else COLOR["red"] for m in means]
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.32 * len(names))))
    y = np.arange(len(names))
    ax.barh(y, means, color=colors, alpha=0.85,
             xerr=np.vstack([np.asarray(means) - los, his - np.asarray(means)]),
             error_kw=dict(ecolor=COLOR["grey"], lw=1.0, capsize=2.5))
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("advantage = best_quantum mean - best_classical mean")
    ax.set_title("Quantum advantage signal per experiment", color=COLOR["deep_blue"])
    fig.tight_layout()
    fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# Entanglement diagnostics
# -----------------------------------------------------------------------

def entanglement_diagnostics_plot(diagnostics: Dict[str, Dict[str, float]],
                                    out_path: Path | str) -> None:
    """Bar chart of mean linear entropy per feature map."""
    names = list(diagnostics.keys())
    means = [diagnostics[n]["linear_entropy_mean"] for n in names]
    stds = [diagnostics[n]["linear_entropy_std"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color=COLOR["deep_blue"], alpha=0.9, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("linear entropy 1 - Tr(rho_A^2)")
    ax.set_title("Entanglement (linear entropy) per feature map",
                  color=COLOR["deep_blue"])
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# Experiment grid (one tile per experiment)
# -----------------------------------------------------------------------

def experiment_grid(rows: List[Dict[str, Any]], out_path: Path | str,
                     title: str = "Experiment grid: 20 configurations") -> None:
    by_exp: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.get("method") == "ADVANTAGE_SIGNAL":
            by_exp[r["experiment"]] = r
    items = sorted(by_exp.items(), key=lambda kv: kv[0])
    n = len(items)
    if n == 0:
        fig = plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No experiments available.", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path); plt.close(fig)
        return
    cols = 4
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.2, rows_n * 1.6))
    if rows_n == 1:
        axes = np.array([axes])
    for k, (name, r) in enumerate(items):
        ax = axes.flat[k]
        m = r.get("mean", 0.0)
        color = COLOR["green"] if m > 0 else COLOR["red"]
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0.02, 0.06), 0.96, 0.88,
                                     boxstyle="round,pad=0.02",
                                     facecolor=color, alpha=0.12, edgecolor=color))
        ax.text(0.5, 0.78, name, ha="center", va="center", fontsize=9,
                 color=COLOR["deep_blue"], weight="bold",
                 wrap=True)
        ax.text(0.5, 0.55, f"advantage = {m:+.3f}", ha="center", va="center",
                 fontsize=11, color=color, weight="bold")
        ax.text(0.5, 0.30, f"q: {r.get('best_quantum', '')}", ha="center",
                 va="center", fontsize=8, color=COLOR["grey"])
        ax.text(0.5, 0.16, f"c: {r.get('best_classical', '')}", ha="center",
                 va="center", fontsize=8, color=COLOR["grey"])
    for k in range(n, rows_n * cols):
        axes.flat[k].set_axis_off()
    fig.suptitle(title, color=COLOR["deep_blue"], y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path); plt.close(fig)


# -----------------------------------------------------------------------
# Pipeline overview diagram
# -----------------------------------------------------------------------

def pipeline_overview(out_path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(16, 4.4), dpi=170)
    ax.set_xlim(0, 16); ax.set_ylim(0, 4.4)
    ax.set_axis_off()
    stages = [
        ("Image $x$",        "$H\\times W$ patch",                       COLOR["deep_blue"]),
        ("Group orbit",      "$Gx = \\{g x : g\\in G\\}$ (D4: $|G|=8$)", COLOR["blue"]),
        ("Features",          "low-res / Sobel / quadrants",              COLOR["green"]),
        ("Quantum state",    "$|\\psi(\\theta)\\rangle = U(\\theta)|0\\rangle^{n_q}$", COLOR["purple"]),
        ("Fidelity kernel",  "$K(x, y) = |\\langle\\psi(x)|\\psi(y)\\rangle|^2$", COLOR["orange"]),
        ("Group avg",         "$K_G = \\frac{1}{|G|^2}\\sum_{g,h}K(g x, h y)$", COLOR["orange"]),
        ("SVM / anomaly",     "kernel-SVM or NN distance",                COLOR["deep_blue"]),
    ]
    n = len(stages)
    x0 = 0.35; gap = 0.30
    box_w = (16 - 2 * x0 - gap * (n - 1)) / n
    y_center = 2.30; box_h = 2.05
    centers = []
    for i, (title, body, c) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        ax.add_patch(FancyBboxPatch((x, y_center - box_h / 2), box_w, box_h,
                                     boxstyle="round,pad=0.04,rounding_size=0.10",
                                     facecolor=c, edgecolor="none", alpha=0.94))
        ax.text(x + box_w / 2, y_center + 0.55, title, ha="center", va="center",
                 fontsize=11.5, weight="bold", color="white", linespacing=1.0)
        ax.text(x + box_w / 2, y_center - 0.35, body, ha="center", va="center",
                 fontsize=8.6, color="white", linespacing=1.2)
        centers.append(x + box_w / 2)
    for cx0, cx1 in zip(centers[:-1], centers[1:]):
        ax.annotate("", xy=(cx1 - box_w / 2 - 0.04, y_center),
                     xytext=(cx0 + box_w / 2 + 0.04, y_center),
                     arrowprops=dict(arrowstyle="-|>,head_length=0.28,head_width=0.16",
                                      color=COLOR["grey"], lw=1.8))
    ax.text(8.0, 4.05, "QEGK pipeline overview", ha="center",
            fontsize=17, weight="bold", color=COLOR["deep_blue"])
    ax.text(8.0, 0.20,
            "Simulator-based quantum kernel  *  group averaging over a finite symmetry group  *  no true quantum-advantage claim",
            ha="center", fontsize=9.5, color=COLOR["grey"], style="italic")
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -----------------------------------------------------------------------
# README hero
# -----------------------------------------------------------------------

def readme_hero(images: np.ndarray, K_quantum: np.ndarray,
                 K_classical: np.ndarray, advantage_rows: List[Dict[str, Any]],
                 out_path: Path | str) -> None:
    fig = plt.figure(figsize=(13.5, 6.5), dpi=180, facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 1.25],
                           height_ratios=[1.0, 0.55],
                           hspace=0.55, wspace=0.30,
                           left=0.05, right=0.99, top=0.80, bottom=0.08)
    fig.text(0.05, 0.95, "QEGK: Entangled Group-Equivariant Quantum Kernels",
              fontsize=19, weight="bold", color=COLOR["deep_blue"])
    fig.text(0.05, 0.91, "for visual anomaly and symmetry detection",
              fontsize=12.5, color=COLOR["grey"])
    fig.text(0.05, 0.876,
              "simulator-based prototype  *  quantum-advantage signal, not true quantum advantage",
              fontsize=9.5, color=COLOR["grey"], style="italic")

    # Panel 1: D4 orbit
    ax1 = fig.add_subplot(gs[0, 0])
    actions = d4_actions()
    side = images.shape[1]
    pad = 2
    grid_w = 4; grid_h = 2
    canvas = np.zeros((grid_h * side + (grid_h + 1) * pad,
                        grid_w * side + (grid_w + 1) * pad), dtype=np.float32)
    for k, act in enumerate(actions.values()):
        r, c = divmod(k, grid_w)
        canvas[pad + r * (side + pad):pad + r * (side + pad) + side,
                pad + c * (side + pad):pad + c * (side + pad) + side] = act(images[0])
    ax1.imshow(canvas, cmap="gray")
    ax1.set_axis_off()
    ax1.set_title("D4 orbit of one input patch", color=COLOR["deep_blue"], pad=8)

    # Panel 2: quantum vs classical kernel
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(K_quantum, cmap="viridis", origin="lower")
    ax2.set_axis_off()
    ax2.set_title("Entangled quantum kernel", color=COLOR["deep_blue"], pad=8)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Panel 3: advantage signal preview
    ax3 = fig.add_subplot(gs[0, 2])
    sig = [r for r in advantage_rows if r.get("method") == "ADVANTAGE_SIGNAL"]
    sig = sig[:8]
    if sig:
        names = [r["experiment"].split("_", 1)[0] + "..." for r in sig]
        means = [r["mean"] for r in sig]
        colors = [COLOR["green"] if m > 0 else COLOR["red"] for m in means]
        y = np.arange(len(sig))
        ax3.barh(y, means, color=colors, alpha=0.85)
        ax3.set_yticks(y); ax3.set_yticklabels(names, fontsize=8.5)
        ax3.axvline(0, color="black", lw=0.7)
        ax3.set_xlabel("advantage", fontsize=9)
        ax3.set_title("Quantum advantage signal (subset)",
                       color=COLOR["deep_blue"], pad=8)
    else:
        ax3.set_axis_off()
        ax3.text(0.5, 0.5, "advantage signal pending", ha="center", va="center")

    # Bottom row: classical kernel as a counter-example
    ax4 = fig.add_subplot(gs[1, :])
    n_show = min(20, images.shape[0])
    strip = np.concatenate([images[i] for i in
                              np.linspace(0, n_show - 1, n_show).astype(int)],
                             axis=1)
    ax4.imshow(strip, cmap="gray")
    ax4.set_axis_off()
    ax4.set_title("Input samples (gray patches)", color=COLOR["deep_blue"], pad=8)

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -----------------------------------------------------------------------
# Cinematic GIF / MP4
# -----------------------------------------------------------------------

def cinematic_gif(images: np.ndarray, K_quantum: np.ndarray, K_classical: np.ndarray,
                   advantage_rows: List[Dict[str, Any]],
                   out_dir: Path, mp4_path: Path | None = None,
                   gif_path: Path | None = None,
                   fps_mp4: int = 24, fps_gif: int = 6) -> None:
    """Build a short cinematic summary in mp4 + gif form."""
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"imageio missing - skipping cinematic: {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = mp4_path or (out_dir / "cinematic_summary.mp4")
    gif_path = gif_path or (out_dir / "cinematic_summary.gif")

    def _slide() -> tuple[plt.Figure, plt.Axes]:
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=COLOR["panel"])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(COLOR["panel"])
        ax.set_xlim(0, 1280); ax.set_ylim(0, 720)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return fig, ax

    def _to_arr(fig: plt.Figure) -> np.ndarray:
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        return rgba[..., :3]

    mp4 = imageio.get_writer(mp4_path, fps=fps_mp4, codec="libx264", quality=9,
                              macro_block_size=1)
    captured: List[np.ndarray] = []
    counter = 0
    every = 4  # for GIF: every 4th frame -> ~6 fps from 24 fps

    def emit(frame: np.ndarray) -> None:
        nonlocal counter
        mp4.append_data(frame)
        if counter % every == 0:
            captured.append(frame)
        counter += 1

    try:
        # 1) Title, 3s
        for _ in range(fps_mp4 * 3):
            fig, ax = _slide()
            ax.add_patch(FancyBboxPatch((40, 280), 18, 320,
                                         boxstyle="round,pad=0",
                                         facecolor=COLOR["amber"], edgecolor="none"))
            ax.text(110, 570, "QEGK", color=COLOR["ink"], fontsize=70, weight="bold")
            ax.text(110, 510, "Entangled Group-Equivariant Quantum Kernels",
                    color=COLOR["ink"], fontsize=26)
            ax.text(110, 470, "for visual anomaly and symmetry detection",
                    color=COLOR["ink"], fontsize=26)
            ax.text(110, 390, "Simulator prototype  *  quantum advantage signal",
                    color=COLOR["amber"], fontsize=20)
            ax.text(110, 320, "20 experiments  *  D4 symmetry group  *  Qiskit",
                    color=COLOR["grey"], fontsize=18)
            emit(_to_arr(fig)); plt.close(fig)

        # 2) Orbit grid, 4s
        actions = list(d4_actions().items())
        side = images.shape[1]
        for k in range(fps_mp4 * 4):
            fig, ax = _slide()
            ax.add_patch(FancyBboxPatch((40, 620), 18, 80,
                                         boxstyle="round,pad=0",
                                         facecolor=COLOR["amber"], edgecolor="none"))
            ax.text(80, 660, "D4 orbit of a single image",
                    color=COLOR["ink"], fontsize=28, weight="bold", va="center")
            cols = 4; rows = 2
            tile_w = 220; tile_h = 220
            x0 = 130; y0 = 80
            highlight = k % len(actions)
            for j, (lab, act) in enumerate(actions):
                r, c = divmod(j, cols)
                xx = x0 + c * (tile_w + 18)
                yy = y0 + (rows - 1 - r) * (tile_h + 36)
                ax.imshow(act(images[0]), cmap="gray",
                          extent=(xx, xx + tile_w, yy, yy + tile_h))
                ec = COLOR["amber"] if j == highlight else COLOR["grey"]
                ax.add_patch(FancyBboxPatch((xx - 4, yy - 4), tile_w + 8, tile_h + 8,
                                              boxstyle="round,pad=0",
                                              fill=False, edgecolor=ec, lw=2))
                ax.text(xx + tile_w / 2, yy - 18, f"${lab}$", ha="center",
                          color=COLOR["ink"], fontsize=18)
            emit(_to_arr(fig)); plt.close(fig)

        # 3) Quantum kernel reveal, 4s
        for k in range(fps_mp4 * 4):
            t = k / (fps_mp4 * 4 - 1)
            fig, ax = _slide()
            ax.add_patch(FancyBboxPatch((40, 620), 18, 80,
                                         boxstyle="round,pad=0",
                                         facecolor=COLOR["amber"], edgecolor="none"))
            ax.text(80, 660, "Entangled quantum fidelity kernel",
                    color=COLOR["ink"], fontsize=26, weight="bold", va="center")
            cut = int((t ** 2) * K_quantum.shape[0])
            partial = K_quantum.copy()
            if cut < K_quantum.shape[0]:
                partial = np.where(np.indices(K_quantum.shape)[0] <= cut, partial, 0.0)
            ax.imshow(partial, cmap="viridis", origin="lower",
                       extent=(380, 900, 70, 590))
            ax.text(420, 70 - 36,
                    "$K(x,y) = |\\langle\\psi(x)|\\psi(y)\\rangle|^2$",
                    color=COLOR["ink"], fontsize=18)
            ax.text(940, 480,
                    "Bright blocks at\nthe class boundary\nseparate inputs.",
                    color=COLOR["ink"], fontsize=18)
            emit(_to_arr(fig)); plt.close(fig)

        # 4) Advantage chart, 5s
        sig = [r for r in advantage_rows if r.get("method") == "ADVANTAGE_SIGNAL"]
        for k in range(fps_mp4 * 5):
            fig, ax = _slide()
            ax.add_patch(FancyBboxPatch((40, 620), 18, 80,
                                         boxstyle="round,pad=0",
                                         facecolor=COLOR["amber"], edgecolor="none"))
            ax.text(80, 660, "Quantum advantage signal per experiment",
                    color=COLOR["ink"], fontsize=24, weight="bold", va="center")
            n = min(12, len(sig))
            x0 = 200; y0 = 100; bar_w = 600; row_h = 32
            ax.text(x0, y0 + (n + 0.5) * row_h, "experiment",
                     color=COLOR["grey"], fontsize=13)
            ax.text(x0 + bar_w + 30, y0 + (n + 0.5) * row_h, "advantage",
                     color=COLOR["grey"], fontsize=13)
            ax.plot([x0 + bar_w / 2, x0 + bar_w / 2],
                     [y0 - row_h * 0.5, y0 + n * row_h], color=COLOR["grey"], lw=0.6)
            for i in range(n):
                r = sig[i]
                m = r["mean"]
                color = COLOR["green"] if m > 0 else COLOR["red"]
                bx = x0 + bar_w / 2
                length = (m / 0.4) * (bar_w / 2)   # 0.4 acc full-range
                length = max(min(length, bar_w / 2 - 6), -(bar_w / 2 - 6))
                yy = y0 + (n - 1 - i) * row_h
                ax.add_patch(FancyBboxPatch((min(bx, bx + length), yy),
                                             abs(length), row_h * 0.75,
                                             boxstyle="round,pad=0",
                                             facecolor=color, edgecolor="none",
                                             alpha=0.9))
                ax.text(x0 - 10, yy + row_h * 0.4, r["experiment"], ha="right",
                          color=COLOR["ink"], fontsize=10)
                ax.text(x0 + bar_w + 30, yy + row_h * 0.4, f"{m:+.3f}",
                          ha="left", color=color, fontsize=11, weight="bold")
            emit(_to_arr(fig)); plt.close(fig)

        # 5) Closing, 3s
        for _ in range(fps_mp4 * 3):
            fig, ax = _slide()
            ax.add_patch(FancyBboxPatch((40, 280), 18, 320,
                                         boxstyle="round,pad=0",
                                         facecolor=COLOR["amber"], edgecolor="none"))
            ax.text(110, 540, "Simulator prototype.",
                    color=COLOR["ink"], fontsize=36, weight="bold")
            ax.text(110, 490, "No true quantum advantage claim.",
                    color=COLOR["amber"], fontsize=24)
            ax.text(110, 390, "Quantum advantage signal: difference vs matched classical baseline.",
                    color=COLOR["ink"], fontsize=16)
            ax.text(110, 330, "Reproduce:", color=COLOR["grey"], fontsize=16)
            ax.text(110, 295, "    python scripts/run_all_experiments.py --quick",
                    color=COLOR["ink"], fontsize=16, family="monospace")
            ax.text(110, 265, "    pytest -q",
                    color=COLOR["ink"], fontsize=16, family="monospace")
            emit(_to_arr(fig)); plt.close(fig)
    finally:
        mp4.close()

    # Compress to GIF: downsample and re-encode.
    try:
        target_w = 640
        out_frames: List[np.ndarray] = []
        for f in captured:
            img = Image.fromarray(f)
            ratio = target_w / img.width
            img = img.resize((target_w, int(img.height * ratio)), Image.BILINEAR)
            out_frames.append(np.array(img))
        imageio.mimsave(gif_path, out_frames, fps=fps_gif, loop=0)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"GIF write failed: {exc}")
