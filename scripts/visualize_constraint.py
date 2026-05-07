"""
Visualize CFO constraint satisfaction before and after training.

Usage:
    python scripts/visualize_constraint.py

Saves figure to docs/figures/constraint_visualization.pdf.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Triangle vertices defining the feasible region (must match cfo/constraint.py)
TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])


def _is_safe_2d(point):
    """Point-in-triangle test (barycentric sign method). Matches cfo/constraint.py."""
    x = np.asarray(point, dtype=float)
    v0, v1, v2 = TRIANGLE

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(x, v0, v1)
    d2 = sign(x, v1, v2)
    d3 = sign(x, v2, v0)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _draw_triangle(ax):
    """Draw the feasible-region triangle boundary in red."""
    triangle = Polygon(TRIANGLE, closed=True, fill=False,
                       edgecolor="red", linewidth=2, linestyle="--", zorder=3)
    ax.add_patch(triangle)


def plot_constraint_visualization(samples_before, samples_after,
                                  output_path="docs/figures/constraint_visualization.pdf"):
    """
    Plot two side-by-side scatter plots showing sample distribution
    before and after CFO training, with the feasible region drawn in red.

    Args:
        samples_before: list/array of (x, y) 2D points — samples before training
        samples_after:  list/array of (x, y) 2D points — samples after training
        output_path:    path to save the PDF figure
    """
    samples_before = np.asarray(samples_before, dtype=float)
    samples_after  = np.asarray(samples_after,  dtype=float)

    safe_before = np.array([_is_safe_2d(p) for p in samples_before])
    safe_after  = np.array([_is_safe_2d(p) for p in samples_after])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    fig.suptitle("CFO: Sample Distribution Before and After Training", fontsize=13)

    titles   = ["Before Training", "After Training"]
    datasets = [(samples_before, safe_before), (samples_after, safe_after)]

    for ax, title, (pts, safe) in zip(axes, titles, datasets):
        # Unsafe points (red), safe points (blue)
        if (~safe).any():
            ax.scatter(pts[~safe, 0], pts[~safe, 1],
                       c="red", alpha=0.6, s=25, label="Unsafe", zorder=2)
        if safe.any():
            ax.scatter(pts[safe, 0], pts[safe, 1],
                       c="steelblue", alpha=0.6, s=25, label="Safe", zorder=2)

        _draw_triangle(ax)

        safe_frac = safe.mean() * 100
        ax.set_title(f"{title}  ({safe_frac:.0f}% safe)", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Red dashed line label
        red_patch = mpatches.Patch(facecolor="none", edgecolor="red",
                                   linestyle="--", linewidth=2,
                                   label="Feasible region")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [red_patch],
                  labels=labels + ["Feasible region"],
                  loc="upper right", fontsize=9)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Before training: samples spread uniformly over [-0.2, 1.2]^2
    before = rng.uniform(-0.2, 1.2, size=(200, 2))

    # After training: samples pushed toward the triangle centre (0.5, 0.33)
    centre = np.array([0.5, 0.33])
    after  = rng.normal(loc=centre, scale=0.25, size=(200, 2))

    plot_constraint_visualization(before, after)
