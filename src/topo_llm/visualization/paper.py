"""
Publication-quality figure generation.

Sets matplotlib RC params for NeurIPS/ICML format and provides
composite figure generators for key paper figures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def _require_matplotlib():
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        return plt, matplotlib
    except ImportError:
        raise ImportError("matplotlib required. Install with: pip install topo-llm[viz]") from None


def set_paper_style() -> None:
    """Configure matplotlib for publication-quality figures.

    Sets RC params suitable for NeurIPS/ICML format:
    - 3.25" single-column, 6.75" double-column width
    - LaTeX-like fonts
    - Appropriate font sizes
    """
    plt, matplotlib = _require_matplotlib()

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.figsize": (3.25, 2.5),
        }
    )

    # Try LaTeX fonts if available
    try:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            }
        )
    except (RuntimeError, ValueError):
        plt.rcParams.update(
            {
                "font.family": "serif",
                "mathtext.fontset": "cm",
            }
        )


# ── Column width constants ────────────────────────────────────
SINGLE_COL = 3.25  # inches
DOUBLE_COL = 6.75  # inches


def figure_intrinsic_dimension(
    layer_indices: list[int],
    dims_by_model: dict[str, list[float]],
    output_path: str | Path | None = None,
) -> Figure:
    """Paper Figure: Intrinsic dimensionality across layers.

    Parameters
    ----------
    layer_indices : list[int]
        Layer numbers.
    dims_by_model : dict[str, list[float]]
        Maps model name to intrinsic dimensions per layer.
    output_path : str | Path | None
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt, _ = _require_matplotlib()
    set_paper_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))

    for model_name, dims in dims_by_model.items():
        ax.plot(layer_indices[: len(dims)], dims, "-o", label=model_name, markersize=3)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Intrinsic Dimension")
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        logger.info("Saved figure to %s", output_path)

    return fig


def figure_curvature_profile(
    layer_indices: list[int],
    curvature_stats: dict[str, dict[str, list[float]]],
    output_path: str | Path | None = None,
) -> Figure:
    """Paper Figure: Curvature profiles across layers and models.

    Parameters
    ----------
    layer_indices : list[int]
        Layer numbers.
    curvature_stats : dict[str, dict[str, list[float]]]
        Maps model name to {"mean": [...], "std": [...]}.
    output_path : str | Path | None
        If provided, save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt, _ = _require_matplotlib()
    set_paper_style()

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.5, 2.5))

    for model_name, stats in curvature_stats.items():
        means = np.array(stats["mean"])
        stds = np.array(stats["std"])
        layers = layer_indices[: len(means)]

        ax.plot(layers, means, "-o", label=model_name, markersize=3)
        ax.fill_between(layers, means - stds, means + stds, alpha=0.15)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Scalar Curvature")
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)

    return fig


def figure_hallucination_comparison(
    method_names: list[str],
    auroc_values: list[float],
    auprc_values: list[float],
    output_path: str | Path | None = None,
) -> Figure:
    """Paper Figure: Hallucination detection AUROC/AUPRC comparison.

    Parameters
    ----------
    method_names : list[str]
        Method names (e.g., ["Ours", "Entropy", "Density", "Cosine"]).
    auroc_values : list[float]
        AUROC per method.
    auprc_values : list[float]
        AUPRC per method.
    output_path : str | Path | None
        If provided, save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt, _ = _require_matplotlib()
    set_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    x = np.arange(len(method_names))
    width = 0.6

    # AUROC
    axes[0].bar(x, auroc_values, width, color="steelblue", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_names, rotation=30, ha="right")
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(a) AUROC")

    # AUPRC
    axes[1].bar(x, auprc_values, width, color="coral", edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_names, rotation=30, ha="right")
    axes[1].set_ylabel("AUPRC")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("(b) AUPRC")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)

    return fig


def save_all_figures(
    output_dir: str | Path,
    **figure_data,
) -> list[Path]:
    """Generate and save all paper figures.

    Parameters
    ----------
    output_dir : str | Path
        Directory to save figures.
    **figure_data
        Data for each figure (see individual figure functions).

    Returns
    -------
    list[Path]
        Paths to saved figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    logger.info("Saving paper figures to %s", output_dir)

    # Generate each figure if data is provided
    if "intrinsic_dim" in figure_data:
        d = figure_data["intrinsic_dim"]
        path = output_dir / "fig_intrinsic_dimension.pdf"
        figure_intrinsic_dimension(d["layers"], d["dims_by_model"], path)
        saved.append(path)

    if "curvature" in figure_data:
        d = figure_data["curvature"]
        path = output_dir / "fig_curvature_profile.pdf"
        figure_curvature_profile(d["layers"], d["stats"], path)
        saved.append(path)

    if "hallucination" in figure_data:
        d = figure_data["hallucination"]
        path = output_dir / "fig_hallucination_comparison.pdf"
        figure_hallucination_comparison(d["methods"], d["auroc"], d["auprc"], path)
        saved.append(path)

    logger.info("Saved %d figures", len(saved))
    return saved
