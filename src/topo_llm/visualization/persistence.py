"""
Visualization tools for persistent homology results.

Plot persistence diagrams, barcodes, landscapes, and Betti curves.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError("matplotlib required. Install with: pip install topo-llm[viz]") from None


def plot_persistence_diagram(
    diagrams: list[np.ndarray],
    title: str = "Persistence Diagram",
    max_dimension: int = 2,
    figsize: tuple[int, int] = (8, 8),
) -> Figure:
    """Plot persistence diagram (birth vs death).

    Parameters
    ----------
    diagrams : list[np.ndarray]
        Persistence diagrams per dimension.
    title : str
        Plot title.
    max_dimension : int
        Maximum homology dimension to plot.
    figsize : tuple[int, int]
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    labels = [f"$H_{k}$" for k in range(max_dimension + 1)]

    all_vals = []
    for k in range(min(len(diagrams), max_dimension + 1)):
        dgm = diagrams[k]
        if len(dgm) > 0:
            ax.scatter(
                dgm[:, 0],
                dgm[:, 1],
                c=colors[k % len(colors)],
                label=labels[k],
                s=30,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
            )
            all_vals.extend(dgm.ravel().tolist())

    if all_vals:
        lim_min = min(all_vals) - 0.1
        lim_max = max(all_vals) + 0.1
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.3, label="Diagonal")

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig


def plot_barcode(
    diagrams: list[np.ndarray],
    title: str = "Persistence Barcode",
    max_dimension: int = 2,
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """Plot persistence barcode (horizontal bars for each feature).

    Parameters
    ----------
    diagrams : list[np.ndarray]
        Persistence diagrams per dimension.
    title : str
        Plot title.
    max_dimension : int
        Maximum homology dimension to plot.
    figsize : tuple[int, int]
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    y_offset = 0

    for k in range(min(len(diagrams), max_dimension + 1)):
        dgm = diagrams[k]
        if len(dgm) == 0:
            continue

        # Sort by lifetime descending
        lifetimes = dgm[:, 1] - dgm[:, 0]
        order = np.argsort(-lifetimes)

        for i, idx in enumerate(order):
            b, d = dgm[idx]
            ax.barh(
                y_offset + i,
                d - b,
                left=b,
                height=0.8,
                color=colors[k % len(colors)],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

        # Add dimension label
        mid = y_offset + len(dgm) / 2
        ax.text(-0.1, mid, f"$H_{k}$", fontsize=12, ha="right", va="center")

        y_offset += len(dgm) + 2  # gap between dimensions

    ax.set_xlabel("Filtration Value")
    ax.set_title(title)
    ax.set_yticks([])
    plt.tight_layout()
    return fig


def plot_betti_curve(
    betti_curves: dict[int, tuple[np.ndarray, np.ndarray]],
    title: str = "Betti Curves",
    figsize: tuple[int, int] = (10, 5),
) -> Figure:
    """Plot Betti numbers as a function of scale.

    Parameters
    ----------
    betti_curves : dict[int, tuple[np.ndarray, np.ndarray]]
        Maps dimension to ``(epsilons, betti_values)``.
    title : str
        Plot title.
    figsize : tuple[int, int]
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["tab:blue", "tab:orange", "tab:green"]

    for k, (eps, vals) in betti_curves.items():
        color = colors[k % len(colors)]
        ax.plot(eps, vals, color=color, linewidth=2, label=f"$\\beta_{k}$")

    ax.set_xlabel("Scale ($\\epsilon$)")
    ax.set_ylabel("Betti Number")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_persistence_landscape(
    grid: np.ndarray,
    landscapes: np.ndarray,
    title: str = "Persistence Landscapes",
    figsize: tuple[int, int] = (10, 5),
) -> Figure:
    """Plot persistence landscape functions.

    Parameters
    ----------
    grid : np.ndarray
        Grid values, shape ``(resolution,)``.
    landscapes : np.ndarray
        Landscape functions, shape ``(n_landscapes, resolution)``.
    title : str
        Plot title.
    figsize : tuple[int, int]
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    n_k = min(5, landscapes.shape[0])
    for k in range(n_k):
        ax.fill_between(grid, 0, landscapes[k], alpha=0.3, label=f"$\\lambda_{k + 1}$")
        ax.plot(grid, landscapes[k], linewidth=1)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\lambda_k(t)$")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig
