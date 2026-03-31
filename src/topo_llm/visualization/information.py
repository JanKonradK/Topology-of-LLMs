"""
Visualization tools for information geometry results.

Plot Fisher information heatmaps, entropy surfaces, and KL matrices.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib required. Install with: pip install topo-llm[viz]")


def plot_fisher_heatmap(
    fisher_matrix: np.ndarray,
    title: str = "Fisher Information Matrix",
    figsize: tuple[int, int] = (8, 6),
) -> object:
    """Plot the Fisher Information Matrix as a heatmap.

    Parameters
    ----------
    fisher_matrix : np.ndarray
        Fisher matrix, shape ``(d, d)``.
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
    im = ax.imshow(fisher_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Dimension $j$")
    ax.set_ylabel("Dimension $i$")
    plt.tight_layout()
    return fig


def plot_entropy_scatter(
    embeddings_2d: np.ndarray,
    entropies: np.ndarray,
    title: str = "Entropy Surface",
    cmap: str = "YlOrRd",
    figsize: tuple[int, int] = (10, 8),
) -> object:
    """Plot entropy as colored scatter on 2D embeddings.

    Parameters
    ----------
    embeddings_2d : np.ndarray
        2D coordinates, shape ``(N, 2)``.
    entropies : np.ndarray
        Entropy at each point, shape ``(N,)``.
    title : str
        Plot title.
    cmap : str
        Colormap.
    figsize : tuple[int, int]
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=entropies, cmap=cmap, s=20, alpha=0.7, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Entropy (nats)")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()
    return fig


def plot_kl_matrix(
    kl_matrix: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Jensen-Shannon Divergence Matrix",
    figsize: tuple[int, int] = (10, 8),
) -> object:
    """Plot pairwise JSD as a heatmap.

    Parameters
    ----------
    kl_matrix : np.ndarray
        Symmetric JSD matrix, shape ``(n, n)``.
    labels : list[str] | None
        Optional labels for rows/columns.
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
    im = ax.imshow(kl_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="JSD")

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_fisher_trace_by_layer(
    layer_indices: list[int],
    traces: list[float],
    title: str = "Fisher Trace Across Layers",
    figsize: tuple[int, int] = (10, 5),
) -> object:
    """Plot Fisher trace as a function of layer depth.

    Parameters
    ----------
    layer_indices : list[int]
        Layer numbers.
    traces : list[float]
        Fisher trace per layer.
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
    ax.plot(layer_indices, traces, "b-o", linewidth=2)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Fisher Trace")
    ax.set_title(title)
    plt.tight_layout()
    return fig
