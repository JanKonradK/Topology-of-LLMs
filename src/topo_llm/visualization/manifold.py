"""
Visualization tools for Riemannian manifold analysis.

Plot curvature fields, geodesic paths, and metric ellipses.
All functions return matplotlib Figure objects for flexible display.
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


def plot_curvature_field(
    embeddings_2d: np.ndarray,
    curvatures: np.ndarray,
    title: str = "Scalar Curvature Field",
    cmap: str = "RdBu_r",
    figsize: tuple[int, int] = (10, 8),
) -> object:
    """Plot scalar curvature as a colored scatter on 2D embeddings.

    Parameters
    ----------
    embeddings_2d : np.ndarray
        2D embedding coordinates, shape ``(N, 2)``.
    curvatures : np.ndarray
        Scalar curvature at each point, shape ``(N,)``.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap.
    figsize : tuple[int, int]
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # Symmetric color scale centered at 0
    vmax = max(abs(curvatures.min()), abs(curvatures.max()))
    sc = ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=curvatures, cmap=cmap, vmin=-vmax, vmax=vmax,
        s=20, alpha=0.7, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Scalar Curvature")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()
    return fig


def plot_geodesic(
    embeddings_2d: np.ndarray,
    geodesic_2d: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "Geodesic Path",
    figsize: tuple[int, int] = (10, 8),
) -> object:
    """Plot a geodesic path over the embedding point cloud.

    Parameters
    ----------
    embeddings_2d : np.ndarray
        Background point cloud, shape ``(N, 2)``.
    geodesic_2d : np.ndarray
        Geodesic path points, shape ``(n_steps, 2)``.
    labels : np.ndarray | None
        Optional labels for coloring background points.
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

    if labels is not None:
        scatter = ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=labels, cmap="tab10", s=10, alpha=0.3,
        )
    else:
        ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c="lightgray", s=10, alpha=0.3,
        )

    # Geodesic path
    ax.plot(
        geodesic_2d[:, 0], geodesic_2d[:, 1],
        "r-", linewidth=2, label="Geodesic",
    )
    ax.plot(geodesic_2d[0, 0], geodesic_2d[0, 1], "go", markersize=10, label="Start")
    ax.plot(geodesic_2d[-1, 0], geodesic_2d[-1, 1], "rs", markersize=10, label="End")

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_metric_ellipses(
    embeddings_2d: np.ndarray,
    metric_tensors: list[np.ndarray],
    n_ellipses: int = 50,
    title: str = "Local Metric Ellipses",
    figsize: tuple[int, int] = (10, 8),
) -> object:
    """Plot metric tensor ellipses at selected points.

    Each ellipse visualizes the local metric: elongated in directions
    where the metric stretches distances, contracted where it compresses.

    Parameters
    ----------
    embeddings_2d : np.ndarray
        2D coordinates, shape ``(N, 2)``.
    metric_tensors : list[np.ndarray]
        Metric tensors (must be 2x2 for visualization).
    n_ellipses : int
        Number of ellipses to draw.
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
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c="lightgray", s=5, alpha=0.3,
    )

    # Select subset of points
    N = len(embeddings_2d)
    rng = np.random.default_rng(42)
    indices = rng.choice(N, size=min(n_ellipses, N), replace=False)

    for idx in indices:
        g = metric_tensors[idx]
        if g.shape != (2, 2):
            continue

        center = embeddings_2d[idx]

        # Eigendecompose metric
        eigenvalues, eigenvectors = np.linalg.eigh(g)
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        # Ellipse axes are proportional to sqrt(eigenvalues)
        width = 2 * np.sqrt(eigenvalues[0]) * 0.1  # scale factor
        height = 2 * np.sqrt(eigenvalues[1]) * 0.1
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        ellipse = Ellipse(
            center, width, height, angle=angle,
            fill=False, edgecolor="blue", linewidth=0.5, alpha=0.6,
        )
        ax.add_patch(ellipse)

    ax.set_title(title)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig


def plot_layer_curvature_profile(
    layer_indices: list[int],
    mean_curvatures: list[float],
    std_curvatures: list[float],
    title: str = "Curvature Across Layers",
    figsize: tuple[int, int] = (10, 5),
) -> object:
    """Plot mean scalar curvature as a function of layer depth.

    Parameters
    ----------
    layer_indices : list[int]
        Layer numbers.
    mean_curvatures : list[float]
        Mean scalar curvature per layer.
    std_curvatures : list[float]
        Standard deviation per layer.
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

    means = np.array(mean_curvatures)
    stds = np.array(std_curvatures)

    ax.plot(layer_indices, means, "b-o", linewidth=2, label="Mean curvature")
    ax.fill_between(
        layer_indices, means - stds, means + stds,
        alpha=0.2, color="blue", label="1 std",
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Scalar Curvature")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig
