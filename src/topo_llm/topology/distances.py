"""
Distances between persistence diagrams.

Provides Wasserstein and bottleneck distances for comparing
topological signatures across layers, models, or datasets.
"""

from __future__ import annotations

import logging

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _require_persim():
    try:
        import persim

        return persim
    except ImportError:
        raise ImportError(
            "persim is required for diagram distances. Install with: pip install topo-llm[tda]"
        ) from None


class DiagramDistances:
    """Compute distances between persistence diagrams.

    All methods are static.

    Examples
    --------
    >>> d = DiagramDistances.wasserstein(dgm_a, dgm_b, p=2.0)
    >>> D = DiagramDistances.distance_matrix(diagrams, metric="wasserstein")
    """

    @staticmethod
    def wasserstein(
        dgm_a: np.ndarray,
        dgm_b: np.ndarray,
        p: float = 2.0,
    ) -> float:
        """Compute p-Wasserstein distance between persistence diagrams.

        W_p(D1, D2) = (inf_{γ} Σ ||x - γ(x)||_∞^p)^{1/p}

        where γ ranges over all matchings (including matching to diagonal).

        Parameters
        ----------
        dgm_a : np.ndarray
            First diagram, shape ``(n, 2)``.
        dgm_b : np.ndarray
            Second diagram, shape ``(m, 2)``.
        p : float
            Wasserstein order.

        Returns
        -------
        float
            Wasserstein distance.
        """
        persim = _require_persim()

        # Handle empty diagrams
        if len(dgm_a) == 0 and len(dgm_b) == 0:
            return 0.0

        # persim expects (n, 2) arrays
        if len(dgm_a) == 0:
            dgm_a = np.empty((0, 2))
        if len(dgm_b) == 0:
            dgm_b = np.empty((0, 2))

        return float(persim.wasserstein(dgm_a, dgm_b, order=p))

    @staticmethod
    def bottleneck(
        dgm_a: np.ndarray,
        dgm_b: np.ndarray,
    ) -> float:
        """Compute bottleneck distance between persistence diagrams.

        d_B(D1, D2) = inf_{γ} sup_x ||x - γ(x)||_∞

        This is the L^∞ analog of Wasserstein distance.

        Parameters
        ----------
        dgm_a : np.ndarray
            First diagram, shape ``(n, 2)``.
        dgm_b : np.ndarray
            Second diagram, shape ``(m, 2)``.

        Returns
        -------
        float
            Bottleneck distance.
        """
        persim = _require_persim()

        if len(dgm_a) == 0 and len(dgm_b) == 0:
            return 0.0

        if len(dgm_a) == 0:
            dgm_a = np.empty((0, 2))
        if len(dgm_b) == 0:
            dgm_b = np.empty((0, 2))

        return float(persim.bottleneck(dgm_a, dgm_b))

    @staticmethod
    def distance_matrix(
        diagrams: list[np.ndarray],
        metric: str = "wasserstein",
        p: float = 2.0,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute pairwise distance matrix between persistence diagrams.

        Parameters
        ----------
        diagrams : list[np.ndarray]
            List of persistence diagrams.
        metric : str
            ``"wasserstein"`` or ``"bottleneck"``.
        p : float
            Wasserstein order (ignored for bottleneck).
        show_progress : bool
            Whether to show a progress bar.

        Returns
        -------
        np.ndarray
            Symmetric distance matrix, shape ``(n, n)``.
        """
        n = len(diagrams)
        D = np.zeros((n, n))

        total_pairs = n * (n - 1) // 2

        if show_progress:
            pbar = tqdm(total=total_pairs, desc=f"{metric} distances", unit="pair")
        else:
            pbar = None

        for i in range(n):
            for j in range(i + 1, n):
                if metric == "wasserstein":
                    d = DiagramDistances.wasserstein(diagrams[i], diagrams[j], p=p)
                elif metric == "bottleneck":
                    d = DiagramDistances.bottleneck(diagrams[i], diagrams[j])
                else:
                    raise ValueError(f"Unknown metric: {metric!r}")

                D[i, j] = d
                D[j, i] = d

                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        return D
