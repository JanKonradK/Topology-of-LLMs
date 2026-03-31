"""
Simplicial complex filtration builders for persistent homology.

Provides Vietoris-Rips and Alpha complex filtrations with automatic
subsampling for large point clouds via furthest point sampling.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.spatial.distance import pdist, squareform

from topo_llm.types import PersistenceResult

logger = logging.getLogger(__name__)


def _require_ripser():
    try:
        import ripser
        return ripser
    except ImportError:
        raise ImportError(
            "ripser is required for Vietoris-Rips filtration. "
            "Install with: pip install topo-llm[tda]"
        )


def _require_gudhi():
    try:
        import gudhi
        return gudhi
    except ImportError:
        raise ImportError(
            "GUDHI is required for Alpha complex filtration. "
            "Install with: pip install topo-llm[tda]"
        )


class FiltrationBuilder:
    """Build simplicial complex filtrations and compute persistent homology.

    All methods are static — they take point clouds and return
    persistence results directly.

    Examples
    --------
    >>> result = FiltrationBuilder.vietoris_rips(points, max_dimension=2)
    >>> print(f"H_1 features: {len(result.diagrams[1])}")
    """

    @staticmethod
    def vietoris_rips(
        point_cloud: np.ndarray,
        max_dimension: int = 2,
        max_edge_length: float | None = None,
        n_points: int | None = None,
    ) -> PersistenceResult:
        """Build Vietoris-Rips filtration and compute persistent homology.

        Parameters
        ----------
        point_cloud : np.ndarray
            Point cloud of shape ``(N, D)``.
        max_dimension : int
            Maximum homology dimension to compute (e.g., 2 for H_0, H_1, H_2).
        max_edge_length : float | None
            Maximum filtration value. None uses the 95th percentile of
            pairwise distances.
        n_points : int | None
            If provided and less than N, subsample using furthest point
            sampling before computing.

        Returns
        -------
        PersistenceResult
            Persistence diagrams and metadata.
        """
        ripser = _require_ripser()

        N = point_cloud.shape[0]

        # Subsample if needed
        if n_points is not None and n_points < N:
            point_cloud, _ = FiltrationBuilder.maxmin_subsample(
                point_cloud, n_points
            )
            N = n_points

        # Auto max_edge_length
        if max_edge_length is None:
            # Use a subset for efficiency
            n_sample = min(500, N)
            rng = np.random.default_rng(42)
            idx = rng.choice(N, size=n_sample, replace=False)
            sample_dists = pdist(point_cloud[idx])
            max_edge_length = float(np.percentile(sample_dists, 95))

        logger.info(
            "Computing Vietoris-Rips: N=%d, max_dim=%d, max_edge=%.3f",
            N, max_dimension, max_edge_length,
        )

        t0 = time.time()
        result = ripser.ripser(
            point_cloud,
            maxdim=max_dimension,
            thresh=max_edge_length,
        )
        elapsed = time.time() - t0

        diagrams = result["dgms"]
        # Replace inf with max_edge_length
        processed = []
        for dgm in diagrams:
            dgm = dgm.copy()
            dgm[dgm[:, 1] == np.inf, 1] = max_edge_length
            processed.append(dgm)

        return PersistenceResult(
            diagrams=processed,
            max_edge_length=max_edge_length,
            n_points_used=N,
            computation_time=elapsed,
            backend="ripser",
        )

    @staticmethod
    def alpha_complex(
        point_cloud: np.ndarray,
        max_dimension: int = 2,
    ) -> PersistenceResult:
        """Alpha complex filtration via GUDHI.

        More geometrically meaningful than Vietoris-Rips, and more
        efficient for low intrinsic dimension.

        Parameters
        ----------
        point_cloud : np.ndarray
            Point cloud of shape ``(N, D)``.
        max_dimension : int
            Maximum homology dimension.

        Returns
        -------
        PersistenceResult
            Persistence diagrams and metadata.
        """
        gudhi = _require_gudhi()

        N = point_cloud.shape[0]
        logger.info("Computing Alpha complex: N=%d, max_dim=%d", N, max_dimension)

        t0 = time.time()
        alpha = gudhi.AlphaComplex(points=point_cloud.tolist())
        simplex_tree = alpha.create_simplex_tree()
        simplex_tree.compute_persistence()
        elapsed = time.time() - t0

        # Extract diagrams per dimension
        diagrams = []
        max_val = 0.0
        for dim in range(max_dimension + 1):
            pairs = simplex_tree.persistence_intervals_in_dimension(dim)
            if len(pairs) > 0:
                dgm = np.array(pairs)
                # Replace inf
                finite_max = dgm[dgm[:, 1] != np.inf, 1]
                if len(finite_max) > 0:
                    max_val = max(max_val, finite_max.max())
                diagrams.append(dgm)
            else:
                diagrams.append(np.empty((0, 2)))

        # Replace inf values
        for dgm in diagrams:
            if len(dgm) > 0:
                dgm[dgm[:, 1] == np.inf, 1] = max_val * 1.1 if max_val > 0 else 1.0

        return PersistenceResult(
            diagrams=diagrams,
            max_edge_length=max_val,
            n_points_used=N,
            computation_time=elapsed,
            backend="gudhi",
        )

    @staticmethod
    def maxmin_subsample(
        point_cloud: np.ndarray,
        n_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Furthest point sampling (maxmin / greedy).

        Iteratively selects the point farthest from the current
        selected set, producing a well-spread subsample.

        Parameters
        ----------
        point_cloud : np.ndarray
            Full point cloud, shape ``(N, D)``.
        n_points : int
            Number of points to select.

        Returns
        -------
        subsampled : np.ndarray
            Subsampled point cloud, shape ``(n_points, D)``.
        indices : np.ndarray
            Indices of selected points in the original cloud.
        """
        N = point_cloud.shape[0]
        n_points = min(n_points, N)

        # Start with a random point
        rng = np.random.default_rng(42)
        selected = [rng.integers(N)]
        min_distances = np.full(N, np.inf)

        for _ in range(n_points - 1):
            # Update minimum distances
            last = point_cloud[selected[-1]]
            dists = np.linalg.norm(point_cloud - last, axis=1)
            min_distances = np.minimum(min_distances, dists)

            # Select the farthest point
            min_distances[selected] = -1  # exclude already selected
            next_idx = np.argmax(min_distances)
            selected.append(next_idx)

        indices = np.array(selected)
        return point_cloud[indices], indices
