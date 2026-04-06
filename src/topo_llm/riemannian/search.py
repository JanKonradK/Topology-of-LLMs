"""
Geodesic-aware nearest neighbor search.

Provides nearest neighbor queries using geodesic distance instead of
Euclidean or cosine distance. Uses a two-stage approach: Euclidean
pre-filtering followed by geodesic refinement.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import kendalltau

from topo_llm.riemannian.geodesic import GeodesicSolver
from topo_llm.riemannian.metric import MetricTensorEstimator
from topo_llm.types import ComparisonResult

logger = logging.getLogger(__name__)


class RiemannianSearch:
    """Geodesic-aware nearest neighbor search.

    Parameters
    ----------
    geodesic_solver : GeodesicSolver
        Solver for computing geodesic distances.
    metric_estimator : MetricTensorEstimator
        Fitted metric tensor estimator.

    Examples
    --------
    >>> search = RiemannianSearch(geodesic_solver, metric_est)
    >>> neighbors = search.query(query_idx=0, k=5)
    >>> for idx, dist in neighbors:
    ...     print(f"  Point {idx}: geodesic distance {dist:.4f}")
    """

    def __init__(
        self,
        geodesic_solver: GeodesicSolver,
        metric_estimator: MetricTensorEstimator,
    ) -> None:
        self.geodesic = geodesic_solver
        self.metric = metric_estimator

    def query(
        self,
        query_idx: int,
        k: int = 10,
        candidates: int = 50,
    ) -> list[tuple[int, float]]:
        """Find k nearest neighbors using geodesic distance.

        Strategy for efficiency:
        1. Find top ``candidates`` by Euclidean distance
        2. Compute geodesic distance to each candidate
        3. Return top ``k`` by geodesic distance

        Parameters
        ----------
        query_idx : int
            Index of the query point.
        k : int
            Number of nearest neighbors to return.
        candidates : int
            Number of Euclidean candidates to pre-filter.

        Returns
        -------
        list[tuple[int, float]]
            List of ``(index, geodesic_distance)`` pairs,
            sorted by ascending distance.
        """
        point_cloud = self.metric.point_cloud_
        N = point_cloud.shape[0]
        candidates = min(candidates, N - 1)
        k = min(k, candidates)

        # Euclidean pre-filtering
        tree = self.metric.nn_tree_
        distances, indices = tree.query(point_cloud[query_idx], k=candidates + 1)

        # Remove self
        mask = indices != query_idx
        indices = indices[mask][:candidates]

        # Compute geodesic distances
        geo_distances = []
        for idx in indices:
            d = self.geodesic.geodesic_distance(query_idx, int(idx), n_shooting=3)
            geo_distances.append((int(idx), d))

        # Sort by geodesic distance
        geo_distances.sort(key=lambda x: x[1])
        return geo_distances[:k]

    def query_euclidean(
        self,
        query_idx: int,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Find k nearest neighbors using Euclidean distance.

        Parameters
        ----------
        query_idx : int
            Index of the query point.
        k : int
            Number of neighbors.

        Returns
        -------
        list[tuple[int, float]]
            List of ``(index, euclidean_distance)`` pairs.
        """
        tree = self.metric.nn_tree_
        distances, indices = tree.query(self.metric.point_cloud_[query_idx], k=k + 1)

        result = []
        for d, idx in zip(distances, indices):
            if idx != query_idx:
                result.append((int(idx), float(d)))

        return result[:k]

    def query_cosine(
        self,
        query_idx: int,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Find k nearest neighbors using cosine distance.

        Parameters
        ----------
        query_idx : int
            Index of the query point.
        k : int
            Number of neighbors.

        Returns
        -------
        list[tuple[int, float]]
            List of ``(index, cosine_distance)`` pairs.
        """
        point_cloud = self.metric.point_cloud_
        query = point_cloud[query_idx]

        # Compute cosine distances to all points
        norms = np.linalg.norm(point_cloud, axis=1)
        query_norm = np.linalg.norm(query)

        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        query_norm = max(query_norm, 1e-10)

        cosine_sim = (point_cloud @ query) / (norms * query_norm)
        cosine_dist = 1.0 - cosine_sim

        # Get top-k (excluding self)
        cosine_dist[query_idx] = float("inf")
        top_k = np.argsort(cosine_dist)[:k]

        return [(int(idx), float(cosine_dist[idx])) for idx in top_k]

    def compare_metrics(
        self,
        query_idx: int,
        k: int = 10,
        candidates: int = 50,
    ) -> ComparisonResult:
        """Compare nearest neighbors under different distance metrics.

        Parameters
        ----------
        query_idx : int
            Index of the query point.
        k : int
            Number of neighbors to compare.
        candidates : int
            Euclidean candidates for geodesic search.

        Returns
        -------
        ComparisonResult
            Dictionary with:

            - ``"euclidean_neighbors"``: list of neighbor indices
            - ``"cosine_neighbors"``: list of neighbor indices
            - ``"geodesic_neighbors"``: list of neighbor indices
            - ``"rank_correlation_euclid_geo"``: Kendall tau
            - ``"rank_correlation_cosine_geo"``: Kendall tau
        """
        euclid = self.query_euclidean(query_idx, k)
        cosine = self.query_cosine(query_idx, k)
        geodesic = self.query(query_idx, k, candidates)

        euclid_idx = [idx for idx, _ in euclid]
        cosine_idx = [idx for idx, _ in cosine]
        geo_idx = [idx for idx, _ in geodesic]

        # Compute rank correlations
        # For Kendall tau, we need shared elements ranked
        def rank_map(neighbors: list[int]) -> dict[int, int]:
            return {idx: rank for rank, idx in enumerate(neighbors)}

        euclid_ranks = rank_map(euclid_idx)
        cosine_ranks = rank_map(cosine_idx)
        geo_ranks = rank_map(geo_idx)

        # Kendall tau on shared elements
        shared_eg = list(set(euclid_idx) & set(geo_idx))
        shared_cg = list(set(cosine_idx) & set(geo_idx))

        tau_eg = 0.0
        if len(shared_eg) > 1:
            r1 = [euclid_ranks[i] for i in shared_eg]
            r2 = [geo_ranks[i] for i in shared_eg]
            tau_eg, _ = kendalltau(r1, r2)
            if not np.isfinite(tau_eg):
                tau_eg = 0.0

        tau_cg = 0.0
        if len(shared_cg) > 1:
            r1 = [cosine_ranks[i] for i in shared_cg]
            r2 = [geo_ranks[i] for i in shared_cg]
            tau_cg, _ = kendalltau(r1, r2)
            if not np.isfinite(tau_cg):
                tau_cg = 0.0

        return {
            "euclidean_neighbors": euclid_idx,
            "cosine_neighbors": cosine_idx,
            "geodesic_neighbors": geo_idx,
            "rank_correlation_euclid_geo": float(tau_eg),
            "rank_correlation_cosine_geo": float(tau_cg),
        }
