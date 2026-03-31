"""
Local Riemannian metric tensor estimation from point clouds.

Estimates the induced Riemannian metric on a data manifold by analyzing
local neighborhoods via PCA. The metric field is interpolated between
sample points to create a smooth field suitable for differentiation.

Mathematical Background
-----------------------
Given a point cloud P ⊂ R^D sampling a k-dimensional submanifold M,
at each point x_i we:

1. Find k nearest neighbors → local neighborhood N(x_i)
2. Compute local covariance Σ_i = (1/k) Σ_{j ∈ N(i)} (x_j - x̄)(x_j - x̄)^T
3. Take top-m eigenvectors as tangent basis: T_i ∈ R^{D×m}
4. Metric tensor in tangent coords: g_i = T_i^T @ T_i ∈ R^{m×m}
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class MetricTensorEstimator:
    """Estimate local Riemannian metric tensors from a point cloud.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbors for local geometry estimation.
    intrinsic_dim : int | None
        Intrinsic dimensionality of the manifold. If None, estimated
        automatically via MLE.
    regularization : float
        Ridge regularization added to metric tensors for numerical
        stability: ``g_ij += regularization * I``.

    Attributes
    ----------
    point_cloud_ : np.ndarray
        The fitted point cloud, shape ``(N, D)``.
    tangent_bases_ : list[np.ndarray]
        Tangent basis at each point, each shape ``(D, m)``.
    metric_tensors_ : list[np.ndarray]
        Metric tensor at each point, each shape ``(m, m)``.
    metric_inverses_ : list[np.ndarray]
        Inverse metric tensor at each point, each shape ``(m, m)``.
    intrinsic_dim_ : int
        Estimated or provided intrinsic dimension.
    nn_tree_ : KDTree
        Fitted nearest-neighbor tree for queries.

    Examples
    --------
    >>> points = np.random.randn(500, 100)  # 500 points in R^100
    >>> estimator = MetricTensorEstimator(n_neighbors=30, intrinsic_dim=10)
    >>> estimator.fit(points)
    >>> g = estimator.get_metric_at(0)
    >>> g.shape
    (10, 10)
    """

    def __init__(
        self,
        n_neighbors: int = 50,
        intrinsic_dim: int | None = None,
        regularization: float = 1e-6,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.intrinsic_dim = intrinsic_dim
        self.regularization = regularization

        # Fitted attributes (set by fit())
        self.point_cloud_: np.ndarray | None = None
        self.tangent_bases_: list[np.ndarray] = []
        self.metric_tensors_: list[np.ndarray] = []
        self.metric_inverses_: list[np.ndarray] = []
        self.local_coords_: list[np.ndarray] = []
        self.intrinsic_dim_: int = 0
        self.nn_tree_: KDTree | None = None

    def fit(self, point_cloud: np.ndarray) -> MetricTensorEstimator:
        """Fit the metric tensor field to a point cloud.

        For each point, estimates the local tangent space and metric
        tensor using k-nearest-neighbor covariance analysis.

        Parameters
        ----------
        point_cloud : np.ndarray
            Point cloud of shape ``(N, D)``.

        Returns
        -------
        MetricTensorEstimator
            Self, for method chaining.
        """
        N, D = point_cloud.shape
        self.point_cloud_ = point_cloud.copy()
        k = min(self.n_neighbors, N - 1)

        # Estimate intrinsic dimension if not provided
        if self.intrinsic_dim is None:
            from topo_llm.extraction.layers import LayerAnalyzer
            est_dim = LayerAnalyzer.intrinsic_dimensionality(point_cloud, method="mle")
            self.intrinsic_dim_ = max(2, min(int(round(est_dim)), D - 1))
            logger.info("Auto-estimated intrinsic dimension: %d", self.intrinsic_dim_)
        else:
            self.intrinsic_dim_ = self.intrinsic_dim

        m = self.intrinsic_dim_

        # Build KD-tree
        logger.info("Building KD-tree for %d points in R^%d", N, D)
        self.nn_tree_ = KDTree(point_cloud)

        # Query all neighborhoods at once
        distances, indices = self.nn_tree_.query(point_cloud, k=k + 1)
        # Remove self (first column)
        neighbor_indices = indices[:, 1:]

        # Compute metric at each point
        self.tangent_bases_ = []
        self.metric_tensors_ = []
        self.metric_inverses_ = []
        self.local_coords_ = []

        logger.info("Computing local metric tensors (m=%d)...", m)
        for i in range(N):
            # Local neighborhood
            nbr_idx = neighbor_indices[i]
            neighbors = point_cloud[nbr_idx]  # (k, D)
            center = neighbors.mean(axis=0)  # local centroid

            # Local covariance
            centered = neighbors - center  # (k, D)
            cov = (centered.T @ centered) / k  # (D, D)

            # Eigendecompose for tangent basis
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort descending
            idx_sorted = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx_sorted]
            eigenvectors = eigenvectors[:, idx_sorted]

            # Top-m eigenvectors as tangent basis
            T_i = eigenvectors[:, :m]  # (D, m)

            # Metric tensor: g = T^T @ T (= I for orthonormal T,
            # but we keep it general for non-orthonormal bases)
            g_i = T_i.T @ T_i  # (m, m)

            # Add regularization
            g_i += self.regularization * np.eye(m)

            # Inverse metric
            g_inv = np.linalg.inv(g_i)  # (m, m)

            # Project neighbors to tangent space
            local = centered @ T_i  # (k, m)

            self.tangent_bases_.append(T_i)
            self.metric_tensors_.append(g_i)
            self.metric_inverses_.append(g_inv)
            self.local_coords_.append(local)

        logger.info("Metric tensor estimation complete for %d points", N)
        return self

    def get_metric_at(self, idx: int) -> np.ndarray:
        """Return the metric tensor at point index ``idx``.

        Parameters
        ----------
        idx : int
            Index into the fitted point cloud.

        Returns
        -------
        np.ndarray
            Metric tensor of shape ``(m, m)``.
        """
        return self.metric_tensors_[idx].copy()

    def get_metric_inverse_at(self, idx: int) -> np.ndarray:
        """Return the inverse metric tensor at point index ``idx``.

        Parameters
        ----------
        idx : int
            Index into the fitted point cloud.

        Returns
        -------
        np.ndarray
            Inverse metric tensor of shape ``(m, m)``.
        """
        return self.metric_inverses_[idx].copy()

    def get_tangent_basis_at(self, idx: int) -> np.ndarray:
        """Return the tangent basis at point index ``idx``.

        Parameters
        ----------
        idx : int
            Index into the fitted point cloud.

        Returns
        -------
        np.ndarray
            Tangent basis of shape ``(D, m)``.
        """
        return self.tangent_bases_[idx].copy()

    def interpolate_metric(
        self,
        point: np.ndarray,
        n_interp: int = 5,
    ) -> np.ndarray:
        """Estimate the metric tensor at an arbitrary point.

        Uses inverse-distance weighting of nearby fitted metrics.

        Parameters
        ----------
        point : np.ndarray
            Point in ambient space, shape ``(D,)``.
        n_interp : int
            Number of nearest fitted points to use for interpolation.

        Returns
        -------
        np.ndarray
            Interpolated metric tensor, shape ``(m, m)``.
        """
        distances, indices = self.nn_tree_.query(point, k=n_interp)

        # Handle exact match (distance = 0)
        if np.isscalar(distances):
            distances = np.array([distances])
            indices = np.array([indices])

        # Inverse-distance weights
        eps = 1e-10
        weights = 1.0 / (distances + eps)
        weights /= weights.sum()

        # Weighted average of metric tensors
        m = self.intrinsic_dim_
        g_interp = np.zeros((m, m))
        for w, idx in zip(weights, indices):
            g_interp += w * self.metric_tensors_[idx]

        return g_interp

    def interpolate_metric_inverse(
        self,
        point: np.ndarray,
        n_interp: int = 5,
    ) -> np.ndarray:
        """Estimate the inverse metric at an arbitrary point.

        Parameters
        ----------
        point : np.ndarray
            Point in ambient space, shape ``(D,)``.
        n_interp : int
            Number of nearest fitted points for interpolation.

        Returns
        -------
        np.ndarray
            Interpolated inverse metric, shape ``(m, m)``.
        """
        g = self.interpolate_metric(point, n_interp)
        return np.linalg.inv(g)

    def interpolate_tangent_basis(
        self,
        point: np.ndarray,
        n_interp: int = 5,
    ) -> np.ndarray:
        """Estimate the tangent basis at an arbitrary point.

        Uses weighted average of nearby tangent bases, followed by
        QR orthogonalization.

        Parameters
        ----------
        point : np.ndarray
            Point in ambient space, shape ``(D,)``.
        n_interp : int
            Number of nearest fitted points for interpolation.

        Returns
        -------
        np.ndarray
            Interpolated tangent basis, shape ``(D, m)``.
        """
        distances, indices = self.nn_tree_.query(point, k=n_interp)

        if np.isscalar(distances):
            distances = np.array([distances])
            indices = np.array([indices])

        eps = 1e-10
        weights = 1.0 / (distances + eps)
        weights /= weights.sum()

        D = self.point_cloud_.shape[1]
        m = self.intrinsic_dim_
        T_interp = np.zeros((D, m))
        for w, idx in zip(weights, indices):
            T_interp += w * self.tangent_bases_[idx]

        # Re-orthogonalize
        Q, _ = np.linalg.qr(T_interp)
        return Q[:, :m]

    def volume_element(self, idx: int) -> float:
        """Compute the Riemannian volume element at a point.

        The volume element is ``sqrt(det(g_ij))``, which measures how
        much the local geometry stretches or compresses volume relative
        to flat space.

        Parameters
        ----------
        idx : int
            Index into the fitted point cloud.

        Returns
        -------
        float
            Volume element ``sqrt(det(g))``.
        """
        g = self.metric_tensors_[idx]
        det_g = np.linalg.det(g)
        return float(np.sqrt(max(det_g, 0.0)))

    def all_volume_elements(self) -> np.ndarray:
        """Compute volume elements at all fitted points.

        Returns
        -------
        np.ndarray
            Array of shape ``(N,)`` with volume elements.
        """
        return np.array([self.volume_element(i) for i in range(len(self.metric_tensors_))])

    def project_to_tangent(self, idx: int, vector: np.ndarray) -> np.ndarray:
        """Project an ambient-space vector onto the tangent space at a point.

        Parameters
        ----------
        idx : int
            Index of the base point.
        vector : np.ndarray
            Vector in ambient space, shape ``(D,)``.

        Returns
        -------
        np.ndarray
            Tangent vector, shape ``(m,)``.
        """
        T = self.tangent_bases_[idx]  # (D, m)
        return T.T @ vector  # (m,)

    def lift_from_tangent(self, idx: int, tangent_vector: np.ndarray) -> np.ndarray:
        """Lift a tangent vector back to ambient space.

        Parameters
        ----------
        idx : int
            Index of the base point.
        tangent_vector : np.ndarray
            Vector in tangent space, shape ``(m,)``.

        Returns
        -------
        np.ndarray
            Vector in ambient space, shape ``(D,)``.
        """
        T = self.tangent_bases_[idx]  # (D, m)
        return T @ tangent_vector  # (D,)
