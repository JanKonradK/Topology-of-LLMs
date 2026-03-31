"""
Christoffel symbol estimation via finite differences.

The Christoffel symbols Γ^k_{ij} encode the Levi-Civita connection of the
Riemannian metric — they describe how the coordinate system "twists" from
point to point and are needed for geodesic computation and curvature.

Γ^k_{ij} = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})

We approximate the metric derivatives via central finite differences.
"""

from __future__ import annotations

import logging

import numpy as np
from tqdm import tqdm

from topo_llm.riemannian.metric import MetricTensorEstimator

logger = logging.getLogger(__name__)


class ChristoffelEstimator:
    """Compute Christoffel symbols of the second kind.

    Parameters
    ----------
    metric_estimator : MetricTensorEstimator
        A fitted metric tensor estimator.
    h : float
        Finite difference step size for metric derivatives.

    Attributes
    ----------
    christoffel_symbols_ : list[np.ndarray] | None
        Cached Christoffel symbols at all points. Each array has
        shape ``(m, m, m)`` where ``result[k, i, j] = Γ^k_{ij}``.

    Examples
    --------
    >>> christoffel = ChristoffelEstimator(metric_estimator, h=1e-3)
    >>> gamma = christoffel.compute_at(0)
    >>> gamma.shape  # (m, m, m)
    (10, 10, 10)
    """

    def __init__(
        self,
        metric_estimator: MetricTensorEstimator,
        h: float = 1e-3,
    ) -> None:
        if metric_estimator.point_cloud_ is None:
            raise ValueError("MetricTensorEstimator must be fitted before use")

        self.metric = metric_estimator
        self.h = h
        self.christoffel_symbols_: list[np.ndarray] | None = None

    def compute_at(self, idx: int) -> np.ndarray:
        """Compute Christoffel symbols at a single point.

        Uses central finite differences to approximate metric derivatives:
        ``∂_i g_{jl} ≈ (g_{jl}(x+) - g_{jl}(x-)) / (2h)``

        where ``x± = x ± h * T @ e_i`` (perturbation along the i-th
        tangent direction).

        Parameters
        ----------
        idx : int
            Index of the point in the fitted point cloud.

        Returns
        -------
        np.ndarray
            Christoffel symbols of shape ``(m, m, m)`` where
            ``result[k, i, j] = Γ^k_{ij}``. Symmetric in the
            lower indices: ``Γ^k_{ij} = Γ^k_{ji}``.
        """
        m = self.metric.intrinsic_dim_
        x = self.metric.point_cloud_[idx]  # (D,)
        T = self.metric.tangent_bases_[idx]  # (D, m)
        g_inv = self.metric.metric_inverses_[idx]  # (m, m)

        # Compute metric derivatives: dg[i, j, l] = ∂_i g_{jl}
        dg = np.zeros((m, m, m))

        for i in range(m):
            # Perturbation direction in ambient space
            direction = T[:, i]  # (D,)

            # Perturbed points
            x_plus = x + self.h * direction
            x_minus = x - self.h * direction

            # Metric at perturbed points
            g_plus = self.metric.interpolate_metric(x_plus)
            g_minus = self.metric.interpolate_metric(x_minus)

            # Central difference
            dg[i] = (g_plus - g_minus) / (2.0 * self.h)

        # Compute Christoffel symbols
        # Γ^k_{ij} = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
        gamma = np.zeros((m, m, m))

        for k in range(m):
            for i in range(m):
                for j in range(i, m):  # exploit symmetry
                    value = 0.0
                    for l in range(m):
                        value += g_inv[k, l] * (
                            dg[i, j, l] + dg[j, i, l] - dg[l, i, j]
                        )
                    value *= 0.5
                    gamma[k, i, j] = value
                    gamma[k, j, i] = value  # symmetric in lower indices

        return gamma

    def compute_at_point(self, point: np.ndarray) -> np.ndarray:
        """Compute Christoffel symbols at an arbitrary point.

        Uses interpolated metric and tangent basis.

        Parameters
        ----------
        point : np.ndarray
            Point in ambient space, shape ``(D,)``.

        Returns
        -------
        np.ndarray
            Christoffel symbols, shape ``(m, m, m)``.
        """
        m = self.metric.intrinsic_dim_
        T = self.metric.interpolate_tangent_basis(point)
        g_inv = self.metric.interpolate_metric_inverse(point)

        dg = np.zeros((m, m, m))

        for i in range(m):
            direction = T[:, i]
            x_plus = point + self.h * direction
            x_minus = point - self.h * direction

            g_plus = self.metric.interpolate_metric(x_plus)
            g_minus = self.metric.interpolate_metric(x_minus)

            dg[i] = (g_plus - g_minus) / (2.0 * self.h)

        gamma = np.zeros((m, m, m))
        for k in range(m):
            for i in range(m):
                for j in range(i, m):
                    value = 0.0
                    for l in range(m):
                        value += g_inv[k, l] * (
                            dg[i, j, l] + dg[j, i, l] - dg[l, i, j]
                        )
                    value *= 0.5
                    gamma[k, i, j] = value
                    gamma[k, j, i] = value

        return gamma

    def compute_all(self, show_progress: bool = True) -> list[np.ndarray]:
        """Compute Christoffel symbols at all fitted points.

        Results are cached in ``self.christoffel_symbols_``.

        Parameters
        ----------
        show_progress : bool
            Whether to show a progress bar.

        Returns
        -------
        list[np.ndarray]
            List of ``(m, m, m)`` arrays, one per point.
        """
        if self.christoffel_symbols_ is not None:
            return self.christoffel_symbols_

        N = len(self.metric.metric_tensors_)
        logger.info("Computing Christoffel symbols at %d points...", N)

        iterator = range(N)
        if show_progress:
            iterator = tqdm(iterator, desc="Christoffel symbols", unit="pt")

        self.christoffel_symbols_ = [self.compute_at(i) for i in iterator]
        return self.christoffel_symbols_

    def verify_symmetry(self, idx: int, tol: float = 1e-8) -> bool:
        """Check that Christoffel symbols are symmetric in lower indices.

        Parameters
        ----------
        idx : int
            Point index.
        tol : float
            Tolerance for symmetry check.

        Returns
        -------
        bool
            True if ``|Γ^k_{ij} - Γ^k_{ji}| < tol`` for all i, j, k.
        """
        gamma = self.compute_at(idx)
        m = gamma.shape[0]
        for k in range(m):
            diff = np.abs(gamma[k] - gamma[k].T).max()
            if diff > tol:
                return False
        return True
