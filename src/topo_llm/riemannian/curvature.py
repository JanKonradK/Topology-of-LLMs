"""
Curvature tensor computations on the estimated Riemannian manifold.

Computes the full curvature hierarchy:
- Riemann curvature tensor R^l_{ijk}
- Ricci tensor Ric_{ij} = R^k_{ikj}
- Scalar curvature S = g^{ij} Ric_{ij}
- Sectional curvature K(v1, v2)

All from the estimated metric and Christoffel symbols.
"""

from __future__ import annotations

import logging

import numpy as np
from tqdm import tqdm

from topo_llm.riemannian.connection import ChristoffelEstimator
from topo_llm.riemannian.metric import MetricTensorEstimator
from topo_llm.types import CurvatureStats

logger = logging.getLogger(__name__)


class CurvatureAnalyzer:
    """Compute curvature tensors on the estimated manifold.

    Parameters
    ----------
    metric_estimator : MetricTensorEstimator
        Fitted metric tensor estimator.
    christoffel_estimator : ChristoffelEstimator
        Christoffel symbol estimator (same metric).

    Examples
    --------
    >>> curvature = CurvatureAnalyzer(metric_est, christoffel_est)
    >>> S = curvature.scalar_curvature_at(0)
    >>> print(f"Scalar curvature: {S:.4f}")
    """

    def __init__(
        self,
        metric_estimator: MetricTensorEstimator,
        christoffel_estimator: ChristoffelEstimator,
    ) -> None:
        self.metric = metric_estimator
        self.christoffel = christoffel_estimator

    def riemann_tensor_at(self, idx: int) -> np.ndarray:
        """Compute the Riemann curvature tensor at a point.

        R^l_{ijk} = ∂_j Γ^l_{ik} - ∂_k Γ^l_{ij}
                   + Γ^l_{jm} Γ^m_{ik} - Γ^l_{km} Γ^m_{ij}

        Uses finite differences on Christoffel symbols for the
        derivative terms.

        Parameters
        ----------
        idx : int
            Index of the point.

        Returns
        -------
        np.ndarray
            Riemann tensor of shape ``(m, m, m, m)`` where
            ``result[l, i, j, k] = R^l_{ijk}``.

        Notes
        -----
        Complexity is O(d^4) where d is the intrinsic dimensionality,
        due to the four-index tensor. Use PCA to reduce dimensionality
        before calling (d=50 is typical).
        """
        m = self.metric.intrinsic_dim_
        h = self.christoffel.h
        x = self.metric.point_cloud_[idx]
        T = self.metric.tangent_bases_[idx]
        gamma = self.christoffel.compute_at(idx)

        # Compute derivatives of Christoffel symbols: d_gamma[j, l, i, k] = ∂_j Γ^l_{ik}
        d_gamma = np.zeros((m, m, m, m))

        for j in range(m):
            direction = T[:, j]
            x_plus = x + h * direction
            x_minus = x - h * direction

            gamma_plus = self.christoffel.compute_at_point(x_plus)
            gamma_minus = self.christoffel.compute_at_point(x_minus)

            d_gamma[j] = (gamma_plus - gamma_minus) / (2.0 * h)

        # Build Riemann tensor
        R = np.zeros((m, m, m, m))

        for l in range(m):
            for i in range(m):
                for j in range(m):
                    for k in range(m):
                        # Derivative terms
                        val = d_gamma[j, l, i, k] - d_gamma[k, l, i, j]

                        # Connection terms
                        for mm in range(m):
                            val += gamma[l, j, mm] * gamma[mm, i, k]
                            val -= gamma[l, k, mm] * gamma[mm, i, j]

                        R[l, i, j, k] = val

        # Enforce antisymmetry in last two indices: R^l_{ijk} = -R^l_{ikj}
        R_antisym = 0.5 * (R - np.swapaxes(R, 2, 3))

        return R_antisym

    def ricci_tensor_at(self, idx: int) -> np.ndarray:
        """Compute the Ricci tensor at a point.

        Ric_{ij} = R^k_{ikj} (contraction over first and third indices).

        Parameters
        ----------
        idx : int
            Index of the point.

        Returns
        -------
        np.ndarray
            Ricci tensor, shape ``(m, m)``. Symmetric.
        """
        R = self.riemann_tensor_at(idx)
        m = self.metric.intrinsic_dim_

        # Contract: Ric_{ij} = sum_k R^k_{ikj}
        Ric = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    Ric[i, j] += R[k, i, k, j]

        # Enforce symmetry
        Ric = 0.5 * (Ric + Ric.T)
        return Ric

    def scalar_curvature_at(self, idx: int) -> float:
        """Compute scalar curvature at a point.

        S = g^{ij} Ric_{ij}

        Parameters
        ----------
        idx : int
            Index of the point.

        Returns
        -------
        float
            Scalar curvature.
        """
        Ric = self.ricci_tensor_at(idx)
        g_inv = self.metric.metric_inverses_[idx]
        S = np.sum(g_inv * Ric)
        return float(S)

    def sectional_curvature_at(
        self,
        idx: int,
        v1: np.ndarray,
        v2: np.ndarray,
    ) -> float:
        """Compute sectional curvature of the plane spanned by v1, v2.

        K(v1, v2) = R_{ijkl} v1^i v2^j v1^k v2^l
                   / (g(v1,v1) g(v2,v2) - g(v1,v2)^2)

        where R_{ijkl} = g_{lm} R^m_{ijk}.

        Parameters
        ----------
        idx : int
            Index of the point.
        v1 : np.ndarray
            First tangent vector, shape ``(m,)``.
        v2 : np.ndarray
            Second tangent vector, shape ``(m,)``.

        Returns
        -------
        float
            Sectional curvature.
        """
        R_upper = self.riemann_tensor_at(idx)  # R^l_{ijk}
        g = self.metric.metric_tensors_[idx]

        # Lower the first index: R_{ijkl} = g_{lm} R^m_{ijk}
        # Actually: R_{lijk} = g_{lm} R^m_{ijk}
        # We want R_{ijkl} with specific contraction for sectional curvature
        R_lower = np.einsum("lm,mijk->lijk", g, R_upper)

        # Numerator: R_{ijkl} v1^i v2^j v1^k v2^l
        numerator = np.einsum("ijkl,i,j,k,l->", R_lower, v1, v2, v1, v2)

        # Denominator: g(v1,v1) g(v2,v2) - g(v1,v2)^2
        g_v1v1 = v1 @ g @ v1
        g_v2v2 = v2 @ g @ v2
        g_v1v2 = v1 @ g @ v2
        denominator = g_v1v1 * g_v2v2 - g_v1v2**2

        if abs(denominator) < 1e-12:
            return 0.0

        return float(numerator / denominator)

    def compute_all_scalar_curvatures(
        self,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute scalar curvature at every fitted point.

        Parameters
        ----------
        show_progress : bool
            Whether to show a progress bar.

        Returns
        -------
        np.ndarray
            Scalar curvatures, shape ``(N,)``.
        """
        N = len(self.metric.metric_tensors_)
        logger.info("Computing scalar curvatures at %d points...", N)

        iterator = range(N)
        if show_progress:
            iterator = tqdm(iterator, desc="Scalar curvatures", unit="pt")

        curvatures = np.array([self.scalar_curvature_at(i) for i in iterator])
        return curvatures

    def curvature_statistics(
        self,
        show_progress: bool = True,
    ) -> CurvatureStats:
        """Compute comprehensive curvature statistics.

        Parameters
        ----------
        show_progress : bool
            Whether to show progress bar during computation.

        Returns
        -------
        CurvatureStats
            Dictionary with keys:

            - ``"scalar_curvatures"``: ``(N,)`` array
            - ``"mean"``, ``"std"``, ``"median"``, ``"min"``, ``"max"``: float
            - ``"positive_fraction"``: fraction of points with S > 0
            - ``"curvature_entropy"``: entropy of the curvature distribution
        """
        curvatures = self.compute_all_scalar_curvatures(show_progress=show_progress)

        # Entropy of curvature distribution (via histogram)
        hist, _ = np.histogram(curvatures, bins=50, density=True)
        hist = hist[hist > 0]
        bin_width = (
            (curvatures.max() - curvatures.min()) / 50
            if curvatures.max() != curvatures.min()
            else 1.0
        )
        probs = hist * bin_width
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-12))

        return {
            "scalar_curvatures": curvatures,
            "mean": float(curvatures.mean()),
            "std": float(curvatures.std()),
            "median": float(np.median(curvatures)),
            "min": float(curvatures.min()),
            "max": float(curvatures.max()),
            "positive_fraction": float((curvatures > 0).mean()),
            "curvature_entropy": float(entropy),
        }
