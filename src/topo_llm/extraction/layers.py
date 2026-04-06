"""
Preliminary analysis tools for layer-wise embedding structure.

Provides intrinsic dimensionality estimation, inter-layer similarity
metrics, and anisotropy measurements — the first step in understanding
embedding geometry before full Riemannian analysis.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy.spatial import KDTree
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

from topo_llm.types import AnisotropyResult

logger = logging.getLogger(__name__)


class LayerAnalyzer:
    """Analyze structural properties of embedding layers.

    All methods are static — they operate directly on embedding matrices.
    Use these for preliminary analysis before committing to full
    Riemannian or topological computation.

    Examples
    --------
    >>> embeddings = np.random.randn(500, 768)
    >>> dim = LayerAnalyzer.intrinsic_dimensionality(embeddings, method="mle")
    >>> print(f"Intrinsic dimension: {dim:.1f}")
    """

    @staticmethod
    def intrinsic_dimensionality(
        embeddings: np.ndarray,
        method: Literal["mle", "twonn", "pca_95"] = "mle",
        k: int = 20,
    ) -> float:
        """Estimate the intrinsic dimensionality of a point cloud.

        Parameters
        ----------
        embeddings : np.ndarray
            Point cloud of shape ``(n_samples, ambient_dim)``.
        method : str
            Estimation method:

            - ``"mle"``: Maximum Likelihood Estimation
              (Levina & Bickel, 2004). Uses k-nearest neighbor
              distances to estimate local dimension.
            - ``"twonn"``: Two Nearest Neighbors (Facco et al., 2017).
              Uses ratio of 2nd to 1st nearest neighbor distances.
            - ``"pca_95"``: Number of PCA components explaining 95%
              of variance.
        k : int
            Number of neighbors for MLE method.

        Returns
        -------
        float
            Estimated intrinsic dimensionality.
        """
        if method == "mle":
            return LayerAnalyzer._id_mle(embeddings, k)
        elif method == "twonn":
            return LayerAnalyzer._id_twonn(embeddings)
        elif method == "pca_95":
            return LayerAnalyzer._id_pca(embeddings, threshold=0.95)
        else:
            raise ValueError(f"Unknown method: {method!r}")

    @staticmethod
    def _id_mle(embeddings: np.ndarray, k: int = 20) -> float:
        """MLE intrinsic dimension (Levina & Bickel, 2004).

        d_MLE = [1/N * sum_i 1/(k-1) * sum_{j=1}^{k-1} log(r_k(x_i)/r_j(x_i))]^{-1}

        where r_j(x_i) is the distance to the j-th nearest neighbor.
        """
        n_samples = embeddings.shape[0]
        k = min(k, n_samples - 1)

        tree = KDTree(embeddings)
        # Query k+1 because the first neighbor is the point itself
        distances, _ = tree.query(embeddings, k=k + 1)
        # Remove self-distance (column 0)
        distances = distances[:, 1:]  # shape: (n, k)

        # Avoid log(0) by clamping
        distances = np.maximum(distances, 1e-10)

        # For each point, compute the MLE contribution
        r_k = distances[:, -1]  # distance to k-th neighbor
        log_ratios = np.log(r_k[:, np.newaxis] / distances[:, :-1])  # (n, k-1)

        # Average over neighbors, then over points
        m_hat = log_ratios.mean(axis=1)  # (n,)

        # Filter out any invalid values
        valid = np.isfinite(m_hat) & (m_hat > 0)
        if valid.sum() == 0:
            return 0.0

        d_mle = 1.0 / m_hat[valid].mean()
        return float(d_mle)

    @staticmethod
    def _id_twonn(embeddings: np.ndarray) -> float:
        """Two Nearest Neighbors dimension estimate (Facco et al., 2017).

        mu_i = r_2(x_i) / r_1(x_i)
        d = N / sum_i log(mu_i)
        """
        tree = KDTree(embeddings)
        distances, _ = tree.query(embeddings, k=3)  # self + 2 neighbors
        r1 = distances[:, 1]  # nearest neighbor
        r2 = distances[:, 2]  # second nearest

        # Filter out zeros to avoid division by zero
        valid = r1 > 1e-10
        r1 = r1[valid]
        r2 = r2[valid]

        mu = r2 / r1
        log_mu = np.log(mu)

        # Filter invalid
        valid_log = np.isfinite(log_mu) & (log_mu > 0)
        if valid_log.sum() == 0:
            return 0.0

        d = len(log_mu[valid_log]) / log_mu[valid_log].sum()
        return float(d)

    @staticmethod
    def _id_pca(embeddings: np.ndarray, threshold: float = 0.95) -> float:
        """PCA-based dimension: number of components for threshold variance."""
        max_components = min(embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=max_components)
        pca.fit(embeddings)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, threshold) + 1)
        return float(n_components)

    @staticmethod
    def layer_similarity(
        embeddings_layer_i: np.ndarray,
        embeddings_layer_j: np.ndarray,
        method: Literal["cka", "procrustes", "cca"] = "cka",
    ) -> float:
        """Compute similarity between representations at two layers.

        Parameters
        ----------
        embeddings_layer_i : np.ndarray
            Embeddings from layer i, shape ``(n, d)``.
        embeddings_layer_j : np.ndarray
            Embeddings from layer j, shape ``(n, d)``.
        method : str
            Similarity metric:

            - ``"cka"``: Centered Kernel Alignment (Kornblith et al., 2019).
              ``CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)``
            - ``"procrustes"``: Procrustes distance after optimal alignment.
              Returns ``1 - distance`` so higher = more similar.
            - ``"cca"``: Mean correlation from Canonical Correlation Analysis.

        Returns
        -------
        float
            Similarity score in [0, 1]. Higher = more similar.
        """
        if method == "cka":
            return LayerAnalyzer._cka(embeddings_layer_i, embeddings_layer_j)
        elif method == "procrustes":
            return LayerAnalyzer._procrustes_similarity(embeddings_layer_i, embeddings_layer_j)
        elif method == "cca":
            return LayerAnalyzer._cca_similarity(embeddings_layer_i, embeddings_layer_j)
        else:
            raise ValueError(f"Unknown method: {method!r}")

    @staticmethod
    def _cka(X: np.ndarray, Y: np.ndarray) -> float:
        """Linear CKA (Kornblith et al., 2019).

        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

        Uses the linear kernel (dot product), which is equivalent to
        the feature-space CKA formulation.
        """
        # Center the matrices
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        # Cross-covariance
        YtX = Y.T @ X  # (d_y, d_x)
        XtX = X.T @ X  # (d_x, d_x)
        YtY = Y.T @ Y  # (d_y, d_y)

        numerator = np.linalg.norm(YtX, "fro") ** 2
        denominator = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")

        if denominator < 1e-10:
            return 0.0

        return float(numerator / denominator)

    @staticmethod
    def _procrustes_similarity(X: np.ndarray, Y: np.ndarray) -> float:
        """Procrustes similarity (1 - normalized Procrustes distance).

        Finds the optimal orthogonal transformation Q that minimizes
        ||X Q - Y||_F, then returns 1 - (distance / scale).
        """
        from scipy.linalg import orthogonal_procrustes

        # Center and normalize
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        norm_X = np.linalg.norm(X, "fro")
        norm_Y = np.linalg.norm(Y, "fro")

        if norm_X < 1e-10 or norm_Y < 1e-10:
            return 0.0

        X = X / norm_X
        Y = Y / norm_Y

        # Find optimal rotation
        # Reduce dimensions if they differ
        min_d = min(X.shape[1], Y.shape[1])
        X_r = X[:, :min_d]
        Y_r = Y[:, :min_d]

        R, scale = orthogonal_procrustes(X_r, Y_r)
        X_aligned = X_r @ R

        distance = np.linalg.norm(X_aligned - Y_r, "fro")
        # Normalized distance is in [0, 2] for unit-norm matrices
        similarity = max(0.0, 1.0 - distance / 2.0)
        return float(similarity)

    @staticmethod
    def _cca_similarity(X: np.ndarray, Y: np.ndarray) -> float:
        """CCA-based similarity: mean of canonical correlations."""
        # Reduce to manageable number of components
        n_components = min(20, X.shape[0] // 2, X.shape[1], Y.shape[1])
        if n_components < 1:
            return 0.0

        # PCA first to avoid singular matrices
        pca_x = PCA(n_components=n_components)
        pca_y = PCA(n_components=n_components)
        X_r = pca_x.fit_transform(X)
        Y_r = pca_y.fit_transform(Y)

        cca = CCA(n_components=n_components, max_iter=1000)
        try:
            cca.fit(X_r, Y_r)
            X_c, Y_c = cca.transform(X_r, Y_r)

            # Canonical correlations
            correlations = np.array(
                [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
            )
            # Filter NaN
            valid = np.isfinite(correlations)
            if valid.sum() == 0:
                return 0.0
            return float(np.abs(correlations[valid]).mean())
        except (ValueError, np.linalg.LinAlgError):
            logger.warning("CCA failed, returning 0.0")
            return 0.0

    @staticmethod
    def compute_anisotropy(embeddings: np.ndarray, seed: int = 42) -> AnisotropyResult:
        """Measure how non-uniform (anisotropic) the embedding space is.

        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix of shape ``(n_samples, hidden_dim)``.

        Returns
        -------
        AnisotropyResult
            Dictionary with:

            - ``"mean_cosine"`` (float): Average cosine similarity between
              random pairs. High values indicate anisotropy (embeddings
              cluster in a narrow cone).
            - ``"isotropy_score"`` (float): Approximate IsoScore — the
              partition function-based isotropy metric. 1.0 = perfectly
              isotropic, 0.0 = all vectors identical.
            - ``"explained_variance_ratio"`` (np.ndarray): PCA explained
              variance ratio for the first 10 components.
            - ``"effective_rank"`` (float): ``exp(entropy(normalized_singular_values))``.
              High value = more uniformly spread across dimensions.
        """
        n, d = embeddings.shape

        # ── Mean cosine similarity ────────────────────────────
        # Sample random pairs for efficiency
        n_pairs = min(5000, n * (n - 1) // 2)
        rng = np.random.default_rng(seed)
        idx_a = rng.integers(0, n, size=n_pairs)
        idx_b = rng.integers(0, n, size=n_pairs)
        # Avoid self-pairs
        mask = idx_a != idx_b
        idx_a = idx_a[mask]
        idx_b = idx_b[mask]

        norms_a = np.linalg.norm(embeddings[idx_a], axis=1, keepdims=True)
        norms_b = np.linalg.norm(embeddings[idx_b], axis=1, keepdims=True)
        norms_a = np.maximum(norms_a, 1e-10)
        norms_b = np.maximum(norms_b, 1e-10)

        cosines = np.sum(
            (embeddings[idx_a] / norms_a) * (embeddings[idx_b] / norms_b),
            axis=1,
        )
        mean_cosine = float(cosines.mean())

        # ── PCA explained variance ────────────────────────────
        n_components = min(10, n, d)
        pca = PCA(n_components=n_components)
        pca.fit(embeddings)
        explained_variance_ratio = pca.explained_variance_ratio_

        # ── Effective rank ────────────────────────────────────
        # Based on entropy of normalized singular values
        centered = embeddings - embeddings.mean(axis=0)
        _, sigma, _ = np.linalg.svd(centered, full_matrices=False)
        sigma = sigma[sigma > 1e-10]  # filter near-zero
        p = sigma / sigma.sum()
        entropy = -np.sum(p * np.log(p))
        effective_rank = float(np.exp(entropy))

        # ── Isotropy score (partition function approx) ────────
        # IsoScore: Z = sum_i exp(e_i · e_min_eigvec)
        # Simplified: use the ratio of min to max eigenvalue
        eigenvalues = pca.explained_variance_
        if eigenvalues[-1] > 0 and eigenvalues[0] > 0:
            isotropy_score = float(eigenvalues[-1] / eigenvalues[0])
        else:
            isotropy_score = 0.0

        return {
            "mean_cosine": mean_cosine,
            "isotropy_score": isotropy_score,
            "explained_variance_ratio": explained_variance_ratio,
            "effective_rank": effective_rank,
        }
