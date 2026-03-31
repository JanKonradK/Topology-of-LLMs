"""
Persistent homology analysis and feature extraction from persistence diagrams.

Provides Betti number computation, persistence entropy, significance
filtering, and comprehensive summary statistics.
"""

from __future__ import annotations

import logging

import numpy as np

from topo_llm.types import TopologicalSummary

logger = logging.getLogger(__name__)


class PersistentHomologyAnalyzer:
    """Analyze persistence diagrams from persistent homology computation.

    Parameters
    ----------
    diagrams : list[np.ndarray]
        Persistence diagrams. ``diagrams[k]`` has shape ``(n_k, 2)``
        with columns ``[birth, death]`` for homology dimension k.
    max_edge_length : float
        Maximum filtration value (used to replace infinite death times).

    Examples
    --------
    >>> analyzer = PersistentHomologyAnalyzer(result.diagrams, result.max_edge_length)
    >>> betti = analyzer.betti_numbers(epsilon=0.5)
    >>> print(f"H_0 = {betti[0]}, H_1 = {betti[1]}")
    """

    def __init__(
        self,
        diagrams: list[np.ndarray],
        max_edge_length: float = 1.0,
    ) -> None:
        self.max_edge_length = max_edge_length

        # Process diagrams: replace inf with max_edge_length
        self.diagrams: list[np.ndarray] = []
        for dgm in diagrams:
            dgm = dgm.copy()
            if len(dgm) > 0:
                dgm[~np.isfinite(dgm[:, 1]), 1] = max_edge_length
            self.diagrams.append(dgm)

    @property
    def max_dimension(self) -> int:
        """Maximum homology dimension available."""
        return len(self.diagrams) - 1

    def lifetimes(self, dimension: int) -> np.ndarray:
        """Compute lifetimes (persistence) for a given dimension.

        Parameters
        ----------
        dimension : int
            Homology dimension.

        Returns
        -------
        np.ndarray
            Array of lifetimes ``death - birth``.
        """
        dgm = self.diagrams[dimension]
        if len(dgm) == 0:
            return np.array([])
        return dgm[:, 1] - dgm[:, 0]

    def betti_numbers(self, epsilon: float) -> dict[int, int]:
        """Compute Betti numbers at a given scale.

        β_k(ε) = |{(b,d) ∈ PD_k : b ≤ ε < d}|

        Parameters
        ----------
        epsilon : float
            Filtration scale.

        Returns
        -------
        dict[int, int]
            Maps dimension k to Betti number β_k.
        """
        result = {}
        for k, dgm in enumerate(self.diagrams):
            if len(dgm) == 0:
                result[k] = 0
            else:
                alive = (dgm[:, 0] <= epsilon) & (dgm[:, 1] > epsilon)
                result[k] = int(alive.sum())
        return result

    def betti_curve(
        self,
        n_steps: int = 100,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Compute Betti numbers across all scales.

        Parameters
        ----------
        n_steps : int
            Number of evenly spaced filtration values.

        Returns
        -------
        dict[int, tuple[np.ndarray, np.ndarray]]
            Maps dimension k to ``(epsilons, betti_values)`` arrays.
        """
        # Determine scale range
        all_vals = np.concatenate([dgm.ravel() for dgm in self.diagrams if len(dgm) > 0])
        if len(all_vals) == 0:
            eps = np.linspace(0, 1, n_steps)
            return {k: (eps, np.zeros(n_steps, dtype=int)) for k in range(len(self.diagrams))}

        eps = np.linspace(all_vals.min(), all_vals.max(), n_steps)

        result = {}
        for k in range(len(self.diagrams)):
            betti_vals = np.array([self.betti_numbers(e)[k] for e in eps])
            result[k] = (eps, betti_vals)
        return result

    def persistence_entropy(self, dimension: int = 1) -> float:
        """Compute persistence entropy for a given dimension.

        E_k = -Σ_i (l_i / L) log(l_i / L)

        where l_i = death_i - birth_i and L = Σ l_i.

        Parameters
        ----------
        dimension : int
            Homology dimension.

        Returns
        -------
        float
            Persistence entropy. High = many features of similar
            persistence. Low = dominated by few long-lived features.
        """
        lt = self.lifetimes(dimension)
        lt = lt[lt > 0]

        if len(lt) == 0:
            return 0.0

        L = lt.sum()
        if L <= 0:
            return 0.0

        p = lt / L
        return float(-np.sum(p * np.log(p + 1e-12)))

    def significant_features(
        self,
        dimension: int,
        threshold: str = "otsu",
    ) -> np.ndarray:
        """Filter persistence diagram to significant features.

        Parameters
        ----------
        dimension : int
            Homology dimension.
        threshold : str
            Thresholding method:

            - ``"otsu"``: Otsu's method on lifetimes.
            - ``"percentile_90"``: Keep features with lifetime > 90th percentile.
            - ``"mean_lifetime"``: Keep features with lifetime > mean.

        Returns
        -------
        np.ndarray
            Filtered diagram, shape ``(n_sig, 2)``.
        """
        dgm = self.diagrams[dimension]
        if len(dgm) == 0:
            return dgm

        lt = dgm[:, 1] - dgm[:, 0]

        if threshold == "otsu":
            thresh_val = self._otsu_threshold(lt)
        elif threshold == "percentile_90":
            thresh_val = np.percentile(lt, 90)
        elif threshold == "mean_lifetime":
            thresh_val = lt.mean()
        else:
            raise ValueError(f"Unknown threshold method: {threshold!r}")

        mask = lt > thresh_val
        return dgm[mask]

    @staticmethod
    def _otsu_threshold(values: np.ndarray) -> float:
        """Otsu's method for automatic thresholding.

        Finds the threshold that minimizes intra-class variance
        of the two groups (below/above threshold).
        """
        if len(values) < 2:
            return 0.0

        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        best_thresh = sorted_vals[0]
        best_score = float("inf")

        for i in range(1, n):
            group1 = sorted_vals[:i]
            group2 = sorted_vals[i:]

            w1 = len(group1) / n
            w2 = len(group2) / n

            var1 = group1.var() if len(group1) > 1 else 0
            var2 = group2.var() if len(group2) > 1 else 0

            score = w1 * var1 + w2 * var2
            if score < best_score:
                best_score = score
                best_thresh = sorted_vals[i]

        return float(best_thresh)

    def summary_statistics(self) -> TopologicalSummary:
        """Compute comprehensive summary of persistent homology.

        Returns
        -------
        TopologicalSummary
            Summary with feature counts, persistence statistics,
            and entropy for each dimension.
        """
        n_features = {}
        n_significant = {}
        max_persistence = {}
        mean_persistence = {}
        persistence_entropy = {}
        total_persistence = {}

        for k in range(len(self.diagrams)):
            lt = self.lifetimes(k)

            n_features[k] = len(lt)

            sig = self.significant_features(k)
            n_significant[k] = len(sig)

            if len(lt) > 0:
                max_persistence[k] = float(lt.max())
                mean_persistence[k] = float(lt.mean())
                total_persistence[k] = float(lt.sum())
            else:
                max_persistence[k] = 0.0
                mean_persistence[k] = 0.0
                total_persistence[k] = 0.0

            persistence_entropy[k] = self.persistence_entropy(k)

        return TopologicalSummary(
            n_features=n_features,
            n_significant=n_significant,
            max_persistence=max_persistence,
            mean_persistence=mean_persistence,
            persistence_entropy=persistence_entropy,
            total_persistence=total_persistence,
        )
