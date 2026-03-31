"""
Persistence landscapes for statistical analysis of topological features.

Persistence landscapes (Bubenik, 2015) are functional summaries of
persistence diagrams that live in a Banach space, enabling proper
statistical operations (means, distances, hypothesis tests).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class PersistenceLandscape:
    """Persistence landscape of a persistence diagram.

    For each persistence pair (b_i, d_i), defines the tent function:
        f_i(t) = max(0, min(t - b_i, d_i - t))

    The k-th landscape function λ_k(t) is the k-th largest
    value of {f_i(t)} at each point t.

    Parameters
    ----------
    diagram : np.ndarray
        Persistence diagram, shape ``(n, 2)`` with columns [birth, death].
    n_landscapes : int
        Number of landscape functions to compute.
    resolution : int
        Number of discretization points.

    Attributes
    ----------
    grid_ : np.ndarray
        Grid of t values, shape ``(resolution,)``.
    landscapes_ : np.ndarray
        Landscape functions, shape ``(n_landscapes, resolution)``.

    Examples
    --------
    >>> landscape = PersistenceLandscape(diagram, n_landscapes=5)
    >>> print(f"L^2 norm of first landscape: {landscape.norm(k=0):.4f}")
    """

    def __init__(
        self,
        diagram: np.ndarray,
        n_landscapes: int = 5,
        resolution: int = 1000,
    ) -> None:
        self.n_landscapes = n_landscapes
        self.resolution = resolution

        if len(diagram) == 0:
            self.grid_ = np.linspace(0, 1, resolution)
            self.landscapes_ = np.zeros((n_landscapes, resolution))
            return

        # Compute grid
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        t_min = births.min()
        t_max = deaths.max()
        self.grid_ = np.linspace(t_min, t_max, resolution)

        # Compute tent functions for all pairs at all grid points
        # f_i(t) = max(0, min(t - b_i, d_i - t))
        n_pairs = len(diagram)
        tent_values = np.zeros((n_pairs, resolution))

        for i in range(n_pairs):
            b, d = diagram[i]
            tent_values[i] = np.maximum(
                0.0,
                np.minimum(self.grid_ - b, d - self.grid_),
            )

        # The k-th landscape is the k-th largest tent value at each t
        # Sort descending at each grid point
        sorted_vals = np.sort(tent_values, axis=0)[::-1]  # (n_pairs, resolution)

        # Take top n_landscapes
        n_avail = min(n_landscapes, n_pairs)
        self.landscapes_ = np.zeros((n_landscapes, resolution))
        self.landscapes_[:n_avail] = sorted_vals[:n_avail]

    def integrate(self, k: int = 0) -> float:
        """Integral of the k-th landscape function.

        ∫ λ_k(t) dt, approximated by the trapezoidal rule.

        Parameters
        ----------
        k : int
            Landscape index (0-based).

        Returns
        -------
        float
            Integral value.
        """
        if k >= self.n_landscapes:
            return 0.0
        return float(np.trapezoid(self.landscapes_[k], self.grid_))

    def norm(self, k: int = 0, p: float = 2.0) -> float:
        """L^p norm of the k-th landscape function.

        (∫ |λ_k(t)|^p dt)^{1/p}

        Parameters
        ----------
        k : int
            Landscape index.
        p : float
            Norm order.

        Returns
        -------
        float
            L^p norm.
        """
        if k >= self.n_landscapes:
            return 0.0
        values = np.abs(self.landscapes_[k]) ** p
        integral = np.trapezoid(values, self.grid_)
        return float(integral ** (1.0 / p))

    @staticmethod
    def distance(
        landscape_a: PersistenceLandscape,
        landscape_b: PersistenceLandscape,
        p: float = 2.0,
    ) -> float:
        """L^p distance between two persistence landscapes.

        d_p(λ_a, λ_b) = (Σ_k ∫ |λ_a_k(t) - λ_b_k(t)|^p dt)^{1/p}

        Interpolates to a common grid if necessary.

        Parameters
        ----------
        landscape_a : PersistenceLandscape
            First landscape.
        landscape_b : PersistenceLandscape
            Second landscape.
        p : float
            Norm order.

        Returns
        -------
        float
            Distance between the landscapes.
        """
        # Interpolate to common grid
        t_min = min(landscape_a.grid_[0], landscape_b.grid_[0])
        t_max = max(landscape_a.grid_[-1], landscape_b.grid_[-1])
        resolution = max(landscape_a.resolution, landscape_b.resolution)
        common_grid = np.linspace(t_min, t_max, resolution)

        n_k = max(landscape_a.n_landscapes, landscape_b.n_landscapes)
        total = 0.0

        for k in range(n_k):
            # Interpolate landscape_a
            if k < landscape_a.n_landscapes:
                vals_a = np.interp(common_grid, landscape_a.grid_, landscape_a.landscapes_[k])
            else:
                vals_a = np.zeros(resolution)

            # Interpolate landscape_b
            if k < landscape_b.n_landscapes:
                vals_b = np.interp(common_grid, landscape_b.grid_, landscape_b.landscapes_[k])
            else:
                vals_b = np.zeros(resolution)

            diff_p = np.abs(vals_a - vals_b) ** p
            total += np.trapezoid(diff_p, common_grid)

        return float(total ** (1.0 / p))

    @staticmethod
    def mean_landscape(
        landscapes: list[PersistenceLandscape],
    ) -> PersistenceLandscape:
        """Compute the mean landscape (pointwise average).

        Well-defined since landscapes live in a Banach space.

        Parameters
        ----------
        landscapes : list[PersistenceLandscape]
            Landscapes to average.

        Returns
        -------
        PersistenceLandscape
            Mean landscape.
        """
        if not landscapes:
            raise ValueError("Cannot compute mean of empty list")

        # Common grid
        t_min = min(l.grid_[0] for l in landscapes)
        t_max = max(l.grid_[-1] for l in landscapes)
        resolution = max(l.resolution for l in landscapes)
        n_k = max(l.n_landscapes for l in landscapes)
        common_grid = np.linspace(t_min, t_max, resolution)

        # Average
        mean_vals = np.zeros((n_k, resolution))
        for l in landscapes:
            for k in range(n_k):
                if k < l.n_landscapes:
                    interp = np.interp(common_grid, l.grid_, l.landscapes_[k])
                    mean_vals[k] += interp

        mean_vals /= len(landscapes)

        # Create a new landscape with the mean values
        result = PersistenceLandscape.__new__(PersistenceLandscape)
        result.n_landscapes = n_k
        result.resolution = resolution
        result.grid_ = common_grid
        result.landscapes_ = mean_vals
        return result

    @staticmethod
    def permutation_test(
        landscapes_group_a: list[PersistenceLandscape],
        landscapes_group_b: list[PersistenceLandscape],
        n_permutations: int = 1000,
        seed: int = 42,
    ) -> dict[str, float]:
        """Test H_0: two groups have the same mean landscape.

        Test statistic: ||mean_a - mean_b||_2

        Parameters
        ----------
        landscapes_group_a : list[PersistenceLandscape]
            First group of landscapes.
        landscapes_group_b : list[PersistenceLandscape]
            Second group of landscapes.
        n_permutations : int
            Number of permutations for the null distribution.
        seed : int
            Random seed.

        Returns
        -------
        dict[str, float]
            ``{"test_statistic": float, "p_value": float}``
        """
        rng = np.random.default_rng(seed)

        mean_a = PersistenceLandscape.mean_landscape(landscapes_group_a)
        mean_b = PersistenceLandscape.mean_landscape(landscapes_group_b)
        observed = PersistenceLandscape.distance(mean_a, mean_b)

        # Pool all landscapes
        all_landscapes = landscapes_group_a + landscapes_group_b
        n_a = len(landscapes_group_a)
        n_total = len(all_landscapes)

        null_distribution = []
        for _ in range(n_permutations):
            perm = rng.permutation(n_total)
            group_a_perm = [all_landscapes[i] for i in perm[:n_a]]
            group_b_perm = [all_landscapes[i] for i in perm[n_a:]]

            mean_a_perm = PersistenceLandscape.mean_landscape(group_a_perm)
            mean_b_perm = PersistenceLandscape.mean_landscape(group_b_perm)
            null_stat = PersistenceLandscape.distance(mean_a_perm, mean_b_perm)
            null_distribution.append(null_stat)

        null_distribution = np.array(null_distribution)
        p_value = float((null_distribution >= observed).mean())

        return {"test_statistic": observed, "p_value": p_value}
