"""
Fixed-size feature vector extraction from persistence diagrams.

Provides multiple vectorization methods for feeding topological
signatures into machine learning pipelines.
"""

from __future__ import annotations

import logging

import numpy as np

from topo_llm.topology.homology import PersistentHomologyAnalyzer
from topo_llm.topology.landscapes import PersistenceLandscape

logger = logging.getLogger(__name__)


class TopologicalFeatures:
    """Extract fixed-size feature vectors from persistence diagrams.

    All methods are static — they take diagrams and return vectors.

    Examples
    --------
    >>> features = TopologicalFeatures.statistics_vector(diagrams)
    >>> features.shape
    (30,)
    """

    @staticmethod
    def statistics_vector(
        diagrams: list[np.ndarray],
    ) -> np.ndarray:
        """Compute a fixed-size statistics vector from persistence diagrams.

        For each dimension k in {0, 1, 2} (padded if fewer dimensions):
        - n_features: number of persistence pairs
        - mean_birth, std_birth
        - mean_death, std_death
        - mean_lifetime, std_lifetime, max_lifetime
        - persistence_entropy
        - n_significant (Otsu-filtered)

        Total: 10 features x 3 dimensions = 30-dimensional vector.

        Parameters
        ----------
        diagrams : list[np.ndarray]
            One diagram per homology dimension.

        Returns
        -------
        np.ndarray
            Feature vector of shape ``(30,)``.
        """
        analyzer = PersistentHomologyAnalyzer(diagrams)
        features = []

        for k in range(3):
            if k < len(diagrams) and len(diagrams[k]) > 0:
                dgm = diagrams[k]
                births = dgm[:, 0]
                deaths = dgm[:, 1]
                lifetimes = deaths - births

                features.extend(
                    [
                        float(len(dgm)),  # n_features
                        float(births.mean()),  # mean_birth
                        float(births.std()),  # std_birth
                        float(deaths.mean()),  # mean_death
                        float(deaths.std()),  # std_death
                        float(lifetimes.mean()),  # mean_lifetime
                        float(lifetimes.std()),  # std_lifetime
                        float(lifetimes.max()),  # max_lifetime
                        analyzer.persistence_entropy(k),  # entropy
                        float(len(analyzer.significant_features(k))),  # n_significant
                    ]
                )
            else:
                features.extend([0.0] * 10)

        return np.array(features, dtype=np.float64)

    @staticmethod
    def persistence_image(
        diagram: np.ndarray,
        resolution: tuple[int, int] = (20, 20),
        sigma: float | None = None,
        weight_fn: str = "linear",
    ) -> np.ndarray:
        """Compute persistence image (Adams et al., 2017).

        1. Transform (b, d) → (b, d-b) [birth, persistence]
        2. Weight each point by weight function
        3. Place weighted Gaussian at each point
        4. Discretize onto grid

        Parameters
        ----------
        diagram : np.ndarray
            Persistence diagram, shape ``(n, 2)``.
        resolution : tuple[int, int]
            Grid resolution (rows, cols).
        sigma : float | None
            Gaussian bandwidth. None auto-selects based on diagram spread.
        weight_fn : str
            Weight function: ``"linear"`` (w = persistence) or
            ``"persistence"`` (w = persistence^2).

        Returns
        -------
        np.ndarray
            Persistence image, shape ``resolution``.
        """
        if len(diagram) == 0:
            return np.zeros(resolution)

        # Transform to birth-persistence coordinates
        births = diagram[:, 0]
        persistences = diagram[:, 1] - diagram[:, 0]

        # Filter out zero-persistence
        mask = persistences > 1e-10
        births = births[mask]
        persistences = persistences[mask]

        if len(births) == 0:
            return np.zeros(resolution)

        # Weights
        if weight_fn == "linear":
            weights = persistences
        elif weight_fn == "persistence":
            weights = persistences**2
        else:
            raise ValueError(f"Unknown weight function: {weight_fn!r}")

        # Auto bandwidth
        if sigma is None:
            spread = max(births.max() - births.min(), persistences.max() - persistences.min())
            sigma = spread / (2 * max(resolution))
            sigma = max(sigma, 1e-6)

        # Grid
        b_min, b_max = births.min() - sigma, births.max() + sigma
        p_min, p_max = 0, persistences.max() + sigma

        b_grid = np.linspace(b_min, b_max, resolution[1])
        p_grid = np.linspace(p_min, p_max, resolution[0])
        B, P = np.meshgrid(b_grid, p_grid)

        # Compute image
        image = np.zeros(resolution)
        for b, p, w in zip(births, persistences, weights):
            gaussian = np.exp(-((B - b) ** 2 + (P - p) ** 2) / (2 * sigma**2))
            image += w * gaussian

        return image

    @staticmethod
    def combined_feature_vector(
        diagrams: list[np.ndarray],
        include_statistics: bool = True,
        include_landscapes: bool = True,
        include_images: bool = True,
        landscape_features: int = 20,
        image_resolution: tuple[int, int] = (10, 10),
    ) -> np.ndarray:
        """Combine multiple topological features into one vector.

        Parameters
        ----------
        diagrams : list[np.ndarray]
            Persistence diagrams per dimension.
        include_statistics : bool
            Include the 30-dim statistics vector.
        include_landscapes : bool
            Include landscape integrals and norms.
        include_images : bool
            Include flattened persistence images.
        landscape_features : int
            Resolution for landscape discretization.
        image_resolution : tuple[int, int]
            Resolution for persistence images.

        Returns
        -------
        np.ndarray
            Combined feature vector.
        """
        parts = []

        if include_statistics:
            parts.append(TopologicalFeatures.statistics_vector(diagrams))

        if include_landscapes:
            for k in range(min(3, len(diagrams))):
                landscape = PersistenceLandscape(
                    diagrams[k], n_landscapes=3, resolution=landscape_features
                )
                # Add integrals and norms for each landscape
                for i in range(3):
                    parts.append(
                        np.array(
                            [
                                landscape.integrate(i),
                                landscape.norm(i, p=1.0),
                                landscape.norm(i, p=2.0),
                            ]
                        )
                    )

        if include_images:
            for k in range(min(3, len(diagrams))):
                img = TopologicalFeatures.persistence_image(
                    diagrams[k], resolution=image_resolution
                )
                parts.append(img.ravel())

        if not parts:
            return np.array([])

        return np.concatenate(parts)
