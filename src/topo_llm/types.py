"""
Shared type definitions and dataclasses for topo-llm.

All public APIs exchange data through these types. Using dataclasses
rather than dicts ensures type safety and self-documenting interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Embedding Types
# ═══════════════════════════════════════════════════════════════

@dataclass
class EmbeddingResult:
    """Result of extracting embeddings from a single text input.

    Parameters
    ----------
    text : str
        The original input text.
    token_ids : np.ndarray
        Token IDs, shape (seq_len,).
    tokens : list[str]
        Decoded token strings.
    layer_embeddings : dict[int, np.ndarray]
        Per-layer token embeddings. Maps layer index to array of
        shape (seq_len, hidden_dim).
    pooled_embeddings : dict[int, np.ndarray]
        Per-layer pooled embeddings. Maps layer index to array of
        shape (hidden_dim,).
    model_name : str
        Name/path of the model used for extraction.
    """

    text: str
    token_ids: np.ndarray
    tokens: list[str]
    layer_embeddings: dict[int, np.ndarray]
    pooled_embeddings: dict[int, np.ndarray]
    model_name: str


@dataclass
class DatasetInfo:
    """Metadata about a generated or loaded dataset.

    Parameters
    ----------
    name : str
        Dataset identifier (e.g., "semantic_categories", "truthfulqa").
    n_samples : int
        Total number of text samples.
    n_categories : int
        Number of distinct categories/labels (0 if unlabeled).
    category_names : list[str]
        Names of the categories, empty if unlabeled.
    description : str
        Human-readable description of the dataset.
    """

    name: str
    n_samples: int
    n_categories: int = 0
    category_names: list[str] = field(default_factory=list)
    description: str = ""


# ═══════════════════════════════════════════════════════════════
# Geometry Types
# ═══════════════════════════════════════════════════════════════

@dataclass
class CurvatureResult:
    """Curvature computation results at a single point.

    Parameters
    ----------
    scalar_curvature : float
        Scalar curvature S = g^{ij} Ric_{ij}.
    ricci_tensor : np.ndarray
        Ricci tensor, shape (m, m).
    riemann_tensor : np.ndarray | None
        Full Riemann tensor, shape (m, m, m, m). None if not computed.
    sectional_curvatures : dict[tuple[int, int], float]
        Sectional curvatures for coordinate plane pairs.
    """

    scalar_curvature: float
    ricci_tensor: np.ndarray
    riemann_tensor: np.ndarray | None = None
    sectional_curvatures: dict[tuple[int, int], float] = field(default_factory=dict)


@dataclass
class GeodesicResult:
    """Result of solving the geodesic equation.

    Parameters
    ----------
    tangent_path : np.ndarray
        Path in tangent/local coordinates, shape (n_steps, m).
    ambient_path : np.ndarray
        Path mapped back to ambient space, shape (n_steps, D).
    velocities : np.ndarray
        Velocity vectors along the path, shape (n_steps, m).
    arc_length : float
        Total geodesic arc length.
    n_steps : int
        Number of integration steps taken.
    converged : bool
        Whether the solver converged (for BVP/shooting methods).
    """

    tangent_path: np.ndarray
    ambient_path: np.ndarray
    velocities: np.ndarray
    arc_length: float
    n_steps: int
    converged: bool = True


# ═══════════════════════════════════════════════════════════════
# Topology Types
# ═══════════════════════════════════════════════════════════════

@dataclass
class PersistenceResult:
    """Result of persistent homology computation.

    Parameters
    ----------
    diagrams : list[np.ndarray]
        Persistence diagrams. diagrams[k] has shape (n_k, 2) with
        columns [birth, death] for homology dimension k.
    max_edge_length : float
        Maximum filtration value used.
    n_points_used : int
        Number of points in the computation (after subsampling).
    computation_time : float
        Wall-clock time in seconds.
    backend : str
        Which TDA library was used.
    """

    diagrams: list[np.ndarray]
    max_edge_length: float
    n_points_used: int
    computation_time: float
    backend: str = "ripser"


@dataclass
class TopologicalSummary:
    """Summary statistics of persistent homology.

    Parameters
    ----------
    n_features : dict[int, int]
        Number of features per homology dimension.
    n_significant : dict[int, int]
        Number of significant features per dimension.
    max_persistence : dict[int, float]
        Maximum persistence per dimension.
    mean_persistence : dict[int, float]
        Mean persistence per dimension.
    persistence_entropy : dict[int, float]
        Persistence entropy per dimension.
    total_persistence : dict[int, float]
        Total persistence (sum of lifetimes) per dimension.
    """

    n_features: dict[int, int]
    n_significant: dict[int, int]
    max_persistence: dict[int, float]
    mean_persistence: dict[int, float]
    persistence_entropy: dict[int, float]
    total_persistence: dict[int, float]


# ═══════════════════════════════════════════════════════════════
# Information Geometry Types
# ═══════════════════════════════════════════════════════════════

@dataclass
class FisherResult:
    """Result of Fisher Information estimation.

    Parameters
    ----------
    fisher_matrix : np.ndarray
        Fisher Information Matrix, shape (d, d) or (k, k) reduced.
    fisher_trace : float
        Trace of the Fisher matrix: tr(G_F).
    fisher_eigenvalues : np.ndarray
        Eigenvalues of G_F in descending order.
    effective_dimension : float
        Participation ratio of eigenvalues.
    entropy : float
        Shannon entropy of the output distribution.
    top_k_probs : np.ndarray
        Probability distribution over top-k tokens.
    """

    fisher_matrix: np.ndarray
    fisher_trace: float
    fisher_eigenvalues: np.ndarray
    effective_dimension: float
    entropy: float
    top_k_probs: np.ndarray


# ═══════════════════════════════════════════════════════════════
# Application Types
# ═══════════════════════════════════════════════════════════════

@dataclass
class HallucinationScore:
    """Hallucination detection score for a single text.

    Parameters
    ----------
    hallucination_score : float
        Combined score in [0, 1]. 0 = safe, 1 = likely hallucination.
    curvature_score : float
        Curvature anomaly component.
    topological_score : float
        Topological isolation component.
    information_score : float
        Information geometry component.
    density_score : float
        Riemannian-corrected density component.
    embedding_layer : int
        Which layer was used for analysis.
    nearest_reference : str
        Nearest text in the reference corpus.
    confidence : float
        Confidence in the score (based on local sample density).
    """

    hallucination_score: float
    curvature_score: float
    topological_score: float
    information_score: float
    density_score: float
    embedding_layer: int
    nearest_reference: str
    confidence: float


@dataclass
class EvaluationResult:
    """Evaluation metrics for hallucination detection.

    Parameters
    ----------
    auroc : float
        Area Under ROC Curve.
    auprc : float
        Area Under Precision-Recall Curve.
    f1 : float
        F1 score at optimal threshold.
    threshold : float
        Optimal threshold for binary classification.
    """

    auroc: float
    auprc: float
    f1: float
    threshold: float


# ═══════════════════════════════════════════════════════════════
# Protocols
# ═══════════════════════════════════════════════════════════════

@runtime_checkable
class MetricField(Protocol):
    """Protocol for a smooth Riemannian metric field.

    Any object implementing this protocol can be used where a metric
    tensor field is needed (e.g., in Christoffel/curvature computation).
    """

    def evaluate(self, point: np.ndarray) -> np.ndarray:
        """Evaluate the metric tensor at a point.

        Parameters
        ----------
        point : np.ndarray
            Point in the ambient space, shape (D,).

        Returns
        -------
        np.ndarray
            Metric tensor, shape (m, m).
        """
        ...

    def evaluate_inverse(self, point: np.ndarray) -> np.ndarray:
        """Evaluate the inverse metric tensor at a point.

        Parameters
        ----------
        point : np.ndarray
            Point in the ambient space, shape (D,).

        Returns
        -------
        np.ndarray
            Inverse metric tensor, shape (m, m).
        """
        ...
