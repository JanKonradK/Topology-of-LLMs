"""
Geodesic-distance-based document retrieval.

A drop-in replacement for cosine-similarity-based retrieval that uses
geodesic distance on the embedding manifold. Accounts for the curved
geometry of the embedding space.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

from topo_llm.types import RetrievalResult

logger = logging.getLogger(__name__)


class GeodesicRetrieval:
    """Manifold-aware document retrieval using geodesic distance.

    Parameters
    ----------
    model_name : str
        HuggingFace model name.
    device : str
        Compute device.

    Examples
    --------
    >>> retriever = GeodesicRetrieval("gpt2-medium")
    >>> retriever.index(documents)
    >>> results = retriever.query("What is machine learning?", k=5)
    """

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = device

        self._fitted = False
        self._documents: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._tree: KDTree | None = None
        self._pca: PCA | None = None
        self._extractor = None
        self._metric_estimator = None
        self._geodesic_solver = None
        self._layer = -2

    def _ensure_extractor(self) -> None:
        if self._extractor is None:
            from topo_llm.extraction.extractor import EmbeddingExtractor

            self._extractor = EmbeddingExtractor(self.model_name, device=self.device)

    def index(
        self,
        documents: list[str],
        layer: int = -2,
        reduced_dim: int = 50,
        n_neighbors: int = 30,
    ) -> GeodesicRetrieval:
        """Build the manifold-aware retrieval index.

        Parameters
        ----------
        documents : list[str]
            Documents to index.
        layer : int
            Embedding layer to use.
        reduced_dim : int
            PCA reduction dimension.
        n_neighbors : int
            Neighbors for metric estimation.

        Returns
        -------
        GeodesicRetrieval
            Self, for method chaining.
        """
        self._ensure_extractor()
        self._documents = list(documents)
        self._layer = layer

        logger.info("Indexing %d documents...", len(documents))

        # Extract embeddings
        embeddings_dict = self._extractor.extract_dataset(
            documents, layers=[layer], show_progress=True
        )
        layer_key = list(embeddings_dict.keys())[0]
        raw = embeddings_dict[layer_key]

        # PCA reduction
        self._pca = PCA(n_components=min(reduced_dim, raw.shape[1]))
        self._embeddings = self._pca.fit_transform(raw)

        # Build KD-tree
        self._tree = KDTree(self._embeddings)

        # Fit Riemannian metric
        logger.info("Fitting manifold geometry...")
        from topo_llm.riemannian.connection import ChristoffelEstimator
        from topo_llm.riemannian.geodesic import GeodesicSolver
        from topo_llm.riemannian.metric import MetricTensorEstimator

        effective_neighbors = min(n_neighbors, len(documents) - 1)
        intrinsic_dim = min(10, self._embeddings.shape[1] - 1)

        self._metric_estimator = MetricTensorEstimator(
            n_neighbors=effective_neighbors,
            intrinsic_dim=intrinsic_dim,
        )
        self._metric_estimator.fit(self._embeddings)

        christoffel = ChristoffelEstimator(self._metric_estimator, h=1e-3)
        self._geodesic_solver = GeodesicSolver(
            self._metric_estimator,
            christoffel,
            dt=0.01,
            max_steps=200,
        )

        self._fitted = True
        logger.info("Indexing complete")
        return self

    def _embed_query(self, query_text: str) -> np.ndarray:
        """Embed and reduce a query text."""
        self._ensure_extractor()
        result = self._extractor.extract(query_text)
        layer_key = list(result.pooled_embeddings.keys())[self._layer]
        raw = result.pooled_embeddings[layer_key].reshape(1, -1)
        return self._pca.transform(raw)[0]

    def query(
        self,
        query_text: str,
        k: int = 5,
        method: str = "geodesic",
    ) -> list[RetrievalResult]:
        """Retrieve nearest documents.

        Parameters
        ----------
        query_text : str
            Query text.
        k : int
            Number of results.
        method : str
            Distance metric: ``"geodesic"``, ``"cosine"``, ``"euclidean"``.

        Returns
        -------
        list[dict]
            Results with ``"text"``, ``"distance"``, ``"rank"`` keys.
        """
        if not self._fitted:
            raise RuntimeError("Must call index() first")
        if not query_text or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")

        query_emb = self._embed_query(query_text)
        k = min(k, len(self._documents))

        if method == "euclidean":
            dists, indices = self._tree.query(query_emb, k=k)
            if np.isscalar(dists):
                dists = np.array([dists])
                indices = np.array([indices])

            return [
                {"text": self._documents[int(idx)], "distance": float(d), "rank": i + 1}
                for i, (d, idx) in enumerate(zip(dists, indices))
            ]

        elif method == "cosine":
            norms = np.linalg.norm(self._embeddings, axis=1)
            norm_q = np.linalg.norm(query_emb)
            sims = (self._embeddings @ query_emb) / (norms * norm_q + 1e-10)
            top_k = np.argsort(-sims)[:k]

            return [
                {
                    "text": self._documents[int(idx)],
                    "distance": float(1.0 - sims[idx]),
                    "rank": i + 1,
                }
                for i, idx in enumerate(top_k)
            ]

        elif method == "geodesic":
            # Pre-filter with Euclidean, refine with geodesic
            candidates = min(k * 5, len(self._documents))
            _, cand_indices = self._tree.query(query_emb, k=candidates)
            if np.isscalar(cand_indices):
                cand_indices = np.array([cand_indices])

            # Find nearest fitted point to query
            _, nearest_idx = self._tree.query(query_emb, k=1)
            nearest_idx = int(nearest_idx)

            geo_dists = []
            for idx in cand_indices:
                idx = int(idx)
                try:
                    d = self._geodesic_solver.geodesic_distance(nearest_idx, idx, n_shooting=3)
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    d = float(np.linalg.norm(self._embeddings[nearest_idx] - self._embeddings[idx]))
                geo_dists.append((idx, d))

            geo_dists.sort(key=lambda x: x[1])
            top_k_geo = geo_dists[:k]

            return [
                {
                    "text": self._documents[idx],
                    "distance": float(d),
                    "rank": i + 1,
                }
                for i, (idx, d) in enumerate(top_k_geo)
            ]

        else:
            raise ValueError(f"Unknown method: {method!r}")

    def benchmark(
        self,
        queries: list[str],
        relevant_docs: list[list[int]],
        k_values: list[int] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compare retrieval quality across distance metrics.

        Parameters
        ----------
        queries : list[str]
            Query texts.
        relevant_docs : list[list[int]]
            Ground truth relevant document indices per query.
        k_values : list[int] | None
            K values for recall@K. Default: [1, 5, 10, 20].

        Returns
        -------
        dict
            Nested dict: ``{method: {recall@k: float}}``.
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]

        results = {}
        for method in ["geodesic", "cosine", "euclidean"]:
            recalls = {}
            for k in k_values:
                if k > len(self._documents):
                    continue

                total_recall = 0.0
                for query, relevant in zip(queries, relevant_docs):
                    retrieved = self.query(query, k=k, method=method)
                    retrieved_idx = set()
                    for r in retrieved:
                        # Find original index
                        idx = self._documents.index(r["text"])
                        retrieved_idx.add(idx)

                    if len(relevant) > 0:
                        recall = len(retrieved_idx & set(relevant)) / len(relevant)
                        total_recall += recall

                recalls[f"recall@{k}"] = total_recall / max(len(queries), 1)

            results[method] = recalls

        return results
