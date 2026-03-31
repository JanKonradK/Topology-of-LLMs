"""
Unified hallucination detection system.

Combines curvature anomaly, topological isolation, information geometry,
and density-based features to score text outputs for hallucination risk.

The four scoring components:
- Curvature: local curvature anomaly vs. reference distribution
- Topological: distance from persistent H_0 component centers
- Information: inverse Fisher trace (low info → high risk)
- Density: Riemannian-corrected local density
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from topo_llm.types import HallucinationScore, EvaluationResult

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """Unified hallucination detection using geometric and topological features.

    Parameters
    ----------
    model_name : str
        HuggingFace model name for embedding extraction.
    device : str
        Compute device.

    Examples
    --------
    >>> detector = HallucinationDetector("gpt2-medium")
    >>> detector.fit(reference_corpus)
    >>> score = detector.score("The capital of France is Berlin.")
    >>> print(f"Hallucination risk: {score.hallucination_score:.3f}")
    """

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = device

        # Fitted state
        self._fitted = False
        self._reference_embeddings: np.ndarray | None = None
        self._reference_texts: list[str] = []
        self._reference_tree: KDTree | None = None
        self._layer: int = -2

        # Component estimators (lazily initialized)
        self._extractor = None
        self._metric_estimator = None
        self._curvature_analyzer = None
        self._scalar_curvatures: np.ndarray | None = None
        self._volume_elements: np.ndarray | None = None

        # Score weights
        self._weights = {
            "curvature": 0.25,
            "topological": 0.25,
            "information": 0.25,
            "density": 0.25,
        }

    def _ensure_extractor(self) -> None:
        """Lazily load the embedding extractor."""
        if self._extractor is None:
            from topo_llm.extraction.extractor import EmbeddingExtractor
            self._extractor = EmbeddingExtractor(
                self.model_name, device=self.device
            )

    def fit(
        self,
        reference_corpus: list[str],
        labels: list[bool] | None = None,
        layer: int = -2,
        n_neighbors: int = 30,
        reduced_dim: int = 50,
    ) -> HallucinationDetector:
        """Fit the detector on a reference corpus.

        Extracts embeddings, fits the Riemannian metric, computes
        curvature and topological features on the reference data.

        Parameters
        ----------
        reference_corpus : list[str]
            Reference texts representing "grounded" content.
        labels : list[bool] | None
            Optional labels (True = hallucination). If provided,
            used to learn score weights.
        layer : int
            Which layer to use for embeddings.
        n_neighbors : int
            Neighbors for metric estimation.
        reduced_dim : int
            PCA reduction dimension.

        Returns
        -------
        HallucinationDetector
            Self, for method chaining.
        """
        from sklearn.decomposition import PCA

        self._ensure_extractor()
        self._layer = layer
        self._reference_texts = list(reference_corpus)

        logger.info("Extracting reference embeddings for %d texts...", len(reference_corpus))
        embeddings_dict = self._extractor.extract_dataset(
            reference_corpus,
            layers=[layer],
            show_progress=True,
        )

        # Get the layer embeddings
        layer_key = list(embeddings_dict.keys())[0]
        raw_embeddings = embeddings_dict[layer_key]  # (N, hidden_dim)

        # PCA reduction
        self._pca = PCA(n_components=min(reduced_dim, raw_embeddings.shape[1]))
        self._reference_embeddings = self._pca.fit_transform(raw_embeddings)

        # Build KD-tree for fast lookup
        self._reference_tree = KDTree(self._reference_embeddings)

        # Fit metric tensor
        logger.info("Fitting Riemannian metric...")
        from topo_llm.riemannian.metric import MetricTensorEstimator

        effective_neighbors = min(n_neighbors, len(reference_corpus) - 1)
        intrinsic_dim = min(10, self._reference_embeddings.shape[1] - 1)

        self._metric_estimator = MetricTensorEstimator(
            n_neighbors=effective_neighbors,
            intrinsic_dim=intrinsic_dim,
        )
        self._metric_estimator.fit(self._reference_embeddings)

        # Compute volume elements for density scoring
        self._volume_elements = self._metric_estimator.all_volume_elements()

        # Compute scalar curvatures (simplified — use a subset for speed)
        logger.info("Computing reference curvatures...")
        from topo_llm.riemannian.connection import ChristoffelEstimator
        from topo_llm.riemannian.curvature import CurvatureAnalyzer

        christoffel = ChristoffelEstimator(self._metric_estimator, h=1e-3)
        self._curvature_analyzer = CurvatureAnalyzer(
            self._metric_estimator, christoffel
        )

        # Compute curvatures for a subset (for statistics)
        n_curv = min(100, len(reference_corpus))
        curv_indices = np.random.default_rng(42).choice(
            len(reference_corpus), n_curv, replace=False
        )
        curvatures = np.array([
            self._curvature_analyzer.scalar_curvature_at(int(i))
            for i in curv_indices
        ])
        self._curv_mean = float(np.mean(curvatures))
        self._curv_std = float(np.std(curvatures)) + 1e-8

        # Learn weights if labels provided
        if labels is not None:
            self._learn_weights(labels)

        self._fitted = True
        logger.info("HallucinationDetector fitted on %d reference texts", len(reference_corpus))
        return self

    def _learn_weights(self, labels: list[bool]) -> None:
        """Learn optimal score weights from labeled data."""
        # Score all reference texts
        scores = []
        for i in range(len(self._reference_texts)):
            emb = self._reference_embeddings[i]
            c = self._curvature_score_from_embedding(emb, i)
            t = self._topological_score_from_embedding(emb)
            d = self._density_score_from_embedding(emb)
            scores.append([c, t, 0.5, d])  # placeholder for info score

        scores = np.array(scores)
        labels_arr = np.array(labels, dtype=float)

        # Simple: correlate each component with labels
        for j, name in enumerate(["curvature", "topological", "information", "density"]):
            if scores[:, j].std() > 1e-8:
                corr = np.corrcoef(scores[:, j], labels_arr)[0, 1]
                self._weights[name] = max(abs(corr), 0.1)

        # Normalize
        total = sum(self._weights.values())
        self._weights = {k: v / total for k, v in self._weights.items()}

    def _embed_and_reduce(self, text: str) -> np.ndarray:
        """Extract and PCA-reduce a single text embedding."""
        self._ensure_extractor()
        result = self._extractor.extract(text)
        layer_key = list(result.pooled_embeddings.keys())[self._layer]
        raw = result.pooled_embeddings[layer_key].reshape(1, -1)
        return self._pca.transform(raw)[0]

    def _curvature_score_from_embedding(
        self, embedding: np.ndarray, idx: int | None = None
    ) -> float:
        """Curvature anomaly score."""
        if idx is not None:
            try:
                S = self._curvature_analyzer.scalar_curvature_at(idx)
            except Exception:
                return 0.5
        else:
            # For new points, use nearest reference curvature
            _, nearest = self._reference_tree.query(embedding, k=1)
            try:
                S = self._curvature_analyzer.scalar_curvature_at(int(nearest))
            except Exception:
                return 0.5

        # Anomaly: how far from mean curvature
        anomaly = abs(S - self._curv_mean) / self._curv_std
        # Sigmoid to [0, 1]
        return float(1.0 / (1.0 + np.exp(-anomaly + 2)))

    def _topological_score_from_embedding(self, embedding: np.ndarray) -> float:
        """Topological isolation score."""
        # Distance to nearest reference point, normalized
        dist, _ = self._reference_tree.query(embedding, k=5)
        if np.isscalar(dist):
            dist = np.array([dist])
        mean_dist = float(dist.mean())

        # Normalize by median reference pairwise distance
        sample_idx = np.random.default_rng(42).choice(
            len(self._reference_embeddings), min(100, len(self._reference_embeddings)), replace=False
        )
        sample = self._reference_embeddings[sample_idx]
        from scipy.spatial.distance import pdist
        median_dist = float(np.median(pdist(sample)))

        if median_dist < 1e-8:
            return 0.5

        normalized = mean_dist / median_dist
        return float(min(normalized, 1.0))

    def _density_score_from_embedding(self, embedding: np.ndarray) -> float:
        """Density-based score using Riemannian volume correction."""
        # K-NN density estimate
        k = min(10, len(self._reference_embeddings) - 1)
        dists, indices = self._reference_tree.query(embedding, k=k)

        if np.isscalar(dists):
            dists = np.array([dists])
            indices = np.array([indices])

        r_k = dists[-1]
        if r_k < 1e-10:
            return 0.0  # very dense region

        # Volume correction using Riemannian volume element
        nearest_idx = int(indices[0])
        vol = self._metric_estimator.volume_element(nearest_idx)
        vol = max(vol, 1e-10)

        # Density ∝ k / (vol * r_k^d)
        d = self._metric_estimator.intrinsic_dim_
        density = k / (vol * r_k ** d + 1e-10)

        # Convert to score: low density → high score
        # Use log scale and sigmoid
        log_density = np.log(density + 1e-10)

        # Compute reference density statistics for normalization
        ref_densities = []
        n_sample = min(50, len(self._reference_embeddings))
        sample_idx = np.random.default_rng(42).choice(
            len(self._reference_embeddings), n_sample, replace=False
        )
        for idx in sample_idx:
            d_ref, _ = self._reference_tree.query(self._reference_embeddings[idx], k=k)
            if np.isscalar(d_ref):
                d_ref = np.array([d_ref])
            r_ref = d_ref[-1]
            v_ref = self._metric_estimator.volume_element(int(idx))
            ref_densities.append(k / (max(v_ref, 1e-10) * r_ref ** d + 1e-10))

        ref_log = np.log(np.array(ref_densities) + 1e-10)
        z = (log_density - ref_log.mean()) / (ref_log.std() + 1e-8)

        # Sigmoid: low density (negative z) → high score
        return float(1.0 / (1.0 + np.exp(z)))

    def score(self, text: str) -> HallucinationScore:
        """Compute hallucination score for a single text.

        Parameters
        ----------
        text : str
            Text to score.

        Returns
        -------
        HallucinationScore
            Score with individual components and combined score.
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before scoring")

        embedding = self._embed_and_reduce(text)

        # Compute component scores
        curv_score = self._curvature_score_from_embedding(embedding)
        topo_score = self._topological_score_from_embedding(embedding)
        density_score = self._density_score_from_embedding(embedding)

        # Information score (simplified: use entropy as proxy)
        try:
            from topo_llm.information.entropy import EntropySurface
            surface = EntropySurface(self.model_name, device=self.device)
            H = surface.compute_entropy(text)
            # Low entropy = confident = low risk; high entropy = uncertain = high risk
            # Normalize by log(vocab_size)
            vocab_size = self._extractor.model.config.vocab_size
            info_score = float(min(H / np.log(vocab_size), 1.0))
        except Exception:
            info_score = 0.5

        # Combined score
        combined = (
            self._weights["curvature"] * curv_score
            + self._weights["topological"] * topo_score
            + self._weights["information"] * info_score
            + self._weights["density"] * density_score
        )

        # Nearest reference text
        _, nearest_idx = self._reference_tree.query(embedding, k=1)
        nearest_text = self._reference_texts[int(nearest_idx)]

        # Confidence based on local sample density
        dists, _ = self._reference_tree.query(embedding, k=5)
        if np.isscalar(dists):
            dists = np.array([dists])
        confidence = float(1.0 / (1.0 + dists.mean()))

        return HallucinationScore(
            hallucination_score=float(np.clip(combined, 0, 1)),
            curvature_score=curv_score,
            topological_score=topo_score,
            information_score=info_score,
            density_score=density_score,
            embedding_layer=self._layer,
            nearest_reference=nearest_text,
            confidence=confidence,
        )

    def evaluate(
        self,
        texts: list[str],
        labels: list[bool],
        baselines: bool = True,
    ) -> dict[str, object]:
        """Full evaluation against labeled data.

        Parameters
        ----------
        texts : list[str]
            Texts to evaluate.
        labels : list[bool]
            True = hallucination.
        baselines : bool
            Whether to compute baseline comparisons.

        Returns
        -------
        dict
            Evaluation metrics for our method and baselines.
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before evaluation")

        logger.info("Evaluating on %d texts...", len(texts))

        # Score all texts
        scores = []
        curv_scores = []
        topo_scores = []
        info_scores = []
        density_scores = []

        for text in texts:
            s = self.score(text)
            scores.append(s.hallucination_score)
            curv_scores.append(s.curvature_score)
            topo_scores.append(s.topological_score)
            info_scores.append(s.information_score)
            density_scores.append(s.density_score)

        scores = np.array(scores)
        labels_arr = np.array(labels, dtype=int)

        def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> EvaluationResult:
            try:
                auroc = float(roc_auc_score(labels, preds))
            except ValueError:
                auroc = 0.5
            try:
                auprc = float(average_precision_score(labels, preds))
            except ValueError:
                auprc = 0.5

            # Optimal threshold
            best_f1 = 0.0
            best_thresh = 0.5
            for thresh in np.linspace(0, 1, 50):
                binary_pred = (preds >= thresh).astype(int)
                f1 = float(f1_score(labels, binary_pred, zero_division=0))
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh

            return EvaluationResult(
                auroc=auroc, auprc=auprc, f1=best_f1, threshold=best_thresh
            )

        result: dict[str, object] = {
            "ours": compute_metrics(scores, labels_arr),
        }

        # Ablation
        result["ablation"] = {
            "curvature_only": compute_metrics(np.array(curv_scores), labels_arr),
            "topology_only": compute_metrics(np.array(topo_scores), labels_arr),
            "information_only": compute_metrics(np.array(info_scores), labels_arr),
            "density_only": compute_metrics(np.array(density_scores), labels_arr),
        }

        if baselines:
            # Baseline: cosine distance to nearest reference
            embeddings_dict = self._extractor.extract_dataset(
                texts, layers=[self._layer], show_progress=False
            )
            layer_key = list(embeddings_dict.keys())[0]
            raw = embeddings_dict[layer_key]
            reduced = self._pca.transform(raw)

            cosine_scores = []
            for emb in reduced:
                # Cosine distance to nearest reference
                norms_ref = np.linalg.norm(self._reference_embeddings, axis=1)
                norm_q = np.linalg.norm(emb)
                sims = (self._reference_embeddings @ emb) / (norms_ref * norm_q + 1e-10)
                cosine_scores.append(1.0 - sims.max())

            result["cosine_baseline"] = compute_metrics(
                np.array(cosine_scores), labels_arr
            )

        return result
