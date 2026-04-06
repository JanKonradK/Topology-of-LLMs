"""Tests for shared type definitions and dataclasses."""

from __future__ import annotations

import numpy as np

from topo_llm.types import (
    AnisotropyResult,
    ComparisonResult,
    CurvatureResult,
    CurvatureStats,
    DatasetInfo,
    DeviceInfo,
    EmbeddingResult,
    EvaluationResult,
    GeodesicResult,
    HallucinationScore,
    MetricField,
    PersistenceResult,
    RetrievalResult,
)


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""

    def test_basic_construction(self):
        result = EmbeddingResult(
            text="hello",
            token_ids=np.array([1, 2, 3]),
            tokens=["hel", "lo"],
            layer_embeddings={0: np.zeros((2, 10))},
            pooled_embeddings={0: np.zeros(10)},
            model_name="gpt2",
        )
        assert result.text == "hello"
        assert result.model_name == "gpt2"
        assert result.token_ids.shape == (3,)

    def test_layer_embeddings_dict(self):
        emb = {0: np.ones((5, 768)), 6: np.ones((5, 768))}
        result = EmbeddingResult(
            text="test",
            token_ids=np.array([1]),
            tokens=["test"],
            layer_embeddings=emb,
            pooled_embeddings={},
            model_name="gpt2",
        )
        assert 0 in result.layer_embeddings
        assert 6 in result.layer_embeddings


class TestDatasetInfo:
    """Test DatasetInfo dataclass."""

    def test_minimal_construction(self):
        info = DatasetInfo(name="test", n_samples=100)
        assert info.n_categories == 0
        assert info.category_names == []
        assert info.description == ""

    def test_full_construction(self):
        info = DatasetInfo(
            name="semantic",
            n_samples=500,
            n_categories=5,
            category_names=["a", "b", "c", "d", "e"],
            description="Test dataset",
        )
        assert info.n_categories == 5


class TestCurvatureResult:
    """Test CurvatureResult dataclass."""

    def test_construction(self):
        result = CurvatureResult(
            scalar_curvature=2.0,
            ricci_tensor=np.eye(3),
        )
        assert result.scalar_curvature == 2.0
        assert result.riemann_tensor is None
        assert result.sectional_curvatures == {}


class TestGeodesicResult:
    """Test GeodesicResult dataclass."""

    def test_construction(self):
        result = GeodesicResult(
            tangent_path=np.zeros((10, 3)),
            ambient_path=np.zeros((10, 100)),
            velocities=np.zeros((10, 3)),
            arc_length=1.5,
            n_steps=10,
        )
        assert result.converged is True
        assert result.arc_length == 1.5


class TestPersistenceResult:
    """Test PersistenceResult dataclass."""

    def test_construction_with_empty_diagrams(self):
        result = PersistenceResult(
            diagrams=[np.array([]).reshape(0, 2)],
            max_edge_length=2.0,
            n_points_used=100,
            computation_time=0.5,
        )
        assert result.backend == "ripser"
        assert len(result.diagrams) == 1

    def test_construction_with_data(self):
        dgm = np.array([[0.0, 0.5], [0.1, 0.8]])
        result = PersistenceResult(
            diagrams=[dgm],
            max_edge_length=1.0,
            n_points_used=50,
            computation_time=0.1,
            backend="gudhi",
        )
        assert result.diagrams[0].shape == (2, 2)


class TestHallucinationScore:
    """Test HallucinationScore dataclass."""

    def test_construction(self):
        score = HallucinationScore(
            hallucination_score=0.75,
            curvature_score=0.8,
            topological_score=0.6,
            information_score=0.7,
            density_score=0.9,
            embedding_layer=-2,
            nearest_reference="reference text",
            confidence=0.85,
        )
        assert 0 <= score.hallucination_score <= 1


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_construction(self):
        result = EvaluationResult(auroc=0.95, auprc=0.90, f1=0.85, threshold=0.5)
        assert result.auroc == 0.95


class TestMetricFieldProtocol:
    """Test MetricField protocol compliance."""

    def test_conforming_class(self):
        class MockMetric:
            def evaluate(self, point: np.ndarray) -> np.ndarray:
                return np.eye(3)

            def evaluate_inverse(self, point: np.ndarray) -> np.ndarray:
                return np.eye(3)

        mock = MockMetric()
        assert isinstance(mock, MetricField)

    def test_non_conforming_class(self):
        class NotAMetric:
            pass

        assert not isinstance(NotAMetric(), MetricField)


class TestTypedDicts:
    """Test TypedDict definitions."""

    def test_device_info_typed_dict(self):
        info: DeviceInfo = {
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": None,
            "mps_available": False,
            "selected": "cpu",
        }
        assert info["selected"] == "cpu"

    def test_anisotropy_result(self):
        result: AnisotropyResult = {
            "mean_cosine": 0.5,
            "isotropy_score": 0.3,
            "explained_variance_ratio": np.array([0.5, 0.3, 0.2]),
            "effective_rank": 2.5,
        }
        assert result["mean_cosine"] == 0.5

    def test_curvature_stats(self):
        stats: CurvatureStats = {
            "scalar_curvatures": np.array([1.0, 2.0]),
            "mean": 1.5,
            "std": 0.5,
            "median": 1.5,
            "min": 1.0,
            "max": 2.0,
            "positive_fraction": 1.0,
            "curvature_entropy": 0.5,
        }
        assert stats["mean"] == 1.5

    def test_comparison_result(self):
        result: ComparisonResult = {
            "euclidean_neighbors": [1, 2, 3],
            "cosine_neighbors": [1, 3, 2],
            "geodesic_neighbors": [2, 1, 3],
            "rank_correlation_euclid_geo": 0.8,
            "rank_correlation_cosine_geo": 0.7,
        }
        assert len(result["euclidean_neighbors"]) == 3

    def test_retrieval_result(self):
        result: RetrievalResult = {
            "text": "hello world",
            "distance": 0.5,
            "rank": 1,
        }
        assert result["rank"] == 1
