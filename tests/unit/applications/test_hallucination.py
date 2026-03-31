"""
Tests for HallucinationDetector.

Uses synthetic data to verify that the scoring pipeline works correctly
without requiring a real LLM.
"""

from __future__ import annotations

import numpy as np
import pytest

from topo_llm.types import HallucinationScore, EvaluationResult


class TestHallucinationScore:
    """Tests for the HallucinationScore dataclass."""

    def test_fields_present(self) -> None:
        """All fields should be accessible."""
        score = HallucinationScore(
            hallucination_score=0.7,
            curvature_score=0.6,
            topological_score=0.8,
            information_score=0.5,
            density_score=0.9,
            embedding_layer=-2,
            nearest_reference="The sky is blue.",
            confidence=0.8,
        )
        assert score.hallucination_score == 0.7
        assert score.curvature_score == 0.6
        assert score.embedding_layer == -2

    def test_score_range(self) -> None:
        """Score should be clipped to [0, 1]."""
        score = HallucinationScore(
            hallucination_score=0.5,
            curvature_score=0.3,
            topological_score=0.4,
            information_score=0.5,
            density_score=0.6,
            embedding_layer=-2,
            nearest_reference="test",
            confidence=0.7,
        )
        assert 0 <= score.hallucination_score <= 1


class TestEvaluationResult:
    """Tests for the EvaluationResult dataclass."""

    def test_fields(self) -> None:
        """All metric fields should be present."""
        result = EvaluationResult(
            auroc=0.85,
            auprc=0.80,
            f1=0.75,
            threshold=0.5,
        )
        assert result.auroc == 0.85
        assert result.f1 == 0.75


class TestHallucinationDetectorIntegration:
    """Integration tests requiring model loading."""

    @pytest.mark.slow
    def test_fit_and_score(self, tiny_model_name: str) -> None:
        """End-to-end: fit on reference, score a text."""
        from topo_llm.applications.hallucination import HallucinationDetector

        detector = HallucinationDetector(tiny_model_name, device="cpu")

        reference = [
            "The sky is blue.",
            "Water is wet.",
            "The sun rises in the east.",
            "Paris is in France.",
            "Dogs are mammals.",
            "Python is a programming language.",
            "The earth orbits the sun.",
            "Music can evoke emotions.",
            "Gravity pulls objects down.",
            "Plants need sunlight.",
        ]

        detector.fit(reference, layer=-1, n_neighbors=5, reduced_dim=10)

        score = detector.score("The sky is blue and clear.")
        assert isinstance(score, HallucinationScore)
        assert 0 <= score.hallucination_score <= 1
        assert score.confidence > 0
