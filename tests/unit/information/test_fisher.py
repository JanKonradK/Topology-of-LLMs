"""
Tests for FisherInformationEstimator.

All tests require model loading and are marked slow.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestFisherInformation:
    """Tests for Fisher information estimation."""

    @pytest.mark.slow
    def test_trace_positive(self, tiny_model_name: str) -> None:
        """Fisher trace should be positive."""
        from topo_llm.information.fisher import FisherInformationEstimator

        fisher = FisherInformationEstimator(tiny_model_name, device="cpu", n_samples=20)
        result = fisher.estimate_at("The capital of France is", top_k=50)
        assert result.fisher_trace >= 0

    @pytest.mark.slow
    def test_matrix_psd(self, tiny_model_name: str) -> None:
        """Fisher matrix should be positive semi-definite."""
        from topo_llm.information.fisher import FisherInformationEstimator

        fisher = FisherInformationEstimator(tiny_model_name, device="cpu", n_samples=20)
        result = fisher.estimate_at("Hello world", top_k=50)

        eigenvalues = np.linalg.eigvalsh(result.fisher_matrix)
        assert np.all(eigenvalues >= -1e-6), f"Negative eigenvalues: {eigenvalues[eigenvalues < 0]}"

    @pytest.mark.slow
    def test_entropy_nonnegative(self, tiny_model_name: str) -> None:
        """Entropy should be non-negative."""
        from topo_llm.information.fisher import FisherInformationEstimator

        fisher = FisherInformationEstimator(tiny_model_name, device="cpu", n_samples=10)
        result = fisher.estimate_at("Test prompt", top_k=50)
        assert result.entropy >= 0

    @pytest.mark.slow
    def test_effective_dimension_bounded(self, tiny_model_name: str) -> None:
        """Effective dimension should be between 0 and reduced_dim."""
        from topo_llm.information.fisher import FisherInformationEstimator

        fisher = FisherInformationEstimator(tiny_model_name, device="cpu", n_samples=20)
        result = fisher.estimate_at("A test", top_k=50)
        assert 0 <= result.effective_dimension <= result.fisher_matrix.shape[0]

    @pytest.mark.slow
    def test_batch_traces(self, tiny_model_name: str) -> None:
        """Batch trace computation should return correct shape."""
        from topo_llm.information.fisher import FisherInformationEstimator

        fisher = FisherInformationEstimator(tiny_model_name, device="cpu", n_samples=10)
        traces = fisher.fisher_trace_batch(
            ["Hello", "World", "Test"], top_k=50, show_progress=False
        )
        assert traces.shape == (3,)
        assert np.all(traces >= 0)
