"""
Tests for KLGeometry.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestKLGeometry:
    """Tests for KL divergence and JSD."""

    @pytest.mark.slow
    def test_kl_self_zero(self, tiny_model_name: str) -> None:
        """KL(p || p) should be 0."""
        from topo_llm.information.divergence import KLGeometry

        kl = KLGeometry(tiny_model_name, device="cpu")
        d = kl.kl_divergence("Hello world", "Hello world")
        assert abs(d) < 0.01, f"KL(p||p) = {d}"

    @pytest.mark.slow
    def test_kl_nonnegative(self, tiny_model_name: str) -> None:
        """KL divergence is non-negative."""
        from topo_llm.information.divergence import KLGeometry

        kl = KLGeometry(tiny_model_name, device="cpu")
        d = kl.kl_divergence("The sky is", "The ocean is")
        assert d >= 0, f"Negative KL: {d}"

    @pytest.mark.slow
    def test_jsd_symmetric(self, tiny_model_name: str) -> None:
        """JSD(p, q) = JSD(q, p)."""
        from topo_llm.information.divergence import KLGeometry

        kl = KLGeometry(tiny_model_name, device="cpu")
        d1 = kl.symmetric_kl("The sky is", "The ocean is")
        d2 = kl.symmetric_kl("The ocean is", "The sky is")
        assert abs(d1 - d2) < 0.01, f"JSD not symmetric: {d1} vs {d2}"

    @pytest.mark.slow
    def test_jsd_nonnegative(self, tiny_model_name: str) -> None:
        """JSD is non-negative."""
        from topo_llm.information.divergence import KLGeometry

        kl = KLGeometry(tiny_model_name, device="cpu")
        d = kl.symmetric_kl("Hello", "Goodbye")
        assert d >= 0

    @pytest.mark.slow
    def test_distance_matrix_symmetric(self, tiny_model_name: str) -> None:
        """JSD matrix should be symmetric."""
        from topo_llm.information.divergence import KLGeometry

        kl = KLGeometry(tiny_model_name, device="cpu")
        D = kl.kl_distance_matrix(["Hello", "World", "Test"], show_progress=False)
        np.testing.assert_allclose(D, D.T, atol=0.01)

    @pytest.mark.slow
    def test_distance_matrix_diagonal_zero(self, tiny_model_name: str) -> None:
        """Diagonal of JSD matrix should be 0."""
        from topo_llm.information.divergence import KLGeometry

        kl = KLGeometry(tiny_model_name, device="cpu")
        D = kl.kl_distance_matrix(["Hello", "World"], show_progress=False)
        np.testing.assert_allclose(np.diag(D), 0, atol=0.01)
