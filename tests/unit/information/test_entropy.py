"""
Tests for EntropySurface.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestEntropySurface:
    """Tests for entropy computation."""

    @pytest.mark.slow
    def test_entropy_nonnegative(self, tiny_model_name: str) -> None:
        """Shannon entropy is always non-negative."""
        from topo_llm.information.entropy import EntropySurface

        surface = EntropySurface(tiny_model_name, device="cpu")
        H = surface.compute_entropy("The quick brown fox")
        assert H >= 0, f"Negative entropy: {H}"

    @pytest.mark.slow
    def test_entropy_finite(self, tiny_model_name: str) -> None:
        """Entropy should be finite."""
        from topo_llm.information.entropy import EntropySurface

        surface = EntropySurface(tiny_model_name, device="cpu")
        H = surface.compute_entropy("Hello world")
        assert np.isfinite(H), f"Non-finite entropy: {H}"

    @pytest.mark.slow
    def test_entropy_map_shape(self, tiny_model_name: str) -> None:
        """entropy_map should return correct shape."""
        from topo_llm.information.entropy import EntropySurface

        surface = EntropySurface(tiny_model_name, device="cpu")
        prompts = ["Hello", "World", "Test"]
        H = surface.entropy_map(prompts, show_progress=False)
        assert H.shape == (3,)
        assert np.all(H >= 0)

    @pytest.mark.slow
    def test_gradient_shape(self, tiny_model_name: str) -> None:
        """Entropy gradient should have hidden_dim shape."""
        from topo_llm.information.entropy import EntropySurface

        surface = EntropySurface(tiny_model_name, device="cpu")
        grad = surface.entropy_gradient("Hello", n_directions=10, epsilon=1e-3)
        assert len(grad.shape) == 1
        assert np.all(np.isfinite(grad))
