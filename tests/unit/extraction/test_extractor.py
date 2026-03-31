"""
Tests for EmbeddingExtractor.

Tests marked @pytest.mark.slow require loading a real model and are
skipped by default with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestEmbeddingExtractorShapes:
    """Verify embedding shapes and types with a tiny model."""

    @pytest.mark.slow
    def test_single_extract_shapes(self, tiny_model_name: str) -> None:
        """Extract from a single text and check shapes."""
        from topo_llm.extraction.extractor import EmbeddingExtractor

        extractor = EmbeddingExtractor(tiny_model_name, device="cpu")
        result = extractor.extract("The cat sat on the mat.")

        # Should have layer 0 (embedding) through layer n_layers
        assert len(result.layer_embeddings) == extractor.n_layers + 1
        assert len(result.pooled_embeddings) == extractor.n_layers + 1

        # Check shapes
        for layer_idx, emb in result.layer_embeddings.items():
            seq_len = len(result.tokens)
            assert emb.shape == (seq_len, extractor.hidden_dim), (
                f"Layer {layer_idx}: expected ({seq_len}, {extractor.hidden_dim}), "
                f"got {emb.shape}"
            )

        for layer_idx, emb in result.pooled_embeddings.items():
            assert emb.shape == (extractor.hidden_dim,), (
                f"Layer {layer_idx}: expected ({extractor.hidden_dim},), got {emb.shape}"
            )

    @pytest.mark.slow
    def test_pooled_is_finite(self, tiny_model_name: str) -> None:
        """All pooled embeddings should be finite and non-zero."""
        from topo_llm.extraction.extractor import EmbeddingExtractor

        extractor = EmbeddingExtractor(tiny_model_name, device="cpu")
        result = extractor.extract("Hello world")

        for layer_idx, emb in result.pooled_embeddings.items():
            assert np.all(np.isfinite(emb)), f"Layer {layer_idx}: non-finite values"
            assert np.any(emb != 0), f"Layer {layer_idx}: all zeros"

    @pytest.mark.slow
    def test_different_pooling_strategies(self, tiny_model_name: str) -> None:
        """Different pooling methods produce different (but valid) results."""
        from topo_llm.extraction.extractor import EmbeddingExtractor

        extractor = EmbeddingExtractor(tiny_model_name, device="cpu")
        text = "The quick brown fox jumps over the lazy dog."

        results = {}
        for strategy in ["mean", "cls", "last", "max"]:
            r = extractor.extract(text, pooling=strategy)
            results[strategy] = r.pooled_embeddings[0]

        # All should be finite
        for name, emb in results.items():
            assert np.all(np.isfinite(emb)), f"{name} has non-finite values"
            assert np.any(emb != 0), f"{name} is all zeros"

        # Mean and CLS should differ (unless single token)
        if len(extractor.tokenizer.encode(text)) > 1:
            assert not np.allclose(results["mean"], results["cls"]), (
                "Mean and CLS should differ for multi-token input"
            )

    @pytest.mark.slow
    def test_batch_extract(self, tiny_model_name: str) -> None:
        """Batch extraction produces consistent results."""
        from topo_llm.extraction.extractor import EmbeddingExtractor

        extractor = EmbeddingExtractor(tiny_model_name, device="cpu")
        texts = ["Hello world.", "The sky is blue.", "Math is fun."]

        results = extractor.extract_batch(texts, batch_size=2, show_progress=False)
        assert len(results) == 3

        for result in results:
            assert len(result.pooled_embeddings) == extractor.n_layers + 1
            for emb in result.pooled_embeddings.values():
                assert emb.shape == (extractor.hidden_dim,)
                assert np.all(np.isfinite(emb))

    @pytest.mark.slow
    def test_extract_dataset(self, tiny_model_name: str) -> None:
        """extract_dataset returns properly shaped matrices."""
        from topo_llm.extraction.extractor import EmbeddingExtractor

        extractor = EmbeddingExtractor(tiny_model_name, device="cpu")
        texts = ["Hello", "World", "Test", "Input"]

        result = extractor.extract_dataset(
            texts, layers=[0, -1], batch_size=2, show_progress=False
        )

        # Should have exactly 2 layers
        assert len(result) == 2

        for layer_idx, matrix in result.items():
            assert matrix.shape == (4, extractor.hidden_dim), (
                f"Layer {layer_idx}: expected (4, {extractor.hidden_dim}), "
                f"got {matrix.shape}"
            )

    @pytest.mark.slow
    def test_save_and_load(self, tiny_model_name: str, tmp_path) -> None:
        """Embeddings survive save/load round-trip."""
        from topo_llm.extraction.extractor import EmbeddingExtractor

        extractor = EmbeddingExtractor(tiny_model_name, device="cpu")
        texts = ["Hello", "World"]

        original = extractor.extract_dataset(
            texts, layers=[0], batch_size=2, show_progress=False
        )

        # Save and reload
        path = extractor.save_embeddings(original, tmp_path / "test_emb")
        loaded = EmbeddingExtractor.load_embeddings(path)

        for layer_idx in original:
            np.testing.assert_array_almost_equal(
                original[layer_idx], loaded[layer_idx]
            )


class TestPoolingFunction:
    """Test the internal _pool method with known data."""

    @pytest.mark.slow
    def test_mean_single_token(self, tiny_model_name: str) -> None:
        """Mean pooling of a single token equals that token's embedding."""
        from topo_llm.extraction.extractor import EmbeddingExtractor

        extractor = EmbeddingExtractor(tiny_model_name, device="cpu")

        # Create a single-element embedding
        emb = np.random.randn(1, 64).astype(np.float32)
        pooled = extractor._pool(emb, None, "mean")
        np.testing.assert_array_almost_equal(pooled, emb[0])

    @pytest.mark.slow
    def test_cls_returns_first(self, tiny_model_name: str) -> None:
        """CLS pooling returns the first token."""
        from topo_llm.extraction.extractor import EmbeddingExtractor

        extractor = EmbeddingExtractor(tiny_model_name, device="cpu")

        emb = np.random.randn(5, 64).astype(np.float32)
        pooled = extractor._pool(emb, None, "cls")
        np.testing.assert_array_almost_equal(pooled, emb[0])
