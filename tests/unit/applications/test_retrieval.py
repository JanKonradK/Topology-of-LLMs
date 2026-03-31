"""
Tests for GeodesicRetrieval.
"""

from __future__ import annotations

import pytest


class TestGeodesicRetrieval:
    """Integration tests for geodesic retrieval."""

    @pytest.mark.slow
    def test_index_and_query(self, tiny_model_name: str) -> None:
        """Index documents and retrieve results."""
        from topo_llm.applications.retrieval import GeodesicRetrieval

        retriever = GeodesicRetrieval(tiny_model_name, device="cpu")

        documents = [
            "The cat sat on the mat.",
            "Dogs enjoy playing fetch.",
            "Python is great for data science.",
            "JavaScript runs in the browser.",
            "The sun is a star.",
        ]

        retriever.index(documents, layer=-1, reduced_dim=10, n_neighbors=3)

        # Euclidean query
        results = retriever.query("A dog played in the park.", k=3, method="euclidean")
        assert len(results) == 3
        for r in results:
            assert "text" in r
            assert "distance" in r
            assert "rank" in r

    @pytest.mark.slow
    def test_cosine_query(self, tiny_model_name: str) -> None:
        """Cosine retrieval should return results."""
        from topo_llm.applications.retrieval import GeodesicRetrieval

        retriever = GeodesicRetrieval(tiny_model_name, device="cpu")
        docs = ["Hello world.", "Goodbye world.", "Test document."]
        retriever.index(docs, layer=-1, reduced_dim=10, n_neighbors=2)

        results = retriever.query("Hello there.", k=2, method="cosine")
        assert len(results) == 2

    @pytest.mark.slow
    def test_query_methods_all_work(self, tiny_model_name: str) -> None:
        """All three query methods should work without error."""
        from topo_llm.applications.retrieval import GeodesicRetrieval

        retriever = GeodesicRetrieval(tiny_model_name, device="cpu")
        docs = ["Alpha.", "Beta.", "Gamma.", "Delta."]
        retriever.index(docs, layer=-1, reduced_dim=10, n_neighbors=2)

        for method in ["euclidean", "cosine", "geodesic"]:
            results = retriever.query("Test.", k=2, method=method)
            assert len(results) == 2
