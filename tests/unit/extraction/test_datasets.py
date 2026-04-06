"""
Tests for DatasetGenerator.

Validates that generated datasets have correct formats, balanced labels,
and expected category structure.
"""

from __future__ import annotations

from topo_llm.extraction.datasets import DatasetGenerator


class TestSemanticCategories:
    """Tests for DatasetGenerator.semantic_categories()."""

    def test_correct_counts(self) -> None:
        """Each category produces n_per_category texts."""
        n_per = 20
        texts, labels, info = DatasetGenerator.semantic_categories(n_per_category=n_per)
        assert len(texts) == len(labels) == n_per * 10

    def test_all_categories_present(self) -> None:
        """All 10 expected categories appear in labels."""
        texts, labels, info = DatasetGenerator.semantic_categories(n_per_category=10)
        unique_labels = set(labels)
        expected = {
            "animals",
            "vehicles",
            "emotions",
            "colors",
            "countries",
            "mathematical_concepts",
            "musical_instruments",
            "diseases",
            "programming_languages",
            "foods",
        }
        assert unique_labels == expected

    def test_balanced_labels(self) -> None:
        """Each category has exactly n_per_category entries."""
        n_per = 15
        texts, labels, info = DatasetGenerator.semantic_categories(n_per_category=n_per)
        from collections import Counter

        counts = Counter(labels)
        for category, count in counts.items():
            assert count == n_per, f"{category} has {count} instead of {n_per}"

    def test_no_empty_texts(self) -> None:
        """All generated texts are non-empty strings."""
        texts, labels, info = DatasetGenerator.semantic_categories(n_per_category=5)
        for text in texts:
            assert isinstance(text, str)
            assert len(text) > 0

    def test_info_metadata(self) -> None:
        """DatasetInfo has correct metadata."""
        texts, labels, info = DatasetGenerator.semantic_categories(n_per_category=10)
        assert info.name == "semantic_categories"
        assert info.n_samples == 100
        assert info.n_categories == 10
        assert len(info.category_names) == 10

    def test_reproducibility(self) -> None:
        """Same seed produces identical output."""
        t1, l1, _ = DatasetGenerator.semantic_categories(n_per_category=10, seed=123)
        t2, l2, _ = DatasetGenerator.semantic_categories(n_per_category=10, seed=123)
        assert t1 == t2
        assert l1 == l2


class TestFactualVsFabricated:
    """Tests for DatasetGenerator.factual_vs_fabricated()."""

    def test_correct_total_count(self) -> None:
        """Total texts = 2 * n_pairs (one factual + one fabricated each)."""
        n_pairs = 30
        texts, is_factual, info = DatasetGenerator.factual_vs_fabricated(n_pairs=n_pairs)
        assert len(texts) == 2 * n_pairs
        assert len(is_factual) == 2 * n_pairs

    def test_balanced_true_false(self) -> None:
        """Exactly half are factual, half fabricated."""
        texts, is_factual, info = DatasetGenerator.factual_vs_fabricated(n_pairs=50)
        assert sum(is_factual) == 50
        assert sum(not f for f in is_factual) == 50

    def test_all_booleans(self) -> None:
        """Labels are all booleans."""
        texts, is_factual, info = DatasetGenerator.factual_vs_fabricated(n_pairs=10)
        for flag in is_factual:
            assert isinstance(flag, bool)

    def test_no_empty_texts(self) -> None:
        """All texts are non-empty."""
        texts, is_factual, info = DatasetGenerator.factual_vs_fabricated(n_pairs=10)
        for text in texts:
            assert len(text) > 0


class TestGraduatedSimilarity:
    """Tests for DatasetGenerator.graduated_similarity()."""

    def test_correct_total(self) -> None:
        """Each anchor produces 5 comparisons (one per level)."""
        n = 10
        anchors, comps, scores, info = DatasetGenerator.graduated_similarity(n_anchors=n)
        assert len(anchors) == len(comps) == len(scores) == n * 5

    def test_score_range(self) -> None:
        """All scores are in {0.0, 0.25, 0.5, 0.75, 1.0}."""
        anchors, comps, scores, info = DatasetGenerator.graduated_similarity(n_anchors=5)
        valid_scores = {0.0, 0.25, 0.5, 0.75, 1.0}
        for s in scores:
            assert s in valid_scores, f"Invalid score: {s}"

    def test_all_levels_present(self) -> None:
        """All five similarity levels appear."""
        anchors, comps, scores, info = DatasetGenerator.graduated_similarity(n_anchors=5)
        assert set(scores) == {0.0, 0.25, 0.5, 0.75, 1.0}

    def test_anchors_and_comparisons_differ(self) -> None:
        """Anchor and comparison are different strings."""
        anchors, comps, scores, info = DatasetGenerator.graduated_similarity(n_anchors=5)
        for a, c in zip(anchors, comps):
            assert a != c
