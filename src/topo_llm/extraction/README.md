# extraction — Embedding Extraction Pipeline

## Purpose

Extract hidden-layer embeddings from HuggingFace transformer models (GPT-2, LLaMA-2,
Mistral, etc.) and prepare curated datasets for geometric and topological analysis.

## Modules

### `extractor.py` — EmbeddingExtractor

The core extraction engine. Loads a HuggingFace model and extracts embeddings from
ALL hidden layers for given text inputs.

**Key features:**
- Supports any HuggingFace causal LM or encoder model
- Multiple pooling strategies: mean, CLS, last-token, max
- Batch processing with automatic padding and attention masking
- Memory-efficient: processes one layer at a time for large models
- Device auto-detection (CUDA > MPS > CPU)
- Precision control (float32, float16, bfloat16)

**Output:** `EmbeddingResult` dataclass containing per-layer embeddings as NumPy arrays.

### `datasets.py` — DatasetGenerator

Curated text datasets designed for geometric/topological experiments:

| Method | Description | Use Case |
|--------|-------------|----------|
| `semantic_categories()` | 10+ semantic categories with varied templates | Cluster structure analysis |
| `factual_vs_fabricated()` | Paired true/false statements | Hallucination geometry |
| `graduated_similarity()` | Anchor-comparison pairs at 5 similarity levels | Distance metric validation |
| `load_truthfulqa()` | TruthfulQA from HuggingFace | Hallucination benchmark |

### `layers.py` — LayerAnalyzer

Preliminary analysis of embedding spaces before full geometric analysis:

| Method | What it Measures |
|--------|-----------------|
| `intrinsic_dimensionality()` | MLE, TwoNN, or PCA-based dimension estimate |
| `layer_similarity()` | CKA, Procrustes, or CCA between layer pairs |
| `compute_anisotropy()` | Mean cosine, isotropy score, effective rank |

## Data Flow

```
Text inputs
    → EmbeddingExtractor.extract_dataset()
    → {layer_idx: np.ndarray (n_texts, hidden_dim)}
    → Save as .npz files
    → LayerAnalyzer for preliminary analysis
    → Downstream: Riemannian, TDA, Information modules
```

## Framework Boundary

This package is the **only** place PyTorch is imported. All outputs are `np.ndarray`.
