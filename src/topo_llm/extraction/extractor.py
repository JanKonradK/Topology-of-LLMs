"""
Embedding extraction from HuggingFace transformer models.

Extracts hidden-layer embeddings from all layers of a transformer model,
supporting batch processing, multiple pooling strategies, and memory-
efficient extraction for large models.

This is the only module in topo-llm that imports PyTorch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np

from topo_llm.types import EmbeddingResult

logger = logging.getLogger(__name__)


def _require_torch():
    """Lazy import guard for PyTorch."""
    try:
        import torch

        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for embedding extraction. "
            "Install with: pip install topo-llm[torch]"
        ) from None


def _require_transformers():
    """Lazy import guard for HuggingFace Transformers."""
    try:
        import transformers

        return transformers
    except ImportError:
        raise ImportError(
            "HuggingFace Transformers is required for embedding extraction. "
            "Install with: pip install topo-llm[torch]"
        ) from None


class EmbeddingExtractor:
    """Extract hidden-layer embeddings from HuggingFace transformer models.

    Loads a causal language model (GPT-2, LLaMA-2, Mistral, etc.) and
    extracts embeddings from all hidden layers for given text inputs.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or path. Examples: ``"gpt2"``,
        ``"gpt2-medium"``, ``"meta-llama/Llama-2-7b-hf"``.
    device : str
        Compute device. ``"auto"`` detects the best available device.
    precision : str
        Model precision. ``"float16"`` halves memory usage.

    Attributes
    ----------
    n_layers : int
        Number of hidden layers in the model.
    hidden_dim : int
        Dimensionality of hidden states.
    model_name : str
        The model name used for loading.

    Examples
    --------
    >>> extractor = EmbeddingExtractor("gpt2")
    >>> result = extractor.extract("The cat sat on the mat.")
    >>> result.pooled_embeddings[0].shape
    (768,)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        precision: Literal["float32", "float16", "bfloat16"] = "float32",
    ) -> None:
        torch = _require_torch()
        transformers = _require_transformers()

        from topo_llm.device import get_device

        self.model_name = model_name
        self._device_str = get_device(device)
        self._precision = precision

        # Determine torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self._torch_dtype = dtype_map[precision]

        # Load tokenizer
        logger.info("Loading tokenizer for %s", model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        logger.info(
            "Loading model %s (precision=%s, device=%s)", model_name, precision, self._device_str
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._torch_dtype,
            output_hidden_states=True,
        )
        self.model.to(self._device_str)
        self.model.eval()

        # Extract model dimensions
        config = self.model.config
        self.n_layers = config.num_hidden_layers
        self.hidden_dim = config.hidden_size

        logger.info(
            "Model loaded: %d layers, %d hidden dim, %s",
            self.n_layers,
            self.hidden_dim,
            self._device_str,
        )

    def extract(
        self,
        text: str,
        pooling: Literal["mean", "cls", "last", "max"] = "mean",
    ) -> EmbeddingResult:
        """Extract embeddings from all layers for a single text.

        Parameters
        ----------
        text : str
            Input text to embed.
        pooling : str
            Pooling strategy for reducing token dimension:
            - ``"mean"``: Average across token positions (exclude padding).
            - ``"cls"``: First token embedding.
            - ``"last"``: Last non-padding token.
            - ``"max"``: Element-wise max across positions.

        Returns
        -------
        EmbeddingResult
            Structured result with per-layer embeddings.
        """
        torch = _require_torch()

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self._device_str) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # hidden_states is a tuple of (n_layers + 1) tensors
        # Index 0 = embedding layer output, 1..n_layers = transformer layers
        hidden_states = outputs.hidden_states

        token_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask[0].cpu()

        layer_embeddings: dict[int, np.ndarray] = {}
        pooled_embeddings: dict[int, np.ndarray] = {}

        for layer_idx, hs in enumerate(hidden_states):
            # hs shape: (1, seq_len, hidden_dim)
            layer_emb = hs[0].cpu().float().numpy()  # (seq_len, hidden_dim)
            layer_embeddings[layer_idx] = layer_emb
            pooled_embeddings[layer_idx] = self._pool(layer_emb, attention_mask, pooling)

        return EmbeddingResult(
            text=text,
            token_ids=token_ids,
            tokens=tokens,
            layer_embeddings=layer_embeddings,
            pooled_embeddings=pooled_embeddings,
            model_name=self.model_name,
        )

    def extract_batch(
        self,
        texts: list[str],
        pooling: Literal["mean", "cls", "last", "max"] = "mean",
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[EmbeddingResult]:
        """Batch extraction with padding and attention masking.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.
        pooling : str
            Pooling strategy (see :meth:`extract`).
        batch_size : int
            Number of texts per batch.
        show_progress : bool
            Whether to show a tqdm progress bar.

        Returns
        -------
        list[EmbeddingResult]
            One result per input text.
        """
        torch = _require_torch()

        if show_progress:
            from tqdm import tqdm

            batches = range(0, len(texts), batch_size)
            batches = tqdm(batches, desc="Extracting embeddings", unit="batch")
        else:
            batches = range(0, len(texts), batch_size)

        results: list[EmbeddingResult] = []

        for start in batches:
            batch_texts = texts[start : start + batch_size]

            # Tokenize batch with padding
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self._device_str) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            hidden_states = outputs.hidden_states
            attention_masks = inputs.get("attention_mask")

            # Process each item in the batch
            for i, text in enumerate(batch_texts):
                token_ids = inputs["input_ids"][i].cpu().numpy()
                # Find actual length (non-padding)
                if attention_masks is not None:
                    mask = attention_masks[i].cpu()
                    actual_len = mask.sum().item()
                else:
                    mask = None
                    actual_len = len(token_ids)

                token_ids_trimmed = token_ids[:actual_len]
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids_trimmed.tolist())

                layer_embeddings: dict[int, np.ndarray] = {}
                pooled_embeddings: dict[int, np.ndarray] = {}

                for layer_idx, hs in enumerate(hidden_states):
                    layer_emb = hs[i].cpu().float().numpy()  # (padded_len, hidden_dim)
                    # Trim to actual length
                    layer_emb_trimmed = layer_emb[:actual_len]
                    layer_embeddings[layer_idx] = layer_emb_trimmed
                    pooled_embeddings[layer_idx] = self._pool(layer_emb, mask, pooling)

                results.append(
                    EmbeddingResult(
                        text=text,
                        token_ids=token_ids_trimmed,
                        tokens=tokens,
                        layer_embeddings=layer_embeddings,
                        pooled_embeddings=pooled_embeddings,
                        model_name=self.model_name,
                    )
                )

        return results

    def extract_dataset(
        self,
        texts: list[str],
        pooling: Literal["mean", "cls", "last", "max"] = "mean",
        layers: list[int] | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> dict[int, np.ndarray]:
        """Optimized extraction returning pooled embedding matrices.

        Extracts only pooled embeddings (not per-token) for efficiency.
        Returns matrices ready for geometric analysis.

        Parameters
        ----------
        texts : list[str]
            Input texts.
        pooling : str
            Pooling strategy.
        layers : list[int] | None
            Layer indices to extract. ``None`` extracts all layers.
            Use negative indices (e.g., ``[-1, -2]``) for layers
            relative to the last.
        batch_size : int
            Batch size for processing.
        show_progress : bool
            Whether to show progress bar.

        Returns
        -------
        dict[int, np.ndarray]
            Maps layer index to array of shape ``(n_texts, hidden_dim)``.
        """
        torch = _require_torch()

        # Resolve layer indices
        total_layers = self.n_layers + 1  # +1 for embedding layer
        if layers is not None:
            resolved_layers = set()
            for l in layers:
                if l < 0:
                    resolved_layers.add(total_layers + l)
                else:
                    resolved_layers.add(l)
        else:
            resolved_layers = set(range(total_layers))

        # Pre-allocate output arrays
        result: dict[int, list[np.ndarray]] = {l: [] for l in resolved_layers}

        if show_progress:
            from tqdm import tqdm

            batches = range(0, len(texts), batch_size)
            batches = tqdm(batches, desc="Extracting dataset", unit="batch")
        else:
            batches = range(0, len(texts), batch_size)

        for start in batches:
            batch_texts = texts[start : start + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self._device_str) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            hidden_states = outputs.hidden_states
            attention_masks = inputs.get("attention_mask")

            for layer_idx in resolved_layers:
                hs = hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
                for i in range(len(batch_texts)):
                    layer_emb = hs[i].cpu().float().numpy()
                    mask = attention_masks[i].cpu() if attention_masks is not None else None
                    pooled = self._pool(layer_emb, mask, pooling)
                    result[layer_idx].append(pooled)

        # Stack into matrices
        return {l: np.stack(vecs) for l, vecs in result.items()}

    def _pool(
        self,
        embeddings: np.ndarray,
        attention_mask: object | None,
        strategy: str,
    ) -> np.ndarray:
        """Apply pooling strategy to token embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Token embeddings, shape (seq_len, hidden_dim).
        attention_mask : Tensor | None
            Attention mask (1 = real token, 0 = padding).
        strategy : str
            Pooling strategy name.

        Returns
        -------
        np.ndarray
            Pooled embedding, shape (hidden_dim,).
        """
        if attention_mask is not None:
            mask = np.array(attention_mask, dtype=bool)
        else:
            mask = np.ones(embeddings.shape[0], dtype=bool)

        if strategy == "mean":
            # Average over non-padding positions
            masked = embeddings[mask]
            if len(masked) == 0:
                return np.zeros(embeddings.shape[1], dtype=embeddings.dtype)
            return masked.mean(axis=0)

        elif strategy == "cls":
            # First token
            return embeddings[0]

        elif strategy == "last":
            # Last non-padding token
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                return np.zeros(embeddings.shape[1], dtype=embeddings.dtype)
            return embeddings[valid_indices[-1]]

        elif strategy == "max":
            # Element-wise max over non-padding positions
            masked = embeddings[mask]
            if len(masked) == 0:
                return np.zeros(embeddings.shape[1], dtype=embeddings.dtype)
            return masked.max(axis=0)

        else:
            raise ValueError(f"Unknown pooling strategy: {strategy!r}")

    def save_embeddings(
        self,
        embeddings: dict[int, np.ndarray],
        path: str | Path,
        metadata: dict | None = None,
    ) -> Path:
        """Save extracted embeddings to a .npz file.

        Parameters
        ----------
        embeddings : dict[int, np.ndarray]
            Layer-indexed embedding matrices from :meth:`extract_dataset`.
        path : str | Path
            Output file path (will add .npz extension if missing).
        metadata : dict | None
            Optional metadata to store alongside embeddings.

        Returns
        -------
        Path
            Path to the saved file.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert layer keys to string for npz
        save_dict = {f"layer_{k}": v for k, v in embeddings.items()}
        save_dict["_layer_indices"] = np.array(list(embeddings.keys()))

        if metadata:
            save_dict["_model_name"] = np.array([self.model_name])

        np.savez_compressed(path, **save_dict)
        logger.info("Saved embeddings to %s", path)
        return path

    @staticmethod
    def load_embeddings(path: str | Path) -> dict[int, np.ndarray]:
        """Load embeddings from a .npz file.

        Parameters
        ----------
        path : str | Path
            Path to the .npz file.

        Returns
        -------
        dict[int, np.ndarray]
            Layer-indexed embedding matrices.
        """
        data = np.load(path)
        layer_indices = data["_layer_indices"]
        return {int(idx): data[f"layer_{idx}"] for idx in layer_indices}
