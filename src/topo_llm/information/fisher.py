"""
Fisher Information Matrix estimation for LLM output distributions.

The Fisher Information Matrix G_F measures how sensitive the output
distribution is to perturbations in the embedding. Low Fisher information
indicates the model is "uncertain" — correlating with hallucination risk.

Method: Empirical Fisher via embedding perturbation.
"""

from __future__ import annotations

import logging

import numpy as np

from topo_llm.types import FisherResult

logger = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for Fisher information estimation. "
            "Install with: pip install topo-llm[torch]"
        )


class FisherInformationEstimator:
    """Estimate Fisher Information Matrix for LLM output distributions.

    Uses empirical Fisher estimation via embedding perturbation:
    perturb the final hidden state, observe the change in output
    distribution, and estimate the Fisher matrix from the scores.

    Parameters
    ----------
    model_name : str
        HuggingFace model name.
    device : str
        Compute device (``"auto"``, ``"cpu"``, ``"cuda"``).
    n_samples : int
        Number of Monte Carlo samples for estimation.

    Examples
    --------
    >>> fisher = FisherInformationEstimator("gpt2")
    >>> result = fisher.estimate_at("The capital of France is")
    >>> print(f"Fisher trace: {result.fisher_trace:.4f}")
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        n_samples: int = 100,
    ) -> None:
        torch = _require_torch()
        import transformers
        from topo_llm.device import get_device

        self.model_name = model_name
        self._device = get_device(device)
        self.n_samples = n_samples

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.model.to(self._device)
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

    def _get_logits_from_hidden(
        self,
        hidden_state: object,  # torch.Tensor
    ) -> object:  # torch.Tensor
        """Get output logits from a hidden state using the LM head."""
        torch = _require_torch()
        # Most HF models have lm_head
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head(hidden_state)
        # GPT-2 uses transformer.wte for tying weights
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            return torch.nn.functional.linear(
                hidden_state, self.model.transformer.wte.weight
            )
        else:
            raise RuntimeError("Cannot find LM head for this model architecture")

    def estimate_at(
        self,
        prompt: str,
        top_k: int = 100,
    ) -> FisherResult:
        """Estimate the Fisher Information Matrix at a prompt.

        Parameters
        ----------
        prompt : str
            Input prompt.
        top_k : int
            Number of top tokens to consider (for tractability).

        Returns
        -------
        FisherResult
            Fisher matrix, trace, eigenvalues, effective dimension,
            entropy, and top-k probabilities.
        """
        torch = _require_torch()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states
            last_hidden = hidden_states[-1][0, -1]  # (hidden_dim,)

            # Get base logits and probabilities
            base_logits = self._get_logits_from_hidden(last_hidden)
            base_probs = torch.softmax(base_logits, dim=-1)

            # Get top-k tokens
            top_k_probs, top_k_indices = torch.topk(base_probs, top_k)
            top_k_probs_np = top_k_probs.cpu().numpy()

        # Entropy
        probs_np = base_probs.cpu().numpy()
        probs_np = probs_np[probs_np > 1e-12]
        entropy = float(-np.sum(probs_np * np.log(probs_np)))

        # Empirical Fisher via random perturbations
        # Reduce dimensionality for tractability
        reduced_dim = min(50, self.hidden_dim)
        rng = np.random.default_rng(42)

        # Random projection matrix
        projection = rng.standard_normal((self.hidden_dim, reduced_dim))
        projection /= np.linalg.norm(projection, axis=0, keepdims=True)
        projection_t = torch.tensor(projection, dtype=torch.float32, device=self._device)

        scores = []
        epsilon = 1e-3

        for _ in range(self.n_samples):
            # Random direction in reduced space
            delta_reduced = rng.standard_normal(reduced_dim)
            delta_reduced /= np.linalg.norm(delta_reduced)
            delta_t = torch.tensor(delta_reduced, dtype=torch.float32, device=self._device)

            # Lift to full space
            delta_full = projection_t @ delta_t  # (hidden_dim,)

            with torch.no_grad():
                # Perturbed hidden state
                h_perturbed = last_hidden + epsilon * delta_full
                logits_perturbed = self._get_logits_from_hidden(h_perturbed)
                probs_perturbed = torch.softmax(logits_perturbed, dim=-1)

                # Score: d/dε log p(v|h+εδ) evaluated at ε→0
                # Approximate: (log p(h+εδ) - log p(h)) / ε for observed tokens
                log_ratio = (
                    torch.log(probs_perturbed[top_k_indices] + 1e-12)
                    - torch.log(base_probs[top_k_indices] + 1e-12)
                ) / epsilon  # (top_k,)

                # Weighted score (weighted by base probability)
                weighted_score = (top_k_probs * log_ratio).sum()
                score_vec = weighted_score.item() * delta_reduced
                scores.append(score_vec)

        scores = np.array(scores)  # (n_samples, reduced_dim)

        # Fisher matrix: G_F ≈ (1/n) Σ s s^T
        fisher_matrix = (scores.T @ scores) / self.n_samples  # (reduced_dim, reduced_dim)

        # Eigendecompose
        eigenvalues = np.linalg.eigvalsh(fisher_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues, 0)  # ensure non-negative

        fisher_trace = float(eigenvalues.sum())

        # Effective dimension (participation ratio)
        if fisher_trace > 1e-12:
            p = eigenvalues / fisher_trace
            p = p[p > 1e-12]
            effective_dim = float(1.0 / np.sum(p ** 2)) if len(p) > 0 else 0.0
        else:
            effective_dim = 0.0

        return FisherResult(
            fisher_matrix=fisher_matrix,
            fisher_trace=fisher_trace,
            fisher_eigenvalues=eigenvalues,
            effective_dimension=effective_dim,
            entropy=entropy,
            top_k_probs=top_k_probs_np,
        )

    def fisher_trace_batch(
        self,
        prompts: list[str],
        top_k: int = 100,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Efficiently compute Fisher trace for many prompts.

        Parameters
        ----------
        prompts : list[str]
            Input prompts.
        top_k : int
            Top-k tokens for estimation.
        show_progress : bool
            Whether to show a progress bar.

        Returns
        -------
        np.ndarray
            Fisher traces, shape ``(n_prompts,)``.
        """
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(prompts, desc="Fisher traces", unit="prompt")
        else:
            iterator = prompts

        traces = []
        for prompt in iterator:
            result = self.estimate_at(prompt, top_k=top_k)
            traces.append(result.fisher_trace)

        return np.array(traces)
