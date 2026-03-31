"""
Entropy surface computation for LLM output distributions.

Maps the Shannon entropy of the next-token distribution across
the embedding space, revealing regions of high/low uncertainty.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch required. Install with: pip install topo-llm[torch]"
        )


class EntropySurface:
    """Compute entropy landscapes over LLM output distributions.

    Parameters
    ----------
    model_name : str
        HuggingFace model name.
    device : str
        Compute device.

    Examples
    --------
    >>> surface = EntropySurface("gpt2")
    >>> H = surface.compute_entropy("The meaning of life is")
    >>> print(f"Entropy: {H:.4f} nats")
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
    ) -> None:
        torch = _require_torch()
        import transformers
        from topo_llm.device import get_device

        self.model_name = model_name
        self._device = get_device(device)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self._device)
        self.model.eval()

    def compute_entropy(self, prompt: str) -> float:
        """Compute Shannon entropy of the next-token distribution.

        H = -Σ_v p(v|prompt) log p(v|prompt)

        Parameters
        ----------
        prompt : str
            Input prompt.

        Returns
        -------
        float
            Shannon entropy in nats.
        """
        torch = _require_torch()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1]  # last token logits
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-12)
            entropy = -torch.sum(probs * log_probs)

        return float(entropy.cpu().item())

    def entropy_map(
        self,
        prompts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute entropy for each prompt.

        Parameters
        ----------
        prompts : list[str]
            Input prompts.
        show_progress : bool
            Whether to show progress bar.

        Returns
        -------
        np.ndarray
            Entropies, shape ``(n_prompts,)``.
        """
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(prompts, desc="Entropy map", unit="prompt")
        else:
            iterator = prompts

        entropies = [self.compute_entropy(p) for p in iterator]
        return np.array(entropies)

    def entropy_gradient(
        self,
        prompt: str,
        layer: int = -1,
        n_directions: int = 50,
        epsilon: float = 1e-3,
    ) -> np.ndarray:
        """Estimate the gradient of entropy w.r.t. the embedding.

        Uses finite differences along random directions in the hidden
        state space, then reconstructs the gradient via pseudo-inverse.

        Parameters
        ----------
        prompt : str
            Input prompt.
        layer : int
            Which layer's hidden state to perturb.
        n_directions : int
            Number of random directions for gradient estimation.
        epsilon : float
            Perturbation magnitude.

        Returns
        -------
        np.ndarray
            Estimated gradient, shape ``(hidden_dim,)``.
        """
        torch = _require_torch()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Resolve layer index
            n_layers = len(hidden_states)
            if layer < 0:
                layer = n_layers + layer
            base_hidden = hidden_states[layer][0, -1]  # (hidden_dim,)
            hidden_dim = base_hidden.shape[0]

        # Base entropy
        H_base = self.compute_entropy(prompt)

        # Random directions and finite differences
        rng = np.random.default_rng(42)
        directions = rng.standard_normal((n_directions, hidden_dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        dH = np.zeros(n_directions)

        for i, d in enumerate(directions):
            # Perturb the hidden state and compute new logits
            d_tensor = torch.tensor(d, dtype=torch.float32, device=self._device)
            with torch.no_grad():
                perturbed = base_hidden + epsilon * d_tensor

                # Get logits from perturbed hidden state
                if hasattr(self.model, "lm_head"):
                    logits_p = self.model.lm_head(perturbed)
                else:
                    logits_p = torch.nn.functional.linear(
                        perturbed, self.model.transformer.wte.weight
                    )

                probs_p = torch.softmax(logits_p, dim=-1)
                H_perturbed = -torch.sum(probs_p * torch.log(probs_p + 1e-12))
                dH[i] = (H_perturbed.item() - H_base) / epsilon

        # Reconstruct gradient: dH ≈ D @ grad → grad ≈ D^+ @ dH
        gradient = np.linalg.lstsq(directions, dH, rcond=None)[0]
        return gradient
