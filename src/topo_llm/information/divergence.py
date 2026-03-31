"""
KL divergence and Jensen-Shannon distance geometry.

Computes divergences between LLM output distributions at different
prompts, revealing the information-geometric structure of the
output manifold.
"""

from __future__ import annotations

import logging

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch required. Install with: pip install topo-llm[torch]"
        )


class KLGeometry:
    """KL divergence and Jensen-Shannon geometry of LLM outputs.

    Parameters
    ----------
    model_name : str
        HuggingFace model name.
    device : str
        Compute device.

    Examples
    --------
    >>> kl = KLGeometry("gpt2")
    >>> d = kl.symmetric_kl("The sky is", "The ocean is")
    >>> print(f"JSD: {d:.4f}")
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

    def _get_probs(self, prompt: str, top_k: int = 1000) -> np.ndarray:
        """Get the full probability distribution for next token.

        Parameters
        ----------
        prompt : str
            Input prompt.
        top_k : int
            Number of top tokens (for efficiency). Set high to get
            near-full distribution.

        Returns
        -------
        np.ndarray
            Probability vector over vocabulary.
        """
        torch = _require_torch()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1]
            probs = torch.softmax(logits, dim=-1)

        return probs.cpu().numpy()

    def kl_divergence(
        self,
        prompt_a: str,
        prompt_b: str,
        top_k: int = 1000,
    ) -> float:
        """Compute KL(p_a || p_b) between output distributions.

        KL(p_a || p_b) = Σ_v p_a(v) log(p_a(v) / p_b(v))

        Uses Laplace smoothing to handle zeros.

        Parameters
        ----------
        prompt_a : str
            First prompt (defines p_a).
        prompt_b : str
            Second prompt (defines p_b).
        top_k : int
            Restrict to union of top-k tokens from each distribution.

        Returns
        -------
        float
            KL divergence (non-negative, asymmetric).
        """
        p = self._get_probs(prompt_a)
        q = self._get_probs(prompt_b)

        # Laplace smoothing
        eps = 1e-10
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()

        # Restrict to top-k from union
        if top_k < len(p):
            top_p = set(np.argsort(p)[-top_k:])
            top_q = set(np.argsort(q)[-top_k:])
            union_idx = np.array(sorted(top_p | top_q))
            p_sub = p[union_idx]
            q_sub = q[union_idx]
            p_sub = p_sub / p_sub.sum()
            q_sub = q_sub / q_sub.sum()
        else:
            p_sub = p
            q_sub = q

        kl = float(np.sum(p_sub * np.log(p_sub / q_sub)))
        return max(kl, 0.0)  # numerical safety

    def symmetric_kl(
        self,
        prompt_a: str,
        prompt_b: str,
        top_k: int = 1000,
    ) -> float:
        """Compute Jensen-Shannon divergence (symmetric KL).

        JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
        where m = 0.5 * (p + q)

        Parameters
        ----------
        prompt_a : str
            First prompt.
        prompt_b : str
            Second prompt.
        top_k : int
            Top-k tokens for restriction.

        Returns
        -------
        float
            Jensen-Shannon divergence (symmetric, non-negative).
        """
        p = self._get_probs(prompt_a)
        q = self._get_probs(prompt_b)

        # Laplace smoothing
        eps = 1e-10
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()

        # Mixture
        m = 0.5 * (p + q)

        # JSD
        kl_pm = float(np.sum(p * np.log(p / m)))
        kl_qm = float(np.sum(q * np.log(q / m)))
        jsd = 0.5 * kl_pm + 0.5 * kl_qm

        return max(jsd, 0.0)

    def kl_distance_matrix(
        self,
        prompts: list[str],
        top_k: int = 1000,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Pairwise JSD matrix between prompts.

        Parameters
        ----------
        prompts : list[str]
            Input prompts.
        top_k : int
            Top-k tokens for restriction.
        show_progress : bool
            Whether to show progress bar.

        Returns
        -------
        np.ndarray
            Symmetric JSD matrix, shape ``(n, n)``.
        """
        n = len(prompts)
        D = np.zeros((n, n))

        # Pre-compute all distributions
        logger.info("Computing output distributions for %d prompts...", n)
        all_probs = []
        iterator = prompts
        if show_progress:
            iterator = tqdm(prompts, desc="Getting distributions", unit="prompt")
        for prompt in iterator:
            all_probs.append(self._get_probs(prompt))

        # Compute pairwise JSD
        total = n * (n - 1) // 2
        if show_progress:
            pbar = tqdm(total=total, desc="JSD matrix", unit="pair")
        else:
            pbar = None

        eps = 1e-10
        for i in range(n):
            for j in range(i + 1, n):
                p = all_probs[i] + eps
                q = all_probs[j] + eps
                p = p / p.sum()
                q = q / q.sum()
                m = 0.5 * (p + q)

                jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
                D[i, j] = max(jsd, 0.0)
                D[j, i] = D[i, j]

                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        return D
