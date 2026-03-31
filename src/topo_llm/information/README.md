# information — Information Geometry

## Purpose

Analyze the information-geometric structure of LLM output distributions. This
connects the embedding space geometry to the model's output uncertainty, providing
a bridge between representation structure and predictive behavior.

## Modules

### `fisher.py` — FisherInformationEstimator

Estimates the Fisher Information Matrix G_F for the LLM's next-token distribution.
The Fisher metric measures how sensitive the output distribution is to perturbations
in the embedding — low Fisher information indicates the model is "uncertain" about
its output, which correlates with hallucination risk.

### `entropy.py` — EntropySurface

Computes Shannon entropy of the next-token distribution across the embedding space.
High entropy = uniform/uncertain predictions. Low entropy = confident predictions.
The entropy gradient shows which directions in embedding space most affect uncertainty.

### `divergence.py` — KLGeometry

KL divergence and Jensen-Shannon distance between output distributions at different
points in embedding space. The JSD distance matrix reveals which prompts produce
similar vs. different output distributions, independent of their embedding similarity.

## Connection to Hallucination Detection

The information-geometric score for hallucination is:

    F(x) = 1 / (tr(G_F(x)) + epsilon)

Low Fisher trace → high F(x) → high hallucination risk. This captures the intuition
that hallucinated outputs come from regions where the model is insensitive to the
input (low information content).
