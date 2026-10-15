# Bounded analytical audit: does median-split entropy identify temporal order?

Recorded 2026-10-15 before numerical checks. This is a targeted analytical audit selected after reading a published formula and its source code; it is not an independently preregistered discovery.

## Motivation and frozen source

Toro-Hernández, Vieira Filho and Cabral-Carvalho, *Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space*, arXiv:2602.05971v1 (ICLR 2026 paper per public record), defines binary entropy of step distances split at their within-trajectory median. Its text discusses predictability. The released implementation divides the binary marginal entropy by the number of valid steps and uses `>` rather than the paper's `>=`.

- Paper: https://arxiv.org/html/2602.05971v1, section 2.2, equations 1–2.
- Code: https://github.com/jesuinovieira/semtraj-iclr2026/blob/7506e25c03036909ab1ad381c678c7b45d19e50e/src/metrics.py
- Code SHA256: `63f776ab45069bdfc6d8f3141edbc61d49e7af4bff8521444f4044f6d0859733`.

## Predictions

1. For an even number n > 3 of distinct finite step distances, the released metric equals exactly 1/n in real arithmetic. For odd n, it equals h2(floor(n/2)/n)/n. Thus it is determined by sample size in the absence of ties.
2. Permuting the **step-distance sequence** preserves the metric, even when its temporal pattern changes. This does not claim that permuting texts and re-embedding them preserves step distances.
3. Ties can change the value; the degenerate constant-distance case gives zero. This changes the interpretation to a length/tie descriptor, not an order statistic.

## Check and decision

Evaluate fixed lengths [4, 5, 8, 16, 32, 64, 128], 100 seeded continuous draws per length, and three differently ordered copies of one 16-step multiset. Construct valid 2D unit-vector paths realizing each distance sequence to rule out impossible geometric inputs. Compare the metric with its exact expression. Lag-1 correlation is descriptive only and is not proposed as a replacement estimator.

No fitting, evaluation labels, participant data or model calls. No statistical confidence interval is required for a mathematical identity. Numerical checks demonstrate implementation and floating-point tolerance; they are not evidence of empirical prevalence in the original study.

If predictions hold, classify this particular metric as unsuitable for claims about temporal unpredictability without further definition/validation. Do not infer that the full paper's other measures or empirical comparisons are invalid. The relevance to CMRI is a concrete example motivating calibration of semantic-motion measurements, beyond merely renaming finite differences.

## Reproducibility

Use Python, NumPy, pytest and Matplotlib already available locally. Record versions, seed, source/config hashes and runtime; save JSON and standalone PNG/SVG figures. Unit tests must cover odd/even lengths, ties, small samples, non-finite/invalid inputs and realization of the stated step distances; integration test verifies deterministic report generation and output schema.
