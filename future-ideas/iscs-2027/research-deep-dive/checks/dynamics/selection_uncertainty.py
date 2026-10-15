"""Post-review prefix uncertainty with lens refitting and identity exclusion.

Only saved scalar scores are read. The original results.json remains unchanged.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from audit_dynamics import leave_one_out, prefix_score_matrix

HERE = Path(__file__).resolve().parent


def refit_from_counts(scores, counts, baseline_index):
    """Evaluate each multinomial sample by leaving out all copies of each case.

    scores is (original conversations, candidate lenses); counts is
    (bootstrap draws, original conversations). Original identity, not row position
    in a repeated sample, determines exclusion. Counts retain training and test
    multiplicity. A one-identity draw has no training sample and is rejected.
    """
    scores = np.asarray(scores, dtype=float)
    counts = np.asarray(counts)
    if scores.ndim != 2 or counts.ndim != 2 or counts.shape[1] != len(scores):
        raise ValueError("scores and counts must be aligned two-dimensional matrices")
    if not np.isfinite(scores).all() or not np.isfinite(counts).all():
        raise ValueError("scores and counts must be finite")
    if np.any(counts < 0) or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("counts must be nonnegative integers")
    if not 0 <= baseline_index < scores.shape[1]:
        raise ValueError("baseline index outside candidate columns")
    if np.any(np.count_nonzero(counts, axis=1) < 2):
        raise ValueError("each draw needs at least two distinct conversations")
    counts = counts.astype(np.int64, copy=False)
    totals = counts @ scores
    # Subtract all copies of the held-out original identity. For each held-out
    # identity the remaining denominator is common to every candidate, so argmax
    # of these sums equals argmax of the training means.
    training_sums = totals[:, None, :] - counts[:, :, None] * scores[None, :, :]
    selected = training_sums.argmax(axis=2)
    held_out_scores = scores[np.arange(len(scores))[None, :], selected]
    gains = held_out_scores - scores[None, :, baseline_index]
    means = (counts * gains).sum(axis=1) / counts.sum(axis=1)
    return {"mean_gains": means, "selected_indices": selected}


def sample_valid_counts(n_conversations, draws, rng):
    """Draw multinomial conversation multiplicities; reject singleton cohorts."""
    if n_conversations < 2 or draws < 1:
        raise ValueError("need at least two conversations and one draw")
    probabilities = np.full(n_conversations, 1 / n_conversations)
    chunks = []
    remaining, rejected = draws, 0
    while remaining:
        counts = rng.multinomial(n_conversations, probabilities, size=remaining)
        valid = np.count_nonzero(counts, axis=1) >= 2
        rejected += int((~valid).sum())
        chunks.append(counts[valid])
        remaining -= int(valid.sum())
    return np.concatenate(chunks), rejected


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config):
    start = time.perf_counter()
    source = Path(config["source_root"]) / config["prefix_result_relative_path"]
    original_path = HERE / "results.json"
    hashes_before = {str(path): sha256(path) for path in [source, original_path]}
    scores, labels, _ = prefix_score_matrix(json.loads(source.read_text()))
    baseline_index = labels.index(config["baseline_lens"])
    original = json.loads(original_path.read_text())["prefix"]["loocv"]
    original_loo = leave_one_out(scores, labels)
    point_gain = float(np.mean(np.array(original_loo["held_out_scores"]) - scores[:, baseline_index]))
    counts, rejected = sample_valid_counts(len(scores), config["valid_bootstrap_draws"], np.random.default_rng(config["seed"]))
    fitted = refit_from_counts(scores, counts, baseline_index)
    gains = fitted["mean_gains"]
    interval = np.quantile(gains, [.025, .975]).tolist()
    result = {
        "analysis_status": config["analysis_status"],
        "config": config,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "n_conversations": len(scores),
        "n_candidate_lenses": len(labels),
        "unresampled_loocv_mean_gain": point_gain,
        "original_point_estimate_discrepancy": point_gain - original["gain_over_uniform_mass"]["mean"],
        "fixed_loss_bootstrap_ci95_preserved": original["gain_over_uniform_mass"]["ci95"],
        "selection_aware_bootstrap": {
            "valid_draws": len(gains),
            "rejected_single_identity_draws": rejected,
            "maximum_observed_multiplicity": int(counts.max()),
            "gain_mean": float(gains.mean()),
            "gain_std": float(gains.std(ddof=1)),
            "percentile_interval95": interval,
            "nonpositive_fraction": float(np.mean(gains <= 0)),
            "interval_contains_zero": bool(interval[0] <= 0 <= interval[1]),
        },
        "interpretation": "Point estimate remains positive, but lens-reselection sensitivity spans zero; no robust positive interval or population effect is established.",
        "limitations": [
            "post-review sensitivity with reviewer result known before reproduction",
            "12 selected conversations from one user; model, study design and archetype selection remain conditioned upon",
            "overlapping LOOCV folds; refitting is included in this empirical bootstrap, but this is not a population confidence guarantee",
            "nonpositive bootstrap fraction is descriptive, not a formal null-hypothesis p-value",
        ],
        "input_sha256_before": hashes_before,
        "input_hashes_unchanged": {path: sha256(Path(path)) == digest for path, digest in hashes_before.items()},
        "implementation_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(HERE / "selection-uncertainty-config.json"),
        "protocol_sha256": sha256(HERE / "PROTOCOL.md"),
        "runtime_seconds": time.perf_counter() - start,
    }
    return result


if __name__ == "__main__":
    config = json.loads((HERE / "selection-uncertainty-config.json").read_text())
    result = run(config)
    (HERE / "selection-uncertainty-results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
