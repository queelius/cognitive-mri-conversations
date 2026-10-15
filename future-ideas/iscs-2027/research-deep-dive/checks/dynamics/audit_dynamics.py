"""Bounded, aggregate-only scientific checks for the dynamics audit.

Run from this directory with its pinned environment. No API calls or source writes.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import platform
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
ROOT = Path(CONFIG["source_root"])
EMBFLOW_ROOT = Path(CONFIG["embflow_root"])
sys.path.insert(0, str(EMBFLOW_ROOT))
import embflow as ef  # noqa: E402


def unit_rows(points):
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=-1, keepdims=True)
    return np.divide(points, norms, out=np.zeros_like(points), where=norms > 0)


def exchangeable_triple(points):
    """Uniform permutations are an exactly exchangeable, non-i.i.d. law."""
    turns, dots, squared_norms = [], [], []
    for order in itertools.permutations(range(3)):
        seq = points[list(order)]
        velocities = np.diff(seq, axis=0)
        turns.append(float(ef.turning_cosines(seq)[0]))
        dots.append(float(velocities[0] @ velocities[1]))
        squared_norms.append(float(velocities[0] @ velocities[0]))
    return {
        "n_permutations": len(turns),
        "mean_turning_cosine": float(np.mean(turns)),
        "ratio_of_expectations": float(np.mean(dots) / np.mean(squared_norms)),
    }


def turning_check(seed, n):
    points = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, size=(n, 3))
    triples = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    velocities = np.diff(triples, axis=1)
    turns = (unit_rows(velocities[:, 0]) * unit_rows(velocities[:, 1])).sum(axis=1)
    mean, se = float(turns.mean()), float(turns.std(ddof=1) / np.sqrt(n))
    return {
        "exact_exchangeable_unit_triangle": exchangeable_triple(points),
        "same_triangle_embedded_in_256_dimensions": exchangeable_triple(np.pad(points, ((0, 0), (0, 254)))),
        "iid_circle": {
            "n_independent_triples": n, "mean": mean, "monte_carlo_se": se,
            "monte_carlo_ci95": [mean - 1.96 * se, mean + 1.96 * se],
            "analytic_expectation": float(-4 / np.pi**2),
        },
        "universal_minus_half_claim_falsified": True,
    }


def calculus_checks(seed):
    rng = np.random.default_rng(seed)
    points = unit_rows(rng.normal(size=(12, 5)))
    alpha = 0.85
    state = ef.leaky_state(points, alpha)
    pure_sum = ef.leaky_state(points, 1.)
    path = ef.trajectory(points, alpha)
    velocities = ef.velocity(path)
    gram = ef.velocity_gram(path)
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    offset = np.arange(5, dtype=float) + 1
    straight = np.array([[0., 0.], [1., 0.], [3., 0.]])
    perm = np.arange(len(points))[::-1]
    masses = np.arange(1, len(points) + 1, dtype=float)
    return {
        "alpha": alpha,
        "leaky_recurrence_identity_error": float(np.max(np.abs(np.diff(state, axis=0) - ((alpha - 1) * state[:-1] + points[1:])))),
        "leaky_force_equals_increment_error": float(np.max(np.abs(np.diff(state, axis=0) - points[1:]))),
        "pure_sum_force_equals_increment_error": float(np.max(np.abs(np.diff(pure_sum, axis=0) - points[1:]))),
        "normalized_force_equals_increment_error": float(np.max(np.abs(np.diff(ef.trajectory(points, 1.), axis=0) - points[1:]))),
        "mean_velocity_endpoint_identity_error": float(np.max(np.abs(velocities.mean(axis=0) - (path[-1] - path[0]) / (len(path) - 1)))),
        "mean_second_difference_boundary_identity_error": float(np.max(np.abs(ef.curvature(path).mean(axis=0) - (velocities[-1] - velocities[0]) / (len(path) - 2)))),
        "unit_adjacent_cosine_identity_error": float(np.max(np.abs(np.square(ef.speed(points)) - (2 - 2 * (points[:-1] * points[1:]).sum(axis=1))))),
        "velocity_gram_rigid_transform_error": float(np.max(np.abs(gram - ef.velocity_gram(path @ rotation + offset)))),
        "normalized_trajectory_translation_error": float(np.max(np.abs(ef.velocity_gram(ef.trajectory(points + offset, alpha)) - gram))),
        "straight_path_second_difference_norm": float(np.linalg.norm(ef.curvature(straight))),
        "straight_path_curvature_radius_is_infinite": bool(np.isinf(ef.local_curvature_radius(straight)[0])),
        "uniform_reordering_error": float(np.max(np.abs(ef.smooth_uniform(points)[-1] - ef.smooth_uniform(points[perm])[-1]))),
        "mass_weighted_reordering_error": float(np.max(np.abs(ef.trajectory(points, 1., masses)[-1] - ef.trajectory(points[perm], 1., masses[perm])[-1]))),
        "exponential_reordering_error": float(np.max(np.abs(ef.trajectory(points, alpha)[-1] - ef.trajectory(points[perm], alpha)[-1]))),
        "decay_at_lambda_dt_one": {"exact": float(np.exp(-1)), "explicit_euler": 0.},
        "half_lives": {str(a): ef.alpha_to_half_life(a) for a in [0.5, 0.76, 0.782, 0.83, 0.84, 0.85, 0.95, 0.999]},
    }


def mean_interval(values, rng, draws):
    values = np.asarray(values, dtype=float)
    sample_means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {"n": len(values), "mean": float(values.mean()),
            "ci95": np.quantile(sample_means, [0.025, 0.975]).tolist(),
            "positive_fraction": float((values > 0).mean())}


def prefix_score_matrix(saved):
    labels = list(saved["lens_means_real"])
    identifiers = [cid for cid, case in saved["per_case"].items()
                   if not case["archetype"].startswith("synthetic_") and case["K"] >= 2]
    scores = np.array([[np.mean(saved["per_case"][cid]["sims"][label][1:])
                        for label in labels] for cid in identifiers])
    return scores, labels, identifiers


def leave_one_out(scores, labels):
    selected, held_out = [], []
    for index in range(len(scores)):
        training = np.delete(scores, index, axis=0)
        choice = int(np.argmax(training.mean(axis=0)))
        selected.append(labels[choice])
        held_out.append(float(scores[index, choice]))
    return {"chosen_labels": selected, "held_out_scores": held_out}


def prefix_check(saved, seed, draws):
    scores, labels, identifiers = prefix_score_matrix(saved)
    means = scores.mean(axis=0)
    selected = int(means.argmax())
    uniform, mass = labels.index("uniform"), labels.index("uniform+mass")
    rng = np.random.default_rng(seed)
    loo = leave_one_out(scores, labels)
    alpha_pairs = np.array([[a["best_fit_alpha"], a["adaptive_alpha"]] for a in saved["alpha_coherence"]])
    spearman = spearmanr(alpha_pairs[:, 0], alpha_pairs[:, 1])
    archetypes = defaultdict(list)
    for index, cid in enumerate(identifiers):
        archetypes[saved["per_case"][cid]["archetype"]].append(index)
    return {
        "n_real_conversations": len(identifiers),
        "n_eligible_prefixes": sum(saved["per_case"][cid]["K"] - 1 for cid in identifiers),
        "n_lenses_selected_over": len(labels),
        "max_discrepancy_from_archived_lens_means": float(np.max(np.abs(means - [saved["lens_means_real"][label] for label in labels]))),
        "best_lens": labels[selected], "best_mean": float(means[selected]),
        "uniform_mean": float(means[uniform]), "uniform_mass_mean": float(means[mass]),
        "gain_over_uniform": mean_interval(scores[:, selected] - scores[:, uniform], rng, draws),
        "gain_over_uniform_mass": mean_interval(scores[:, selected] - scores[:, mass], rng, draws),
        "loocv": {
            "selected_lens_counts": dict(Counter(loo["chosen_labels"])),
            "mean": float(np.mean(loo["held_out_scores"])),
            "gain_over_uniform_mass": mean_interval(np.array(loo["held_out_scores"]) - scores[:, mass], rng, draws),
            "in_sample_minus_loocv_mean": float(means[selected] - np.mean(loo["held_out_scores"])),
        },
        "by_archetype": {archetype: {
            "n": len(indices),
            "best_global_lens_mean": float(scores[indices, selected].mean()),
            "uniform_mass_mean": float(scores[indices, mass].mean()),
            "paired_gain": float((scores[indices, selected] - scores[indices, mass]).mean()),
        } for archetype, indices in archetypes.items()},
        "alpha_coherence": {
            "n": len(alpha_pairs), "reconstruction_mean": float(alpha_pairs[:, 0].mean()),
            "prediction_mean": float(alpha_pairs[:, 1].mean()),
            "reconstruction_range": [float(alpha_pairs[:, 0].min()), float(alpha_pairs[:, 0].max())],
            "reconstruction_minus_prediction": mean_interval(alpha_pairs[:, 0] - alpha_pairs[:, 1], rng, draws),
            "pearson_r": float(np.corrcoef(alpha_pairs.T)[0, 1]),
            "spearman_r": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
            "n_reconstruction_alphas_above_095": int((alpha_pairs[:, 0] > .95).sum()),
        },
        "archived_null_mismatched_mean": saved["null_mismatched_mean"],
        "archived_null_crossprefix_mean": saved["null_crossprefix_mean"],
        "archived_retrieval_top1_exact": saved["retrieval_top1_exact"],
        "archived_retrieval_top1_conversation": saved["retrieval_top1_conv"],
        "exact_retrieval_prediction_met": saved["retrieval_top1_exact"] >= .5,
        "archived_gate": saved["gate"],
        "uniform_mass_also_clears_numeric_gate_using_archived_best_lens_null": bool(means[mass] >= .8 and means[mass] >= saved["null_mismatched_mean"] + .25),
        "uniform_null_caveat": "Comparator uses the saved best-lens null; lens-specific nulls are unavailable.",
        "order_sensitivity_tested": False,
    }


def prediction_scores(points, grid, epsilon=1e-8):
    states = np.repeat(points[0][None].astype(float), len(grid), axis=0)
    scores = []
    for target in points[1:]:
        norms = np.linalg.norm(states, axis=1, keepdims=True)
        normalized = np.divide(states, norms + epsilon, out=np.zeros_like(states), where=(norms + epsilon) > 0)
        scores.append(normalized @ target)
        states = grid[:, None] * states + target[None]
    return np.array(scores)


def chronological_evaluation(points, grid, fixed_alpha=.85, train_fraction=.5, epsilon=1e-8):
    all_grid = np.concatenate([grid, [fixed_alpha, 1., 0.]])
    scores = prediction_scores(points, all_grid, epsilon)
    split = int(len(points) * train_fraction)
    train, test = scores[:split - 1], scores[split - 1:]
    selected = int(train[:, :len(grid)].mean(axis=0).argmax())
    oracle = int(scores[:, :len(grid)].mean(axis=0).argmax())
    return {
        "selected_alpha": float(grid[selected]),
        "oracle_full_sequence_alpha": float(grid[oracle]),
        "n_train_pairs": len(train), "n_test_pairs": len(test),
        "held_out_selected": float(test[:, selected].mean()),
        "held_out_fixed": float(test[:, -3].mean()),
        "held_out_running_mean": float(test[:, -2].mean()),
        "held_out_last_message": float(test[:, -1].mean()),
        "held_out_full_sequence_oracle": float(test[:, oracle].mean()),
    }


def database_check(config):
    import sqlite_vec
    db_path = ROOT / "papers/operational-memex/data/analysis.db"
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.execute("PRAGMA query_only = ON")
    model_counts = con.execute("SELECT model_id, COUNT(*) FROM message_embeddings GROUP BY model_id ORDER BY COUNT(*) DESC").fetchall()
    model_id = model_counts[0][0]
    model = con.execute("SELECT provider, model, dimensions FROM embedding_models WHERE id=?", (model_id,)).fetchone()
    metadata = {
        "conversation_rows": con.execute("SELECT COUNT(*) FROM conversations").fetchone()[0],
        "message_embedding_rows": con.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0],
        "n_models_with_embeddings": len(model_counts),
        "selected_model": dict(zip(["provider", "model", "dimensions"], model)),
        "embedding_rows_per_source": dict(con.execute("SELECT c.source, COUNT(*) FROM message_embeddings me JOIN conversations c ON c.id=me.conversation_id WHERE me.model_id=? GROUP BY c.source", (model_id,))),
        "distinct_embedded_timestamps": con.execute("SELECT COUNT(DISTINCT embedded_at) FROM message_embeddings WHERE model_id=?", (model_id,)).fetchone()[0],
    }
    groups = defaultdict(list)
    query = """SELECT c.source, me.conversation_id, COUNT(*) FROM message_embeddings me
        JOIN conversations c ON c.id=me.conversation_id
        WHERE me.model_id=? AND me.role IN ('user','assistant')
        GROUP BY c.source, me.conversation_id HAVING COUNT(*) >= ?"""
    for source, cid, count in con.execute(query, (model_id, config["min_messages"])):
        groups[source].append(cid)
    selected = []
    source_for = {}
    for source in sorted(groups):
        ordered = sorted(groups[source], key=lambda cid: hashlib.sha256(f"{config['seed']}:{cid}".encode()).digest())
        for cid in ordered[:config["max_conversations_per_source"]]:
            selected.append(cid)
            source_for[cid] = source
    if len(selected) > 240:
        raise ValueError("More than three platforms; protocol cap would be exceeded")
    placeholders = ','.join('?' for _ in selected)
    query = f"""WITH ranked AS (
        SELECT conversation_id, message_id, model_id, role, char_count, embedded_at,
               ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY rowid) AS seq
        FROM message_embeddings
        WHERE model_id=? AND role IN ('user','assistant') AND conversation_id IN ({placeholders})
    ) SELECT r.conversation_id, r.role, r.char_count, r.embedded_at, mv.embedding
      FROM ranked r JOIN message_vectors mv ON mv.conversation_id=r.conversation_id
        AND mv.message_id=r.message_id AND mv.model_id=r.model_id
      WHERE r.seq <= ? ORDER BY r.conversation_id, r.seq"""
    records = defaultdict(list)
    for cid, role, chars, embedded, blob in con.execute(query, (model_id, *selected, config["max_messages"])):
        records[cid].append((role, chars, embedded, np.frombuffer(blob, dtype=np.float32)))
    con.close()
    rng = np.random.default_rng(config["seed"])
    by_source = defaultdict(list)
    for cid, rows in records.items():
        points = unit_rows(np.array([row[3] for row in rows]))
        result = chronological_evaluation(points, np.array(config["alpha_grid"]), config["fixed_alpha"], config["train_fraction"], config["normalization_epsilon"])
        result["user_fraction"] = float(np.mean([row[0] == "user" for row in rows]))
        result["role_transition_fraction"] = float(np.mean([a[0] != b[0] for a, b in zip(rows, rows[1:])]))
        result["unique_embedded_timestamp_fraction"] = len({row[2] for row in rows}) / len(rows)
        by_source[source_for[cid]].append(result)
    summary = {}
    for source, rows in by_source.items():
        alpha = np.array([row["selected_alpha"] for row in rows])
        selected_scores = np.array([row["held_out_selected"] for row in rows])
        summary[source] = {
            "n": len(rows), "eligible_pool": len(groups[source]),
            "train_selected_alpha_mean": float(alpha.mean()),
            "train_selected_alpha_median": float(np.median(alpha)),
            "upper_alpha_boundary_fraction": float((alpha == max(config["alpha_grid"])).mean()),
            "mean_user_fraction": float(np.mean([row["user_fraction"] for row in rows])),
            "mean_role_transition_fraction": float(np.mean([row["role_transition_fraction"] for row in rows])),
            "mean_unique_embedded_timestamp_fraction": float(np.mean([row["unique_embedded_timestamp_fraction"] for row in rows])),
            "held_out_mean_scores": {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key.startswith("held_out_")},
            "paired_differences": {key: mean_interval(selected_scores - [row[key] for row in rows], rng, config["prediction_bootstrap_draws"])
                                   for key in ["held_out_fixed", "held_out_running_mean", "held_out_last_message", "held_out_full_sequence_oracle"]},
        }
    return {"metadata": metadata, "n_selected": len(selected), "n_loaded": len(records),
            "cohort_digest_sha256": hashlib.sha256('\n'.join(sorted(selected)).encode()).hexdigest(),
            "by_source": summary, "private_identifiers_exported": False,
            "order_provenance": "message_embeddings.rowid; insertion order, not verified event time",
            "sampling_limitation": "Within-user, selected long-conversation diagnostic; source effects are confounded."}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def provenance():
    files = [
        ROOT / "CLAUDE.md",
        ROOT / "papers/embedding-dynamics/CLAUDE.md",
        ROOT / "papers/embedding-dynamics/paper/SPEC.md",
        ROOT / "papers/embedding-dynamics/paper/main.tex",
        ROOT / "papers/embedding-dynamics/paper/sections/04_validation.tex",
        ROOT / "papers/embedding-dynamics/.papermill/state.md",
        ROOT / "papers/embedding-dynamics/experiments/EXPERIMENTS.md",
        ROOT / "papers/embedding-dynamics/experiments/motionlib.py",
        ROOT / "papers/embedding-dynamics/experiments/exp01_shuffle_validation.py",
        ROOT / "papers/embedding-dynamics/experiments/exp02_platform_separability.py",
        ROOT / "papers/embedding-dynamics/experiments/exp03_motion_motifs.py",
        ROOT / "papers/embedding-dynamics/experiments/exp04_prefix_experiment.py",
        ROOT / "papers/embedding-dynamics/experiments/results/SUMMARY.md",
        ROOT / "papers/embedding-dynamics/experiments/results/exp01_shuffle_validation.json",
        ROOT / "papers/embedding-dynamics/experiments/results/exp02_platform_separability.json",
        ROOT / "papers/embedding-dynamics/experiments/results/exp03_motifs.json",
        ROOT / "papers/embedding-dynamics/experiments/results/exp04_prefix.json",
        ROOT / "papers/semantic-dynamics/CLAUDE.md",
        ROOT / "papers/semantic-dynamics/paper/paper.md",
        ROOT / "papers/semantic-dynamics/paper/paper.tex",
        ROOT / "papers/semantic-dynamics/experiments/exp01_velocity_graph.py",
        ROOT / "papers/semantic-dynamics/experiments/exp03_strengthen.py",
        ROOT / "papers/semantic-dynamics/experiments/results/velocity_graph.json",
        ROOT / "papers/semantic-dynamics/experiments/results/exp03_strengthen.json",
        ROOT / "papers/operational-memex/CLAUDE.md",
        ROOT / "papers/operational-memex/experiments/exp17_exp_embedding_deep.py",
        ROOT / "papers/operational-memex/experiments/exp19_changepoints.py",
        ROOT / "papers/operational-memex/data/analysis.db",
        EMBFLOW_ROOT / "CLAUDE.md", EMBFLOW_ROOT / "pyproject.toml",
    ]
    files += sorted((EMBFLOW_ROOT / "embflow").glob("*.py"))
    files += sorted((EMBFLOW_ROOT / "tests").glob("*.py"))
    repos = [ROOT / "papers/embedding-dynamics", ROOT / "papers/semantic-dynamics", ROOT / "papers/operational-memex", EMBFLOW_ROOT]
    return {
        "files": {str(path): {"sha256": file_sha256(path), "bytes": path.stat().st_size} for path in files if path.exists()},
        "repos": {str(repo): {"head": subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                              "status_porcelain": subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).splitlines()}
                  for repo in repos},
        "protocol_sha256": file_sha256(HERE / "PROTOCOL.md"),
    }


def main():
    start = time.perf_counter()
    source_provenance = provenance()
    (HERE / 'source_hashes.json').write_text(json.dumps(source_provenance, indent=2) + '\n')
    result = {"config": CONFIG, "python": platform.python_version(),
              "dependencies": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn", "sqlite-vec", "pytest", "pytest-cov"]}}
    tasks = {
        "turning": lambda: turning_check(CONFIG["seed"], CONFIG["circle_triples"]),
        "calculus": lambda: calculus_checks(CONFIG["seed"]),
        "prefix": lambda: prefix_check(json.loads((ROOT / "papers/embedding-dynamics/experiments/results/exp04_prefix.json").read_text()), CONFIG["seed"], CONFIG["prefix_bootstrap_draws"]),
        "prediction": lambda: database_check(CONFIG),
    }
    for name, fn in tasks.items():
        task_start = time.perf_counter()
        try:
            result[name] = fn()
        except Exception as exc:
            result[name] = {"failed": True, "error_type": type(exc).__name__, "error": str(exc)}
        result[name]["runtime_seconds"] = time.perf_counter() - task_start
        print(json.dumps({"stage": name, "failed": result[name].get("failed", False), "runtime_seconds": result[name]["runtime_seconds"]}), flush=True)
        (HERE / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    result["runtime_seconds"] = time.perf_counter() - start
    result["source_integrity_after"] = {path: file_sha256(Path(path)) == info["sha256"] for path, info in source_provenance["files"].items()}
    (HERE / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({"all_source_hashes_unchanged": all(result["source_integrity_after"].values()), "runtime_seconds": result["runtime_seconds"]}))


if __name__ == '__main__':
    main()
