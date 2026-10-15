"""Read-only, aggregate-only diagnostics fixed in the adjacent PROTOCOL.md."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json
import platform
import resource
import subprocess
import time

import networkx as nx
import numpy as np
import scipy
from scipy.optimize import curve_fit


HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[4]
HIERARCHY = ROOT / "papers/hierarchical-memory"
RESULTS = HIERARCHY / "experiments/results/hierarchy_v2"


def hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def margins(rows: list[set[int]]) -> tuple[list[int], dict[int, int]]:
    return [len(row) for row in rows], dict(Counter(c for row in rows for c in row))


def trade(rows: list[set[int]], first: int, second: int,
          rng: np.random.Generator) -> None:
    """Uniform Curveball trade; the unchanged allocation is a possible draw."""
    left, right = rows[first], rows[second]
    common = left & right
    exclusive = sorted(left ^ right)
    take = len(left - common)
    if not exclusive:
        return
    chosen = set(int(c) for c in rng.choice(exclusive, size=take, replace=False))
    rows[first] = common | chosen
    rows[second] = common | (set(exclusive) - chosen)


def randomize(rows: list[set[int]], n_trades: int,
              rng: np.random.Generator) -> None:
    for _ in range(n_trades):
        first, second = rng.choice(len(rows), size=2, replace=False)
        trade(rows, int(first), int(second), rng)


def vocabulary(rows: list[set[int]]) -> np.ndarray:
    seen: set[int] = set()
    counts = []
    for row in rows:
        seen |= row
        counts.append(len(seen))
    return np.asarray(counts, dtype=float)


def expected_vocabulary(rows: list[set[int]]) -> np.ndarray:
    """Exact mean over all N! episode orderings, via no-hit probabilities."""
    n = len(rows)
    frequencies = np.array(list(margins(rows)[1].values()), dtype=float)
    unseen = np.ones_like(frequencies)
    counts = []
    for k in range(1, n + 1):
        unseen *= np.maximum(n - frequencies - k + 1, 0) / (n - k + 1)
        counts.append(np.sum(1 - unseen))
    return np.asarray(counts)


def fit_beta(counts: np.ndarray) -> float:
    x = np.arange(1, len(counts) + 1, dtype=float)
    params, _ = curve_fit(lambda t, scale, exponent: scale * np.power(t, exponent),
                          x, counts, p0=[1., .5], maxfev=10000)
    return float(params[1])


def projection(rows: list[set[int]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(c for row in rows for c in row)
    for row in rows:
        graph.add_edges_from(combinations(sorted(row), 2))
    return graph


@dataclass(frozen=True)
class GraphMetrics:
    total_nodes: int
    total_edges: int
    giant_nodes: int
    giant_edges: int
    clustering: float
    transitivity: float
    mean_path: float


def graph_metrics(rows: list[set[int]]) -> GraphMetrics:
    graph = projection(rows)
    giant = graph.subgraph(max(nx.connected_components(graph), key=len))
    return GraphMetrics(graph.number_of_nodes(), graph.number_of_edges(),
                        giant.number_of_nodes(), giant.number_of_edges(),
                        nx.average_clustering(giant), nx.transitivity(giant),
                        nx.average_shortest_path_length(giant))


def role_entropy(roles: list[int]) -> float:
    counts = Counter(zip(roles, roles[1:]))
    totals = Counter(a for a, _ in zip(roles, roles[1:]))
    n_edges = sum(counts.values())
    return float(-sum(count / n_edges * np.log2(count / totals[a])
                      for (a, _), count in counts.items())) if n_edges else 0.


def load_rows() -> tuple[list[set[int]], dict[str, object]]:
    # Text is used in memory only to join membership; outputs contain no text/IDs.
    extraction = json.loads((RESULTS / "extraction_state.json").read_text())
    hierarchy = json.loads((RESULTS / "hierarchy_semantic.json").read_text())
    membership = hierarchy["levels"][0]["cluster_membership"]
    mapping = {concept.lower(): index
               for index, concepts in enumerate(membership.values())
               for concept in concepts}
    episodes = extraction["episodes"]
    rows = [{mapping[c.lower()] for c in ep["concepts"] if c.lower() in mapping}
            for ep in episodes]
    return rows, {
        "n_episodes": len(rows), "n_meta_concepts": len(set(mapping.values())),
        "n_raw_concept_members": len(mapping),
        "bipartite_edges": sum(map(len, rows)),
        "unmapped_mentions": sum(c.lower() not in mapping
                                  for ep in episodes for c in ep["concepts"]),
        "ids_in_lexicographic_order": [ep["episode_id"] for ep in episodes]
                                     == sorted(ep["episode_id"] for ep in episodes),
    }


def source_manifest() -> dict[str, str]:
    files = [
        "future-ideas/iscs-2027/EVIDENCE.md", "future-ideas/iscs-2027/audit-results.json",
        "future-ideas/iscs-2027/audit_existing_results.py",
        "future-ideas/percolation-thresholds.md", "future-ideas/turn-taking-motifs.md",
        "future-ideas/compsac-2026/OUTLINE.md", "future-ideas/compsac-2026/paper/paper.tex",
        "papers/knowledge-networks/README.md", "papers/knowledge-networks/paper/paper.tex",
        "papers/temporal-networks/README.md", "papers/temporal-networks/paper/PLOS/paper.tex",
        "papers/temporal-networks/code/temporal_permutation_null.py",
        "papers/temporal-networks/code/temporal_preferential_attachment.py",
        "papers/temporal-networks/code/temporal_community_tracker.py",
        "papers/temporal-networks/code/exploration_index.py",
        "papers/hierarchical-memory/CLAUDE.md", "papers/hierarchical-memory/paper/paper.tex",
        "papers/hierarchical-memory/hierarchy_semantic.py",
        "papers/hierarchical-memory/concept_extraction_v2.py",
        "papers/hierarchical-memory/experiments/figures/heaps_null_model.py",
        "papers/hierarchical-memory/experiments/figures/cls_exploration.py",
        "papers/hierarchical-memory/experiments/figures/domain_flow.py",
        "papers/selective-forgetting/REPORT.md", "papers/selective-forgetting/REVIEW.md",
        "papers/selective-forgetting/LITERATURE.md",
    ]
    files += [str((RESULTS / name).relative_to(ROOT)) for name in [
        "extraction_state.json", "hierarchy_semantic.json", "concept_embeddings.npy",
        "heaps_null_model.json", "cls_exploration.json", "clustering_sensitivity.json",
        "flow_normalization.json", "domain_flow_analysis.json",
    ]]
    files += [str((HERE / name).relative_to(ROOT))
              for name in ["PROTOCOL.md", "audit_checks.py", "test_audit_checks.py"]]
    return {name: hash_file(ROOT / name) for name in files}


def main() -> None:
    start = time.perf_counter()
    sources = source_manifest()
    prior = json.loads((ROOT / "future-ideas/iscs-2027/audit-results.json").read_text())
    prior_identity = {name: hash_file(ROOT / name) == expected
                      for name, expected in prior["source_sha256"].items()}
    rows, provenance = load_rows()
    baseline_margins = margins(rows)
    observed = graph_metrics(rows)
    sampled = []
    for seed in range(114001, 114017):
        rng = np.random.default_rng(seed)
        randomized = [set(row) for row in rows]
        previous = 0
        for sweeps in (10, 40):
            randomize(randomized, (sweeps - previous) * len(rows), rng)
            assert margins(randomized) == baseline_margins
            sampled.append({"seed": seed, "trades_per_episode": sweeps,
                            **asdict(graph_metrics(randomized))})
            previous = sweeps
        print(f"Completed projection-null seed {seed}", flush=True)
    null_summary = {}
    for sweeps in (10, 40):
        group = [r for r in sampled if r["trades_per_episode"] == sweeps]
        summary = {}
        for key in ("total_edges", "giant_nodes", "clustering", "transitivity", "mean_path"):
            values = np.array([r[key] for r in group])
            summary[key] = {"mean": float(values.mean()), "min": float(values.min()),
                            "max": float(values.max()), "sd": float(values.std(ddof=1))}
        summary["relative_sigma"] = ((observed.clustering / summary["clustering"]["mean"])
                                     / (observed.mean_path / summary["mean_path"]["mean"]))
        summary["clustering_null_ge_observed"] = sum(r["clustering"] >= observed.clustering for r in group)
        null_summary[str(sweeps)] = summary
    expected = expected_vocabulary(rows)
    stored = json.loads((RESULTS / "heaps_null_model.json").read_text())
    chronology = json.loads((RESULTS / "cls_exploration.json").read_text())["E2_ordering"]
    role_checks = []
    for length in range(2, 21):
        roles = [i % 2 for i in range(length)]
        windows = {str(k): len({tuple(roles[i:i+k]) for i in range(length-k+1)})
                   for k in (3, 4, 5)}
        role_checks.append({"length": length, "conditional_entropy": role_entropy(roles),
                            "distinct_role_windows": windows})
    result = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Prewritten diagnostic audit; finite-chain null is exploratory; no raw outputs",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "networkx": nx.__version__},
        "source_sha256": sources,
        "git_heads": {name: subprocess.check_output(["git", "rev-parse", "HEAD"],
                       cwd=ROOT / "papers" / name, text=True).strip()
                      for name in ("knowledge-networks", "temporal-networks",
                                   "hierarchical-memory", "selective-forgetting")},
        "prior_temporal_source_identity": prior_identity,
        "hierarchy_provenance": provenance,
        "observed_projection": asdict(observed),
        "vocabulary": {"recomputed_saved_order_beta": fit_beta(vocabulary(rows)),
                       "stored_saved_order_beta": stored["real_beta"],
                       "exact_random_order_mean_curve_beta": fit_beta(expected),
                       "note": "Fit to expected curve is not the expectation of fitted exponents",
                       "stored_chronology_not_rerun": chronology},
        "projection_null": {"summary": null_summary, "draws": sampled,
                            "all_margins_preserved": True,
                            "limitation": "16 draws; 10/40-sweep stability is not a mixing proof"},
        "synthetic_role_checks": role_checks,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    for name, digest in sources.items():
        assert hash_file(ROOT / name) == digest, f"Input changed during check: {name}"
    (HERE / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"elapsed_seconds": result["elapsed_seconds"],
                      "observed": asdict(observed), "projection_null": null_summary,
                      "vocabulary": result["vocabulary"]}, indent=2))


if __name__ == "__main__":
    main()
