#!/usr/bin/env python3
"""Read-only checks supporting the ISCS proposal comparison.

Run from any directory with numpy, scipy, networkx, and python-louvain installed.
Imports the existing temporal code, uses synthetic nodes only, and prints JSON.
This records diagnostic observations; it is not a passing test suite for that code.
"""

import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def main():
    root = Path(__file__).resolve().parents[2]
    temporal = root / "papers/temporal-networks"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(temporal / "code"))
    from exploration_index import indices
    from temporal_community_tracker import track_communities

    sources = [
        temporal / "code/temporal_community_tracker.py",
        temporal / "code/exploration_index.py",
        temporal / "code/temporal_permutation_null.py",
        temporal / "data/temporal/permutation_null.json",
        temporal / "data/wildchat/multiuser_summary.csv",
        temporal / "data/wildchat/exploration_index.csv",
        temporal / "paper/PLOS/paper.tex",
    ]
    out = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic fixtures and stored-output checks; no full pipeline rerun.",
        "temporal_git_head": subprocess.check_output(
            ["git", "-C", str(temporal), "rev-parse", "HEAD"], text=True
        ).strip(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "scipy", "networkx", "python-louvain"]},
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    }

    a, b = set("abc"), set("def")
    fixtures = [
        ("continuation", {0: a, 1: b}, {0: a, 1: b}, "continuation"),
        ("merge", {0: a, 1: b}, {0: a | b}, "merge"),
        ("split", {0: a | b}, {0: a, 1: b}, "split"),
    ]
    out["community_fixtures"] = {}
    for name, before, after, expected in fixtures:
        _, events = track_communities([
            ("2000-01", before, None), ("2000-02", after, None)
        ])
        counts = Counter(e.event_type for e in events if e.month == "2000-02")
        out["community_fixtures"][name] = {
            "expected_event": expected,
            "observed_counts": dict(counts),
            "expected_event_detected": counts[expected] > 0,
        }

    out["sequence_fixtures"] = {}
    for name, order in [("recent", [0, 0, 1, 1, 2, 2]),
                        ("delayed", [0, 1, 2, 0, 1, 2])]:
        step, revisit = indices(np.eye(3)[order])
        out["sequence_fixtures"][name] = {
            "topic_order": order, "step_index": step, "revisit_index": revisit
        }

    with (temporal / "data/wildchat/exploration_index.csv").open() as f:
        rows = list(csv.DictReader(f))
    out["stored_index_correlations"] = {"n_archives": len(rows)}
    for field in ["step_index", "revisit_index"]:
        result = spearmanr([float(r[field]) for r in rows],
                           [float(r["beta_z"]) for r in rows])
        out["stored_index_correlations"][field] = {
            "spearman_rho": float(result.statistic), "p_value": float(result.pvalue)
        }

    with (temporal / "data/wildchat/multiuser_summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    out["stored_multiarchive_summary"] = {
        "n_archives": len(rows),
        "gamma_range": [min(float(r["gamma"]) for r in rows),
                        max(float(r["gamma"]) for r in rows)],
        "gamma_p_above_0_05": sum(float(r["gamma_p"]) > .05 for r in rows),
        "degenerate_beta_count": sum(float(r["beta"]) == 0 for r in rows),
    }
    data = json.loads((temporal / "data/temporal/permutation_null.json").read_text())
    out["stored_case_permutation_null"] = {
        "meta": data["meta"],
        "scalars": {k: data["scalars"][k] for k in ["gamma", "beta"]},
    }
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
