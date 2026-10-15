"""Exact scalar-cosine degeneracy in the library's shape-distance path."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
sys.path.insert(0, CONFIG["embflow_root"])
import embflow as ef


def check():
    start = time.perf_counter()
    angle_a = np.array([0., .1, .2, .3])
    angle_b = np.array([0., .6, 1.8, 2.0])
    a = np.column_stack([np.cos(angle_a), np.sin(angle_a)])
    b = np.column_stack([np.cos(angle_b), np.sin(angle_b)])
    profile_a = ef.angular_velocity(a)
    profile_b = ef.angular_velocity(b)
    costs = 1 - cosine_similarity(profile_a[:, None], profile_b[:, None])
    source = Path(CONFIG["embflow_root"]) / "embflow/compare.py"
    return {
        "angles_a": angle_a.tolist(), "angles_b": angle_b.tolist(),
        "profile_a": profile_a.tolist(), "profile_b": profile_b.tolist(),
        "profile_euclidean_difference": float(np.linalg.norm(profile_a - profile_b)),
        "all_profile_values_strictly_positive": bool((profile_a > 0).all() and (profile_b > 0).all()),
        "maximum_absolute_scalar_cosine_cost": float(np.max(np.abs(costs))),
        "library_shape_distance": ef.trajectory_distance(a, b, method="shape"),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "protocol_with_addendum_sha256": hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest(),
        "runtime_seconds": time.perf_counter() - start,
    }


if __name__ == "__main__":
    result = check()
    (HERE / "shape-results.json").write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
