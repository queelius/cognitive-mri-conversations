"""Exploratory aggregate queries, not a new evaluation or production pipeline.

Run from cmri with papers/temporal-networks-calibration/.venv/bin/python.
No private texts, concept names, identifiers or incidence rows are exported.
Uses existing audited loaders and cross-checks saved measurement results.
"""
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import sys
import time

import numpy as np

root = Path(__file__).resolve().parents[5]
started = time.perf_counter()
source = root / 'future-ideas/iscs-2027/research-deep-dive/checks/networks/audit_checks.py'
spec = importlib.util.spec_from_file_location('saved_network_audit', source)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)
rows, meta = audit.load_rows()
pairs = Counter(pair for row in rows for pair in combinations(sorted(row), 2))
total = sum(pairs.values())
assert total == sum(len(row) * (len(row) - 1) // 2 for row in rows)
prior = json.loads(source.with_name('results.json').read_text())
assert len(pairs) == prior['observed_projection']['total_edges']
pins = {}
for name in ('extraction_state.json', 'hierarchy_semantic.json'):
    p = audit.RESULTS / name
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual == prior['source_sha256'][str(p.relative_to(root))]
    pins[str(p.relative_to(root))] = actual

# Descriptive within/across-domain split using the existing hierarchy, not a fitted
# or outcome-selected new partition. Refuse to silently assign non-nested clusters.
hierarchy = json.loads((audit.RESULTS / 'hierarchy_semantic.json').read_text())
domains = {c.lower(): i for i, concepts in enumerate(hierarchy['levels'][-1]['cluster_membership'].values())
           for c in concepts}
meta_domains = []
for concepts in hierarchy['levels'][0]['cluster_membership'].values():
    owners = {domains[c.lower()] for c in concepts}
    if len(owners) != 1:
        raise ValueError('Hierarchy is not nested; no automatic domain assignment')
    meta_domains.append(owners.pop())
split = {}
for label, cross in [('within_domain', False), ('cross_domain', True)]:
    counts = [count for (a, b), count in pairs.items() if (meta_domains[a] != meta_domains[b]) == cross]
    split[label] = {'distinct_pairs': len(counts), 'pair_occurrences': sum(counts),
                    'pairs_seen_at_least_twice': sum(x >= 2 for x in counts),
                    'repeat_occurrence_fraction': 1 - len(counts) / sum(counts) if counts else None}
assert sum(x['pair_occurrences'] for x in split.values()) == total
null = prior['projection_null']['summary']['40']['total_edges']
result = {
    'scope': 'Post-pilot exploratory descriptive mining; no fresh holdout or significance test',
    'queried_at_utc': datetime.now(timezone.utc).isoformat(),
    'versions': {'python': platform.python_version(), 'numpy': np.__version__},
    'hierarchy': {'episodes': len(rows), 'pair_occurrences': total, 'distinct_pairs': len(pairs),
        'pairs_seen_once': sum(v == 1 for v in pairs.values()),
        'pairs_seen_at_least_twice': sum(v >= 2 for v in pairs.values()),
        'extra_pair_occurrences_beyond_first': total - len(pairs),
        'repeat_occurrence_fraction': 1 - len(pairs) / total,
        'saved_fixed_margin_null_mean_distinct_pairs': null['mean'],
        'null_implied_mean_repeat_fraction': 1 - null['mean'] / total,
        'existing_hierarchy_domain_split': split, 'no_chronological_reuse_claim': True},
    'public_nearest_returns': {}, 'input_sha256': pins,
}
sys.path.insert(0, str(root / 'papers/temporal-networks-calibration'))
from temporal_calibration.data import load_archive
from temporal_calibration.metrics import return_summary

for key in ('23736046cf70', '9342eba92593'):
    folder = root / 'papers/temporal-networks-calibration/pilot/public-data'
    a = load_archive(folder, key)
    result['input_sha256'][str((folder / f'{key}.npz').relative_to(root))] = a.manifest['sha256']
    order = np.argsort(a.times['primary'], kind='stable')
    times = a.times['primary'][order]
    ages, rankgaps, similarities = [], [], []
    for t, node in enumerate(order):
        stop = np.searchsorted(times, times[t], side='left')
        if not stop:
            continue
        values = a.similarity[node, order[:stop]]
        maximum = values.max()
        if maximum < .9:
            continue
        tied = np.flatnonzero(values == maximum)
        ages.append((times[t] - times[tied].mean()) / 86400)
        rankgaps.append(t - tied.mean())
        similarities.append(maximum)
    ages, rankgaps, similarities = map(np.array, (ages, rankgaps, similarities))
    reference = return_summary(a.similarity, order, times, [.9])[.9]
    assert len(ages) == reference['returns']
    assert np.isclose(ages.mean() * 86400, reference['lag_seconds'])
    result['public_nearest_returns'][key] = {
        'returns': len(ages),
        'calendar_days_quantiles': dict(zip(['q25', 'median', 'q75', 'q90'], np.quantile(ages, [.25, .5, .75, .9]).tolist())),
        'within_one_day': int((ages <= 1).sum()), 'over_30_days': int((ages > 30).sum()),
        'within_20_arrivals': int((rankgaps <= 20).sum()), 'over_100_arrivals': int((rankgaps > 100).sum()),
        'near_identical_vector_matches': int((similarities >= 1 - 1e-8).sum()),
        'reference_metrics_agree': True,
    }
result['elapsed_seconds'] = time.perf_counter() - started
result['probe_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
print(json.dumps(result, indent=2, allow_nan=False))
