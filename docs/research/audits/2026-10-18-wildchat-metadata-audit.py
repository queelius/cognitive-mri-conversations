"""Metadata-only audit of the local WildChat-4.8M mirror (no conversation text read)."""
import collections
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

shards = sorted(Path(sys.argv[1]).glob('shard_*.parquet'))
cols = ['conversation_hash', 'model', 'timestamp', 'turn', 'language', 'toxic', 'redacted', 'hashed_ip']
total = 0
elig = 0                      # English, turn>=3, not toxic
by_model = collections.Counter()
by_turn = collections.Counter()
ips_all = collections.Counter()
ips_elig = collections.Counter()
hashes = set()
tmin = tmax = None
for i, shard in enumerate(shards):
    t = pq.read_table(shard, columns=cols)
    d = t.to_pydict()
    n = len(d['turn'])
    total += n
    for k in range(n):
        turn = d['turn'][k]
        by_turn[min(turn, 20)] += 1
        ip = d['hashed_ip'][k]
        ips_all[ip] += 1
        ts = d['timestamp'][k]
        if ts is not None:
            tmin = ts if tmin is None or ts < tmin else tmin
            tmax = ts if tmax is None or ts > tmax else tmax
        if d['language'][k] == 'English' and turn is not None and turn >= 3 and not d['toxic'][k]:
            elig += 1
            by_model[d['model'][k]] += 1
            ips_elig[ip] += 1
            hashes.add(d['conversation_hash'][k])
    print(f'shard {i+1}/{len(shards)} total={total} eligible={elig}', file=sys.stderr, flush=True)

ip_sizes = collections.Counter(ips_elig.values())
out = {
    'shards': len(shards),
    'conversations': total,
    'eligible_english_turn_ge3_nontoxic': elig,
    'eligible_unique_hashes': len(hashes),
    'eligible_by_model': dict(by_model.most_common()),
    'turn_histogram_capped20': dict(sorted(by_turn.items())),
    'hashed_ips_total': len(ips_all),
    'hashed_ips_with_eligible': len(ips_elig),
    'eligible_ip_size_histogram_first20': {k: ip_sizes[k] for k in sorted(ip_sizes)[:20]},
    'eligible_ips_with_ge4': sum(1 for v in ips_elig.values() if v >= 4),
    'timestamp_min': str(tmin), 'timestamp_max': str(tmax),
}
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print(json.dumps(out, indent=1))
