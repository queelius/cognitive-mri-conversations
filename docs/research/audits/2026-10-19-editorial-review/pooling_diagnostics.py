"""Editorial diagnostics on PRISM development embeddings only (fit/tune/probe; the reserve
is never touched). Asks two questions about the pre-registered contrast MEAN minus WINDOW(2):
  (1) how much of the gain is the user's opening message or opening exchange, and
  (2) whether pooling more vectors raises raw cosine even when the extra vectors are
      unrelated (drawn from other components), i.e. a geometry artifact.
Same pooling, normalization, component aggregation and bootstrap as cmri_wave.
"""
import json, sys
from collections import defaultdict
import numpy as np

ROOT = sys.argv[1]   # a cache directory holding {minilm,e5}.{npz,json} for DEVELOPMENT data only
TARGETS = (4, 6, 8, 10)
DRAWS, SEED = 2000, 104729


def normalize(x):
    x = np.asarray(x, float)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


def comp_values(values, groups, convs):
    by_conv = defaultdict(list)
    for v, g, c in zip(values, groups, convs):
        by_conv[(g, c)].append(float(v))
    by_group = defaultdict(list)
    for (g, _), v in by_conv.items():
        by_group[g].append(np.mean(v))
    ids = sorted(by_group)
    return ids, np.array([np.mean(by_group[g]) for g in ids])


def paired(new, base, groups, convs):
    ids, d = comp_values(np.asarray(new) - np.asarray(base), groups, convs)
    rng = np.random.default_rng(SEED)
    means = d[rng.integers(len(ids), size=(DRAWS, len(ids)))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return float(d.mean()), float(lo), float(hi), len(ids), float(np.mean(d > 0))


def load(enc):
    z = np.load(f'{ROOT}/{enc}.npz')
    recs = json.load(open(f'{ROOT}/{enc}.json'))['records']
    assert all(r['split'] in ('fit', 'tune', 'probe') for r in recs)
    assert all(r.get('sample', 'development') == 'development' for r in recs)
    return z, recs


def run(enc):
    z, recs = load(enc)
    # role-wise donor banks over all development messages
    bank = {'user': [], 'assistant': []}
    bank_comp = {'user': [], 'assistant': []}
    fit_user = []
    for r in recs:
        x = z[f"{r['id']}__messages"]
        for p, role in enumerate(r['roles']):
            bank[role].append(x[p]); bank_comp[role].append(r['component'])
            if r['split'] == 'fit' and role == 'user':
                fit_user.append(x[p])
    bank = {k: np.stack(v) for k, v in bank.items()}
    bank_comp = {k: np.array(v) for k, v in bank_comp.items()}
    center = np.stack(fit_user).mean(axis=0)          # user-role center from FIT only
    centroid_dir = normalize(center)

    cases = []
    rng = np.random.default_rng(SEED)
    for r in recs:
        x = z[f"{r['id']}__messages"]; roles = r['roles']
        for j in range(4, len(x)):
            if roles[j] != 'user' or j not in TARGETS:
                continue
            h = x[:j]
            # donor history: same role pattern, every vector from a different component
            donor = np.empty_like(h)
            for p in range(j):
                role = roles[p]
                while True:
                    i = rng.integers(len(bank[role]))
                    if bank_comp[role][i] != r['component']:
                        donor[p] = bank[role][i]; break
            # within-conversation shuffle null: a random prior user message and a random prior
            # assistant message stand in for the last two (same role mix, same topic, no recency)
            ru = rng.choice([p for p in range(j) if roles[p] == 'user'])
            ra = rng.choice([p for p in range(j) if roles[p] == 'assistant'])
            cases.append(dict(id=r['id'], comp=r['component'], split=r['split'], j=j,
                              h=h, donor=donor, y=x[j], rand2=(ru, ra)))

    def preds(c):
        h, d, j = c['h'], c['donor'], c['j']
        return {
            'MEAN': h.mean(0), 'WIN2': h[-2:].mean(0), 'WIN4': h[-4:].mean(0),
            'FIRST': h[0], 'OPEN2': h[:2].mean(0),
            'FIRST+WIN2': np.vstack([h[:1], h[-2:]]).mean(0),
            'OPEN2+WIN2': np.vstack([h[:2], h[-2:]]).mean(0) if j > 4 else h.mean(0),
            'OWN_MEAN': h[0::2].mean(0),
            'DONOR_MEAN': d.mean(0), 'DONOR_WIN2': d[-2:].mean(0), 'DONOR_FIRST': d[0],
            'WIN2+DONOR_REST': np.vstack([h[-2:], d[:-2]]).mean(0) if j > 2 else h[-2:].mean(0),
            'WIN2+DONOR_FIRST': np.vstack([h[-2:], d[:1]]).mean(0),
            # the user's message before the latest exchange (equals FIRST when j == 4)
            'PREV_USER+WIN2': np.vstack([h[j - 4:j - 3], h[-2:]]).mean(0),
            'RAND2': np.vstack([h[c['rand2'][0]], h[c['rand2'][1]]]).mean(0),
        }

    names = list(preds(cases[0]))
    raw = {n: [] for n in names}; cen = {n: [] for n in names}
    normk = defaultdict(list); cosc = defaultdict(list); dnormk = defaultdict(list); dcosc = defaultdict(list)
    for c in cases:
        p = preds(c); y = normalize(c['y']); yc = normalize(c['y'] - center)
        for n in names:
            raw[n].append(float(normalize(p[n]) @ y))
            cen[n].append(float(normalize(p[n] - center) @ yc))
        m = c['h'].mean(0); dm = c['donor'].mean(0)
        normk[c['j']].append(np.linalg.norm(m)); cosc[c['j']].append(float(normalize(m) @ centroid_dir))
        dnormk[c['j']].append(np.linalg.norm(dm)); dcosc[c['j']].append(float(normalize(dm) @ centroid_dir))
    groups = [c['comp'] for c in cases]; convs = [c['id'] for c in cases]
    js = np.array([c['j'] for c in cases]); splits = np.array([c['split'] for c in cases])

    out = {'encoder': enc, 'cases': len(cases), 'conversations': len(set(convs)),
           'components': len(set(groups)), 'contrasts': {}, 'by_position': {}, 'geometry': {}}
    contrasts = [('MEAN', 'WIN2'), ('WIN4', 'WIN2'), ('FIRST', 'WIN2'), ('OPEN2', 'WIN2'),
                 ('FIRST+WIN2', 'WIN2'), ('OPEN2+WIN2', 'WIN2'), ('MEAN', 'FIRST+WIN2'),
                 ('MEAN', 'OPEN2+WIN2'), ('OWN_MEAN', 'WIN2'), ('FIRST', 'MEAN'),
                 ('DONOR_MEAN', 'DONOR_WIN2'), ('DONOR_MEAN', 'DONOR_FIRST'),
                 ('WIN2+DONOR_REST', 'WIN2'), ('WIN2+DONOR_FIRST', 'WIN2'),
                 ('WIN2', 'RAND2'), ('MEAN', 'RAND2')]
    for metric, table in (('raw', raw), ('centered', cen)):
        out['contrasts'][metric] = {}
        for a, b in contrasts:
            g, lo, hi, k, fpos = paired(table[a], table[b], groups, convs)
            out['contrasts'][metric][f'{a} - {b}'] = dict(gain=g, lo=lo, hi=hi, components=k, frac_pos=fpos)
        # probe-only replication of the pre-registered development value
        sel = splits == 'probe'
        g, lo, hi, k, _ = paired(np.array(table['MEAN'])[sel], np.array(table['WIN2'])[sel],
                                 [x for x, s in zip(groups, sel) if s], [x for x, s in zip(convs, sel) if s])
        out['contrasts'][metric]['MEAN - WIN2 (probe only)'] = dict(gain=g, lo=lo, hi=hi, components=k)
        out['contrasts'][metric]['levels'] = {n: float(np.mean(table[n])) for n in names}
    POS = (('MEAN', 'WIN2'), ('FIRST+WIN2', 'WIN2'), ('MEAN', 'FIRST+WIN2'),
           ('WIN4', 'WIN2'), ('MEAN', 'WIN4'), ('FIRST+WIN2', 'WIN4'), ('FIRST+WIN2', 'PREV_USER+WIN2'),
           ('WIN2+DONOR_REST', 'WIN2'), ('DONOR_MEAN', 'DONOR_WIN2'), ('WIN2', 'RAND2'), ('MEAN', 'RAND2'))
    out['by_position'] = {'raw': {}, 'centered': {}}
    for metric, table in (('raw', raw), ('centered', cen)):
        for j in TARGETS:
            sel = js == j
            row = {}
            for a, b in POS:
                g, lo, hi, k, _ = paired(np.array(table[a])[sel], np.array(table[b])[sel],
                                         [x for x, s in zip(groups, sel) if s], [x for x, s in zip(convs, sel) if s])
                row[f'{a} - {b}'] = dict(gain=g, lo=lo, hi=hi, components=k, cases=int(sel.sum()))
            out['by_position'][metric][j] = row
    for j in TARGETS:
        out['geometry'][j] = dict(mean_norm=float(np.mean(normk[j])), cos_to_centroid=float(np.mean(cosc[j])),
                                  donor_mean_norm=float(np.mean(dnormk[j])), donor_cos_to_centroid=float(np.mean(dcosc[j])))
    out['geometry']['single_message_cos_to_centroid'] = float(np.mean(normalize(bank['user']) @ centroid_dir))
    return out


def fmt(d):
    return f"{d['gain']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"


if __name__ == '__main__':
    results = {enc: run(enc) for enc in ('minilm', 'e5')}
    json.dump(results, open(sys.argv[2], 'w'), indent=1)
    print(f"# Source: {ROOT}")
    lines = []
    for enc, r in results.items():
        lines.append(f"\n## {enc}: {r['cases']} user targets, {r['conversations']} conversations, {r['components']} components (all development splits)\n")
        lines.append('| contrast | raw cosine | centered cosine | frac. comp. > 0 (raw) |')
        lines.append('|---|---|---|---|')
        for k, v in r['contrasts']['raw'].items():
            if k == 'levels':
                continue
            c = r['contrasts']['centered'].get(k)
            lines.append(f"| {k} | {fmt(v)} | {fmt(c) if c else ''} | {v.get('frac_pos', float('nan')):.2f} |")
        for metric in ('raw', 'centered'):
            bp = r['by_position'][metric]
            keys = list(next(iter(bp.values())))
            lines.append(f"\nBy target position, {metric} cosine (components / cases in last column):")
            lines.append('| j | ' + ' | '.join(keys) + ' | n |')
            lines.append('|---|' + '---|' * (len(keys) + 1))
            for j, row in bp.items():
                lines.append(f"| {j} | " + ' | '.join(fmt(row[k]) for k in keys)
                             + f" | {row[keys[0]]['components']}/{row[keys[0]]['cases']} |")
        g = r['geometry']
        lines.append(f"\nGeometry (user-role center from FIT): single user message cos to centroid {g['single_message_cos_to_centroid']:.3f}")
        lines.append('| j | norm of own mean | own mean cos to centroid | norm of donor mean | donor mean cos to centroid |')
        lines.append('|---|---|---|---|---|')
        for j in TARGETS:
            e = g[j]
            lines.append(f"| {j} | {e['mean_norm']:.3f} | {e['cos_to_centroid']:.3f} | {e['donor_mean_norm']:.3f} | {e['donor_cos_to_centroid']:.3f} |")
        lv = r['contrasts']['raw']['levels']
        lines.append('\nRaw-cosine levels: ' + ', '.join(f"{k} {v:.3f}" for k, v in lv.items()))
    print('\n'.join(lines))
