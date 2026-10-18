"""Replay saved aggregate evidence only: no raw conversations or model calls."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parents[2]


def pinned_bytes(root: Path, name: str, pin: dict) -> bytes:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('source path escapes workspace')
    content = path.read_bytes()
    if len(content) != pin['bytes'] or hashlib.sha256(content).hexdigest() != pin['sha256']:
        raise ValueError(f'source pin mismatch: {name}')
    return content


def verify_effect(effect: dict, draws: int, seed: int) -> dict:
    values = np.asarray(effect['component_differences'], dtype=float)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or len(values) != effect['components'] or type(draws) is not int or draws < 1):
        raise ValueError('finite aligned component values and positive draws required')
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    actual = dict(gain=float(values.mean()), ci_low=float(low), ci_high=float(high))
    for key, value in actual.items():
        if not np.isclose(value, effect[key], atol=1e-12, rtol=0):
            raise ValueError(f'recorded {key} differs from component replay')
    if not np.isclose(effect['new_mean'] - effect['baseline_mean'], actual['gain'], atol=1e-12, rtol=0):
        raise ValueError('recorded contrast orientation/means disagree')
    return actual


def extract_effects(encoder: str, result: dict, draws: int, seed: int) -> list[dict]:
    records = []
    for view, value in sorted(result['views'].items()):
        natural = value['regimes']['natural']
        cells = [('user_gain', 'natural', natural['user']['pooling']['contrasts']['mean_minus_window2']['cosine'])]
        cells += [('role_interaction', regime, row['role_interaction']['cosine'])
                  for regime, row in sorted(value['regimes'].items())]
        cells.append(('boundary_change', 'natural_minus_common', value['interaction_attenuation']['cosine']))
        for measure, regime, effect in cells:
            actual = verify_effect(effect, draws, seed)
            records.append(dict(encoder=encoder, view=view, regime=regime, measure=measure,
                                **actual, **{k:effect[k] for k in ('components','conversations','cases')},
                                component_differences=effect['component_differences']))
    return records


def figure(records: list[dict]) -> bytes:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11, 'axes.spines.right':False, 'axes.spines.top':False})
    rows = [r for r in records if r['view']=='native128' and r['measure']=='role_interaction']
    instruments = list(dict.fromkeys(r['encoder'] for r in rows))
    regimes = list(dict.fromkeys(r['regime'] for r in rows))
    fig, ax = plt.subplots(figsize=(7.1,2.8))
    for i, encoder in enumerate(instruments):
        for j, regime in enumerate(regimes):
            r = next(r for r in rows if r['encoder']==encoder and r['regime']==regime)
            y = i + (j - (len(regimes)-1)/2)*.22
            label = regime.replace('_',' ').capitalize() if i==0 else None
            ax.errorbar(r['gain'], y, xerr=[[r['gain']-r['ci_low']],[r['ci_high']-r['gain']]],
                        fmt='o', capsize=4, color=plt.get_cmap('tab10')(j), label=label)
    ax.axvline(0,color='.5',linewidth=1)
    ax.set(yticks=range(len(instruments)),yticklabels=[s.upper() for s in instruments],
           xlabel='Assistant minus user history gain (raw cosine)',
           title='Same targets, different available information',ylim=(-.45,len(instruments)-.55))
    ax.invert_yaxis()
    ax.legend(loc='lower left', bbox_to_anchor=(0,1.15), ncol=2, frameon=False)
    fig.tight_layout()
    stream=io.BytesIO();fig.savefig(stream,format='png',dpi=200,metadata={'Software':'CMRI aggregate evidence replay'})
    plt.close(fig)
    return stream.getvalue()


def execute(verify: bool = False) -> dict:
    started=time.perf_counter()
    manifest=json.loads((PACKAGE/'evidence-inputs.json').read_text())
    source={name:pinned_bytes(WORKSPACE,name,pin) for name,pin in manifest['files'].items()}
    cfg=json.loads(source[manifest['configuration']])
    records=[]
    for item in manifest['forecast_sources']:
        result=json.loads(source[item['path']])['results']
        if result['reserve_components_excluded']!=248:
            raise ValueError('unexpected reserve certificate')
        records += extract_effects(item['encoder'],result,cfg['bootstrap_replicates'],cfg['bootstrap_seed'])
    outputs={'data/forecast-effects.json':(json.dumps(records,indent=2,sort_keys=True)+'\n').encode(),
             'figures/forecast-origin.png':figure(records)}
    native=[r for r in records if r['view']=='native128' and r['measure']=='role_interaction']
    lines=[r'\begin{tabular}{llrrr}',r'Encoder & Forecast origin & Estimate & Lower & Upper \\',r'\hline\hline']
    for r in native:
        lines.append(f"{r['encoder'].upper()} & {r['regime'].replace('_',' ')} & {r['gain']:+.4f} & {r['ci_low']:+.4f} & {r['ci_high']:+.4f} "+r'\\')
    lines.append(r'\end{tabular}')
    outputs['data/forecast-table.tex']=('\n'.join(lines)+'\n').encode()
    for name,content in outputs.items():
        path=PACKAGE/name
        if verify:
            if not path.exists() or path.read_bytes()!=content:raise ValueError(f'output replay differs: {name}')
        else:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    log={'utc':datetime.now(timezone.utc).isoformat(),'verify':verify,'effects_recomputed':len(records),
         'outputs':{n:hashlib.sha256(b).hexdigest() for n,b in outputs.items()},
         'source_files_verified':len(source),'wall_seconds':time.perf_counter()-started,
         'new_natural_observations':0,'model_calls':0}
    with (PACKAGE/'verification-runs.jsonl').open('a') as stream:stream.write(json.dumps(log,sort_keys=True)+'\n')
    return log


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    print(json.dumps(execute(parser.parse_args().verify)))
