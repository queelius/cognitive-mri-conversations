"""Guard against altered sources, copied uncertainty and mismatched group vectors."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('build_evidence') is not None, 'evidence verifier missing'
    import build_evidence
    return build_evidence


def effect():
    return {'component_differences':[0.,1.,2.], 'components':3,
            'gain':1., 'ci_low':0., 'ci_high':2.,
            'new_mean':2., 'baseline_mean':1.}


def test_component_bootstrap_matches_exhaustive_three_group_bounds():
    result=api().verify_effect(effect(),draws=2000,seed=104729)
    assert result['gain']==1.
    assert result['ci_low']==0. and result['ci_high']==2.


@pytest.mark.parametrize('change', [
    {'gain':1.2}, {'ci_low':.5}, {'components':4},
    {'component_differences':[0.,1.,float('nan')]},
    {'new_mean':3.}, {'component_differences':[[0.,1.,2.]]}])
def test_report_inconsistency_is_rejected(change):
    with pytest.raises(ValueError):api().verify_effect({**effect(),**change},2000,104729)


def test_source_pin_rejects_same_size_tampering_and_path_escape(tmp_path):
    a=api();p=tmp_path/'source.json';p.write_text('{"value":1}')
    pin={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    assert json.loads(a.pinned_bytes(tmp_path,'source.json',pin))=={'value':1}
    p.write_text('{"value":2}')
    with pytest.raises(ValueError,match='pin'):a.pinned_bytes(tmp_path,'source.json',pin)
    with pytest.raises(ValueError,match='escape'):a.pinned_bytes(tmp_path,'../outside',pin)


def test_distinct_forecast_regimes_remain_labeled_and_complete():
    a=api();e={**effect(),'conversations':4,'cases':8}
    data={'views':{'native128':{'regimes':{
        'natural':{'user':{'pooling':{'contrasts':{'mean_minus_window2':{'cosine':e}}}},'role_interaction':{'cosine':e}},
        'common_prefix':{'role_interaction':{'cosine':e}}},
        'interaction_attenuation':{'cosine':e}}}}
    records=a.extract_effects('test',data,2000,104729)
    assert len(records)==4
    assert {r['regime'] for r in records if r['measure']=='role_interaction'}=={'natural','common_prefix'}
    assert all(r['components']==3 and r['conversations']==4 and r['cases']==8 for r in records)
    assert all('component_ids' not in r for r in records)


def test_portable_saved_component_effects_replay_without_source_repositories():
    package=Path(__file__).resolve().parents[1]
    records=json.loads((package/'data/forecast-effects.json').read_text())
    assert len(records)==32
    assert {r['encoder'] for r in records}=={'minilm','e5'}
    for r in records:
        # The stored vector is a paired difference: its comparison with zero
        # is sufficient to replay the mean and bootstrap interval.
        actual=api().verify_effect({**r,'new_mean':r['gain'],'baseline_mean':0.},2000,104729)
        assert actual['gain']==pytest.approx(r['gain'],abs=1e-12)


def test_source_backed_replay_is_byte_identical():
    a=api()
    if not (a.WORKSPACE/'papers/history-falsification/results/public-minilm.json').exists():
        pytest.skip('optional full source audit requires sibling CMRI result repositories')
    result=a.execute(verify=True)
    assert result['effects_recomputed']==32 and len(result['outputs'])==3
    assert result['new_natural_observations']==0 and result['model_calls']==0
