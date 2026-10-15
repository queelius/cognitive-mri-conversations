import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest


def module():
    path = Path(__file__).with_name('audit.py')
    assert path.exists(), 'The analytic audit implementation must exist'
    spec = importlib.util.spec_from_file_location('metric_audit', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('n', [4, 5, 8, 9, 16])
def test_distinct_lengths_have_closed_form(n):
    mod = module()
    x = np.linspace(0.1, 1.9, n)
    p = (n // 2) / n
    expected = -(p*np.log2(p)+(1-p)*np.log2(1-p))/n
    assert mod.median_entropy(x) == pytest.approx(expected)
    assert mod.median_entropy(x[::-1]) == pytest.approx(expected)


def test_ties_small_samples_and_invalid_values():
    mod = module()
    assert mod.median_entropy(np.ones(8)) == 0
    assert mod.median_entropy([0.1, 0.2, 0.3]) is None
    for x in ([0.1, np.nan, 0.2, 0.3], [-0.1, 0.2, 0.3, 0.4], [0, 1, 2, 3]):
        with pytest.raises(ValueError):
            mod.median_entropy(x)


def test_paths_realize_arbitrary_valid_step_distances():
    mod = module()
    distances = np.array([0, 0.1, 0.9, 2, 1.1, 0.2])
    points = mod.realize_steps(distances)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1)
    recovered = 1 - np.sum(points[:-1]*points[1:], axis=1)
    np.testing.assert_allclose(recovered, distances, atol=1e-12)
    with pytest.raises(ValueError):
        mod.realize_steps([2.01])


def test_reproducible_full_report(tmp_path):
    mod = module()
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    one, two = mod.evaluate(cfg), mod.evaluate(cfg)
    assert one == two
    assert one['maximum_identity_error'] < 1e-12
    assert len(one['length_checks']) == len(cfg['lengths'])
    assert len({x['entropy'] for x in one['order_examples']}) == 1
    assert len({round(x['lag1'], 3) for x in one['order_examples']}) > 1
    assert all(x['geometry_max_error'] < 1e-12 for x in one['order_examples'])


def test_complete_artifacts(tmp_path):
    module().main(tmp_path)
    report = json.loads((tmp_path/'results.json').read_text())
    runtime = json.loads((tmp_path/'runtime.json').read_text())
    assert report['maximum_identity_error'] < 1e-12
    assert runtime['runtime_seconds'] > 0
    assert len(runtime['config_sha256']) == 64
    assert (tmp_path/'median_entropy.png').read_bytes().startswith(b'\x89PNG')
    assert '<svg' in (tmp_path/'median_entropy.svg').read_text()
