"""Independent implementation of the audited formula, not vendored source code."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import time

import numpy as np


def valid_distances(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 2):
        raise ValueError('Expected a finite vector of cosine distances in [0, 2]')
    return x


def binary_entropy(p):
    return 0.0 if p in (0.0, 1.0) else float(-p*np.log2(p)-(1-p)*np.log2(1-p))


def median_entropy(values):
    x = valid_distances(values)
    if len(x) <= 3:
        return None
    return binary_entropy(float(np.mean(x > np.median(x)))) / len(x)


def realize_steps(values):
    x = valid_distances(values)
    angles = np.concatenate(([0.0], np.cumsum(np.arccos(1-x))))
    return np.column_stack((np.cos(angles), np.sin(angles)))


def evaluate(config):
    rng = np.random.default_rng(config['seed'])
    records = []
    for n in config['lengths']:
        expected = binary_entropy((n // 2)/n) / n
        scores = [median_entropy(rng.uniform(0, 2, n)) for _ in range(config['draws_per_length'])]
        records.append({'n': n, 'draws': len(scores), 'expected': expected,
                        'observed_min': min(scores), 'observed_max': max(scores),
                        'maximum_error': float(np.max(np.abs(np.array(scores)-expected)))})
    n = config['illustration_length']
    ordered = np.linspace(0.1, 1.9, n)
    half = n // 2
    alternating = np.column_stack((ordered[:half], ordered[half:])).ravel()
    examples = []
    for name, sequence in [('ascending', ordered), ('alternating', alternating),
                           ('permuted', rng.permutation(ordered))]:
        points = realize_steps(sequence)
        recovered = 1 - np.sum(points[:-1]*points[1:], axis=1)
        examples.append({'ordering': name, 'distances': sequence.tolist(),
                         'entropy': median_entropy(sequence),
                         'lag1': float(np.corrcoef(sequence[:-1], sequence[1:])[0, 1]),
                         'geometry_max_error': float(np.max(np.abs(recovered-sequence)))})
    return {'source_commit': config['source_commit'], 'seed': config['seed'],
            'interpretation': 'Exact formula check; synthetic illustrations, no population inference',
            'length_checks': records, 'order_examples': examples,
            'maximum_identity_error': max(r['maximum_error'] for r in records),
            'constant_case': median_entropy(np.ones(n)),
            'small_sample_case': median_entropy([0.1, 0.2, 0.3])}


def plot(result, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt': 'cmri-metric-calibration', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    rows = result['length_checks']
    axes[0].plot([r['n'] for r in rows], [r['expected'] for r in rows], 'o-', color='#275d8c')
    axes[0].set(xlabel='Number of distinct step distances', ylabel='Released entropy metric',
                title='Determined by length when distances are distinct', xscale='log', yscale='log')
    for row in result['order_examples']:
        axes[1].plot(np.arange(1, len(row['distances'])+1), row['distances'],
                     label=row['ordering'], linewidth=1.6)
    axes[1].set(xlabel='Step', ylabel='Cosine distance',
                title=f"Different orders, identical entropy = {result['order_examples'][0]['entropy']:.4f}")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.15)
    fig.savefig(directory/'median_entropy.png', dpi=180)
    fig.savefig(directory/'median_entropy.svg', metadata={'Date': None})
    plt.close(fig)


def main(output_dir=None):
    start = time.perf_counter()
    directory = Path(__file__).resolve().parent
    cfg_path = directory/'config.json'
    destination = directory if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result = evaluate(json.loads(cfg_path.read_text()))
    plot(result, destination)
    metadata = {'runtime_seconds': time.perf_counter()-start, 'python': platform.python_version(),
                'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'matplotlib', 'pytest']},
                'config_sha256': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'source_sha256': json.loads(cfg_path.read_text())['source_sha256']}
    (destination/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (destination/'runtime.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'identity_max_error': result['maximum_identity_error'],
                      'order_examples': [{k:v for k,v in r.items() if k!='distances'} for r in result['order_examples']],
                      'runtime_seconds': metadata['runtime_seconds']}))


if __name__ == '__main__':
    main()
