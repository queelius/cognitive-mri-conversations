# Network audit checks: reproducibility and limits

The [prewritten protocol](PROTOCOL.md) fixes the diagnostic before its new
numerical execution. The [audit](../../audits/networks-memory.md) and
[literature review](../../literature/networks-memory.md) distinguish original
results, new checks and proposed work. No original repository was modified.

## Results

* The 7 temporal source/data hashes match the prior audit. Its existing merge/split
  and revisit fixtures are reused; the 10,000 permutations were not rerun.
* The hierarchy reconstruction yields 1,908 episodes, 500 meta-concepts and 7,153
  incidence edges; its projected giant has 499 nodes and 5,935 edges, reproducing
  C=.328730 and L=2.368786. Saved-order β recomputes as .319843.
* At 40 trades per episode, 16 incidence-degree-preserving diagnostic draws have
  mean C=.396513 and L=2.011752. The corresponding observed/null ratio is .704094,
  versus the stored ER ratio 6.57. The 10-sweep ratio is .704415. Both incidence
  margins are exact at all 32 checkpoints.
* The exact expected vocabulary curve under uniform episode permutation has fitted
  exponent .239705. This is a fit to a mean curve, not a mean fitted exponent.
* Synthetic alternating-role paths have zero conditional role entropy and at most
  two distinct colored windows of a given length, as dictated by parity.

[Full numeric output and source hashes](results.json),
[execution log](execution.txt), [verification record](verification.json),
[standalone PNG](projection-null.png), [editable SVG](projection-null.svg).

The historical ablation and flow tables were separately inspected without
rerunning their pipelines: [ablation inspection](stored-ablation-inspection.json),
[flow inspection](stored-flow-inspection.json). Both identify their source hash.

## Exact executed commands

From `/home/spinoza/github/cmri/future-ideas/iscs-2027/research-deep-dive/checks/networks`:

```bash
PYTHONDONTWRITEBYTECODE=1 COVERAGE_FILE=.coverage-audit python -m pytest -p no:cacheprovider -q test_audit_checks.py --cov=audit_checks --cov-report=term-missing --cov-report=json:audit-coverage.json > audit-tests.txt 2>&1
PYTHONDONTWRITEBYTECODE=1 COVERAGE_FILE=.coverage-audit python -m coverage run --append --source=audit_checks audit_checks.py > execution.txt 2>&1
COVERAGE_FILE=.coverage-audit python -m coverage json -o audit-coverage.json
COVERAGE_FILE=.coverage-audit python -m coverage report -m
PYTHONDONTWRITEBYTECODE=1 /home/spinoza/github/cmri/papers/selective-forgetting/.venv/bin/python plot_projection_null.py
```

The numeric environment was Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3,
NetworkX 3.4.2. Exact testing and plotting versions are in verification.json.
No installation was necessary. The instrumented diagnostic took **231.28 seconds**
and peak process RSS was **109.30 MiB**, including loading and both checkpoints.
Plotting was a separate approximately one-second command.

From `/home/spinoza/github/cmri/papers/hierarchical-memory`:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/spinoza/github/cmri/papers/selective-forgetting/.venv/bin/python -m pytest tests/ -p no:cacheprovider -q > /home/spinoza/github/cmri/future-ideas/iscs-2027/research-deep-dive/checks/networks/hierarchy-suite.txt 2>&1
PYTHONDONTWRITEBYTECODE=1 COVERAGE_FILE=/home/spinoza/github/cmri/future-ideas/iscs-2027/research-deep-dive/checks/networks/.coverage-hierarchy /home/spinoza/github/cmri/papers/selective-forgetting/.venv/bin/python -m pytest tests/test_hierarchy.py tests/test_hierarchy_semantic.py tests/test_concept_extraction_v2.py -p no:cacheprovider -q --cov=hierarchy --cov=hierarchy_semantic --cov=concept_extraction_v2 --cov-report=term-missing --cov-report=json:/home/spinoza/github/cmri/future-ideas/iscs-2027/research-deep-dive/checks/networks/hierarchy-coverage.json > /home/spinoza/github/cmri/future-ideas/iscs-2027/research-deep-dive/checks/networks/hierarchy-targeted.txt 2>&1
```

The full original suite exits **2**, with nine collection errors from the obsolete
`agentic` import path. The three relevant suites pass **48 tests in 1.65 seconds**,
with **62%** statement coverage (70% extraction, 58% geometric hierarchy, 62%
semantic hierarchy). Optional Louvain is absent in this environment, so its
branches are not certified. No API-based extraction or full pipeline is covered.
[Full-suite log](hierarchy-suite.txt), [targeted log](hierarchy-targeted.txt),
[coverage](hierarchy-coverage.json).

The new audit passes **9 mathematical/unit tests in .87 seconds**. They validate
both degree margins, self-loop reachability, exact vocabulary expectation by
exhaustive permutation, known clique projection, and alternating-role entropy.
Tests alone cover 59% of audit_checks.py; adding the recorded empirical execution
covers 100% of its statements. This is statement coverage, not branch coverage or
proof of statistical validity. [Test log](audit-tests.txt),
[combined coverage](audit-coverage.json).

## Interpretation limits

The new null uses standard uniform Curveball allocations and allows self-loops;
the original paper's subset-swap implementation is not rerun. Sixteen seeds and
10/40-sweep agreement are a diagnostic, **not a mixing guarantee or a significance
test**. Null projections have about 8,943 edges and 500 giant-component nodes,
versus 5,935 and 499 observed: incidence degrees, not projected density, are held
fixed. All results condition on one saved extraction and clustering; they neither
validate its concepts nor prove an absence of other structure. The ER figures are
historical saved means, not newly simulated ER draws. The figure shows run points,
not confidence intervals, and was visually inspected.

The empirical script emits aggregate numbers only. It uses concept strings in
memory to join saved memberships, but exports no text, titles, episode IDs or
incidence matrix. SHA-256 checks confirm its source/input files are unchanged
during execution. Existing source-control changes in child repositories were left
untouched. No test result here certifies the completed selective-forgetting run;
that pilot's separate reviewed evidence is reused explicitly.
