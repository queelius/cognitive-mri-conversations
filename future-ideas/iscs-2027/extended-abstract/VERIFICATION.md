# Verification record

Executed October 18, 2026. This is a transfer/replay audit of existing evidence,
not a new natural experiment, model evaluation or independent confirmation.

## Numerical and code checks

- `checks/test_evidence.py`: **11 passed**, 1.42 seconds; statement/branch
  combined coverage **94%**. See `verification/tests.txt` and `coverage.json`.
- Eight targeted tests in the original temporal/history implementations passed
  in **1.60 seconds** (17 other tests deselected). They cover exhaustive edge
  and active-node expectations on elementary graphs, withholding the assistant's
  current request, target invariance under changed input views, component-level
  averaging, and rejection of relabeled reserve components.
- The source-backed integration test checked **22 SHA-256-pinned files** and
  replayed **32 contrasts**, including all four input views and both encoders.
  Means and bootstrap endpoints agree to 1e-12; the JSON, LaTeX table and PNG
  reproduce byte for byte. Runtime and output hashes are in
  `verification-runs.jsonl`. Initial extraction took 1.35 seconds.
- The portable test recomputes all saved component means and intervals without
  requiring source repositories. It checks exported calculations, not source
  authenticity or construct validity. Source authentication is a separate check.
  A copied three-file bundle was also tested in a temporary directory outside
  the source tree: **10 passed, 1 skipped** in 0.46 seconds. The skip is precisely
  the optional source-backed integration audit. See `verification/portable-tests.txt`.
- No new natural observations, text encoding, experimental model/API requests,
  annotation judgments or reserve evaluation occurred. Drafting and technical
  review used AI assistance; those are not empirical validation.

Coverage gaps are explicit: the CLI argument entry point, wrong reserve-certificate
branch, and absent/altered output-file branch are not separately fault-injected.
The real integration test exercises valid source/output replay. Matplotlib internals,
LaTeX and the original studies' full inference pipelines are outside this coverage
measurement. Their existing source verification remains linked in the ledger.

The reused source-test command was run from this package:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=/home/spinoza/github/cmri/papers/history-falsification/src:/home/spinoza/github/cmri/papers/portfolio-wave1/src:/home/spinoza/github/cmri/papers/temporal-networks-calibration \
/home/spinoza/github/cmri/papers/selective-forgetting/.venv/bin/python \
  -m pytest -q -c /dev/null -o cache_dir=/tmp/cmri-iscs-source-pytest \
  /home/spinoza/github/cmri/papers/temporal-networks-calibration/tests/test_metrics.py \
  /home/spinoza/github/cmri/papers/history-falsification/tests/test_analysis.py \
  -k 'exact_edge_active_expectation or common_prefix_withholds or changed_input_view or grouped_estimates or reserve_component_cannot_be_relabelled'
```

## Document and venue checks

The PDF compiles using the official unmodified class. Four A4 pages including
references were rendered at 105 dpi and each inspected: body, equations, plot,
labels and references are legible and unclipped. Page four begins with references;
no stranded final body line remains. All citation/reference keys resolve.
One 3.91661-point title-box warning from the supplied class remains; inspection
shows no visible overflow. It is not suppressed or called a clean warning log.

The initial build selected the downloaded example because of TEXINPUTS search
order and failed on its missing image. The failed log is preserved. Putting `.`
first selects the intended source; no class or margin change was used to fit
the page limit. The 500-character abstract and five-keyword limits, class hash,
PDF hash and page count are recorded in the artifact checks.

Primary sources and official venue pages were refreshed. Live submissions
instructions confirm the 2027 abstract category; legacy publication/camera-ready
details do not establish the 2027 publication contract. See
[the source audit](LITERATURE-AND-TERMS.md) and its search/source manifests.

## Final review and preservation

[Fresh whole-package review](REVIEW.md) found no Critical, Important or Minor
issues requiring correction. It independently checked component alignment and
contrast algebra in addition to replaying the reported statistics. Its independent
context is a technical safeguard, not human validation of the research construct.

The final comparison found all **eleven source repository HEADs and Git status
outputs unchanged**, including the pre-existing temporal PDF modification and
static slide deletions. The temporal modified file's hash is unchanged. See
`verification/preservation.json` and the starting `initial-state.json`.
No source repository edits are part of this goal. Existing umbrella edits are
retained; only four linked research-metadata files and this new package changed.
The registry parses and all thirty linked record paths exist. `git diff --check`
passes. The official template class matches the downloaded ZIP byte for byte.

The exact deliverable/source hashes are in `artifacts.json`. Raw public source
snapshots, downloaded examples and build scratch remain ignored. The preserved
failure log contains TeX diagnostics, not private conversation content. No external
release, message, submission, or Git push was performed.
