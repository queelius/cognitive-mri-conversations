# Dynamics falsification and calibration checks

The protocol was written before execution. Read `PROTOCOL.md`, the aggregate
`results.json`, and `../../audits/dynamics.md`. No source projects were edited;
the source hashes before and after the run match. No message text or titles were
queried. No embeddings or private conversation identifiers are exported.

Reproduce from this directory (Python 3.10.20 used):

```bash
python -m venv /tmp/cmri-dynamics-audit-env
/tmp/cmri-dynamics-audit-env/bin/python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /tmp/cmri-dynamics-audit-env/bin/python audit_dynamics.py
PYTHONDONTWRITEBYTECODE=1 /tmp/cmri-dynamics-audit-env/bin/python -m pytest \
  -p no:cacheprovider -q test_audit_dynamics.py
```

`config.json` pins the roots, seed, sampling limits, alpha grid and bootstrap
counts. Update roots on another machine. The original private vector database is
needed for the stored-vector stage, and the source embflow checkout is needed for
the analytic/integration checks. The exact synthetic theorem counterexample does
not need private data. `source_hashes.json` pins the inspected versions; it is not
a substitute for obtaining the private database with authorization.

The pipeline ran in 5.17 seconds excluding setup and testing. The bounded sample
contained 80 ChatGPT, 80 Claude Code, and all 17 eligible Claude browser-chat
conversations. No sample expansion followed the results. Missing packages were
installed in `/tmp/cmri-dynamics-audit-env`; no provider calls were made.

`tests-local.txt` and `tests-embflow.txt` record the test outputs.
`coverage-local.json` is focused-test coverage; `coverage-audit.json` combines
focused tests with the complete bounded pipeline execution. `coverage-embflow.json`
comes from the original 191-test library suite. Coverage says which code ran; it
does not validate a scientific premise. The original theorem tests exercise only
16-dimensional approximately concentrated Gaussian samples with 0.02/0.03
tolerances; this audit supplies exact non-equilateral and low-dimensional tests.

No substantive protocol deviation occurred. The first focused test run contained
9 tests; three saved-artifact/integration checks were subsequently added before
the final verification. Statistical intervals are descriptive within this one
user's selected conversations. They neither establish population platform effects
nor account for study-design/model/archetype selection.

A separate protocol addendum precedes the final synthetic shape-distance check.
Run `python shape_counterexample.py` and
`python -m pytest -p no:cacheprovider -q test_shape_counterexample.py` in the same
environment. `shape-results.json` records the result, source hash, and updated
protocol hash; the original `source_hashes.json` retains the original protocol
hash from before that addendum. The two shape tests supplement the 12 primary tests.

Independent review identified a limitation in the original prefix interval:
resampling the 12 LOOCV gains holds their selected lenses fixed. A post-review
protocol addendum now reproduces uncertainty including lens reselection. Run:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /tmp/cmri-dynamics-audit-env/bin/python selection_uncertainty.py
PYTHONDONTWRITEBYTECODE=1 /tmp/cmri-dynamics-audit-env/bin/python -m pytest \
  -p no:cacheprovider -q test_selection_uncertainty.py
```

`selection-uncertainty-config.json` fixes 20,000 multinomial conversation resamples
and seed 20261015. Within each resample, lens fitting excludes **every copy of the
held-out original identity**; training and held-out averages preserve multiplicity.
Singleton cohorts are rejected and redrawn explicitly. There were zero such draws.
The 11 new tests cover duplicate exclusion, weighting, direct-reference agreement,
tie handling and degenerate draws. Configuration, input/implementation hashes,
runtime and results are in `selection-uncertainty-results.json`; test output is in
`tests-selection.txt`.
`coverage-selection-tests.json` records focused-test coverage and
`coverage-selection-combined.json` combines those tests with the complete
sensitivity run. The original coverage files remain intact.

The LOOCV gain remains **+.017660**, but its selection-aware percentile interval is
**[-.001224, .034807]**. The earlier **[.004830, .033811]** fixed-loss interval is
preserved in the unchanged `results.json` as a conditional descriptive summary.
The positive point estimate does not establish a robust positive learning-procedure
gain. The new sensitivity ran in .055 seconds and matched the reviewer-disclosed
result; it is not a blind preregistration. Neither interval generalizes beyond
the 12 selected conversations from one user or resolves session/design selection.

The script catches stage failures to preserve partial diagnostics in `results.json`.
Its exit code alone is therefore insufficient; the artifact integration test
explicitly rejects any recorded stage failure. Results include the failed scientific
hypotheses, not just checks that pass.
