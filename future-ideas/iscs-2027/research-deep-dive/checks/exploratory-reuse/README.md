# Exploratory reuse queries

These descriptive queries reuse existing audited inputs after the temporal pilot. They were selected during brainstorming, without a fresh holdout or a new preregistered hypothesis test. They do not establish behavioral reuse, a bridge advantage, or a population effect.

Run from the CMRI root with the existing temporal-calibration environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 papers/temporal-networks-calibration/.venv/bin/python future-ideas/iscs-2027/research-deep-dive/checks/exploratory-reuse/probe.py
```

The command prints JSON and does not modify its inputs. The original output is [results.json](results.json): Python 3.11.15, NumPy 2.2.6, elapsed 0.646 seconds. It records source hashes and the probe hash. Private concept incidence inputs remain local; public temporal arrays are available in the child pilot. This combined query is therefore not wholly independently reproducible from public data alone.

The query verifies pair-count conservation, agreement with the previous projection audit, source hashes, nested domain membership and agreement with the calibrated nearest-return routine. Equal-time observations are excluded from predecessors and exact maximum-similarity ties are averaged. It reuses earlier null summaries rather than drawing new null graphs.

Verification on October 16: a fresh execution passed those checks and matched every saved JSON field exactly except the execution timestamp and elapsed runtime. Python syntax parsing and relative-link checks also passed. No production feature was added and no new unit-test/coverage claim is made for this small query; the underlying calibration and audit tests are documented in their original reports.

Important denominator distinctions:

- `repeat_occurrence_fraction` is `(all occurrences - distinct pairs) / all occurrences`. It is order-invariant and cannot demonstrate chronological reuse.
- For cross-domain pairs, 991/3,954 = 25.06% occur at least twice. This is different from the occurrence-weighted repeat fraction, 43.44%.
- The existing full-archive hierarchy supplies the domain partition. It cannot be treated as a prefix-only predictor without a new design.
- The two public archives were previously selected for investigation. Their return-age distributions are hypotheses for follow-up, not independent confirmation.
- Repeated concept extraction and copied text can explain some recurrence. The vector-identity check does not detect all near-duplicate text.

No confidence interval is reported for the post hoc cross-archive contrast: two selected proxies do not supply a defensible population sample. Prior null draws lack established mixing. See [paper options](../../EMPIRICAL-PAPER-OPTIONS.md) for interpretation and next checks.
