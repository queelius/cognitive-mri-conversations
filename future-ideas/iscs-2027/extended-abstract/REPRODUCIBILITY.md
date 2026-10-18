# Reproducibility appendix

This package transfers existing results into a working extended abstract. It
does not fit a model, encode text, sample new outcomes or open reserved data.
The source study remains exposed development evidence.

## Inputs and units

- Source PRISM revision `18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6`, previously
  pinned by `papers/portfolio-wave1/config.json`; source SHA-256
  `55ec67a194ca59d567b57acaeab813cb30abe801e73f9e495597a77ab674d638`.
- History source commit `7a391141f7de3d06aeeca11434baadf6e744a3e0`; first wave
  `e84846590e228fc3457bfd7e8d411118fb6035ae`. Exact repository states are recorded
  in `initial-state.json`; included files are bound by `evidence-inputs.json`.
- MiniLM: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. E5: `intfloat/e5-small-v2`,
  revision `ffb93f3bd4047442299a41ebb6fa998a38507c52`, official qint8 ONNX artifact.
  Model bytes, tokenizers and earlier inference boundary checks remain in the
  source model manifests. E5 and MiniLM are not an architecture-controlled contrast.
- Source grouping: 360 development components, split 216/72/72 fit/tune/probe.
  Probe has 240 conversations and 428 paired targets. The same targets and native
  target embeddings are used across views/regimes; predictor views change. The separate 248 reserve components
  are excluded; source grouping IDs are proxies, not verified people.
- `data/forecast-effects.json` includes all 32 retained contrasts across four
  input views and two instruments. It exports only aggregate component effects
  and counts, without participant IDs, conversation text or target embeddings.

For each contrast, the source first averages paired effects within a conversation,
then averages conversations within a component. Components have equal weight.
The transfer audit recomputes the mean and percentile interval from those saved
component differences using NumPy's default RNG, **2,000 draws, seed 104729**,
and quantiles .025/.975. It checks agreement to 1e-12, not merely rounded display
values. These intervals are conditional descriptive uncertainty, not correction
for all earlier research selection or population confirmation.

## Executed commands

Use Python 3.11 with the already pinned environment. The working interpreter is
`/home/spinoza/github/cmri/papers/selective-forgetting/.venv/bin/python`.
The relevant versions are NumPy 2.2.6 and Matplotlib 3.10.7; no package was installed.
Exact replay/test versions are in `requirements-replay.txt` and platform metadata
in `verification/environment.json`.
From this package directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/spinoza/github/cmri/papers/selective-forgetting/.venv/bin/python \
  checks/build_evidence.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/spinoza/github/cmri/papers/selective-forgetting/.venv/bin/python \
  checks/build_evidence.py --verify
/home/spinoza/github/cmri/papers/selective-forgetting/.venv/bin/python \
  -m pytest -q checks/test_evidence.py
TEXINPUTS=.:template: BIBINPUTS=.: SOURCE_DATE_EPOCH=1792310633 \
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build ./main.tex
pdftoppm -r 105 -png build/main.pdf build/page
```

The first command regenerates only package outputs. The second requires byte
identity of the JSON, generated LaTeX table and PNG. Each writes a runtime record
to `verification-runs.jsonl`. Neither command changes source repositories.
The 22-file source check requires the corresponding local CMRI repositories;
it reads saved results, code and reports, not raw conversations.

The **portable test suite** needs only this package plus NumPy/pytest: its saved-
bundle test replays all 32 component means and intervals without accessing the
source repositories. This validates the exported numerical calculation, not the
authenticity of source preparation, annotation truth or generalization. The
full source-backed command additionally checks original file pins.

The official class is preserved in `template/iscs.cls` with its original hash.
Put `.` before `template` in TEXINPUTS: the opposite ordering selects the
downloaded example named `main.tex`. The initial failed build and the correction
are preserved in `verification/initial-build-failure.log` and PROGRESS.md.
No custom class, font or margin change is used. `\unskip\par` removes a trailing
title-box space without modifying the official style.

## Scope of verification

The new tests cover source tampering/path escape, component alignment, nonfinite
values, altered means/intervals, contrast orientation, labeled forecast regimes,
and portable numerical replay. Existing source tests are reused for the exact
graph expectations and forecast-prefix/group-weight boundaries. Figures and all
PDF pages receive visual inspection; compilation alone does not establish layout.

Full original inference, all old experimental grids and human annotation are not
rerun. Their existing verification remains linked from the source reports. Only
checks needed for the included claims are executed. See VERIFICATION.md for actual
counts, coverage, runtime, preserved states and the final review outcome.

## Availability and release boundary

The draft, code, source pins, aggregate effect vectors and new figure can be
reviewed locally without private conversation text. No public hosting URL or
release DOI for this new package is claimed. Raw source snapshots and template
downloads stay in ignored `external/`; manifests identify official sources.
Before external distribution, authors must choose a stable package location and
resolve its terms and final content. No archive was uploaded in this goal.
