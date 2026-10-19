# ISCS 2027 Short Paper Draft Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A complete, compilable first draft of the ISCS 2027 short paper (8 pages, Springer SSDC `svproc` class) in `papers/iscs-2027/paper/`, with every number traceable to a committed result file in the pilot and confirmation repositories.

**Architecture:** One LaTeX manuscript (`main.tex`, `references.bib`) plus a figure script that reads only committed result files from `papers/iscs-2027/results/` and `papers/wildchat-replication/results/` and records their hashes. Prose is drafted inline by the controlling session (one voice), then reviewed by a fresh reviewer subagent against the two REPORTs and CLAIMS ledgers.

**Tech Stack:** pdflatex + bibtex (TeX Live 2023), `svproc.cls` and `spmpsci.bst` from the official 2027 template (sha256 `c44824ff...` and `013368de...`, identical to the 2026 template), Python 3.11 venv with matplotlib and numpy for figures.

**Spec:** `docs/superpowers/specs/2026-10-18-iscs-2027-paper-design.md` (the paper section, evidence map, goals, non-goals, anti-goals and constraints), amended by the F6 entry and `docs/superpowers/specs/2026-10-18-wildchat-replication-design.md`.

## Global Constraints

- Short paper: 8 pages including references, `svproc` class, English, submission via CMT by 2027-03-01.
- Every number in the paper is copied from a committed result or report file; none is recomputed in prose. Figures may recompute only an ECDF or a difference of recorded values.
- Development values are reported as pre-registration alongside held-out values; the held-out values are the result.
- The paper carries, in its own words, the disclosures the reports mark as binding: the pre-freeze WildChat amendment, the row-group decode of confirmation text, the self-administered freezes, the spent reserve and spent confirmation sample, two instruments on one sample, hashed-IP components, estimates shrinking from development to held-out.
- No em-dashes in any prose or LaTeX source (the soul voice hook).
- No reading, re-drawing or re-scoring of any PRISM reserve or WildChat confirmation data.
- Single author (Towell) in the draft; co-author listing is the owner's decision and is left as a source comment.
- Figures are vector PDFs with fonts no smaller than 8 pt at print width (ISCS 2026 reviewer 1 complained about small figures).

## Review Focus

1. A number in the manuscript that does not match its source file (wrong decimal, wrong sign, wrong sample). The reviewer checks every number against `REPORT.md`/`CLAIMS.md` of both repositories.
2. A claim whose status in the CLAIMS ledgers is "not claimed" or "not tested" stated as a finding (independent encoders, mechanism, sign reversal replicated, verified individuals).
3. A disclosure from the Global Constraints missing from the manuscript.
4. Page overflow beyond 8 pages, or figure text below 8 pt.
5. An em-dash or a sentence that claims priority over CCL, OnePred or Flipping the Dialogue.

---

### Task 1: Figures from committed results

**Files:**
- Create: `papers/iscs-2027/paper/figures/make_figures.py`
- Create (generated): `papers/iscs-2027/paper/figures/{primary-and-budget,role-origin,component-ecdf}.pdf`, `papers/iscs-2027/paper/figures/sources.json`

**Interfaces:**
- Consumes: `papers/wildchat-replication/results/report/primary.csv` (four samples), `results/report/budget.csv` of both repositories, `results/public-{minilm,e5}.json` of both repositories (`primary.component_differences`, `roles.interaction_natural`, `roles.interaction_common`).
- Produces: three PDFs at 12.2 cm width and `sources.json` (path, sha256 of every input).

- [ ] Step 1: Write the script; it refuses to run if any input is missing and writes `sources.json`.
- [ ] Step 2: Run `.venv/bin/python paper/figures/make_figures.py` from `papers/iscs-2027/`; expected: three PDFs and `sources.json`.
- [ ] Step 3: View the PNG previews; fonts legible at print width.

### Task 2: Manuscript

**Files:**
- Create: `papers/iscs-2027/paper/main.tex`, `papers/iscs-2027/paper/references.bib`, `papers/iscs-2027/paper/Makefile`, `papers/iscs-2027/paper/README.md`
- Copy: `svproc.cls`, `spmpsci.bst` from the official template (done).

- [ ] Step 1: Draft all sections per the spec's evidence map: Introduction; Related work; Data, measurement and protocol; Results (history gain, budget, horizon, heterogeneity); Observation choices that resemble dynamics (packaging, revelation identity); Discussion and limitations; Conclusion; Reproducibility.
- [ ] Step 2: `make` builds `main.pdf` with no undefined references or citations.
- [ ] Step 3: Page count is at most 8; trim if not.

### Task 3: Independent review and fix wave

- [ ] Step 1: Dispatch a fresh reviewer (most capable model) with the PDF text, `main.tex`, both `REPORT.md` and `CLAIMS.md`, and the Review Focus list; the reviewer reports number mismatches, status upgrades, missing disclosures and length.
- [ ] Step 2: One fix wave; rebuild; page count re-checked.

### Task 4: Commit and record

- [ ] Step 1: Commit `paper/` in `papers/iscs-2027` (main) and push.
- [ ] Step 2: Umbrella: `DECISIONS.md` current-assessment paragraph (draft exists, what the owner must decide: author list, NetSci abstract, F3), `registry.yaml` entry for the draft, `CLAUDE.md` row note; commit and push.
- [ ] Step 3: Record the WildChat pin consequence: `papers/wildchat-replication` pins `papers/iscs-2027` at `ef845c2`; the new commits make that pin stale as its limitation 13 anticipates; a re-run needs a checkout at `ef845c2`.
