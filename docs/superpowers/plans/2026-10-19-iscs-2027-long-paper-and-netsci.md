# ISCS 2027 Long Paper and NetSci 2027 Abstract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the reviewed eight-page ISCS 2027 short paper in `papers/iscs-2027/paper/` into an eleven-page long paper (limit 12) with full-size figures, the material the short format dropped, and a fuller limitations section; and produce a one-page NetSci 2027 abstract from the same results.

**Architecture:** The same single manuscript (`main.tex`, `references.bib`) and the same figure script, extended: the two merged 2x2 figures become four single-row figures, a subgroup forest plot is added, and a synthetic-process table is generated from a committed CSV so that every number stays traceable. The NetSci abstract is a separate LaTeX file in the same repository using the official NetSci template preamble verbatim and one composite figure built by the same script. Prose is drafted inline by the controlling session (one voice), then reviewed by a fresh reviewer against the REPORT and CLAIMS files.

**Tech Stack:** pdflatex + bibtex (TeX Live 2023), `svproc.cls` and `spmpsci.bst` (official 2027 template), NetSci 2027 template preamble (article 10pt, letterpaper, 20 mm margins, sans-serif), Python 3.11 venv `papers/iscs-2027/.venv` with matplotlib 3.11 and numpy 2.4.

**Spec:** `docs/superpowers/specs/2026-10-18-iscs-2027-paper-design.md` (paper section, evidence map, constraints) and the owner's decisions of 2026-10-19: long paper at about eleven pages; NetSci 2027 abstract to be submitted; authors Towell and Matta.

## Global Constraints

- Long paper: at most 12 pages including references in `svproc`; target 11. No padding: every added paragraph carries a number, a control or a disclosure from a report.
- Every number in the paper is copied from a committed result or report file; none is recomputed in prose. The figure script may compute only an ECDF, a difference of recorded values, or a table rendering of recorded values.
- Development values are reported as pre-registration alongside held-out values; the held-out values are the result.
- The role contrast is a forecast-origin (information-set) effect, never a "horizon" effect; under the natural origin both forecasts are one step ahead.
- The paper carries every binding disclosure of the short draft plus: the PRISM freeze hashed only protocol, config and pins; the reserve rows were structurally parsed by a test before the freeze; heterogeneity code was written after the reserve was opened; one WildChat module outside the frozen set was edited before the run; 806 of 828 confirmation conversations were decoded (not inspected) pre-freeze; the family-by-year cells are post hoc; the survey join is post hoc.
- No em-dashes in any prose or LaTeX source (the soul voice hook).
- No reading, re-drawing or re-scoring of any PRISM reserve or WildChat confirmation data.
- Figures are vector PDFs with fonts no smaller than 8 pt at print width (12.2 cm), including tick labels.
- NetSci abstract: one page including figure and references, template preamble unchanged, PDF via pdflatex, one figure.
- Authors on both documents: Alexander Richard Towell and John Matta, SIUE.

## Review Focus

1. A number that does not match its source file (wrong decimal, sign, sample or encoder); the reviewer checks every number against both `REPORT.md` files, `history-falsification/REPORT.md`, `history-specificity/REPORT.md` and the CSVs the script reads.
2. A claim whose status in a CLAIMS ledger is "not claimed" or "not tested" stated as a finding (independent encoders, mechanism, PRISM reversal replicated, verified individuals, a between-subgroup difference).
3. A disclosure from the Global Constraints missing from the manuscript, or softened.
4. Page overflow beyond 12 pages, figure text below 8 pt, or a reference that does not resolve.
5. An em-dash, a "horizon effect" phrasing, or a sentence that claims priority over CCL, Triple-Encoders, OnePred or Flipping the Dialogue.

---

### Task 1: Figures and generated tables from committed results

**Files:**
- Modify: `papers/iscs-2027/paper/figures/make_figures.py`
- Create (generated): `papers/iscs-2027/paper/figures/{primary-four-samples,budget-curves,role-origin,component-ecdf,subgroups,netsci-composite}.{pdf,png}`, `papers/iscs-2027/paper/figures/synthetic-table.tex`, `papers/iscs-2027/paper/figures/sources.json`
- Delete (generated, superseded): `primary-and-budget.{pdf,png}`, `role-and-ecdf.{pdf,png}`

**Interfaces:**
- Consumes (unchanged inputs): `../wildchat-replication/results/report/primary.csv`; `results/report/budget.csv` of both repositories; `results/public-{minilm,e5}.json` of both repositories.
- Consumes (new inputs): `papers/iscs-2027/results/report/subgroups.csv` (columns `encoder,family,label,gain,ci_low,ci_high,components,conversations`; `family` is `condition` or `length`); `papers/wildchat-replication/results/report/subgroups.csv` (columns `encoder,axis,label,gain,ci_low,ci_high,components,conversations,declared`; `axis` is `condition (model family)`, `year` or `length`); `papers/history-falsification/results/report/synthetic-effects.csv` (columns `scenario,regime,role,contrast,mean,mc_ci_low,mc_ci_high,datasets,dataset_sd,detections,detection_frequency,wilson_low,wilson_high`).
- Produces: the six PDFs, the table fragment and `sources.json`, each figure 12.2 cm wide.

- [ ] Step 1: Replace `primary_and_budget()` and `role_and_ecdf()` by four functions, each a 1x2 panel (MiniLM, E5) of width `TEXT_WIDTH_IN` and height 2.1 in, with the same data and styling as the current panels but all fonts at 8 pt or larger (`font.size` 8, tick labels 8, legend 8, titles 9; x tick labels of the four-sample panel written on two lines at 8 pt): `primary_four_samples()`, `budget_curves()`, `role_origin()`, `component_ecdf()`.
- [ ] Step 2: Add `subgroups()`: a forest plot, 1x2 (MiniLM, E5), height 3.4 in, one row per subgroup cell in this order, top to bottom: PRISM condition (controversy guided, unguided, values guided), PRISM length (6, 8, 10, 12 messages), WildChat family (gpt-3.5, gpt-4, gpt-4o, gpt-4.1), WildChat year (2023, 2024, 2025), WildChat length (6, 8, 10, 12). Marker at `gain`, horizontal bar from `ci_low` to `ci_high`, PRISM rows in `PRISM_COLOR`, WildChat rows in `WILD_COLOR`, a vertical line at 0 and a dashed line at .01, thin separators between the five blocks, row labels on the left panel only with the component count in parentheses, e.g. `gpt-4.1 (22)`. Read WildChat labels as `gpt-4.1` from `axis == "condition (model family)"`; length labels as `6 messages`.
- [ ] Step 3: Add `synthetic_table()` writing `synthetic-table.tex`: a `tabular` with `booktabs` rules and columns Process | User gain | Own second minus latest | Interaction, natural | Interaction, common prefix | Detections, one row per scenario A, B, C, where User gain is `mean [mc_ci_low, mc_ci_high]` of `(scenario, natural, user, mean_minus_window2)`, Own second is `mean` of `(natural, user, own_second_minus_own_latest)`, the two interactions are `mean` of `(natural|common_prefix, role_interaction, mean_minus_window2)`, and Detections is `detections/datasets` of the user natural `mean_minus_window2` row. Format every value to three decimals with a leading sign in text mode (`$+$.120`, `$-$.261`), intervals in brackets. Process labels: A `fixed topic, independent user innovations, reactive assistant`; B `independent normalized messages`; C `planted own-second recurrence`.
- [ ] Step 4: Add `netsci_composite()`: 1x3, width 7.0 in, height 1.9 in, MiniLM only: (a) primary gain in the four samples (as `primary_four_samples` MiniLM panel), (b) role interaction under both origins for both collections (as `role_origin` MiniLM panel), (c) budget curves for both collections (as `budget_curves` MiniLM panel); panel letters `(a)`, `(b)`, `(c)` in the upper left of each axis; fonts 9 pt.
- [ ] Step 5: Update the module docstring and `__main__` to run all six, delete the two superseded output pairs if present, write `sources.json`.
- [ ] Step 6: Run `.venv/bin/python paper/figures/make_figures.py` from `papers/iscs-2027/`; expected output lists the six PDFs; `sources.json` now has eleven entries (the eight previous inputs plus the three new CSVs).
- [ ] Step 7: View every PNG; check no text below 8 pt, no overlapping labels, the forest plot rows readable.
- [ ] Step 8: Commit in `papers/iscs-2027`: `git add paper/figures && git commit -m "Split and extend the paper figures for the long format"`.

### Task 2: Manuscript expansion

**Files:**
- Modify: `papers/iscs-2027/paper/main.tex`, `papers/iscs-2027/paper/references.bib`, `papers/iscs-2027/paper/README.md` (format facts: long paper, 12-page limit)

**Interfaces:**
- Consumes: Task 1's figure names and `figures/synthetic-table.tex` (`\input`).
- Produces: `main.pdf` at 10 to 12 pages.

- [ ] Step 1: Replace the two `\includegraphics` figures by five figures (`primary-four-samples`, `budget-curves`, `role-origin`, `component-ecdf`, `subgroups`), each with its own caption; `[t]` placement; drop the three float-spacing `\setlength` lines of the preamble.
- [ ] Step 2: Add Table 1 (samples): rows PRISM development probe (72 components, 240 conversations, 428 user targets, pre-registration), PRISM reserve (233, 775, 1,482, confirmation), WildChat development (345, 599, 1,490, pre-registration and centers), WildChat confirmation (482, 801, 1,998, confirmation), with the collection period and unit per collection in the caption. Numbers from `iscs-2027/REPORT.md` "Sample", `wildchat-replication/REPORT.md` "Samples", `portfolio-wave1/REPORT.md` (428 probe targets per role).
- [ ] Step 3: Extend the estimates table with rows: centered cosine (P +.022 [.015, .029] / +.017 [.011, .023]; W +.015 [.007, .023] / +.012 [.004, .019]); own mean minus own last two (P +.006 [.004, .008] / +.004 [.003, .004]; W +.010 [.007, .014] / +.005 [.005, .006]); attenuation (P −.109 [−.118, −.101] / −.023 [−.025, −.022]; W −.091 [−.099, −.084] / −.021 [−.023, −.019]). Sources: `iscs-2027/REPORT.md` and `wildchat-replication/REPORT.md` secondary panels.
- [ ] Step 4: Add Table 3 (development stability across visibility views) from `history-falsification/REPORT.md`: native128 .034 [.023, .049] / .019 [.016, .023]; expanded240 .033 [.021, .047] / .019 [.015, .022]; chunk64 restored .033 [.021, .047] / .017 [.014, .021]; all valid past candidates .034 [.021, .048] / .020 [.017, .023]; caption states these are exposed development values.
- [ ] Step 5: Add Section 5.3 "Synthetic processes" with `\input{figures/synthetic-table}` and the reading from `history-falsification/REPORT.md` (A reproduces the gain and the negative natural interaction with no older dependence conditional on topic; C has genuine older dependence and the same negative natural assistant gain; B shows neither; detections 100/100, 0/100, 100/100 with Wilson intervals). Add Section 5.4 "Ranking does not follow cosine" from `history-specificity/REPORT.md`: selected long minus short MRR −.010 [−.041, .018] (MiniLM) and −.002 [−.005, .000] (E5) on 79 conversations in 47 components; genuine minus nearest donor +.025 [−.008, .061] and .000; cosine diagnostic on the same 117 cases +.015 [.006, .024] and +.005 [.003, .007]; the frozen requirement was +.03 MRR with positive lower bounds; the exact-topic donor control failed its own calibration gate (18/100 against a ≤10/100 rule), so the ranking null is a bounded negative.
- [ ] Step 6: Expand Related work into three paragraphs (history in dialogue prediction; role and context ablations; observation-process nulls and pre-registration), adding entries to `references.bib` from the locators in `papers/history-falsification/audits/literature.md`: `erker2024triple` (ACL 2024, DOI 10.18653/v1/2024.acl-long.290), `yi2025chatbotid` (NeurIPS 2025), `ghosal2021context` (Findings ACL-IJCNLP 2021, DOI 10.18653/v1/2021.findings-acl.124), `ohagi2024lookahead` (Interspeech 2024), `chi2017speakerrole` (IJCNLP 2017, anthology I17-2028), `rudolph2026transition` (Findings ACL 2026, anthology 2026.findings-acl.1271), `nosek2018preregistration` (PNAS 115(11):2600–2606, DOI 10.1073/pnas.1708274114), `reimers2019sbert` (EMNLP-IJCNLP 2019, DOI 10.18653/v1/D19-1410). Author lists must be verified against the official records before the entry is written.
- [ ] Step 7: Expand Heterogeneity with the forest plot, the family-by-year post hoc reading (gpt-4o .037 in 2024 against .001 in 2025 in MiniLM; .019 against .011 in E5; small cells, unadjusted) and the survey rho table in prose; expand Discussion into named paragraphs: what the gain is not; why PRISM's reversal did not replicate; instruments and visibility; components are not people; what the freezes did and did not fix (the full disclosure list); what would change the conclusion; and a short reporting checklist (unit retained, feature availability time, denominator, permitted prefix, what each control preserves).
- [ ] Step 8: `make` builds with no undefined references or citations; `pdfinfo main.pdf` reports 10 to 12 pages; a grep for the em-dash character (U+2014) in `main.tex` finds nothing.
- [ ] Step 9: Commit in `papers/iscs-2027`: `git commit -am "Expand the ISCS 2027 draft to the long-paper format"`.

### Task 3: NetSci 2027 abstract

**Files:**
- Create: `papers/iscs-2027/abstracts/netsci-2027/main.tex`, `papers/iscs-2027/abstracts/netsci-2027/README.md`, `papers/iscs-2027/abstracts/netsci-2027/Makefile`
- Uses: `../../paper/figures/netsci-composite.pdf` via `\graphicspath`

**Interfaces:**
- Consumes: the NetSci template preamble (the `DO NOT EDIT` block, copied verbatim from the official Overleaf template `netsci2027-abstract-template/gqryjjnvrczn`), Task 1's composite figure.
- Produces: a one-page PDF.

- [ ] Step 1: Write `main.tex`: template preamble unchanged; title "What the whole conversation predicts: a replicated history gain and two observation nulls for human–AI dialogue"; authors Towell\ts{1} and Matta\ts{1}; affiliation 1. Southern Illinois University Edwardsville, Edwardsville, Illinois, USA; body of about 420 words covering the pre-registered contrast, the two held-out confirmations with their four numbers, the forecast-origin result, the revelation identity (the displayed equation for E[M_k]) and the reporting recommendation; references [1] Towell and Matta, Complex Networks XIV chapter; [2] Kirk et al., PRISM; [3] Zhao et al., WildChat; [4] Gauvin et al., randomized reference models; figure caption naming panels (a), (b), (c).
- [ ] Step 2: `Makefile` with `all: main.pdf` via `pdflatex` twice; README with the call facts (one page including figure and references, PDF via official template, deadline 2026-12-01, notification 2027-01-15, submission at netsci2027.eventclass.it, in-person presentation required, suggested topics: Multi-Agent and LLM Networks; Temporal Networks; Statistical Analysis of Networks and Inference).
- [ ] Step 3: Build; `pdfinfo main.pdf` reports exactly 1 page; view the page image.
- [ ] Step 4: Commit in `papers/iscs-2027`: `git add abstracts && git commit -m "Draft the NetSci 2027 one-page abstract"`.

### Task 4: Independent review and fix wave

- [ ] Step 1: Dispatch a fresh reviewer (most capable model) with `main.tex`, the NetSci `main.tex`, the PDF text of both, both `REPORT.md`, `history-falsification/REPORT.md`, `history-specificity/REPORT.md`, both `CLAIMS.md`, the three CSVs of Task 1 and the Review Focus list; the reviewer reports number mismatches, status upgrades, missing disclosures, length and figure legibility.
- [ ] Step 2: One fix wave; rebuild both documents; page counts re-checked (10 to 12 and exactly 1).
- [ ] Step 3: Commit the fixes in `papers/iscs-2027` and push `main`.

### Task 5: Record

- [ ] Step 1: Umbrella `docs/research/DECISIONS.md`: new dated row and a "Current assessment, October 19" paragraph (owner decisions: authors Towell and Matta; long paper; NetSci abstract drafted; F3 started as its own study); `registry.yaml`: `iscs_2027_long_paper_plan` and `netsci_2027_abstract` entries; `CLAUDE.md` iscs-2027 row: "long-paper draft" and the NetSci abstract; `docs/research/README.md` priority 1 line.
- [ ] Step 2: Commit and push the umbrella (master).
