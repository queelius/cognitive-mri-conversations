# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **umbrella** for the "Cognitive MRI of AI Conversations" research program:
applying complex network analysis to ChatGPT conversation archives. The program began as one
conference paper and grew into a cluster of related papers, all built on the same corpus of
1,908 ChatGPT conversations (December 2022 to April 2025).

The core idea: transform linear conversation logs into semantic similarity networks,
revealing knowledge communities, bridge conversations, and cognitive structure in
AI-assisted knowledge exploration.

This repository is a landing page plus shared working data. Each paper is its own independent
git repository under `papers/` (gitignored here). The umbrella still carries the program's
published compendium DOI (10.5281/zenodo.18778376) and the full pre-split git history.

## Current focus and how we work (read first; update when the focus changes)

**Goal (Alex, 2026-10-26): publish again soon.** Plan papers and experiments, not a thesis or
dissertation chapters, and move fast.

**Real deadlines only.** The live ones are in the board (`python3 tools/workspace.py board`, which
reads the sibling `../pubs` repository): NetSci 2027 abstract (2026-12-01), the Applied Network
Science extension (2026-12-04), ISCS 2027 (2027-03-01). Internal targets are labelled as targets,
never presented as deadlines.

**Now (update this list as work finishes):**
1. The extension (`papers/hierarchical-memory-extended/`): Alex decided on 2026-10-28 to centre it on the
   organisation of concept co-use, with H (who made the connection) as a second result if the evidence supports it.
   Of the two decisive comparisons, J (codebook crossing) survives in every construction and I finds no predictive
   increment beyond recency. See that repository's PLAN.md and `notes/2026-10-28-I-J-results.md`. Alex decided on
   2026-10-28 (evening): H (who made the connection) is one scope paragraph in the discussion (attribution through an
   extractor, composition, and one persistence sentence: read under amendment 8 on 2026-10-29, taken-up connections
   recur more than made ones under the primary extractor, but not across codebooks, extractors or populations;
   `notes/2026-10-29-H-corrected-reading.md`). A second population enters the December 4 submission as a co-equal result: K (WildChat long histories, 77
   identities) with a true-identity case series of three open personal archives (`analyses/K-long-histories/DESIGN.md`,
   amendments 1 to 6). Both extractors were read on 2026-10-29 and every Codex review closed: every WildChat reading
   holds under Qwen3-14B and Mistral-7B (co-use beyond themes 75 to 77 of 77; concentration in time 56 to 66 of 77);
   the three case-series people pass the co-use test under both, and the temporal test under Qwen3-14B (case 1 fails
   it under Mistral-7B with the paper codebook) (`notes/2026-10-29-K-results.md`). Alex decided on 2026-10-29: title
   "Concept networks from AI conversation logs: co-use beyond shared themes, in one archive and in other people's long
   histories" (provisional until settled with Matta), K in the abstract's second sentence, people before prediction.
   The title, abstract, introduction, discussion and conclusion are rewritten (fe0adec); a Codex review of the rewrite
   is running. Left for submission: acknowledgements and authors' contributions (Alex with Matta), the repository URL
   and DOI, and the ISCS editors' note decision.
2. The NetSci abstract (`papers/person-networks/abstracts/netsci-2027/`): its coarser-matching sentence
   was confirmed by the fingerprinted rerun (2026-10-26); it needs Alex's read and submission by Dec 1.
3. Reach in bits (`papers/reach-bits/`, for ISCS 2027): base language models score the next user
   message under Protocol 2's contexts. Development on the spent samples agreed across two scorers and
   both collections; Protocol R (frozen 2026-10-27) PASSED on fresh PRISM and WildChat samples in both
   scorers (all five rules). Next: write it into the ISCS 2027 paper (`papers/iscs-2027/paper/`). The extension also gained G (open-weight extractors run locally).
   Cross-session reach in bits (exploratory, spent samples, 2026-10-27; `DESIGN-cross-session.md`,
   `results/xs/REPORT.md`, `notes/2026-10-27-cross-session-results.md`): a person's own earlier
   conversation predicts the opening of their next one better than a matched stranger's, on PRISM and
   WildChat under both scorers, declining with lag and elapsed time but still positive at lag 8 and
   after 30 days; it survives repetition and timing exclusions. Where to write it up is Alex's call.
   The candidate list and its prior art are in `papers/agenda/` (`PROGRAM-PLAN.md`, a proposal, and
   `reports/CMRI research questions 2026.md`).

**Working rules:**
- Claude implements; Codex (`codex exec -m gpt-6.1-sol -s read-only -c model_reasoning_effort=max ... < /dev/null`,
  sometimes `-m gpt-6-astra` for variation) reviews once per milestone with a closure criterion, not in
  open-ended rounds. Verify its findings before acting.
- Ask Alex only for decisions that are his (co-authors, money, scope, submission), in plain
  language, with a recommendation. Do the rest.
- Design before the first run, then aggregate results committed before they are interpreted;
  failed checks are reported as they are, then explored as labelled exploratory work.
- After every milestone, commit and push every touched repository (several repos, several pushes).
- At every milestone, audit the living documents against the work just done: the agenda ledger
  (`papers/agenda/questions.json`, `studies.json`), the paper's PLAN.md (open decisions, next actions)
  and CLAIMS.md, README and resume notes, and this "Now" list. Record each mistake as a lesson in
  `questions.json` (`lessons`), and encode it as a check or written rule where possible. Open
  questions to revisit live in the ledger (research) and in each PLAN.md's "Open decisions" (papers).
- No spend without a reviewed request record and Alex's committed approval; agents never send
  email or submit.

## Working model: a cluster of independent git repos

This is the most important thing to understand before making changes. The umbrella git repo
tracks **only** the landing page and program-level planning: `README.md`, `CLAUDE.md`,
`CITATION.cff`, `LICENSE`, `.zenodo.json`, `.gitignore`, plus `docs/` and `future-ideas/`.
Everything substantive is a *separate, gitignored git repository* nested inside:

- `papers/<name>/`: independent `cmri-*` repos (papers, pilots and studies; their own history,
  remote and CLAUDE.md), plus two worktrees of them.
- `chatgpt-conversation-corpus/`: the publishable dataset, its own repo.

`workspace.json` is the single structured list of these repos: path, branch, GitHub
visibility, data class (`public` or `private-archive`), a one-line note, and named snapshots
that pin the commits a paper depends on. `tools/workspace.py` derives everything from it:
`clone` sets up a fresh workspace, `status` reports every checkout, `lock`/`verify`/`checkout`
manage snapshots, `remote-check` compares GitHub settings, `scan` checks a snapshot for
forbidden files and secrets before publication, and `render` regenerates the tables below.
`spent.json` is the ledger of every sample drawn from a shared pool (PRISM, WildChat); `spent --verify`
checks that each later draw's committed configuration excludes every earlier one. An exploratory draw may instead name earlier samples it deliberately reuses (`reuses`, with a `reuse_decision` saying who decided and when); it confirms nothing, and later draws must still exclude it. Each sample's
`dataset` references a dataset id, validated against the manifest's datasets source. Unit and size
labels are retained from the draw record; counts are never combined across draws or units.
`board [--as-of YYYY-MM-DD] [--write]` reads the manifest's `sources`: the generated
`pubs.jsonl`, agenda `studies.json`, `questions.json` and `datasets.json`, plus `spent.json` for
dataset draw counts. It keeps publication status, study state, and repository role separate.
A manuscript locator inside a repository (for example a paper subdirectory) is attributed to the
deepest manifest repository containing it; the umbrella matches only its own root.
Missing source files are reported; cross-references remain unverified until their records are
available. Missing exposure records show unverified counts, not zero. Optional publication deadlines
use ISO dates for both `date` and `verified`, with a `source` string. Deadline attention covers
today through 45 days ahead; follow-up attention includes overdue checks and the next 14 days.
`--write` saves the private board to gitignored `STATUS.local.md`.

`questions [--write | --check]` prints every question's state, known findings, remaining work,
study states and candidate datasets with known exposure. `questions --write` regenerates
`questions-ledger.md` beside the configured questions source; `--check` returns non-zero when
that file is missing, stale or its required sources are unavailable. Neither mode needs the
publication index. Writes refuse symlinks, paths outside the workspace and missing required
sources. Edit the JSON sources, never the generated ledger. The questions source also owns
the structured `lessons` entries, whose text is preserved verbatim in the ledger.

Dataset ids (`D-...`) are identifiers; `location` and `documentation` are locators, with relative
paths resolved conceptually from `datasets.json` (the tool does not open them). Dataset records
require `name`, `provenance`, `version`, `access`, `licence`, `location`, `documentation`,
`sampling_unit` and `catalog_sections`. JSON `null` explicitly means unknown for provenance,
version, licence, location, documentation and sampling unit; missing fields are errors.
Access is one of `public`, `gated`, `controlled`, `private-archive` or `unavailable`.
Sampling units currently supported are `component`, `ip`, `identity` and `conversation`.
`catalog_sections` holds stable HTML anchor ids in `public-datasets.md`, whose narrative remains
authoritative for context and constraints. Questions own candidate `datasets` references;
these planning links do not attest to eligibility. Samples own the dataset links used to derive
exposure, so datasets store no duplicate draw lists. Each exposure report says
"remaining eligibility unverified"; zero draws in the ledger does not establish unused data.

`by-stage` builds gitignored relative links to repository directories and study subpaths. Its
`.generated-by-workspace` marker designates the whole subtree as disposable generator-owned links.
Do not put personal links or files there. Rebuilds refuse unmarked directories and real files,
leave empty directories intact, and never traverse directory links. The umbrella is never linked.
The generated view is not an input source for this tool.

Edit the manifest, then run `python3 tools/workspace.py render`; never edit the tables by hand.

Consequences:

- **A `git commit` at the umbrella level captures none of the paper or corpus work.** To change
  a paper, `cd` into that paper's directory and commit *there*. `git status` at the root will
  never show paper/corpus edits; they are gitignored, not untracked.
- **The only umbrella-level code is `tools/workspace.py`** (standard library only), tested by
  `cd tools && uv run --no-project --with pytest pytest tests`. All research code, build steps
  and their tests live inside the child repos. The study code shares one locked environment in
  `papers/iscs-2027/` (`uv sync` there); `papers/history-reach/` and
  `papers/wildchat-replication/` run their tests with that `.venv`.
- When asked to "work on the paper / run the pipeline," operate inside the relevant child repo
  and read its own README and CLAUDE.md first.

## Papers (each an independent repo under papers/)

<!-- workspace:papers:begin -->
<!-- Generated from workspace.json by tools/workspace.py render; edit the manifest. -->
| Local path | Repo | Status |
|------------|------|--------|
| `papers/knowledge-networks/` | cmri-knowledge-networks | Published: Complex Networks 2025, Springer chapter online 2026-02-25. Holds the conference paper plus the shared pipeline (`code/`) and reproducibility data (`data/`). Status after Study 1 (2026-10-22): the numbers stand; the cognitive gloss is superseded by the population result. |
| `papers/temporal-networks/` | cmri-temporal-networks; worktree `papers/temporal-networks-calibration/` on `pilot/iscs-2027-calibration` | Rejected at PLOS Complex Systems 2026-04-27; retired as a standalone paper on 2026-10-22 (withdrawal note in the manuscript). Branch `pilot/iscs-2027-calibration` holds the October 2026 calibration whose exact identities, return-lag statistic and densification exponents survive and merge into the person-networks line. |
| `papers/hierarchical-memory/` | cmri-hierarchical-memory | Accepted at ISCS 2026; Springer SSDC 709 proceedings forthcoming. An addendum correcting the small-world comparison (degree-preserving null) is in `addendum/` (2026-10-22). Formerly `agentic/`. |
| `papers/hierarchical-memory-extended/` | cmri-hierarchical-memory-extended | In preparation: the extended journal version of the ISCS 2026 paper for the Applied Network Science ISCS 2026 collection (due 2026-12-04). Pins the camera-ready and the addendum's results; corrects the small-world claim. |
| `papers/embedding-dynamics/` | cmri-embedding-dynamics | Partial draft; to be folded into a methods note or dissertation chapter (decided 2026-10-22). The false turning-cosine proposition is corrected per the October 2026 audit; see `docs/research/DYNAMICS-CONSOLIDATION.md`. |
| `papers/semantic-dynamics/` | cmri-semantic-dynamics | Full draft, retired 2026-10-22: its headline claims failed the October 2026 audit; two hypotheses (continuation versus revisitation links; within-conversation episode boundaries) are kept for later studies. |
| `papers/operational-memex/` | cmri-operational-memex | Infrastructure (the per-message embedding store), not a paper (decided 2026-10-22). |
<!-- workspace:papers:end -->

Build instructions, pipeline usage, and per-paper notes live in each child repo's README and
CLAUDE.md. The shared embedding/network pipeline is in `papers/knowledge-networks/code/` (a
working mirror of the standalone `chatgpt-complex-net`).

## Pilot repos (October 2026 campaign, also under papers/)

Between 2026-10-14 and 2026-10-18 an autonomous research campaign ran nine bounded pilots,
mostly on **public multi-participant data** (PRISM, ThoughtTrace, CGDIALOG, LongMemEval)
rather than the private archive. Each is its own private `cmri-*` repo, gitignored here like
the papers. Each has a `REPORT.md`, a `CLAIMS.md` ledger and a `VERIFICATION.md`; read the
report, not the surrounding planning prose, for the numbers.

<!-- workspace:pilots:begin -->
<!-- Generated from workspace.json by tools/workspace.py render; edit the manifest. -->
| Local path | Repo (branch) | One-line result |
|------------|---------------|-----------------|
| `papers/portfolio-wave1/` | cmri-portfolio-wave1 (`pilot/first-wave`) | Six PRISM experiments; uniform history pooling beats the last two messages for next-user prediction (+.034 MiniLM, +.019 E5, raw cosine). Holds the PRISM development caches, the source manifest and the `cmri_wave` pipeline every study imports. |
| `papers/history-falsification/` | cmri-history-falsification (`pilot/history-falsification`) | The pooling gain survives four views of the past; the user/assistant asymmetry depends on the forecast origin, not the roles. Supplies the synthetic processes and the fitted centers. |
| `papers/history-specificity/` | cmri-history-specificity (`development-wave`) | HS01 to HS04: no ranking gain over strong short-context controls; the cosine gain persists. |
| `papers/grounded-dependencies/` | cmri-grounded-dependencies (`development-pilot`); archived on GitHub | CGDIALOG human relevance labels: recency was competitive and history-mean augmentations gave no stable increment, though 300 records carry relevant turns three or more back. |
| `papers/integration-measurement/` | cmri-integration-measurement (`pilot/integration-measurement`); worktree `papers/integration-events-worktree/` on `pilot/event-centered-revision`; archived on GitHub | Zero clear same-premise revisions in 100 PRISM windows; no-go on this source. Its AI-labelled cases are excluded from the F3 reliability draw. |
| `papers/semantic-measurement/` | cmri-semantic-measurement (`pilot/public-validation`) | ThoughtTrace: the tested sequence alternative did not earn its complexity over recent-turn pooling (a narrow stopping decision; other panels inconclusive). Supplies the pinned ONNX encoders (MiniLM, E5) every study uses. |
| `papers/selective-forgetting/` | cmri-selective-forgetting (`main`); archived on GitHub | LongMemEval: the benchmark had no cross-topic cases, so the bridge hypothesis went untested. |
<!-- workspace:pilots:end -->

### Confirmatory studies (not counted among the nine pilots)

Frozen single-shot confirmations run after the pilots: the history-pooling contrast on the
PRISM reserve and on WildChat (2026-10-18), then Protocol 2, the reach decomposition (the
person's preceding turn carries the gain), on fresh PRISM and WildChat samples (2026-10-20).

<!-- workspace:studies:begin -->
<!-- Generated from workspace.json by tools/workspace.py render; edit the manifest. -->
| Local path | Repo (branch) | Role and result |
|------------|---------------|-----------------|
| `papers/iscs-2027/` | cmri-iscs-2027 (`main`) | Frozen PRISM reserve confirmation of the history-pooling gain (MiniLM +.025, E5 +.018) and Protocol 2, the reach decomposition confirmed on fresh PRISM and WildChat samples. Home of the ISCS 2027 paper (`paper/`, build with `make`), the NetSci 2027 abstract and the locked environment for all study code (`uv sync`). |
| `papers/wildchat-replication/` | cmri-wildchat-replication (`main`) | Frozen cross-collection replication on WildChat-4.8M passed in both encoders (MiniLM +.021, E5 +.016; 482 IP components, 801 conversations). |
| `papers/history-reach/` | cmri-history-reach (`master`) | Draws and encodes WildChat samples outside every earlier one: the exploratory long-conversation sample (557 conversations of 20 to 40 messages) and the fresh sample for Protocol 2. |
| `papers/reach-bits/` | cmri-reach-bits (`main`) | Reach in bits: base language models (OLMo-2, Qwen3) score the next user message under Protocol 2's contexts; development on the spent samples and Protocol R's frozen confirmation on fresh PRISM and WildChat samples (2026-10-27). |
| `papers/person-networks/` | cmri-person-networks (`main`) | Study 1: how personal is a person's knowledge network? Frozen confirmation on 250 held-out WildChat identities passed in both encoders (2026-10-21): networks are more fragmented, more clustered and more temporally episodic than size- and density-matched pools of strangers, but strangers' conversations chosen as nearest neighbours in embedding space reproduce about 90% of the fragmentation and clustering contrasts (a ratio that depends on the matching rule, not a topic share). Study 1b's frozen confirmation on 449 fresh light users passed (2026-10-22): locality usually shrinks after grouping conversations into sittings, and under the prespecified topic definition shuffling within topics reproduces most sitting-level locality. Home of the NetSci 2027 abstract. |
<!-- workspace:studies:end -->

The PRISM pilots share one 1,191-conversation development partition and a **248-component
reserve that was opened on 2026-10-18 under the frozen protocol in cmri-iscs-2027**; it must
not be reused for further confirmation. `papers/wildchat-replication/` (cmri-wildchat-replication)
re-ran the same frozen primary contrast on a second collection, English WildChat-4.8M from the
temporal pilot's local mirror, with hashed-IP components; its confirmation sample has also been
opened once and must not be reused. Protocol 2's fresh samples (250 never-selected PRISM
components; 478 WildChat IPs outside every other sample, drawn by `papers/history-reach/`) are
spent the same way, as are Study 1's 150 development and 250 confirmation WildChat identities
and Study 1b's 449 light-user identities (`papers/person-networks/`); `spent.json` lists them all. The pilots' large data directories are gitignored;
cloning a pilot repo gives you code, reports and small result tables, not the embeddings.

## Program planning layer (docs/research/)

The research agenda of 2026-10-21 (four ranked studies, the data that make them feasible, the
owner's decisions) and the designs of studies not yet frozen live in the private
`papers/agenda/` (cmri-agenda), not in this public repository; a study's design is published with
its frozen protocol. The owner's private-data inventory (counts only) is in the gitignored
`dev/inventory/`.

`docs/research/` is the umbrella-tracked record of priorities and decisions. Start with
`DECISIONS.md` (dated table separating the owner's decisions from assistant recommendations)
and `registry.yaml` (structured index of every report/claim file). `EXPERIMENT-PORTFOLIO.md`
is the P01–P12 question map; `DYNAMICS-CONSOLIDATION.md` is the plan to merge the two
dynamics drafts. `future-ideas/iscs-2027/` holds the conference history, the research
deep-dive, and a finished four-page extended abstract (`extended-abstract/`) that is the
skeleton for the planned ISCS 2027 full paper.

## Umbrella layout

- `papers/`: independent paper repos (gitignored; clone each `cmri-*` repo into place).
- `chatgpt-conversation-corpus/`: the sanitized, publishable corpus, its own git repo
  (gitignored here). Derived from `dev/`; pre-release, sanitization in progress.
- `dev/`: shared working corpus (gitignored). Raw conversation JSON the pipelines read.
  - `chatgpt-4-11-2025_json_no_embeddings/`: 1,908 raw conversation JSONs.
  - `conversations-sanitized/`: sanitized JSONs and metadata (work in progress).
  - `ablation_study/`: ablation outputs the figure scripts read.
- `future-ideas/`: early-stage directions (compsac-2026 draft, percolation, turn-taking motifs).
- `docs/`: program-level design and planning documents (see `docs/plans/`).

Child code reads the corpus from `$CMRI_CORPUS_DIR`, which defaults to this repo's `dev/`.

## Key Research Parameters

| Parameter | Optimal Value | Source |
|-----------|--------------|--------|
| User:AI weight ratio (alpha) | 2:1 | Ablation study (63 configs) |
| Similarity threshold (theta) | 0.9 | Phase transition analysis |
| Network size (full) | 601 nodes, 1718 edges | At theta=0.9 (journal paper) |
| Network size (giant component) | 449 nodes, 1615 edges | At theta=0.9 (conference paper) |
| Communities | 15 (modularity 0.750) | Louvain method |
| Embedding model | nomic-embed-text | 768-dim, 8192 token context |

The pipeline lives in `papers/knowledge-networks/code/` (see its README); embedding generation needs a reachable Ollama server (`OLLAMA_HOST`, `nomic-embed-text`) and has no local fallback.

## Corpus data-safety guardrail

The raw `dev/` conversations and any files carrying embeddings are **not safe to publish**:
they contain leaked secrets (Google API keys in `safe_urls`) and private conversation titles.
Only the separately-sanitized `chatgpt-conversation-corpus/` repo is publishable, and its
sanitization is still in progress. Before sharing or committing corpus-derived data anywhere
public, run/verify sanitization (`code/sanitize_conversations.py`, itself gitignored and
published in the dataset repo) and never commit `dev/` content or `*secret*.txt` files. The
root `cmri-secrets-to-rotate.txt` is a gitignored rotation list, not a fixture.

## External References

- Pipeline of record: https://github.com/queelius/chatgpt-complex-net (DOI: 10.5281/zenodo.15314235)
- Corpus: https://github.com/queelius/chatgpt-conversation-corpus (1,908 conversations, Dec 2022 to Apr 2025)
- Compendium DOI (this umbrella): 10.5281/zenodo.18778376
- The 2026-06-08 split into a paper cluster is recorded in
  `docs/plans/2026-06-08-papers-cluster-reorg-design.md` and the paired `-plan.md`.
