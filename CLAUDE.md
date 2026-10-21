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
| `papers/knowledge-networks/` | cmri-knowledge-networks | Published: Complex Networks 2025, Springer chapter online 2026-02-25. Holds the conference paper plus the shared pipeline (`code/`) and reproducibility data (`data/`). |
| `papers/temporal-networks/` | cmri-temporal-networks; worktree `papers/temporal-networks-calibration/` on `pilot/iscs-2027-calibration` | Rejected at PLOS Complex Systems 2026-04-27. Branch `pilot/iscs-2027-calibration` holds the October 2026 calibration that supersedes the old growth/beta claims and supplies the densification exponents the ISCS 2027 paper reports. |
| `papers/hierarchical-memory/` | cmri-hierarchical-memory | Accepted at ISCS 2026; Springer SSDC 709 proceedings forthcoming. Formerly `agentic/`. |
| `papers/embedding-dynamics/` | cmri-embedding-dynamics | Partial draft (abstract + validation section). Proposed consolidation with Semantic Dynamics; see `docs/research/DYNAMICS-CONSOLIDATION.md`. |
| `papers/semantic-dynamics/` | cmri-semantic-dynamics | Full draft (LaTeX is more complete than the Markdown). Several of its mathematical claims failed the October 2026 audit. |
| `papers/operational-memex/` | cmri-operational-memex | Outline plus working infrastructure. |
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
| `papers/grounded-dependencies/` | cmri-grounded-dependencies (`development-pilot`); archived on GitHub | CGDIALOG human labels: recency alone matches embedding augmentations. |
| `papers/integration-measurement/` | cmri-integration-measurement (`pilot/integration-measurement`); worktree `papers/integration-events-worktree/` on `pilot/event-centered-revision`; archived on GitHub | Zero clear same-premise revisions in 100 PRISM windows; no-go on this source. Its AI-labelled cases are excluded from the F3 reliability draw. |
| `papers/semantic-measurement/` | cmri-semantic-measurement (`pilot/public-validation`) | ThoughtTrace: sequence-trajectory features are reliably worse than pooling. Supplies the pinned ONNX encoders (MiniLM, E5) every study uses. |
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
<!-- workspace:studies:end -->

The PRISM pilots share one 1,191-conversation development partition and a **248-component
reserve that was opened on 2026-10-18 under the frozen protocol in cmri-iscs-2027**; it must
not be reused for further confirmation. `papers/wildchat-replication/` (cmri-wildchat-replication)
re-ran the same frozen primary contrast on a second collection, English WildChat-4.8M from the
temporal pilot's local mirror, with hashed-IP components; its confirmation sample has also been
opened once and must not be reused. Protocol 2's fresh samples (250 never-selected PRISM
components; 478 WildChat IPs outside every other sample, drawn by `papers/history-reach/`) are
spent the same way. The pilots' large data directories are gitignored;
cloning a pilot repo gives you code, reports and small result tables, not the embeddings.

## Program planning layer (docs/research/)

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
