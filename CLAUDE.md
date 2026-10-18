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

- `papers/<name>/`: six independent `cmri-*` repos (their own history, remote, CLAUDE.md).
- `chatgpt-conversation-corpus/`: the publishable dataset, its own repo.

Consequences:

- **A `git commit` at the umbrella level captures none of the paper or corpus work.** To change
  a paper, `cd` into that paper's directory and commit *there*. `git status` at the root will
  never show paper/corpus edits; they are gitignored, not untracked.
- **There is no umbrella-level build or test.** The root `.pytest_cache/`, `.coverage`, and
  `.benchmarks/` are stale, gitignored tooling artifacts, not a test suite for this repo. All
  runnable code, build steps, and tests live inside the child repos (see below).
- When asked to "work on the paper / run the pipeline," operate inside the relevant child repo
  and read its own README and CLAUDE.md first.

## Papers (each an independent repo under papers/)

| Local path | Repo | Status |
|------------|------|--------|
| `papers/knowledge-networks/` | cmri-knowledge-networks | Published (Complex Networks 2025, Springer). Holds the conference paper plus the shared pipeline (`code/`) and reproducibility data (`data/`). |
| `papers/temporal-networks/` | cmri-temporal-networks | Rejected at PLOS Complex Systems 2026-04-27, seeking a venue. Temporal-evolution extension. |
| `papers/hierarchical-memory/` | cmri-hierarchical-memory | ISCS 2026. Latent hierarchical memory (episodic/concept layers). Formerly `agentic/`. |
| `papers/embedding-dynamics/` | cmri-embedding-dynamics | In progress (spec stage). |
| `papers/semantic-dynamics/` | cmri-semantic-dynamics | In progress. |
| `papers/operational-memex/` | cmri-operational-memex | Early draft. |

Build instructions, pipeline usage, and per-paper notes live in each child repo's README and
CLAUDE.md. The shared embedding/network pipeline is in `papers/knowledge-networks/code/` (a
working mirror of the standalone `chatgpt-complex-net`).

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

## The shared pipeline (papers/knowledge-networks/code/)

A `python cli.py <subcommand>` toolkit. It is not packaged; run it from inside `code/`.

```bash
cd papers/knowledge-networks/code
pip install -r requirements.txt          # networkx, scikit-learn, python-louvain, ...
cp .env.example .env                      # set OLLAMA_HOST, MODEL_NAME, INPUT_DIR, theta, alpha
```

The pipeline is staged; each `cli.py` subcommand consumes the previous stage's output:

```bash
python cli.py node-embeddings --input-dir <dir> --method role-aggregate --embedding-method llm
python cli.py normalize       --input-dir <dir>                 # L2-normalize node embeddings
python cli.py edges           --input-dir <dir> --output-file edges.json     # all-pairs cosine
python cli.py cut-off         --input-file edges.json --output-file filtered.json --cutoff 0.9
python cli.py export          --nodes-dir <dir> --edges-file filtered.json --format gexf
```

`edges-gpu` (PyTorch) replaces `edges` for large runs; `validate-json`, `is-normalized`,
`clean-embeddings`, and `test-chunking` are utilities. The `generate_*.sh` and `run_ablation_*`
scripts orchestrate full sweeps. **Embedding generation requires a reachable Ollama server**
(`OLLAMA_HOST`, default model `nomic-embed-text`). There is no local fallback.

## Architecture: the base pipeline (in papers/knowledge-networks/code/)

1. Embedding: per-message embeddings via Ollama, role-aggregated (user and AI separate means),
   weighted combination, L2-normalized. (`node-embeddings` → `normalize`)
2. Edge generation: all-pairs cosine similarity, threshold filter (theta = 0.9 optimal).
   (`edges` / `edges-gpu` → `cut-off`)
3. Network analysis: Louvain community detection, centrality metrics, core-periphery
   decomposition. (`networks.py` and the `analyze_*.py` scripts)
4. Export: GEXF/GraphML for Gephi visualization. (`export`)

Embeddings use the Ollama API (`nomic-embed-text`, 8192 token context). The pipeline weights
user messages at 2:1 versus AI responses (validated by the ablation study).

## Key Research Parameters

| Parameter | Optimal Value | Source |
|-----------|--------------|--------|
| User:AI weight ratio (alpha) | 2:1 | Ablation study (63 configs) |
| Similarity threshold (theta) | 0.9 | Phase transition analysis |
| Network size (full) | 601 nodes, 1718 edges | At theta=0.9 (journal paper) |
| Network size (giant component) | 449 nodes, 1615 edges | At theta=0.9 (conference paper) |
| Communities | 15 (modularity 0.750) | Louvain method |
| Embedding model | nomic-embed-text | 768-dim, 8192 token context |

## Corpus data-safety guardrail

The raw `dev/` conversations and any files carrying embeddings are **not safe to publish**:
they contain leaked secrets (Google API keys in `safe_urls`) and private conversation titles.
Only the separately-sanitized `chatgpt-conversation-corpus/` repo is publishable, and its
sanitization is still in progress. Before sharing or committing corpus-derived data anywhere
public, run/verify sanitization (`code/sanitize_conversations.py`, itself gitignored and
published in the dataset repo) and never commit `dev/` content or `*secret*.txt` files. The
root `cmri-secrets-to-rotate.txt` is a gitignored rotation list, not a fixture.

## Key Dependencies

These belong to the child pipeline, not the umbrella:

- Python: networkx, scikit-learn, numpy (<2), pandas, scipy, python-louvain, requests
  (Ollama API), python-dotenv, matplotlib, seaborn, tqdm.
- LaTeX: svproc.cls (Springer), llncs.cls (LNCS), beamer.
- Ollama running and reachable for embedding generation.

## External References

- Pipeline of record: https://github.com/queelius/chatgpt-complex-net (DOI: 10.5281/zenodo.15314235)
- Corpus: https://github.com/queelius/chatgpt-conversation-corpus (1,908 conversations, Dec 2022 to Apr 2025)
- Compendium DOI (this umbrella): 10.5281/zenodo.18778376
- The 2026-06-08 split into a paper cluster is recorded in
  `docs/plans/2026-06-08-papers-cluster-reorg-design.md` and the paired `-plan.md`.
