# Cognitive MRI of AI Conversations: Research Compendium

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18778376.svg)](https://doi.org/10.5281/zenodo.18778376)

Applying complex network analysis to ChatGPT conversation archives to reveal knowledge
organization, community structure, and temporal evolution patterns.

**Authors:**
[Alexander Towell](https://orcid.org/0000-0001-6443-9897) and
[John Matta](https://orcid.org/0000-0002-7666-1409),
Southern Illinois University Edwardsville

## Overview

This is the umbrella for a research program that transforms sequential AI conversation logs
into semantic similarity networks, revealing latent cognitive structure in AI-assisted
knowledge exploration. The program analyzes 1,908 ChatGPT conversations spanning December
2022 to April 2025. This repository is a landing page: it indexes the individual paper
repositories and points to the shared data and pipeline.

For current priorities, existing work and future options, start with the
[research plans and metadata](docs/research/README.md). The immediate priority is an
empirical ISCS 2027 paper, with semantic/embedding dynamics retained as a second track.
The [dynamics consolidation assessment](docs/research/DYNAMICS-CONSOLIDATION.md) compares
the two related drafts; their manuscripts and repositories remain separate.

## Papers

Most paper projects are independent Git repositories under `papers/` (not tracked by this
umbrella). The synthesis draft and data-paper manuscript live in the additional locations
listed below. Clone child repositories individually. Statuses below reflect the October 16–17,
2026 review; the [full catalog](future-ideas/iscs-2027/research-deep-dive/PAPER-CATALOG-AND-OPTIONS.md)
links the manuscripts and publication evidence.

| Paper | Venue / status | Path | Repository |
|-------|----------------|------|------------|
| Cognitive MRI of AI Conversations (semantic embedding networks) | Complex Networks 2025; Springer chapter published February 2026 | `papers/knowledge-networks/` | [cmri-knowledge-networks](https://github.com/queelius/cmri-knowledge-networks) |
| Temporal Evolution of Cognitive Knowledge Networks | PLOS Complex Systems (rejected 2026-04-27, seeking venue) | `papers/temporal-networks/` | [cmri-temporal-networks](https://github.com/queelius/cmri-temporal-networks) |
| From Episodes to Abstractions: Latent Hierarchical Memory | ISCS 2026 accepted; full proceedings forthcoming | `papers/hierarchical-memory/` | [cmri-hierarchical-memory](https://github.com/queelius/cmri-hierarchical-memory) |
| Embedding Dynamics | Partial draft; proposed consolidation with Semantic Dynamics | `papers/embedding-dynamics/` | [cmri-embedding-dynamics](https://github.com/queelius/cmri-embedding-dynamics) |
| Semantic Dynamics | Draft; proposed consolidation with Embedding Dynamics | `papers/semantic-dynamics/` | [cmri-semantic-dynamics](https://github.com/queelius/cmri-semantic-dynamics) |
| Multi-Scale Network Organization in AI-Assisted Knowledge Exploration | Unsubmitted synthesis draft | `future-ideas/compsac-2026/paper/` | This umbrella repository |
| Sanitized ChatGPT Conversation Corpus | Unpublished data-paper draft; release status unresolved | `chatgpt-conversation-corpus/paper/` | [chatgpt-conversation-corpus](https://github.com/queelius/chatgpt-conversation-corpus) |
| Operational Memex | Outline and working infrastructure | `papers/operational-memex/` | [cmri-operational-memex](https://github.com/queelius/cmri-operational-memex) |

### Pilot studies (October 2026)

Nine bounded pilots, mostly on public multi-participant data, each in its own private
repository under `papers/` with a `REPORT.md` and claim ledger:
[cmri-portfolio-wave1](https://github.com/queelius/cmri-portfolio-wave1),
[cmri-history-falsification](https://github.com/queelius/cmri-history-falsification),
[cmri-history-specificity](https://github.com/queelius/cmri-history-specificity),
[cmri-grounded-dependencies](https://github.com/queelius/cmri-grounded-dependencies),
[cmri-integration-measurement](https://github.com/queelius/cmri-integration-measurement),
[cmri-semantic-measurement](https://github.com/queelius/cmri-semantic-measurement),
[cmri-selective-forgetting](https://github.com/queelius/cmri-selective-forgetting),
[cmri-wildchat-replication](https://github.com/queelius/cmri-wildchat-replication), plus the
temporal calibration on the `pilot/iscs-2027-calibration` branch of
[cmri-temporal-networks](https://github.com/queelius/cmri-temporal-networks). The
[decision history](docs/research/DECISIONS.md) records what each one established.

The published conference paper and the shared analysis pipeline (`code/` and `data/`) live
in the `knowledge-networks` repository; the temporal-evolution journal extension is
`temporal-networks`.

## How this repository is organized

- `papers/`: the independent paper repositories (gitignored here; clone each `cmri-*` repo
  into place, or work with them separately).
- `chatgpt-conversation-corpus/`: the sanitized, publishable conversation dataset, its own git
  repository (gitignored here). Derived from `dev/`; **pre-release, sanitization in progress.**
- `dev/`: shared working corpus (gitignored; the raw conversation JSON the pipelines read).
- `future-ideas/`: early-stage directions not yet promoted to their own repos.
- `docs/`: program-level design and planning documents.

## Shared data and pipeline

- Conversation corpus (dataset): [chatgpt-conversation-corpus](https://github.com/queelius/chatgpt-conversation-corpus), co-located in this cluster at `chatgpt-conversation-corpus/`. The sanitized, publishable version of the corpus (pre-release; sanitization in progress).
- Analysis pipeline of record: [chatgpt-complex-net](https://github.com/queelius/chatgpt-complex-net) (DOI: [10.5281/zenodo.15314235](https://doi.org/10.5281/zenodo.15314235))

Child code reads the raw corpus from `$CMRI_CORPUS_DIR`, which defaults to this repo's `dev/`.

## Citation

Cite the specific paper you use (each paper repository has its own `CITATION.cff`). To cite
the program as a whole, use the compendium DOI:
[10.5281/zenodo.18778376](https://doi.org/10.5281/zenodo.18778376). See
[CITATION.cff](CITATION.cff) for machine-readable metadata.

## License

[MIT](LICENSE)
