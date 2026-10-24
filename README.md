# Cognitive MRI of AI Conversations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18778376.svg)](https://doi.org/10.5281/zenodo.18778376)

[Alexander Towell](https://orcid.org/0000-0001-6443-9897) and
[John Matta](https://orcid.org/0000-0002-7666-1409), Southern Illinois University Edwardsville

We study what network analysis of people's conversations with AI assistants can and cannot
reveal about how they explore knowledge. The program began with one archive, Alex's 1,908
ChatGPT conversations (December 2022 to April 2025), turned into semantic similarity networks.
It now tests its claims on public collections with many people (WildChat, PRISM), because one
archive is a case, not a population.

This repository is the landing page. Each paper and study is its own repository; this page says
where each one stands.

## Where things stand (October 2026)

| Work | Status | Repository |
|---|---|---|
| Cognitive MRI of AI Conversations: semantic embedding networks | **Published**, Complex Networks 2025 (Springer, February 2026) | [cmri-knowledge-networks](https://github.com/queelius/cmri-knowledge-networks) |
| From Episodes to Abstractions: Latent Hierarchical Memory in 1,908 AI Conversations | **In press**, ISCS 2026 (Springer proceedings). An addendum corrects its small-world comparison | [cmri-hierarchical-memory](https://github.com/queelius/cmri-hierarchical-memory) |
| From Episodes to Abstractions, extended journal version | **In preparation** for the Applied Network Science ISCS 2026 collection, due December 4, 2026 | [cmri-hierarchical-memory-extended](https://github.com/queelius/cmri-hierarchical-memory-extended) |
| How personal is a knowledge network? | **In preparation**, NetSci 2027 abstract, due December 1, 2026 | [cmri-person-networks](https://github.com/queelius/cmri-person-networks) |
| How far back does a person's next message reach? | **In preparation**, ISCS 2027 full paper, due March 1, 2027 | [cmri-iscs-2027](https://github.com/queelius/cmri-iscs-2027) |
| Temporal Evolution of Cognitive Knowledge Networks | **Retired** after rejection at PLOS Complex Systems (April 2026); its surviving methods moved into the person-networks study | [cmri-temporal-networks](https://github.com/queelius/cmri-temporal-networks) |
| Semantic dynamics and embedding dynamics drafts | **Retired** after an internal audit (October 2026); parts may become a methods note | [cmri-semantic-dynamics](https://github.com/queelius/cmri-semantic-dynamics), [cmri-embedding-dynamics](https://github.com/queelius/cmri-embedding-dynamics) |
| Sanitized ChatGPT conversation corpus | **Not yet released**; sanitization in progress | [chatgpt-conversation-corpus](https://github.com/queelius/chatgpt-conversation-corpus) |

Most repositories other than this one are private while the work is unpublished.

## What we have found so far

Results confirmed by frozen, single-shot tests on samples no earlier analysis had touched:

- **A person's recent history predicts their next message better than their last two
  messages do**, on held-out PRISM data and again on English WildChat-4.8M, in both text
  encoders we use. A second protocol located the gain: it is carried by the person's own
  preceding turn, not by deep history.
- **A person's conversation network is more
  fragmented, more clustered and more bunched in time than a matched pool of strangers'
  conversations** (250 held-out WildChat users), but strangers' conversations on the same topics
  reproduce about 90% of the difference. The bunching in time comes mostly from sittings and
  topic periods (449 further users).

Corrections to our own earlier work, carried by the addendum and the journal extension:

- The hierarchy paper's comparison with the small-world topology of human semantic memory does
  not hold: graphs rebuilt to keep every episode's and every concept's degree already reach most
  of the observed small-world coefficient. The comparison is withdrawn.
- Its vocabulary-growth exponent stands, but the sentence calling it robust to ordering does not.

## Program documents

- [Research index](https://github.com/queelius/cognitive-mri-conversations/blob/master/docs/research/README.md): current priorities, existing work and what the pilots established.
- [Decision history](https://github.com/queelius/cognitive-mri-conversations/blob/master/docs/research/DECISIONS.md): what was decided, when, and on what evidence.
- [Program design](https://github.com/queelius/cognitive-mri-conversations/blob/master/docs/superpowers/specs/2026-10-23-cmri-research-program-design.md): how the program keeps track of its papers, studies, questions and datasets.

## How the work is organized

Each paper and study keeps its own design, dated amendments, claims ledger and generated
numbers. Every design is reviewed by an independent model before results are read, and every
manuscript number is generated from committed result files. `workspace.json` lists every
repository; `python3 tools/workspace.py clone` sets up a full working copy for collaborators
with access.

## Shared data and pipeline

- Analysis pipeline of record: [chatgpt-complex-net](https://github.com/queelius/chatgpt-complex-net)
  (DOI [10.5281/zenodo.15314235](https://doi.org/10.5281/zenodo.15314235)).
- Conversation corpus: [chatgpt-conversation-corpus](https://github.com/queelius/chatgpt-conversation-corpus)
  (pre-release).

## Citation

Cite the specific paper you use; each repository has its own `CITATION.cff`. For the program as a
whole, use the compendium DOI [10.5281/zenodo.18778376](https://doi.org/10.5281/zenodo.18778376).

## License

[MIT](https://github.com/queelius/cognitive-mri-conversations/blob/master/LICENSE)
