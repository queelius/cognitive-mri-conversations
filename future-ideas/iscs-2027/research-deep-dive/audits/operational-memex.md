# Operational Memex audit

Inspected 2026-10-15. Source snapshot: [manifest](../results/memex-source-manifest.json). Originals were read only. No raw conversation text, annotations, private titles, or credentials are reproduced here.

## Actual state and reusable assets

`papers/operational-memex/paper/outline.md` is an outline, not a completed manuscript; no `paper/paper.tex` exists. The analysis pipeline and many saved experiments do exist. The outline/state describe four platforms and text-embedding-3-large/3072 dimensions, whereas current code guidance, stored results and the [dynamics audit](dynamics.md) identify three embedded sources and one text-embedding-3-small/256 model. The database contains 3,144 metadata conversations and 149,623 message embeddings; the saved composed graph covers 3,097 conversations. Different counts describe different stages and must not be silently conflated.

Reusable work includes role-aware composition, content-hash provenance, graph/weight sweeps, conversation queries, candidate trails, and per-message vectors. Operational software can support later studies without its current cognitive narrative being valid. Tool functionality is not evidence that its rankings improve retrieval or navigation for users.

The claimed ISCS2026 predecessor in these notes is an agentic/delegation paper, whereas the present hierarchical-memory manuscript is a different concept-hierarchy study. Treat these historical descriptions as stale until tied to a specific publication version. The outline's cross-platform universality claims also inherit temporal-network and hierarchy interpretation problems documented in the companion audit.

## Role weighting is already an explored direction

`findings_summary.json` records different weights maximizing the selected structural objective for ChatGPT and Claude Code (user weights 3 and .5). Maximizing modularity is not an independent demonstration of user-intent fidelity. The project already considered source-dependent weighting, so a new paper cannot claim novelty from merely separating user and assistant text.

The saved `full_corpus.json` at threshold .88 and equal role weights has 3,097 nodes, 9,049 edges, 2,272 isolates and a 357-node giant component (11.5% coverage), with giant-component modularity .249. The accompanying per-source giant components contain 5/129 Anthropic, 352/472 Claude Code, and 50/2,496 ChatGPT nodes. Those are saved outputs, not fresh replications, and illustrate why matching a numeric similarity threshold does not give comparable coverage across sources. Sweeps and other thresholds produce different reported figures; do not cherry-pick them as universal constants.

## Marginalia: real result, very small and observational

`marginalia_analysis.json` records 17 annotated conversations, of which 12 appear in the analyzed giant component at threshold .78 and user weight 3. The saved 10,000-draw random-assignment tests give degree p=.6549, betweenness p=.0029, and inter-community-edge-fraction p=.0034. These are exploratory saved results, not independently reproduced p-values.

The implementation was inspected: it compares annotation flags against positions in a graph of the same author's full archive, with a simple exchangeable-node null. It does not control source, topic, age, length, visit/retrieval frequency, or when an annotation was made. Louvain is invoked without a fixed seed. Its signed rank effect is negative under the chosen convention even when annotated values are larger; the sign should not be narrated as the opposite association. Restriction to the giant component and the tiny selected annotated set further limit inference.

This supports a follow-up question about what the author chose to annotate. It does not establish causal attention allocation, general user behavior, or that bridge retention improves memory. The completed forgetting pilot directly warns against that last leap. Existing annotations are too sparse to anchor a new population paper under the present constraints.

## Decision

Use the pipeline as infrastructure and the archive as a secondary case study. Defer a stand-alone Memex/marginalia paper until an independently evaluated retrieval/navigation task or richer public interaction/annotation data are available. A software paper is possible only after a separate software-comparison and adoption/reproducibility assessment; it is not established by this research audit.

Do not optimize role weights or threshold on a structural metric and then use that same metric as evidence of cognitive validity. A new contribution should instead test representation reliability or usefulness against an external target, with group-disjoint evaluation and a simple baseline.
