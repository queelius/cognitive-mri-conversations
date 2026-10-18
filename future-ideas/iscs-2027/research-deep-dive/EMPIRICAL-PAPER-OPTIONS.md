# Empirical paper options after searching the conversation logs

October 16, 2026. Exploratory discovery following the completed temporal pilot.

**Prioritize a small feasibility study of how cross-topic connections develop through human–AI exchanges and subsequently get used.** The user wants a new empirical finding and a defensible ISCS paper. He describes useful connections as emerging through back-and-forth. This favors the development of an idea through an exchange over attributing it to whichever speaker mentioned it first.

Working question: **Does elaborating and revising a connection predict its later application beyond topic similarity, repetition, and ordinary recency?** This is a candidate question, not an established effect or a recommendation to write a full paper immediately.

## What the local search establishes

Two existing Memex databases contain 5,219 distinct conversation identifiers after removing their 472-identifier overlap. This is an identifier union, not a count of independent human interactions or a cleaned research sample. It includes coding-agent activity and imported context. Targeted full-text queries and surrounding-turn inspection found examples of cross-context recurrence, explicit revision, intended workflow transfer, and a self-reported return to an old manuscript.

The cases are useful because their meanings differ despite sharing vocabulary:

- A conceptual connection recurs in a different project and receives further elaboration.
- A user revisits a creative approach explicitly to change it; recurrence does not imply endorsement of the earlier formulation.
- A request to try a tool on another paper establishes an intention, not completed reuse.
- A question about where an assistant's idea came from receives counterevidence pointing to pre-existing project materials. Without inspecting that history, it cannot establish assistant invention or erroneous attribution.
- Successive revisions of one prompt and generated context summaries can produce apparent repeated user contributions.

Private identifiers, timestamps, excerpts and project-specific case descriptions are excluded from this repository. Local evidence pointers and search notes are retained under `~/.local/share/cmri/research-mining/2026-10-16/`, with restricted permissions. The underlying logs remain local. All inspected histories are development material.

There are newer raw Codex records through October. A header inventory found 928 JSONL files: 910 had the expected first-record schema, comprising 723 subagent sessions, 158 CLI sessions, 28 VS Code sessions and one exec session. Eighteen headers require separate handling. These are file/source classifications, not a human-conversation census. Six purposively selected newer non-CMRI sessions were inspected for short user-role messages. One contained a malformed nonblank record, so this was not a validated import. The full approximately 23.7 GB collection was not searched exhaustively.

**Existing data are sufficient for discovering candidate phenomena and developing a codebook.** Importing and embedding everything is not the next bottleneck. Reliable authorship/provenance, observable reuse, independent evaluation and public-data suitability are.

## Leads from the saved numeric artifacts

[Probe code](checks/exploratory-reuse/probe.py), [saved output](checks/exploratory-reuse/results.json), and [reproduction notes](checks/exploratory-reuse/README.md) preserve descriptive queries on existing inputs. They are post-pilot exploration, not an independent evaluation or a new hypothesis test.

| Existing artifact | Descriptive result | Interpretation limit |
|---|---|---|
| Hierarchy, 1,908 episodes | 11,457 concept-pair occurrences, 5,935 distinct pairs; 48.20% of occurrences exceed one occurrence per distinct pair. | Repeated co-occurrence, not evidence of chronological application, learning or consolidation. |
| Previously saved fixed-margin null | Mean implied repeat fraction 21.94% over the existing diagnostic draws. | Reuses 16 finite-chain samples; mixing and inferential calibration are not established. This is a lead, not a significance result. |
| Fixed eight-domain hierarchy | 991 of 3,954 cross-domain pairs occur at least twice (25.06%); occurrence-weighted repeat fraction 43.44%. Within-domain repeat fraction is higher, 55.64%. | No evidence that bridges are exceptionally reusable. The partition is retrospective and extraction provenance is incomplete. |
| Two already selected public archive proxies | At similarity .90, median nearest-return ages are about 17 minutes and 10 days; 12/130 and 42/107 returns exceed 30 days. | Selection is known; these are not fresh confirmation, verified individuals, or independently annotated idea returns. Task continuation and near-duplicate text remain alternatives. |

The hierarchy result suggests looking at **persistent combinations**, rather than reviving the unsuccessful claim that the projection's clustering alone establishes a memory mechanism. The temporal result motivates separating short continuation from later return. Neither supplies the behavioral outcome needed for the proposed paper.

## Preferred candidate

**Working title:** From Connections to Reuse: How Ideas Develop in Human–AI Conversations.

**Contribution to test.** The published static paper classified bridge conversations; the hierarchy represented shared concepts; the temporal work measured semantic returns. The new contribution would establish what some of those links *do*: whether a connection is proposed, substantively developed, corrected, or applied later. A network-specific claim would require additional value beyond direct text and simple topic/recency predictors.

**Unit and outcome.** An identifiable connection between ideas, followed through subsequent human–AI turns and, where linkage supports it, later episodes. Distinguish application to a task from paraphrase, acknowledgement, rejection, quotation and copied artifacts. Allow unresolved labels and multiple contributors. First observed mention is not origin, ownership or invention. A later application need not preserve the original formulation unchanged.

**Competing hypotheses.** Substantive back-and-forth predicts later application; alternatively, its apparent benefit is explained by attention, repeated exposure, topic importance, task difficulty, or more opportunities to observe an outcome. Corrections might support useful reformulation, or simply identify unproductive suggestions. Both are worth discovering.

**Smallest decisive next step, proposed rather than launched.**

1. Develop an event codebook on the inspected private cases and a small public development sample. Include copied drafts, generated summaries, brief assent, disagreement, abandoned connections and ambiguous source attribution as deliberate counterexamples. Do not label embedding similarity as uptake.
2. Audit whether a public source has enough *substantive later user turns* to distinguish application from immediate preference. PRISM is a plausible source for within-conversation work, not a substitute for long personal histories. Inspect the data before fixing a sample size. No additional participant recruitment is assumed.
3. Freeze a bounded annotation-feasibility protocol, suggested cap 60 public conversation trees with participant-disjoint development/evaluation. Include both retrieved candidate cases and a random audit sample; retrieval-selected cases alone cannot estimate prevalence or recall. Preserve all cases with no usable outcome.
4. Require independent review of the substantive labels before claiming behavioral ground truth. An LLM can help retrieve candidates; agreement between model-generated labels and an embedding score does not validate the construct. If credible annotation or adequate later turns are unavailable, stop or narrow before building a large pipeline.
5. Only after feasibility, preregister a comparison of direct similarity, recency/frequency, length, topic/task, prior pair history, and simple role/turn features against the sequence or network feature. Match observation opportunities and handle end-of-record censoring. Use prefix-only features and participant/archive-level uncertainty; keep whole related histories together.

The proposed 60-tree cap is for feasibility, not a power claim or a frozen final experiment. An informative label audit would tell us whether a larger study is justified. A result based only on this author's history should be presented as a case study; short public conversations would test a related within-conversation question, not independently replicate month-scale reuse.

**Failure criteria.** Defer a full paper if apparent applications mostly collapse into copies/continuations, labels cannot be checked reliably, public linkage or follow-up is inadequate, or the remaining finding is already covered by prior work. A negative result is useful only if it changes a substantive inference, not simply because another embedding score fails.

## Alternatives

| Direction | Why retain it | Decision now |
|---|---|---|
| Burst continuation versus long-gap return | Reuses the calibrated temporal pipeline; asks which observed returns are continuation, reapplication or revision. | Fallback. Must beat recency/frequency explanations and verify meaning; two selected archives cannot justify a paper. |
| Assistant suggestion uptake and rejection | Human agency is directly relevant; public ratings and later turns may support observable outcomes. | Integrate the useful part into the preferred candidate. Generic alignment, endorsement and uptake are already studied. |
| Erasing obsolete code to escape design fixation | A newer log points to a separate, substantial local controlled-erasure project. Its protocol distinguishes source inaccessibility from repair pressure. | Defer as a separate software/agent study. Inspected readiness document says pre-freeze and runner-blocked; no scored effectiveness result was inspected. The CMRI retention pilot's negative result does not settle this different question. |
| Pure platform or topology comparison | Large existing embedding and graph assets. | Do not lead with it. Activity, task, role, context generation and provenance remain confounds. |

## Closest literature and novelty constraints

Searches on October 16 used public concepts and titles only. No private excerpts were submitted to online services.

- **ThinkFlow, CHI 2026:** extracts a cognitive workflow schema from conversations and evaluates reuse across sense-making tasks. This rules out claiming that turning conversations into reusable workflows is itself new. The proposed observational question is which naturally occurring exchanges precede later application. [Author publication page](https://research.adobe.com/publication/from-conversation-to-human-ai-common-ground-extracting-cognitive-workflows-for-reuse-in-sense-making-tasks/). Abstract inspected; full methods remain to audit.
- **CoNode, CHI 2026:** a system for preserving, reusing and recombining process knowledge in team–AI design, with user studies. A new generic graph interface for reuse would overlap substantially. [Primary university record and paper](https://scholars.cityu.edu.hk/en/publications/conode-visualizing-workflows-for-knowledge-reuse-and-recombinatio/), [DOI](https://doi.org/10.1145/3772318.3791216). Abstract and publication status inspected; full study instruments remain to audit.
- **Choi and Nixon, February 2026 preprint:** measures human–AI idea uptake using semantic and endorsement signals with background controls. Its own validation exposes difficulty distinguishing uptake from topical similarity and backchanneling. Generic similarity-based uptake is therefore neither new nor adequately validated by the label alone. [Full text](https://arxiv.org/html/2602.09269v1). Uptake definitions and limitations inspected.
- **Zhou et al., Decision Support Systems 2026:** studies message-level response adoption and linguistic signals in platform logs. This narrows novelty for a generic “which responses do users accept?” study. [Publisher record](https://doi.org/10.1016/j.dss.2026.114734). Publisher search record inspected; direct page retrieval failed, so its methods and data availability remain unverified.
- **Ebrahimi et al., July 2026 preprint:** directly measures deletion avoidance and provides CanItDelete. The separate erasure project would need to add an intervention on access to an old design, not merely report that agents retain code. [Primary paper](https://arxiv.org/abs/2607.28887). Abstract inspected; full mitigation comparison remains to audit.
- **Recency/frequency and recombination:** human-mobility recurrence models and prior work on recombinant lag are necessary baselines for any return-time story. [Barbosa et al.](https://doi.org/10.1140/epjds/s13688-015-0059-8), [Kok et al.](https://doi.org/10.1177/0149206318765926). The analogy motivates controls; it does not establish the same mechanism in conversations.

[PRISM's dataset card](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment) documents participant-linked rated dialogue trees. Both response alternatives are displayed: an unchosen response is not an unexposed control. Ratings/selection are distinct from substantive later use. Licensing, release revision, duplicates and usable follow-up must be pinned before a pilot; no new public corpus was downloaded in this exploration.

Public search queries included `LLM human AI conversation idea uptake rejection persistence user adopts suggestions linguistic entrainment study`, `LLM coding agents fixation existing code deletion refactoring cognitive bias study`, `"human AI" "idea uptake" dataset`, `"conversations" "idea reuse" "AI" study`, `"knowledge recombination" "human-AI" conversations`, and exact-title follow-ups for the studies above. Earlier exploration also searched recency/frequency, recombinant lag, topic return and linguistic alignment. Coverage is targeted rather than systematic. Search hits and informal posts were discovery leads; technical comparisons above use primary author/publisher sources. This search does not establish that the proposed contribution is unique.

## ISCS scope and next investment

The [current submission page](https://iutdijon.u-bourgogne.fr/ccs-france/submissions/) allows an eight-page original short paper and a two-to-four-page extended abstract. A focused empirical claim about collective idea development fits better than another broad framework. The earlier journal-extension/overlap audit remains applicable; do not repackage the published static/hierarchy findings as new results.

The next investment should be the bounded public-data and annotation feasibility check above, with the inspected private cases as motivation. Do not yet commit to a manuscript, import every new log, or train another embedding method. The current temporal [NEXT-GOAL.md](NEXT-GOAL.md) remains a conditional fallback; this memo adds a user-directed empirical candidate without erasing the preceding negative or inconclusive findings.
