# CMRI research plans and existing work

Program owner: Alexander Richard Towell. Updated 2026-10-18.

This directory is the program-level record. The long reverse-chronological version of this
page, with every dated update paragraph from the October 14–18, 2026 campaign, is preserved
in git history (commit `51b87e2`); it was replaced by this index because the stacked updates
had become unreadable. Nothing in the linked reports changed.

## Where to look

| Need | File |
|---|---|
| What was decided, by whom, on what evidence | [DECISIONS.md](DECISIONS.md) (dated table; newest first) |
| Structured index of every report, claim ledger and verification file | [registry.yaml](registry.yaml) |
| The open research questions and the twelve proposed probes P01–P12 | [EXPERIMENT-PORTFOLIO.md](EXPERIMENT-PORTFOLIO.md) |
| How the two dynamics drafts should be merged, and which of their claims survived audit | [DYNAMICS-CONSOLIDATION.md](DYNAMICS-CONSOLIDATION.md) |
| The history-specificity lead and why it was stopped | [HISTORY-SPECIFICITY-LEAD.md](HISTORY-SPECIFICITY-LEAD.md) |
| Last proposed (not launched) next goal from the campaign | [NEXT-GOAL.md](NEXT-GOAL.md) |
| Manuscript catalog with publication evidence | [PAPER-CATALOG-AND-OPTIONS.md](../../future-ideas/iscs-2027/research-deep-dive/PAPER-CATALOG-AND-OPTIONS.md) |
| Conference history and venue terms (ISCS) | [future-ideas/iscs-2027/README.md](../../future-ideas/iscs-2027/README.md), [VENUE.md](../../future-ideas/iscs-2027/research-deep-dive/VENUE.md) |
| Finished four-page extended abstract and its evidence package | [extended-abstract/](../../future-ideas/iscs-2027/extended-abstract/README.md) |

## Current priorities

1. **ISCS 2027 full paper** (Porto, June 2–4, 2027; submission due March 1, 2027; long 12 pp,
   short 8 pp, abstract 2–4 pp; Springer SSDC proceedings). The owner wants a full paper; an
   extended abstract is acceptable only as a companion to a separate full-paper submission.
   Publication/reuse terms for 2027 and companion-submission rules are still unverified.
   A reviewed eight-page short-paper draft is at `papers/iscs-2027/paper/` (October 18, 2026).
2. **Semantic/embedding dynamics** as a second track: one consolidated manuscript lineage
   drawing on both drafts, with the audit corrections carried forward.

## Existing manuscripts

| Line | Status | Reusable contribution |
|---|---|---|
| Cognitive MRI of AI Conversations | Published Springer chapter, 2026-02-25 (Complex Networks 2025) | Conversation graphs, communities, bridge taxonomy |
| From Episodes to Abstractions | Accepted, ISCS 2026; proceedings forthcoming | Episode–concept incidence, hierarchical abstraction |
| Temporal Evolution of Cognitive Knowledge Networks | Rejected at PLOS Complex Systems 2026-04-27; calibration completed | Graph/order controls, return-lag measurements, public-archive plumbing |
| Embedding Dynamics | Partial draft | Prefix approximation, filtering, order experiments |
| Semantic Dynamics | Full draft; several claims failed audit | Episodes, role divergence, trajectory graphs |
| Multi-Scale Network Organization | Unsubmitted synthesis draft | Background text |
| Sanitized ChatGPT Conversation Corpus | Unpublished data-paper draft; release unresolved | Sanitization and corpus statistics |

## What the October 2026 pilots established

Nine pilots ran between October 14 and 18 (see the pilot table in the umbrella `CLAUDE.md`
and the per-pilot `REPORT.md` files). The findings that survived every control:

- **Reserve confirmation passed.** Under a frozen protocol on the held-out 248-component
  PRISM reserve (233 components, 775 conversations scored), uniform history minus last-two
  pooling: +.02546 [.01853, .03247] MiniLM, +.01801 [.01650, .01943] E5; the interval lower
  bound is above zero in both encoders. [Report](../../papers/iscs-2027/REPORT.md).
- **WildChat replication passed.** The same frozen contrast on a second collection (English
  WildChat-4.8M, 482 hashed-IP components, 801 conversations scored): +.02134 [.01358, .02897]
  MiniLM, +.01562 [.01388, .01736] E5; lower bound above zero in both encoders. Smaller than
  the PRISM reserve, and the common-prefix role-interaction reversal did not replicate.
  [Report](../../papers/wildchat-replication/REPORT.md).
- **History pooling helps predict the next user turn.** On 1,191 public PRISM conversations,
  uniform history minus last-two pooling: +.034 [.023, .049] MiniLM, +.019 [.016, .023] E5;
  stable across four observation views.
- **The user/assistant asymmetry is a forecast-horizon artifact.** The role interaction moves
  from −.087 to +.017 (MiniLM) when the assistant forecast is denied the current user request;
  a synthetic process with no history dependence reproduces the original pattern.
- **Observation unit is a confound.** Re-encoding message halves shifts state cosine by .054
  (MiniLM) / .010 (E5).
- **Fixed-graph revelation produces growth by itself.** E[M_k] = M·k(k−1)/(N(N−1)) for a fixed
  endpoint graph revealed in random order. The old preferential-attachment beta is a different
  estimand with 72% power; the direct return-lag statistic is better powered (24/25 vs 18/25)
  and shorter than block-null in all three archives.
- **Clean negatives:** sequence-trajectory features are reliably worse than pooling for
  reason identification (ThoughtTrace); with human relevance labels (CGDIALOG), recency alone
  matches any embedding augmentation.

Dead ends on these sources: same-premise revision is too rare in capped PRISM conversations
to measure (0 clear cases in 100 windows); the LongMemEval sample had no cross-topic cases, so
the bridge-retention hypothesis was never actually tested.

The campaign set its "full paper" bar at journal-of-record level (novelty beyond the closest
prior work, independent confirmation on an unopened reserve, and a human-labeled behavioral
outcome), so every pilot was recorded as "narrow" or "no-go". Treat those verdicts as
conservative. The 248-component PRISM reserve was opened once, on 2026-10-18, under the
frozen protocol in `papers/iscs-2027/`.

## Updating these records

- Add dated rows to `DECISIONS.md` when priorities or recommendations change; record who
  decided and what evidence changed the judgment.
- Update `registry.yaml` when a report, claim ledger or verification file is added.
- Keep this page an index. Put narrative in the linked files.
- Keep private text local; respect the child repositories; define outcome, baselines, split
  unit and decision criteria before each experiment.
