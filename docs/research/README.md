# CMRI research plans and existing work

Program owner: Alexander Richard Towell. Updated 2026-10-24.

This directory is the program-level record. The long reverse-chronological version of this
page, with every dated update paragraph from the October 14–18, 2026 campaign, is preserved
in git history (commit `51b87e2`); it was replaced by this index because the stacked updates
had become unreadable. Nothing in the linked reports changed.

## Where to look

| Need | File |
|---|---|
| What was decided, by whom, on what evidence | [DECISIONS.md](DECISIONS.md) (dated entries; newest first) |
| Historical index of reports, claim ledgers and verification files | [registry.yaml](registry.yaml); current statuses come from `pubs` and the private agenda records through `python3 tools/workspace.py board` |
| The open research questions and the twelve proposed probes P01–P12 | [EXPERIMENT-PORTFOLIO.md](EXPERIMENT-PORTFOLIO.md) |
| Historical dynamics consolidation proposal and claim audit; superseded by the October 22 retirements | [DYNAMICS-CONSOLIDATION.md](DYNAMICS-CONSOLIDATION.md) |
| The history-specificity lead and why it was stopped | [HISTORY-SPECIFICITY-LEAD.md](HISTORY-SPECIFICITY-LEAD.md) |
| Last proposed (not launched) next goal from the campaign | [NEXT-GOAL.md](NEXT-GOAL.md) |
| Manuscript catalog with publication evidence | [PAPER-CATALOG-AND-OPTIONS.md](../../future-ideas/iscs-2027/research-deep-dive/PAPER-CATALOG-AND-OPTIONS.md) |
| Conference history and venue terms (ISCS) | [future-ideas/iscs-2027/README.md](../../future-ideas/iscs-2027/README.md), [VENUE.md](../../future-ideas/iscs-2027/research-deep-dive/VENUE.md) |
| Finished four-page extended abstract and its evidence package | [extended-abstract/](../../future-ideas/iscs-2027/extended-abstract/README.md) |
| Hierarchy extension: decisions, corrections and design notes (private child repository) | [PLAN.md](../../papers/hierarchical-memory-extended/PLAN.md), [CLAIMS.md](../../papers/hierarchical-memory-extended/CLAIMS.md), [notes/](../../papers/hierarchical-memory-extended/notes/), [extractor design](../../papers/hierarchical-memory-extended/analyses/E-hierarchy-robustness/DESIGN.md) |
| Program map and engine design; sources for the board, stage view, questions and datasets | [Program design](../superpowers/specs/2026-10-23-cmri-research-program-design.md), [workspace tool](../../tools/workspace.py) |
| Reach decomposition confirmed after the pilots | [Protocol 2](../../papers/iscs-2027/PROTOCOL2.md), [PRISM result](../../papers/iscs-2027/results/protocol2/prism.md), [WildChat result](../../papers/iscs-2027/results/protocol2/wildchat.md) |
| Person-network confirmations and current NetSci abstract (private child repository) | [Study 1](../../papers/person-networks/RESULTS.md), [Study 1b](../../papers/person-networks/episodes/RESULTS.md), [abstract](../../papers/person-networks/abstracts/netsci-2027/main.tex) |

Child-repository links require the corresponding checkout or repository access. The board and
`by-stage/` view are local and private; this page records public program-level status only.

## Current priorities

The active submissions are listed in program priority order; NetSci's deadline falls three
days before the journal extension's.

1. **Applied Network Science journal extension of the ISCS 2026 paper**, due **2026-12-04**.
   In preparation in `papers/hierarchical-memory-extended/`. The extension audits the original
   hierarchy claims against construction-aware controls and tests extractor dependence. The
   comparison with human semantic memory is withdrawn; pair recurrence (N1) is withheld as a
   validation limitation. Vocabulary-growth and domain-coherence interpretations also remain
   withheld pending adequate controls. The owner chose paid API extraction with a spending cap,
   phased acceptance and a separate batched arm. Component C is exploratory work on an existing
   WildChat development sample, not a new confirmation; its inclusion requires extraction
   approval by November 1 and checked first results by November 13. These are decisions and
   gates, not completed results. Matta's go-ahead remains open.
2. **NetSci 2027 person-networks abstract**, due **2026-12-01**. In preparation in
   `papers/person-networks/abstracts/netsci-2027/`, drawing on Studies 1 and 1b. Their frozen
   confirmations stand, while the interpretation is being re-examined on development data
   before the abstract is rebuilt. The reach abstract in
   `papers/iscs-2027/abstracts/netsci-2027/` is superseded, not a second active submission.
3. **ISCS 2027 full paper, "How far back does a person's next message reach?"**, due
   **2027-03-01**. The twelve-page draft by Towell and Matta is in `papers/iscs-2027/paper/`.
   It reports the pooling confirmations and Protocol 2's reach decomposition, with centered
   cosine carrying the interpretation after the raw-cosine pool-size artifact was identified.
   Further controls and machine-generated manuscript numbers remain part of the October 22
   review requirements. F3 is designed but unrun, re-scoped to person-network bridges with
   blinded model coders. It is not a completed behavioral validation.

The October 22 decisions retired temporal networks as a standalone paper and semantic dynamics,
and folded embedding dynamics into a possible methods note or dissertation chapter. Dynamics is
not a second active manuscript track. Study 3 folds into Study 2; Study 4 is stopped.

## Existing manuscripts

| Line | Status | Reusable contribution |
|---|---|---|
| Cognitive MRI of AI Conversations | Published Springer chapter, 2026-02-25 (Complex Networks 2025) | Conversation graphs, communities, bridge taxonomy |
| From Episodes to Abstractions | In press, ISCS 2026; correction addendum prepared | Episode–concept incidence and descriptive hierarchy results; semantic-memory comparison withdrawn |
| Hierarchy journal extension (working title) | In preparation, Applied Network Science; due 2026-12-04 | Claim corrections, construction-aware controls and extractor comparisons; new interpretations remain conditional |
| Person-networks abstract (working title) | In preparation, NetSci 2027; due 2026-12-01; revision pending | Studies 1 and 1b confirmations, with interpretation under review |
| How far back does a person's next message reach? | In preparation, ISCS 2027; twelve-page draft; due 2027-03-01 | Cross-collection pooling confirmations and reach decomposition |
| Reach-based NetSci 2027 abstract | Superseded by the person-networks abstract | Historical companion draft in `papers/iscs-2027/abstracts/netsci-2027/` |
| Temporal Evolution of Cognitive Knowledge Networks | Rejected at PLOS Complex Systems 2026-04-27; retired as a standalone paper 2026-10-22 | Surviving graph/order controls and return-lag methods fold into person networks |
| Embedding Dynamics | Folded 2026-10-22; methods-note or dissertation destination not settled | Corrected prefix approximation, filtering and order experiments |
| Semantic Dynamics | Retired 2026-10-22; headline claims failed audit | Two hypotheses retained for later studies; no active manuscript |
| Multi-Scale Network Organization (COMPSAC synthesis) | Retired 2026-10-22 | Historical synthesis and background text |
| Sanitized ChatGPT Conversation Corpus | Unpublished data-paper draft; release unresolved | Sanitization and corpus statistics |

Operational Memex is infrastructure, not a paper. Publication status comes from `pubs`; study
state comes from the private agenda's `studies.json`. The table is a dated public summary.

## What the October 2026 pilots established

Nine pilots ran between October 14 and 18 (see the pilot table in the umbrella `CLAUDE.md`
and the per-pilot `REPORT.md` files). The findings below distinguish measured contrasts from
their interpretations; the later frozen confirmations are separate from those nine pilots.

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
- **The user/assistant asymmetry depends on the information available at the forecast origin.**
  The role interaction moves
  from −.087 to +.017 (MiniLM) when the assistant forecast is denied the current user request;
  a synthetic process with no history dependence reproduces the original pattern.
- **Observation unit is a confound.** Re-encoding message halves shifts state cosine by .054
  (MiniLM) / .010 (E5).
- **Fixed-graph revelation produces growth by itself.** E[M_k] = M·k(k−1)/(N(N−1)) for a fixed
  endpoint graph revealed in random order. The old preferential-attachment beta is a different
  estimand with 72% power; the direct return-lag statistic is better powered (24/25 vs 18/25)
  and shorter than block-null in all three archives.
- **Bounded negatives (narrowed 2026-10-25 after an independent review):** on ThoughtTrace's
  reported-purpose retrieval task, the development-selected sequence alternative did not earn its
  complexity over recent-turn pooling (primary difference -4.9 points, selection-aware interval
  -8.6 to +4.8; other panels inconclusive), a stopping decision for that proposal rather than
  evidence that order carries no information; on CGDIALOG's human relevance labels, recency was
  competitive and history-mean augmentations gave no stable increment, although 300 records carry
  annotated relevant turns three or more back.

Dead ends on these sources: the integration pilot identified no clear same-premise revisions
in 100 windows from 36 capped PRISM packets using unvalidated AI drafts, so revision persistence
was unestimable; the LongMemEval sample had no cross-topic cases, so
the bridge-retention hypothesis was never actually tested.

The campaign set its "full paper" bar at journal-of-record level (novelty beyond the closest
prior work, independent confirmation on an unopened reserve, and a human-labeled behavioral
outcome), so every pilot was recorded as "narrow" or "no-go". Treat those verdicts as
conservative. The 248-component PRISM reserve was opened once, on 2026-10-18, under the
frozen protocol in `papers/iscs-2027/`.

### Later confirmations and checks (through 2026-10-24)

- **Protocol 2 passed on fresh PRISM and WildChat samples (October 20).** All three frozen
  centered-cosine contrasts passed in both encoders: adding the person's preceding turn to the
  latest exchange improves the forecast; the full history mean does not improve on that
  three-message recipe; and the preceding user turn contributes more than the preceding
  assistant reply. This narrows the pooling result to a reach decomposition, not evidence that
  arbitrarily distant history is needed. [PRISM](../../papers/iscs-2027/results/protocol2/prism.md),
  [WildChat](../../papers/iscs-2027/results/protocol2/wildchat.md).
- **Study 1 passed (October 21).** All three frozen hypotheses held in both encoders on 250
  held-out WildChat identities. The interpretation remains under development-data review;
  passing the frozen contrasts does not establish a person-specific cognitive mechanism.
  [Decision record](DECISIONS.md).
- **Study 1b passed (October 22).** The validity control and all three frozen hypotheses held
  in both encoders on 449 fresh WildChat identities; Study 1's measures reproduced as a
  pre-specified ungated secondary. Explanatory and trait readings remain qualified. These
  identities are IP-based proxies, not verified people. [Decision record](DECISIONS.md).
- **The grounded-dependency negative was reproduced (October 23).** The corrected result
  reproduced byte for byte; the schema repair left all prefix scores unchanged. This supplies
  the missing reproducibility record, not a new sample or confirmation. The narrow/stop
  decision stands. [Verification](../../papers/grounded-dependencies/VERIFICATION.md).

The October 19 editorial review identified a raw-cosine pool-size artifact; the centered-cosine
confirmations stand and carry the paper's interpretation. No confirmation sample above may be
reused as a fresh holdout. Later exploratory hierarchy corrections and API approvals are recorded
in [DECISIONS.md](DECISIONS.md); they are not additional frozen confirmations.

## Updating these records

- Add dated entries to `DECISIONS.md` when priorities or recommendations change; record who
  decided and what evidence changed the judgment.
- Keep `registry.yaml` as a historical index and add evidence pointers there. Current publication
  statuses belong in `pubs`; study, question and dataset records belong in the private agenda
  (`studies.json`, `questions.json`, `datasets.json`), read by `python3 tools/workspace.py board`.
- Keep this page an index. Put narrative in the linked files.
- Keep private text local; respect the child repositories; define outcome, baselines, split
  unit and decision criteria before each experiment.
