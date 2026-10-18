# ISCS 2027 paper and follow-up program: design

Date: 2026-10-18. Owner: Alexander Richard Towell. Drafted by Claude from its own read of
the October 2026 pilots; the owner opted out of the clarifying-questions round, so every
choice below is Claude's call unless marked otherwise.

## Intent

Produce a full (8-page short) paper for ISCS 2027 (Porto, June 2–4, 2027; due March 1, 2027)
from the evidence the October pilots already produced, plus a small number of follow-up
experiments chosen because they unlock a result Codex's gates declined to promote. Keep the
dynamics consolidation and the temporal-networks revision as separate lines that reuse the
same corrections.

The owner's stated preferences (DECISIONS.md, Oct 16–18): a new empirical finding about
human–AI conversation; "useful connections emerge through back-and-forth"; a full paper is
primary and an extended abstract is only a companion; investigate several questions before
narrowing; do not let deadlines drive content (memory: feedback_no_deadline_pressure).

## The paper

**Working title:** *What the Whole Conversation Predicts: History, Horizon and Observation
in Human–AI Dialogue.*

**Thesis (one sentence):** across many participants, the next thing a person says to an AI is
predicted better by the whole prior exchange than by the latest reply; the apparent opposite
for the assistant is an artifact of what the forecast is allowed to see; and two cheap
observation choices (message packaging and fixed-graph revelation) manufacture "dynamics" on
their own, so any claim about conversational development must report its observation
process.

This is the positive version of Codex's abstract *When Observation Choices Resemble
Dynamics*. Same evidence, led by the finding instead of the caution. It answers both reviewer
pools: multi-participant public data (single-user critique) and exact null models for growth
(cumulative-snapshot critique).

### Evidence map

| Section | Claim | Source (pilot, file) | Status |
|---|---|---|---|
| 3.1 History gain | Uniform history beats last-two for next-user prediction: +.034 [.023,.049] MiniLM, +.019 [.016,.023] E5; stable across four views | portfolio-wave1 P05; history-falsification REPORT §views | Development evidence; **to confirm on reserve** |
| 3.2 Horizon reversal | Role interaction −.087 → +.017 (MiniLM), −.022 → +.001 (E5) under common prefix; synthetic no-history process reproduces the natural pattern | history-falsification REPORT; synthetic scenario A | Complete |
| 3.3 Ranking saturates | No ranking gain over strong opening+latest control (−.010/−.002 MRR); cosine gain persists | history-specificity REPORT | Complete; reported as a boundary, not a negative |
| 3.4 Where the signal is | Budget-response curve and user-vs-assistant middle contribution (+.023/+.016) | history-specificity HS03/HS04 | Complete; **to re-run descriptively on reserve** |
| 4.1 Packaging | Re-encoding halves shifts state cosine .054/.010; fixed-vector identities exact | portfolio-wave1 P06 | Complete |
| 4.2 Revelation | E[M_k] = M·k(k−1)/(N(N−1)); E[A_k] identity; verified by enumeration | temporal-networks-calibration; abstract §2 | Complete (short, cite the temporal revision for the rest) |
| 5 Heterogeneity | Distribution of per-participant / per-component gains; task-type split | **new, cheap** | To run |
| 6 Discussion | Not a memory mechanism; not learning; CCL/OnePred/Flipping-the-Dialogue positioned as neighbours, not blockers | history-falsification audits/literature.md | Write |

Everything in rows marked Complete already has replayable results, claim ledgers and
bootstrap intervals in the pilot repos. The paper's new work is rows 3.1/3.4 (reserve
confirmation) and 5 (heterogeneity).

### Goals

- A paper whose primary result is confirmed on data nobody has looked at (the 248 reserve
  components), with the development numbers reported alongside as the pre-registration.
- Every effect reported with component-level bootstrap intervals and both encoders.
- A reader can reproduce every number from the pilot repos plus one new confirmation repo.

### Non-goals

- Proving a memory mechanism, a semantic half-life, or anything "physics".
- Beating CCL/OnePred on their tasks. We are not building a next-query predictor.
- Human-labelled behavioural constructs (integration, revision). Those are the optional
  companion study below; the paper must stand without them.
- Touching the private 1,908-conversation archive. The paper is public-data only. The private
  archive appears only as motivation (cited prior papers).

### Anti-goals (would make this a failure even if it technically works)

- Reporting the reserve result after peeking or after changing the primary contrast. The
  protocol is frozen before the reserve is encoded, full stop.
- Inflating .03 cosine into a cognitive claim. The honest framing is "small, robust,
  role-specific, horizon-dependent".
- A paper that reads as a list of cautions. The cautions are sections 4 and 6; the finding is
  section 3.

### Constraints

- Public data only: PRISM (CC-BY-ish per dataset card; re-verify), ThoughtTrace/CGDIALOG not
  needed. No participant recruitment, no paid model calls.
- Encoders fixed: MiniLM-L6-v2 and E5-small-v2, CPU, as in the pilots. A third encoder is a
  sensitivity, not a replication.
- The pilot repos are frozen (hash-pinned FREEZE.json, runners refuse to overwrite). Do not
  modify them. New work lives in a new repo that imports their `src/` read-only, the way
  history-falsification imports portfolio-wave1.
- Authorship: Towell; Matta to be asked (credited on the two prior papers). Not decided here.
- Venue terms for 2027 (reuse of the accepted-abstract material, companion submissions) are
  still unverified; verify before February.

## Follow-up experiments, in priority order

The first column says what Codex demanded before it would promote the result; the second
says what this design accepts, and why that is the right bar for a conference short paper.

### F1. Reserve confirmation of the history gain (required for the paper)

- **Codex demanded:** "a distinctive contribution beyond CCL/OnePred" before opening the
  reserve. **Accepted here:** distinctiveness is role × horizon × multi-participant on human–LLM
  data; conference-level distinct. Open the reserve under a frozen protocol.
- **Protocol (freeze before encoding):** primary contrast = uniform-mean minus last-two
  pooling, next-user target, native128 view, both encoders; component-level bootstrap
  (2,000 draws); success = lower bound > 0 in both encoders, with .01 as the practical floor
  reported but not gating. Secondary panel, declared now: the common-prefix role interaction;
  the budget-response curve (HS03) descriptively; user-vs-assistant middle (HS04).
- **Scope:** 248 components, ~800 conversations, two encoders, CPU. The pilots' pipeline did
  1,191 conversations in minutes per encoder; expect under an hour.
- **Home:** new repo `papers/iscs-2027/` (`cmri-iscs-2027`), containing `paper/` and
  `confirmation/` (imports `cmri_history` and wave1 `src/`, pins their commits).

### F2. Participant heterogeneity (new, cheap, complex-systems flavour)

- Distribution of per-component gains on development + reserve; tail behaviour; split by
  PRISM task/topic metadata and conversation length. Question: is the history gain a
  population average hiding a bimodal "some people are predictable from history, some
  aren't"? A heavy tail here is a real complex-systems result; a tight unimodal is also
  worth one figure.
- No new gate; descriptive with intervals.

### F3. Author-coded integration study (optional companion section or separate paper)

- **Codex demanded:** independent coders before using any behavioural label; it never got
  one and wrote off P01/P04/P07. **Accepted here:** two-author coding (Towell + Matta, or
  Towell + a blind second pass) with Cohen's kappa on a 60–100 item sample is the standard
  for a conference paper. The owner is a competent coder of his own construct.
- Use Codex's prepared material: 60 blinded development examples and rubrics
  (portfolio-wave1/annotations), the 20-packet event set (integration-measurement/events).
- Outcome: prevalence of integration vs switching vs listing in multi-topic PRISM
  conversations, and whether the published bridge taxonomy's network scores track the labels
  beyond direct-text baselines. If kappa is poor, report that as the finding and stop.
- Only enters the ISCS paper if done by mid-January; otherwise it is its own short paper.

### F4. Temporal-networks revision (separate line, reuses 4.2)

- Replace beta (72% power, undefined at theta=.95) with the direct return-lag statistic
  (96% power) as the primary; keep block-permutation nulls; add the revelation identities as
  the growth null. Select up to 20 new public archive proxies on metadata only (Codex's
  proposal in temporal NEXT-GOAL.md) to retire the outcome-selected WildChat pair.
- Target: Applied Network Science or the ISCS journal invitation. Not March-bound.

### F5. Dynamics consolidation (track 2, not March-bound)

- One manuscript in `embedding-dynamics`, following DYNAMICS-CONSOLIDATION.md centre A
  (state reconstruction at matched cost), with the audit's corrections (drop the −1/2 turning
  theorem, endpoint-collapsing derivatives, degenerate shape distance). Sections 3.3/4.1 of
  the ISCS paper become its empirical anchor.

## New directions (brainstorm; none scheduled)

Ideas that are related but not rescues of existing pilots. Each is one line of what it
is and one line of why it might matter; a sentence of risk.

1. **Predictability as a participant trait.** PRISM has 1,500 participants with demographics
   and self-reported LLM familiarity. Does the history gain correlate with familiarity or with
   the participant's stated conversation goal (values-guided, controversy, unguided)? Turns a
   measurement into a human finding. Risk: ~5 conversations per participant is thin.
2. **The assistant as a low-pass filter.** The horizon reversal suggests the assistant turn
   is near-deterministic given the request. Measure how much of the user's *next* turn is
   explained by the assistant's reply versus by the user's own prior turns (own-history vs
   partner-history decomposition, already partly in HS04). A clean "who shapes whom" number
   per conversation. Risk: confounded by shared topic; needs the synthetic null again.
3. **Observation-process null models as a reusable library.** The revelation identities,
   packaging identities and common-prefix contrast as a small, tested Python package
   (`convnull`?) that any conversation-network paper can run. A JOSS-able software
   contribution that outlives this paper. Risk: scope creep; keep to three nulls.
4. **Anonymization distortion of network conclusions.** Codex's option 4: apply the corpus
   sanitizer at several strengths to the public PRISM network and the private archive network,
   and measure whether centralities, bridges and communities move. Self-contained, uses the
   corpus repo's work, and gives the data paper a scientific result instead of a release.
   Risk: "privacy loses information" is known; the paper needs a *structural* distortion.
5. **Two-platform, one-person replication.** The Claude Code / Codex session logs (5,219
   conversation ids found Oct 16) are a second archive of the same author on a different
   platform and task mix. Replicating the Cognitive MRI structure claims across the two
   archives does not fix single-user, but it tests platform-dependence, which no reviewer has
   asked for yet and which the Operational Memex outline already wants. Risk: coding-agent
   sessions are not conversations in the same sense; needs a role/turn adapter.
6. **Return-lag on public longitudinal users.** PRISM is cross-sectional. WildChat hashed-IP
   histories are weak proxies. Are there public datasets with verified repeat users over
   months (LMSYS-Chat-1M user ids, ShareGPT accounts, the new 2026 longitudinal releases)?
   A metadata-only audit, half a day, could unlock the "return after a long gap" question
   Codex kept deferring. Risk: licensing and linkage, as always.
7. **Does a hierarchy level predict anything?** The ISCS 2026 hierarchy paper picks 500/50/8
   levels interpretively. On PRISM, test whether concept-level membership at each cut improves
   next-user prediction over the flat embedding (same P05 machinery, different features).
   Links the two accepted papers to the new finding. Risk: concept extraction on PRISM is new
   work.

## What is left to the builder

- Exact file layout of `cmri-iscs-2027` and how it pins the two upstream pilot commits.
- The bootstrap/aggregation code is in `cmri_history.analysis`; reuse, do not rewrite.
- Figure set: one for 3.1 (gain by view and encoder), one for 3.2 (the existing forecast-origin
  figure), one for 5 (component-gain distribution), one for 4.2 (revelation curve vs observed).
- LaTeX: Springer `svproc`/SSDC class as used for the hierarchy paper; the `iscs.cls` in the
  abstract package is for the 2–4-page format only. Confirm which class the 2027 short-paper
  track specifies.

## Sequence

1. Design approved (this document). 2. Plan for F1+F2 (superpowers:writing-plans).
3. Freeze F1 protocol; run; report. 4. F2. 5. Draft the paper from the abstract skeleton
(November–December). 6. F3 decision by mid-January. 7. Internal review rounds in February;
submit by March 1 only if the content is ready (owner's rule: content over deadline).
