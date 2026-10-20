# F3: two-coder integration study on PRISM conversations (design)

Date: 2026-10-19. Owner: Alexander Richard Towell. Status: design for owner review; nothing
built, no sample drawn, no coding started. Follows F3 of
[the ISCS 2027 paper design](2026-10-18-iscs-2027-paper-design.md) and the materials
inventory of 2026-10-19 (scratchpad; summarised in "What exists" below).

## Revision history

- 2026-10-19, after the editorial review (`docs/research/audits/2026-10-19-editorial-review/`,
  `editor-framing.md` Sect. 8 and `REVIEW.md` "F3 and the programme"), five changes:
  1. The cross-participant PRISM network score (betweenness, neighbour-community count) is
     replaced by a within-conversation, construct-matched score computed from the cached
     per-message vectors, defined in full below and frozen before any coding.
  2. The kappa gate is changed from a lower-bound gate to a point-estimate gate with the
     interval reported, after reproducing the arithmetic that shows the lower-bound gate
     cannot pass at plausible prevalence on 80 items.
  3. A non-author third coder is added on a 30-item subset (blind packet and codebook only).
  4. A sequencing section puts the PRISM chosen-reply test before F3, on development data and
     labelled exploratory; its result is recorded in `docs/research/DECISIONS.md` and decides
     among three branches (F3 unchanged, F3 with a revised construct, F3 deferred).
  5. The outcome in which reliability passes but the construct does not separate is a
     workshop note, not a paper; the same holds for a kappa between .4 and .6.

## Intent

The October pilots established a replicated, pre-registered *geometric* fact (the history
gain) and showed that the measurement does not identify what people do with earlier turns.
Every reviewer line of attack on the programme's behavioural vocabulary ("integration",
"bridge", "revision") comes back to the same gap: no human has ever labelled any of it.
F3 closes that gap for one construct, the P01 distinction between a conversation that
*integrates* distinct ideas and one that merely *switches* between or *lists* them, with the
two authors as independent coders and a non-author third coder on a subset.

The question is deliberately modest: **can two trained humans agree on whether a human--AI
conversation integrates distinct ideas, and if so, how common is integration, and does a
frozen within-conversation score computed before anyone saw a label track it?** A
reliability failure is a recordable result; the study is designed so that each outcome is
written up at the length it earns (see "Output and venue").

## Goals, non-goals, anti-goals

Goals
- A pre-registered reliability estimate (Cohen's kappa with a conversation-level bootstrap
  interval) for the P01 summary label on a fresh sample of PRISM conversations.
- A prevalence estimate of integration, switching and listing among multi-topic human--AI
  conversations, with the ambiguous rate reported, not hidden.
- One pre-registered test of whether a frozen, label-blind, within-conversation score (the
  bridging-turn score B and the reach-back margin R defined below, computed from the cached
  message vectors of the sampled conversation itself) is higher for integrating conversations
  than for the rest.
- An independence check: a non-author third coder on a 30-item subset, reported against each
  author's locked first pass.
- Materials and code that a further coder could reuse: rubric v2, practice set, packet
  builder, agreement script, score code with synthetic tests.

Non-goals
- No claim about memory, learning or causal influence; integration is a property of the
  visible text.
- No reuse of the spent PRISM reserve or either WildChat sample.
- No AI-drafted labels in the reliability sample, as primary or as "adjudication aid".
- No attempt to validate the three published bridge *types* (evolutionary, integrative,
  pure); the inventory found no executable definition of them. The test is on continuous
  scores.
- No cross-participant similarity graph. The reason is given under "Within-conversation
  score".

Anti-goals (what would make it a failure even if it "works")
- Reporting agreement after discussion as reliability.
- Tuning the rubric, the sample or the score after seeing labels.
- Letting the coders see the pilot reports' per-case commentary, the AI drafts or the
  review guide that names cases.
- Letting the third coder see anything beyond the blind packet and the codebook.

## What exists (from the inventory)

- A blinded packet of 60 PRISM "fit" conversations (468 messages, 54 components) with a
  JSON schema and rubric v1 (P01, P04, P07), gitignored under
  `papers/portfolio-wave1/data/annotations/wave1/`. **42 of the 60 were read and AI-labelled
  by the coordinating assistant** in the integration-measurement pilots (6 calibration + 36
  first-exchange); only about 18 are untouched. The packet is therefore unusable as the
  reliability sample and is repurposed as the practice set (see below).
- Zero completed human labels anywhere. No P01 prevalence estimate exists.
- Calibration notes identify the rubric's weak point: "integration" as written is satisfied
  by ordinary explanatory answers that relate two ideas the user already supplied, and
  "ambiguous" is the dump category for ordinary continuity, so base-rate skew would depress
  kappa.
- Message-level MiniLM and E5 encodings for the 1,191 development conversations (722 fit,
  229 tune, 240 probe) are cached in `papers/portfolio-wave1/cache/encoded/{minilm,e5}.npz`
  with metadata in the matching `.json`. Per conversation `{id}`: `{id}__messages` is one
  384-dimensional row per message of the chosen path (6, 8, 10 or 12 messages, user first,
  alternating), `{id}__alternatives` is the same shape and holds, at each assistant position,
  the unnormalised mean of the candidate replies the participant rejected at that turn (zero
  rows at user positions and where no candidate was recorded), and
  `{id}__alternative_counts` gives the number of rejected candidates per position (1 to 3 at
  almost every assistant turn). Roles, component, split, participant and token masses are in
  the json `records`. These arrays are sufficient for both the within-conversation score and
  the chosen-reply test; no graph, threshold or Louvain run is needed, and no text is read.
- The 248-component reserve is spent and excluded.

## Sequencing: the PRISM chosen-reply test runs first

F3 spends about five coder-days on the premise that drawing on earlier material is something
PRISM participants visibly do. PRISM records a cheaper behavioural outcome that bears on that
premise directly: at each turn the participant chose one reply among several candidates. The
editorial review's alternative 2 asks whether the pooled history predicts *which reply the
participant chose* better than the latest exchange does. It is a day of assistant work on
vectors that already exist, so it runs before any F3 material is built, and its result decides
how F3 proceeds.

### The test (defined here before it is run)

- Data: the 1,191 development conversations of wave 1 (fit, tune and probe partitions). The
  reserve is spent and WildChat has no candidate replies, so no held-out sample exists; the
  test is **exploratory** and every report of it says so. It is nonetheless written down here
  in full before it is run.
- Unit: an assistant position s of the chosen path (1-based, so s is even) at the wave-1
  forecast positions, s in {6, 8, 10, 12} within the path length, with
  `alternative_counts[s] >= 1`. The outcome at s is the chosen reply x_s against the
  rejected-candidate mean a_s = `{id}__alternatives[s]`.
- Two predictors, as in wave 1: the pooled history p_s = mean(x_1, ..., x_{s-1}) (uniform
  pooling over the whole visible path, the construct the history gain was about) and the
  latest exchange q_s = mean(x_{s-2}, x_{s-1}) (the previous reply and the current request).
- Metric: centred cosine, which the editorial review established as the valid metric (raw
  cosine rewards pooling regardless of content). The centre for this test is c_A, the mean of
  every assistant message in the 722 FIT-split conversations, the wave-1 fit-target mean for
  assistant targets. Raw cosine is reported as a sensitivity only.
- Margin of a predictor v at position s: m_s(v) = cos_c(v, x_s) - cos_c(v, a_s), where
  cos_c(v, w) is the cosine of v - c_A and w - c_A. A positive margin means the predictor is
  closer to the reply the participant chose than to the ones they rejected.
- Contrast: D_s = m_s(p_s) - m_s(q_s), aggregated the wave-1 way (mean over positions within
  conversation, over conversations within component, then equally over components), with a
  2,000-draw component bootstrap percentile interval, in MiniLM and E5. Also reported: the
  pooled predictor's own margin m(p) and its win rate (share of positions with m_s(p_s) > 0)
  against the chance level of one half, because if the chosen reply is not distinguishable
  from the rejected mean by any predictor the contrast is uninformative.
- Caveat fixed in advance: the cache holds the *mean* of the rejected candidates, not each
  candidate, so this is a chosen-versus-rejected-mean contrast. A per-candidate version (which
  of the k candidates was chosen) needs the candidates re-encoded from the pinned PRISM source
  and is a follow-up if the first contrast is positive; it is not required to decide the
  branch. Wave 1's protocol rule against using current-round alternatives applied to the
  forecasting models; here the alternatives are the outcome, not a predictor, and nothing is
  fitted.

### Where the result goes and what it decides

The assistant writes the result table (both encoders, D with interval, m(p) with interval,
win rate, counts of positions, conversations and components) to the wave-1 repository's
`results/` and summarises it in a dated entry in `docs/research/DECISIONS.md`. The owner's
decision is recorded in the same entry. Three branches:

1. **F3 proceeds unchanged.** D is positive with the 95% interval above zero in both encoders,
   and the pooled predictor's win rate is above one half. The pooled history is behaviourally
   usable at the level of choice; F3 asks the finer question of whether humans can see the
   integration of distinct ideas in the text. This document stands as written.
2. **F3 proceeds with a revised construct.** The intervals include zero, or the encoders
   disagree in sign, or the win rate is not above one half. Drawing on older material then has
   no visible footprint in choice, and coding the rarer and stricter construct (integration)
   first would be the wrong order. The revision, made before the sample is drawn and dated in
   the rubric: the primary coded construct becomes *reach-back* (does the user's turn draw on
   material from before the latest exchange: yes or no, with the evidence span), which has a
   higher base rate and matches what the history gain measures; `integration` becomes a
   sub-label of reach-back; the reliability primary becomes the binary reach-back label; and
   the primary score becomes R (defined below), with B secondary. Sample, gate, third coder
   and data rules are unchanged.
3. **F3 is deferred.** D is negative with both intervals below zero. The pooled history's
   advantage is then a property of the next user turn (wave 1) and not of the participant's
   choice among replies, and coder time is not spent until that discrepancy is understood.
   The multi-archive network replication (the editorial review's alternative 1, F4) is
   scheduled first; rubric v2 and schema v2 are kept for later use.

No F3 build step (packet, scores, freeze) starts before the DECISIONS.md entry exists.

## Design

### Sample

A fresh draw of **80 conversations, one per dependence component**, from the 722-conversation
PRISM fit split, excluding every component touched by the 42 AI-exposed cases (and the one
sibling component), by a salted hash recorded before the draw (same selection domain
mechanism as `materials.py`). Eligibility: the chosen path has at least six messages, so that
"distinct ideas" can occur (every cached path has six or more). Eighty keeps each coder's
load near two days. The interval on a binary kappa at n=80 is wide, and wider than this
document first said: the arithmetic under "Reliability statistic and decision rule" gives
about plus or minus .3 at kappa .6 and ten percent prevalence, not plus or minus .2. The
study is written up as a reliability estimate with that width, not as a precise one.
*My call; the owner may prefer 100, which narrows the interval to about plus or minus .28
and does not change the gate logic.*

Not adopted: the editor's stratified draw (half the sample from conversations with a high
label-blind topic-count proxy, half random, prevalence from the random half). It would roughly
double the positive count, which the arithmetic below shows is the binding constraint, but the
proxy would be computed from the same cached vectors as the score under test, so the score
test would run on a sample preselected on the score's own ingredients, and the prevalence
estimate would rest on 40 items. The owner may overrule this; if so, the proxy must be
defined and frozen before the draw and the score test restricted to the random half.

The coder packet is the chosen path only, capped at twelve messages, exactly as the
embeddings saw it, so labels and scores describe the same text. The custodian mapping
(case id to conversation id and component) stays in the ignored data directory.

A **30-item subset** of the 80 for the third coder is drawn by a second salted hash over the
80 case ids, recorded in `FREEZE.json` with the main draw and before any coding.

### Rubric v2 (P01 only)

Rubric v1's four labels are kept; two changes are frozen before coding:

1. **Integration requires a relation introduced in the exchange.** The text must relate
   two distinct ideas in a way that was not already stated in the user's prompt that
   introduced them, and the relation must change an inference, explanation, solution or
   use. An assistant answer that explains how A relates to B when the user asked "how does
   A relate to B" is *not* integration; a later turn that applies the answer about A to a
   new question about B is.
2. **Ordinary continuity gets its own label.** A fifth summary label, `single`, for a
   conversation that develops one idea without a second distinct one, replaces the misuse
   of `ambiguous` for that case; `ambiguous` is reserved for genuinely undecidable cases.

The unit is the **whole-exchange summary label**; local units are optional and not scored
for reliability. *My call: whole-exchange only keeps the task at two days and makes kappa
interpretable; local units can be a follow-up.*

The schema is the v1 `blank-schema.json` with the P04 and P07 blocks removed and `single`
added to the P01 label set (schema v2, committed to the study repository). Under branch 2 of
the sequencing section the schema also gains a binary `reach_back` block with an evidence
span, and `integration` is permitted only when `reach_back` is yes.

### Practice and exposure

- A practice set of 10 conversations drawn from the 42 already-exposed packet cases
  (they cannot be in the reliability sample anyway). Each coder codes them, then the two
  coders discuss, and the rubric may be clarified *once*, with a dated v2.1, before the
  reliability sample is opened. No reliability case is discussed before both first passes
  are locked.
- Both coders have read the pilot reports (counts and paraphrased examples) and the
  rubric. That exposure is recorded in the protocol as a limitation; it does not touch the
  fresh sample, whose cases no one has read.
- The coordinating assistant builds the packet and runs the agreement script but never
  sees a label before both first passes are locked, and never drafts a label.

### Third coder (non-author, 30-item subset)

A reviewer will call a two-author result "author-coded" however the exposure is disclosed.
The third coder answers that at a cost of about one day of a non-author's time.

- Who: a person who is not an author of any CMRI paper and has not read the pilot reports;
  named by the owner (question 6 below). Their participation and the rubric they received are
  recorded in `PROTOCOL.md`.
- What they receive: the **blind packet** for the 30-item subset (chosen path only, capped at
  twelve messages, case ids only, same text the authors see) and the **codebook only**: rubric
  v2.1 as frozen after the practice round, with its worked examples written as paraphrases,
  and the schema. No practice set, no pilot reports, no author labels, no discussion of any
  case. Questions about the wording of the codebook are answered in writing and appended to
  the codebook as a dated clarification log; no case is discussed.
- When: at the third coder's convenience between the freeze and the start of adjudication;
  their labels are locked with a timestamp before adjudication begins and are never changed.
  Their labels do not enter the adjudicated set.
- What is computed, all with the conversation-level bootstrap and all reported, none gated:
  binary and five-category Cohen's kappa of the third coder against each author's locked
  first pass (two pairwise kappas); Fleiss' kappa over the three coders on the 30 items; the
  two authors' kappa on the same 30 items for comparison; and the third coder's agreement
  with the adjudicated label on those items.
- How disagreement is reported: the results file lists, per case id, the three first-pass
  labels and the adjudicated label; the report gives the count of items where the third coder
  differs from both authors when the authors agree, the count where they side with one author
  against the other, and the third coder's label distribution against the authors', so a
  reader can see whether the third coder applies the labels with a different threshold or a
  different understanding. On 30 items at the expected prevalence a kappa has a standard
  error near .25, so this is an independence check and a description of disagreement, not a
  second reliability estimate.

### Reliability statistic and decision rule (frozen before coding)

- Primary: Cohen's kappa on the **binary** collapse `integration` versus everything else,
  with a 2,000-draw bootstrap over conversations (one conversation per component, so the
  component-clustered bootstrap is a conversation bootstrap), percentile interval.
- Secondary: five-category kappa; prevalence-adjusted agreement (PABAK and Gwet's AC1)
  because the base rate is expected to be skewed; the full confusion matrix.

**Why the lower-bound gate in the first draft cannot pass.** The first draft gated on "binary
kappa lower bound above .4 and point estimate at least .6". The editorial review observed that
at a plausible prevalence of ten percent, 80 items yield about eight positives and the lower
bound will not clear .4 even with good coders. Reproducing the arithmetic, with n = 80, each
coder marking a fraction p = .10 of items positive, and a true kappa of .6:

- chance agreement p_e = p^2 + (1 - p)^2 = .01 + .81 = .82;
- observed agreement implied by kappa .6: p_o = p_e + .6 (1 - p_e) = .82 + .108 = .928, about
  5.8 disagreements in 80 items;
- the 2 x 2 table that realises it with symmetric marginals: both-positive 5.12, discordant
  2.88 and 2.88, both-negative 69.12 (so the estimate rides on about five agreed positives);
- large-sample standard error, SE = sqrt(p_o (1 - p_o) / (n (1 - p_e)^2))
  = sqrt(.928 x .072 / (80 x .0324)) = sqrt(.0258) = .161, hence a 95% interval of
  .6 plus or minus 1.96 x .161 = [.285, .915]; the Fleiss-Cohen-Everitt (1969) variance on the
  same table gives SE = .151 and [.303, .897]. Either way the lower bound is below .4.
- Solving for what would pass: the lower bound clears .4 at n = 80 only for a true kappa of
  about .68 or more; at kappa .6 and prevalence .10 it needs about 180 to 200 items (177 by
  Fleiss-Cohen-Everitt, 199 by the simple formula); at kappa .6 and a doubled prevalence of
  .20 (p_e = .68, p_o = .872) it needs about 100 items, and n = 120 to 150 gives lower bounds
  of .41 to .44. The editor's 120 to 150 range is therefore the figure for a stratified draw,
  not for the draw as designed.

**Remedy adopted: gate on the point estimate, report the interval, keep n = 80.** The
lower-bound gate at the design's prevalence needs about 200 items, beyond the two-day load
the design was built around, and the editor ranks a third coder above extra author-coded
items. The interval is always reported, with the arithmetic above, so a reader can see what
the point estimate can and cannot support. The error rates of this gate at n = 80 and
prevalence .10 are stated rather than hidden: a true kappa of .6 is observed below .4 with
probability about .11, and a true kappa of .4 is observed at .6 or above with probability
about .15.

- Decision rule (revised): binary kappa point estimate at least .6 means the construct is
  treated as reliably codable and the prevalence and score analyses proceed as
  pre-registered. Point estimate between .4 and .6 means the reliability result is reported
  as a workshop note, not a paper, and the score test is reported as exploratory. Below .4
  means stop: the finding is that integration as defined is not reliably codable in this
  material, recorded in the study repository's report and in `DECISIONS.md`.
- Adjudication: after both first passes are locked, disagreements are adjudicated by the
  two coders together into a separate file; adjudicated labels are used only for
  prevalence and the score test, never for the reliability estimate.

### Within-conversation score (frozen before any label exists)

**Why the cross-participant graph was the wrong construct.** The first draft scored each
sampled conversation by its betweenness and neighbour-community count in a similarity graph
over the 1,191 development conversations, which come from about 1,500 different participants.
A bridge in the programme was a conversation linking two of *one person's* knowledge
communities; a high-betweenness node in a graph over many participants is a conversation
that resembles other people's conversations on several topics, which is a different thing.
The programme's own catalogue says so ("a global public-corpus graph is not a personal
archive network"). A null AUC from that score would have said nothing about the bridge
taxonomy and a positive one would have been uninterpretable. The replacement is computed
from the sampled conversation's own messages and nothing else, so it can only measure
whether *this* conversation relates distinct ideas, which is the construct the coders label.

**Notation.** For a sampled conversation let x_1, ..., x_n be the cached message vectors of
its chosen path in encoding order (`{id}__messages`; n is 6, 8, 10 or 12), with roles from
the json `records`; the path starts with a user message and alternates, so user messages sit
at odd positions 1, 3, 5, ... and assistant messages at even positions. Let **c** be the
centre: the arithmetic mean of every user-message vector in the 722 FIT-split conversations,
computed once per encoder and recorded by SHA-256 in `FREEZE.json`. Write
cos_c(v, w) for the cosine of v - c and w - c (the centred cosine of wave 1, with the
user-target centre, applied to every vector in the score regardless of role, as wave 1 applied
it to pooled histories of mixed role when the target was a user turn).

**Eligible target positions.** User positions t with t >= 5, that is t in {5, 7, 9, 11} and
t <= n: the wave-1 forecast positions restricted to user targets. Each has at least two
earlier user messages (positions 1 and 3) and two earlier assistant messages (2 and 4). A
6-message path has one eligible position, a 12-message path four.

**Score 1, the bridging-turn score B (primary).** This is the editor's "bridging message"
score made precise without a clustering step. For eligible t let U_t = {i < t : i odd} be
the earlier user positions. The *poles* are the two earlier user messages least alike,

    (i*, j*) = argmin over i < j in U_t of cos_c(x_i, x_j),    d_t = cos_c(x_{i*}, x_{j*}),

ties broken by the smallest (i, j). The bridging margin of turn t is

    b_t = min( cos_c(x_t, x_{i*}), cos_c(x_t, x_{j*}) ) - d_t,

how much closer the turn lies to *both* poles than the poles lie to each other. The
conversation score is

    B = max over eligible t of b_t.

A turn that continues one pole only, switches to a third idea, or sits in a one-idea
conversation (d_t already high) scores near or below zero; a turn that draws on two ideas
that were distinct scores positive. The position attaining the maximum is recorded with the
score for a descriptive comparison with the coders' evidence spans; it is not tested.

**Score 2, the reach-back margin R (secondary).** For eligible t let the pooled older
material be h_t = mean(x_1, ..., x_{t-3}), everything before the latest exchange, of either
role, and the latest exchange e_t = mean(x_{t-2}, x_{t-1}), the user's previous turn and the
assistant's reply. The reach-back margin of turn t is

    r_t = cos_c(x_t, h_t) - cos_c(x_t, e_t),

and the conversation score is the mean over eligible positions,

    R = (1 / |T|) sum over t in T of r_t,    T = {5, 7, 9, 11} intersected with [1, n].

R measures how much the user's continuation draws on material older than the latest exchange.
It differs from the wave-1 contrast (full pool versus last two) in that the pool excludes the
latest exchange, so the margin isolates older material; it is construct-matched to reach-back
rather than to integration, which is why it is secondary here and becomes primary only under
branch 2 of the sequencing section.

**Computation and freeze.**
- Both scores are computed in MiniLM and E5 for the 80 sampled conversations only, from the
  cached arrays and the centre, and written to `results/scores-{encoder}.csv` (case id, B,
  the position attaining B, R, number of eligible positions). The SHA-256 of each file, of
  the two centre vectors and of the score module is committed in `FREEZE.json` before the
  packet is opened. No conversation text is read by code at any point; the sample is drawn
  on metadata and hashes, and the scores come from vectors that already exist.
- Label-blind description, computed and committed at the same time on all 1,191 development
  conversations (not only the 80): the distribution of B and R, their correlation with path
  length and with user token mass, and their agreement between encoders. This is reported as
  context; nothing in it changes the test.
- Synthetic unit tests of the score code, fixed before the freeze: a one-idea path gives
  B <= 0 and R near 0; a path that switches A, B, C gives B near 0; a path with orthogonal
  ideas A and B and a later user turn at their normalised sum gives B = cos 45 degrees =
  .707 and the attaining position at that turn; the wave-1 P07 synthetic episodes (abrupt
  switch, linear mixture, return switching, constant topic) are run through both scores and
  the resulting values committed as expected outputs.

**Test.** Area under the ROC curve of each score for `integration` versus the rest on the
adjudicated labels, with the conversation bootstrap; a pre-registered practical floor of
AUC .65. Two scores times two encoders is four tests; they are reported together with no
correction and the primary is B in MiniLM (R in MiniLM under branch 2). Under the stop
outcome of the kappa gate, and under the .4 to .6 outcome, the test is reported as
exploratory with the adjudicated labels, labelled as such.

### Output and venue

What is written depends on the outcome, and the length is fixed here so that it is not
argued afterwards:

| Kappa point estimate | Primary AUC | Output |
|---|---|---|
| at least .6 | at least .65 | A short paper (8 pages, ISCS format, or a NetSci-style abstract first) titled along the lines of "Can two coders agree that a conversation integrates ideas?" reporting reliability, prevalence, the third-coder check and the score test. |
| at least .6 | below .65 | **A workshop note, not a paper.** Reliability passes and the construct does not separate: the content is a reliable label, a prevalence estimate and a null on one frozen score, which is worth recording and not worth a paper. |
| between .4 and .6 | any | **A workshop note, not a paper.** A kappa of .5 on eighty items supports no claim either way; the note records the estimate, the interval, the confusion matrix and the third-coder check, and the score test as exploratory. |
| below .4 | any | Stop. A report in the study repository and an entry in `DECISIONS.md`; at most a paragraph in a later paper's limitations. |

None of this goes into the ISCS 2027 long paper, which is full at twelve pages. Target for
the paper outcome: decided by the owner once the kappa is known (mid-January). Candidates:
ISCS 2027 as a second, short submission; Complex Networks 2027; a NetSci 2027 abstract is
already taken by the history paper.

### Timeline

- By October 31: the chosen-reply test run and reported (assistant, about one day); the
  owner's branch decision entered in `DECISIONS.md` by November 2.
- By November 15 (branches 1 and 2): rubric v2, schema v2, packet builder, agreement script,
  score code with synthetic tests, scores with committed hashes, the 30-item sub-draw, third
  coder named, protocol frozen and pushed. Assistant work, about two days.
- November 15 to December 15: practice set (half a day each), then the 80-case first pass
  (one to two days each), locked with timestamps.
- Between the freeze and January 5: the third coder's 30-item pass (about one day), locked
  before adjudication begins.
- By January 10: agreement, third-coder check, adjudication, prevalence, score test, report.
- Mid-January: owner decision on venue or note.

### Repository

A new child repository `papers/integration-coding/` (`cmri-integration-coding`, private),
following the pilot layout: `PROTOCOL.md`, `FREEZE.json`, `src/`, `tests/`, `results/`,
with coder packets (including the third coder's) and the mapping under an ignored `data/`.
Labels, including the third coder's, are committed only as JSON keyed by case id with
evidence offsets, never with conversation text (model text is CC BY-NC; human text CC BY).
The chosen-reply test lives in the wave-1 repository, next to the cache it reads.

## What is left to the builder

- The chosen-reply script: positions, centre c_A, margins, contrast and the wave-1
  aggregation and bootstrap, with a test on synthetic vectors where the chosen reply is
  constructed to be closer to the pooled history than the rejected mean is.
- The exact salted-hash selection and its exclusion list (derived from the pilot mappings,
  which the builder may read as ids only), and the second hash for the 30-item subset.
- The score module implementing B and R exactly as written, the two centre vectors, and the
  synthetic tests listed above.
- The agreement script (kappa, PABAK, AC1, Fleiss' kappa for the three-coder subset,
  bootstrap, confusion matrix, the per-case three-label listing) with tests on synthetic
  label pairs and triples of known kappa.
- A practice-set answer key is *not* produced by the assistant; the two coders produce it
  together after their practice passes.

## Questions only the owner can answer

1. Is John Matta available for about two and a half days of coding between mid-November
   and mid-December, and does he agree to be a coder rather than an adjudicator only?
2. Sample size: 80 (two days each) or 100 (two and a half days each)? With the gate now on
   the point estimate, 100 narrows the interval from about plus or minus .31 to .28 at
   kappa .6 and prevalence .10; it does not change what the gate can decide.
3. Unit: whole-exchange summary label only (my recommendation), or also local units?
4. Does the owner accept that the fresh sample must exclude the 42 AI-exposed components,
   and that both coders' prior reading of the pilot reports is disclosed as a limitation
   rather than treated as disqualifying?
5. The first draft asked whether the network half should be built before the kappa is
   known. The editor's answer, adopted here, is that the within-conversation score costs
   about an hour and should be built and frozen first, and the cross-participant graph not
   at all. Does the owner agree?
6. Who is the third coder (a non-author who has not read the pilot reports and can give
   about one day between the freeze and January 5), and are they acknowledged or named on
   the note or paper?
7. Does the owner confirm that the chosen-reply test may run now on the development
   partition (cached vectors only, no text read), labelled exploratory, before any F3
   material is built?
