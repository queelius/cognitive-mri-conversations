# ISCS 2027 paper revision: how far back the next message reaches

Date: 2026-10-19. Status: design, written after the editorial review
(`docs/research/audits/2026-10-19-editorial-review/REVIEW.md`) and the owner's decisions of the
same day. Supersedes the framing in `2026-10-18-iscs-2027-paper-design.md` and the long-paper
plan of 2026-10-19 where they conflict; the data rules in both remain in force.

## Owner decisions this design implements (2026-10-19)

1. Take the most interesting and salient approach to the metric finding: centered cosine
   becomes the reported primary, the raw rule is stated as the gate it was, and the pool-size
   artifact is itself reported, with the donor null that exposes it.
2. Retitle and reframe around the two-user-turn result; apply the editor's ten edits.
3. Condense Section 5 (observation nulls) to one properly cited paragraph.
4. The frozen protocol is not a constraint on what the paper is about. The research and the
   paper may change radically at this stage. Two hygiene rules stay because they are cheap to
   keep and expensive to lose: the PRISM reserve and the WildChat confirmation sample are not
   re-drawn or re-scored for exploration (only their committed result files are read), and
   every new analysis is labelled development or held-out.
5. F3 proceeds as the editor recommended (within-conversation score, point-estimate kappa
   gate, third coder), after the PRISM chosen-reply test; the spec was revised the same day.
6. New experiments and new data sources are allowed.

## Thesis

The paper answers one question, measured so that the metric cannot manufacture the answer:
how far back does what a person says next to an AI assistant reach?

The answer, on development data from two collections and two encoders, is:

- A person's next message is anchored by their own recent turns. Adding the person's previous
  turn to the latest exchange raises centered cosine by about .03; adding the assistant's
  previous reply instead raises it by .01 or nothing. Once two of the person's turns are
  pooled, pooling the whole history adds nothing and slightly hurts, because older assistant
  replies dilute the pool.
- The marginal value of the k-th most recent user turn decays roughly geometrically (about
  .03, .012, .005 for k = 2, 3, 4), while its relation to the next message decays slowly (a
  single turn four exchanges back is still .24 cosine closer than a stranger's turn). History
  is relevant but redundant: older turns say what recent turns already say.
- Recency is real: the latest exchange beats a random earlier pair by about .02.
- Who the person is matters differently by collection. On WildChat, the person's messages
  from other sessions substitute for their previous turn in the current session (adding either
  gives about .03); on PRISM they do not (.04 less than the previous turn). A WildChat IP's
  other sessions are .17 cosine closer to the next message than a stranger's; a PRISM
  participant's other conversations only .03 to .04 closer.
- The gain is homogeneous across components as far as the data can tell: the spread of
  per-component gains and the share of non-positive components match a common-gain
  permutation null in both collections.
- The gain does not predict choice: on 3,396 PRISM later-turn choices, every history
  predictor picks the chosen reply over the rejected one at 50 to 52 percent, and pooling
  more history adds at most .002 of margin.
- Raw cosine rewards pool size regardless of content, by the size of the frozen headline;
  centered cosine does not. The frozen contrast passes under both, and the paper reports the
  centered version as primary.

Every statement above is development evidence. The frozen two-corpus confirmation covers only
"mean of all earlier messages minus mean of the last two, for the next user message, lower
bound above zero in both encoders" (raw, and its centered secondary). The paper says this
plainly and confirms the new headline on fresh held-out samples before submission (Protocol 2
below).

## Evidence ledger

| Claim | Status | Source |
|---|---|---|
| Mean minus last two > 0, raw and centered, both collections, both encoders | Held-out, frozen, once | `papers/iscs-2027/REPORT.md`, `papers/wildchat-replication/REPORT.md` |
| Gain positive at every target position | Held-out, from committed per-case scores | review `referee-methods.md` B1 |
| Raw cosine rewards pool size (donor null); centered does not | Development | review `pooling-diagnostics.md` |
| Previous user turn > previous assistant turn; three-message recipe matches the mean | Development | `papers/iscs-2027/results/development/reach-{prism,wildchat}.md` |
| Reach curve and lag curves | Development | the same files, and `papers/history-reach/results/reach-long.md` |
| Long-conversation extension (20 to 40 messages) | Development, fresh WildChat sample | `papers/history-reach/results/reach-long.md` |
| State versus signature (own other sessions versus topic-matched and random strangers) | Development | `papers/iscs-2027/results/development/reach-{prism,wildchat}.md` |
| Heterogeneity indistinguishable from common gain | Development | the same files |
| Chosen reply not predicted by history | Development | `papers/iscs-2027/results/development/choice-prism.md` |
| Forecast-origin control | Held-out secondaries | both REPORT.md files |
| Revelation null exponent below the uniform band | Author archive, calibration pilot | `temporal-networks-calibration` diagnostics |
| Protocol 2 (below) | Planned held-out, fresh samples | this design |

## Measurement decisions

- Reported primary: centered cosine, center = mean of the FIT-split user messages of the same
  collection, applied to predictor and target. The raw rule is reported as the frozen gate it
  was, with the donor null beside it.
- The success-rule change (lower bound above .01 to above zero, commit 98f5ff1, 24 minutes
  before the freeze) is disclosed in one sentence; both confirmations pass the stricter rule.
- "Frozen protocol" throughout; "pre-registered" only if a registry deposit exists at
  submission.
- Message vectors are unit norm in both caches, so "add one message to the pool" means "add
  one unit vector"; recipes that add a pooled mean normalise it first.
- All intervals: 95 percent component bootstrap, 2,000 draws, seed 104729, targets averaged
  within conversation and conversations within component.

## New experiments

Done on development caches (scripts `papers/iscs-2027/scripts/{reach,choice}_diagnostics.py`,
results in `papers/iscs-2027/results/development/`): role weighting, reach curve with
donor-filled control, lag curves, echo-free reply, state versus signature with a topic-matched
stranger, heterogeneity permutation test, chosen-reply test with a length baseline.

Done, the long-conversation sample (`papers/history-reach/`, 557 conversations of 20 to 40
messages from 384 fresh WildChat IPs, pinned instruments, targets at every even position from
4 to 38, `results/reach-long.md`): the marginal value of the k-th user turn is zero from the
eighth turn back; the single-turn curve decays exponentially (halving every 1.7 turns) to a
floor of .23 (user) and .19 (assistant) above a stranger's turn; pooling all of the person's
turns beats the three-message recipe there by +.015 and +.021, so the reach is about six of
the person's turns with most of it in the first three. The paper's Sect. 4.2 reports it.

Planned, Protocol 2 (held-out confirmation of the new headline). Fixed before any sample is
drawn:

- Samples: PRISM, up to 250 components drawn by salted rank from the 632 components never
  selected by wave 1 (1,240 total, 360 development, 248 reserve); WildChat, 500 IPs drawn by a
  new salt from eligible IPs excluding the development, confirmation and reach samples.
- Contrasts, centered cosine, next user message at positions 4, 6, 8, 10, both encoders:
  (C1) last two plus previous user turn minus last two, lower bound above zero;
  (C2) mean of all minus last two plus previous user turn, upper bound below +.01;
  (C3) last two plus previous user turn minus last three (previous assistant turn in the slot),
  lower bound above zero.
  Pass: all three in both encoders in both collections. Descriptive: reach increments to k = 4,
  lag curves, donor line.
- Frozen by a committed protocol file before the draw; run once; written up whatever the
  result. The owner decides when to commit the freeze; until then Protocol 2 is a plan.

Optional third collection, if time allows: a public multi-turn human-AI collection with no
user identifiers (component = conversation), for example LMSYS-Chat-1M or ShareGPT, for C1 to
C3 only. Not required for submission.

## Paper structure (twelve pages, svproc)

Title (primary candidate): *How Far Back Does a Person's Next Message Reach? Recency, Role
and Redundancy in Human--AI Conversation Logs*. Alternative, more declarative: *A Person's
Next Message to an AI Assistant Reaches Two of Their Own Turns Back*. The owner chooses.

1. Introduction (field-level opening per the editor's edit 1; the programme in one clause;
   the question; what is confirmed and what is development; contributions in four lines).
2. Related work: history in dialogue prediction (keep, trimmed); embedding geometry and
   anisotropy (new); human-AI log analyses (new); context reach and memory length (new, from
   `prior-art-reach.md`); person signature and personalisation (new); observation nulls and
   pre-registration (rewritten with Pedarsani 2008, Leskovec and Faloutsos 2006, van
   Miltenburg 2021, Søgaard 2023).
3. Data, measurement and protocol: collections and components (keep); visibility and
   metric (centered primary, raw gate, donor null); one "Protocol deviations" paragraph of
   about eight lines including the rule change.
4. Results.
   4.1 The frozen contrast replicates (short; Table 2 trimmed; Fig. 1 upper only).
   4.2 What the gain is: role (previous user turn versus previous assistant turn), the
   three-message recipe, position-stratified, the reach curve with the donor line (new Fig. 1
   lower), the long-conversation extension (if ready).
   4.3 Relevant but redundant: lag curves for user and assistant turns (new figure or panel).
   4.4 State and signature: own other sessions versus stranger, by collection.
   4.5 Homogeneity: per-component distribution against the common-gain null (replaces the
   heterogeneity section and Fig. 3; subgroups to one sentence).
   4.6 The forecast-origin control (condensed to one paragraph and Fig. 2 upper).
   4.7 What the gain does not do: choice (new) and ranking (four sentences).
5. Observation processes that resemble dynamics: one paragraph, cited (revelation null as
   Pedarsani 2008 with vertices in place of edges, the identity from exchangeability, the
   archive exponent below the null band on one denominator), packaging in two sentences,
   synthetic processes in four lines (Table 3 kept if space allows).
6. Discussion and limitations: what the gain is and is not; instruments; components are not
   people; development versus held-out; what would change the conclusion.
7. Conclusion. Reproducibility: public at submission, deposit DOI, per-component CSVs.

Figures: Fig. 1 (primary contrast in four samples; reach curve with donor line, both
collections); Fig. 2 (lag curves, user and assistant, both collections, both encoders); Fig. 3
(role under two origins; per-component ECDF against the null band). Tables: samples;
held-out estimates (trimmed); synthetic (if space).

Cuts, per the editor's Section 6: freeze procedure (to repositories), packaging detail,
subgroup forest plot, survey join, attenuation rows, ranking detail.

## F3 branch decision

The chosen-reply test result (history does not predict choice; margins .002 to .008, accuracy
50 to 52 percent, pooling adds at most .002) means embedding-space behavioural outcomes on
PRISM are weak. This raises, not lowers, the value of a human-coded integration construct. F3
proceeds with the within-conversation scores B and R as revised; the result is recorded in
`docs/research/DECISIONS.md`.

## NetSci 2027 abstract

Not decided here. With the reframed paper the abstract has two honest options: lead with the
reach and redundancy result as a statement about the memory of a conversational process (no
network content, likely a poster), or lead with the revelation null applied to conversation
networks with the archive's exponent below the band (network content, needs one figure).
Due 2026-12-01.

## Out of scope

The multi-archive network replication (the editor's alternative 1) is the next paper, not
this one. No change to the WildChat or PRISM frozen repositories beyond reading their
committed results.
