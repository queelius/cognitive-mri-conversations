# F3: two-coder integration study on PRISM conversations (design)

Date: 2026-10-19. Owner: Alexander Richard Towell. Status: design for owner review; nothing
built, no sample drawn, no coding started. Follows F3 of
[the ISCS 2027 paper design](2026-10-18-iscs-2027-paper-design.md) and the materials
inventory of 2026-10-19 (scratchpad; summarised in "What exists" below).

## Intent

The October pilots established a replicated, pre-registered *geometric* fact (the history
gain) and showed that the measurement does not identify what people do with earlier turns.
Every reviewer line of attack on the programme's behavioural vocabulary ("integration",
"bridge", "revision") comes back to the same gap: no human has ever labelled any of it.
F3 closes that gap for one construct, the P01 distinction between a conversation that
*integrates* distinct ideas and one that merely *switches* between or *lists* them, with the
two authors as independent coders.

The question is deliberately modest: **can two trained humans agree on whether a human--AI
conversation integrates distinct ideas, and if so, how common is integration, and does a
frozen network score computed before anyone saw a label track it?** A reliability failure
is a publishable result; the study is designed so that each outcome is written up.

## Goals, non-goals, anti-goals

Goals
- A pre-registered reliability estimate (Cohen's kappa with a component-clustered bootstrap
  interval) for the P01 summary label on a fresh sample of PRISM conversations.
- A prevalence estimate of integration, switching and listing among multi-topic human--AI
  conversations, with the ambiguous rate reported, not hidden.
- One pre-registered test of whether a frozen, label-blind network score (betweenness and
  neighbour-community count of the conversation in a PRISM similarity graph) is higher for
  integrating conversations than for the rest.
- Materials and code that a third coder could reuse: rubric v2, practice set, packet builder,
  agreement script.

Non-goals
- No claim about memory, learning or causal influence; integration is a property of the
  visible text.
- No reuse of the spent PRISM reserve or either WildChat sample.
- No AI-drafted labels in the reliability sample, as primary or as "adjudication aid".
- No attempt to validate the three published bridge *types* (evolutionary, integrative,
  pure); the inventory found no executable definition of them. The test is on continuous
  scores.

Anti-goals (what would make it a failure even if it "works")
- Reporting agreement after discussion as reliability.
- Tuning the rubric, the sample or the network score after seeing labels.
- Letting the coders see the pilot reports' per-case commentary, the AI drafts or the
  review guide that names cases.

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
- No per-conversation network score exists for PRISM. Message-level MiniLM and E5 encodings
  for the 1,191 development conversations are cached in `papers/portfolio-wave1/cache/`.
  The published pipeline's betweenness and community code runs on a conversation graph and
  needs conversation-level vectors, a threshold and Louvain.
- The 248-component reserve is spent and excluded.

## Design

### Sample

A fresh draw of **80 conversations, one per dependence component**, from the 722-conversation
PRISM fit split, excluding every component touched by the 42 AI-exposed cases (and the one
sibling component), by a salted hash recorded before the draw (same selection domain
mechanism as `materials.py`). Eligibility: the chosen path has at least six messages, so that
"distinct ideas" can occur. Eighty rather than 100 keeps each coder's load near two days;
the component-clustered interval on a binary kappa at n=80 is wide (roughly plus or minus .2
at kappa .6) and the study is written up as a reliability estimate with that width, not as a
precise one. *My call; the owner may prefer 100.*

The coder packet is the chosen path only, capped at twelve messages, exactly as the
embeddings saw it, so labels and scores describe the same text. The custodian mapping
(case id to conversation id and component) stays in the ignored data directory.

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
added to the P01 label set (schema v2, committed to the study repository).

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

### Reliability statistic and decision rule (frozen before coding)

- Primary: Cohen's kappa on the **binary** collapse `integration` versus everything else,
  with a 2,000-draw bootstrap over components (one conversation per component, so
  conversation-level resampling).
- Secondary: five-category kappa; prevalence-adjusted agreement (PABAK and Gwet's AC1)
  because the base rate is expected to be skewed; the full confusion matrix.
- Decision rule: binary kappa lower bound above .4 and point estimate at least .6 means the
  construct is reliably codable and the prevalence and network analyses proceed as
  pre-registered. Point estimate between .4 and .6 means the reliability result is the paper
  and the network test is reported as exploratory. Below .4 means stop: the finding is
  that integration as defined is not reliably codable in this material, written up as such.
- Adjudication: after both first passes are locked, disagreements are adjudicated by the
  two coders together into a separate file; adjudicated labels are used only for
  prevalence and the network test, never for the reliability estimate.

### Network score (frozen before any label exists)

- Graph: the 1,191 development conversations, each represented by the role-weighted mean
  (user 2 : assistant 1, as in the published pipeline) of its cached MiniLM message vectors,
  L2-normalised; cosine edges above a threshold chosen on the graph alone by the same
  phase-transition criterion the published paper used (the threshold at which the giant
  component stops absorbing most nodes), recorded with the graph hash. E5 as the second
  instrument.
- Scores per conversation: betweenness centrality in the giant component and the number of
  distinct Louvain communities among its neighbours (the two quantities the bridge
  taxonomy was built from). Scores for the 80 sampled conversations are written to a file
  whose hash is committed before the packet is opened.
- Test: area under the ROC curve of each score for `integration` versus the rest on the
  adjudicated labels, with the component bootstrap; a pre-registered practical floor of
  AUC .65. Two scores times two encoders is four tests; they are reported together with
  no correction and the primary is betweenness in MiniLM.
- The sample is drawn on metadata and hashes only; the graph is built from cached vectors
  that already exist; no PRISM text is read by code before the packet is built.

### Output and venue

- A short paper (8 pages, ISCS format, or a NetSci-style abstract first) titled along the
  lines of "Can two coders agree that a conversation integrates ideas?" reporting
  reliability, prevalence and the network test, whichever way each came out. It does not
  go into the ISCS 2027 long paper, which is full at twelve pages.
- Target: decided by the owner once the kappa is known (mid-January). Candidates: ISCS 2027
  as a second, short submission; Complex Networks 2027; a NetSci 2027 abstract is already
  taken by the history paper.

### Timeline

- By November 15: rubric v2, schema v2, packet builder, agreement script, network scores
  with committed hashes, protocol frozen and pushed. Assistant work, about two days.
- November 15 to December 15: practice set (half a day each), then the 80-case first pass
  (one to two days each), locked with timestamps.
- By January 10: agreement, adjudication, prevalence, network test, report.
- Mid-January: owner decision on venue.

### Repository

A new child repository `papers/integration-coding/` (`cmri-integration-coding`, private),
following the pilot layout: `PROTOCOL.md`, `FREEZE.json`, `src/`, `tests/`, `results/`,
with coder packets and the mapping under an ignored `data/`. Labels are committed only as
JSON keyed by case id with evidence offsets, never with conversation text (model text is
CC BY-NC; human text CC BY).

## What is left to the builder

- The exact salted-hash selection and its exclusion list (derived from the pilot mappings,
  which the builder may read as ids only).
- The threshold criterion's implementation on the PRISM graph and the handling of
  conversations outside the giant component (score zero, flagged).
- The agreement script (kappa, PABAK, AC1, bootstrap, confusion matrix) with tests on
  synthetic label pairs of known kappa.
- A practice-set answer key is *not* produced by the assistant; the two coders produce it
  together after their practice passes.

## Questions only the owner can answer

1. Is John Matta available for about two and a half days of coding between mid-November
   and mid-December, and does he agree to be a coder rather than an adjudicator only?
2. Sample size: 80 (two days each, wide interval) or 100 (two and a half days each)?
3. Unit: whole-exchange summary label only (my recommendation), or also local units?
4. Does the owner accept that the fresh sample must exclude the 42 AI-exposed components,
   and that both coders' prior reading of the pilot reports is disclosed as a limitation
   rather than treated as disqualifying?
5. Should the network half be built at all before the kappa is known? Building it first is
   what makes it pre-registered; it costs about a day of assistant time that is wasted if
   the reliability gate fails.
