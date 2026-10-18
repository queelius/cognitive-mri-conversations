# Consolidating the semantic and embedding dynamics drafts

October 18 grounded-dependency calibration: [report](../../papers/grounded-dependencies/REPORT.md).
Mean/EMA state did not meet the fixed incremental-relevance criterion beyond
current-content controls. Source labels measure response relevance, not semantic
forces or necessity; no human–AI transfer is established. The newly inspected
[CIG 2026 paper](https://aclanthology.org/2026.acl-long.2203/) is close prior work
on semantic-memory dynamics and informational progress. Any consolidation must
distinguish its contribution and validate its information conditions independently.
No original manuscript was merged or rewritten. Earlier claims and negative
results remain intact; [next proposed assessment](NEXT-GOAL.md).

October 18 executed specificity wave: [results](../../papers/history-specificity/REPORT.md) retain cosine accumulation but show no primary ranking gain over stronger short controls. Small learned order readouts add no benefit; budget curves do not define a half-life; role effects depend on exposure/weighting. Stable-topic/style controls and fixed-bank behavior further limit mechanism claims. Preserve these measurement corrections in the proposed consolidated lineage; no manuscript merger, physical mechanism, full paper or confirmation is established. [Claim ledger](../../papers/history-specificity/CLAIMS.md). Earlier dated assessments below remain history.

October 18 lead selection: the user endorses full-paper research on
[history specificity](HISTORY-SPECIFICITY-LEAD.md) while continuing multiple
questions and experiments. Order, predictive horizon and role contributions
provide concrete connections to these drafts. Their status is proposed
development work, not a new dynamics result or a decision to force every
portfolio finding into one paper. The [next goal](NEXT-GOAL.md) keeps the prior
measurement counterexamples and independent-confirmation boundary intact.

October 18 event-centered follow-up: the [100-window assessment](../../papers/integration-measurement/events/REPORT.md)
does not supply a validated revision-persistence outcome. Stop scaling this
PRISM application route; its recorded ambiguity and source coverage miss the
prospective validation floors. Preserve the distinction between text movement,
changed referents, changed premises and revised commitments in the planned
measurement synthesis. This no-go is source/construct-specific, not a rejection
of all semantic dynamics. No source manuscript was modified or merged.

October 18 update: the [revision/later-use pilot](../../papers/integration-measurement/pilot/REPORT.md)
adds controlled measurement evidence, not a validated dynamics mechanism. Shared
response grammar produces perfect text classification that fails identical-prior
counterexamples; simple sequence outperforms the fixed geometric bundles for
later use. Natural AI drafts leave revision persistence unestimable in the frozen
first-exchange window. Retain these results and the earlier positive history
measurements in the same planned manuscript lineage. No intrinsic persistence,
coupled cognition or semantic-physics claim is promoted. Event-centered independent
validation remains a possible application route; source manuscripts stay intact.

Assessment date: October 17, 2026. **Recommend one planned manuscript lineage, drawing from both drafts, with a revised central claim.** Both contain useful, complementary material. Their shared definitions, corpus, software and overlapping claims make two broad papers difficult to distinguish. In a subsequent October 17 clarification, the user requested investigation of multiple questions and many experiments before choosing that central claim. The [experiment portfolio](EXPERIMENT-PORTFOLIO.md) develops this broader program; A–C below remain research strands and possible later paper centers, not an immediate forced choice.

This records the user's proposed merger as research planning. The original manuscripts and independent repositories remain intact. No combined manuscript has been written and neither source is retired.

**Latest evidence:** the [completed useful-history follow-up](../../papers/history-falsification/REPORT.md)
retains robust development pooling gains but withdraws an intrinsic role-memory
interpretation. No confirmation or full paper is promoted. The dated first-wave
nomination below is historical; the final section records the current assessment.

## Source lineage and the earlier merge plan

The [June 13 merged outline](../../papers/embedding-dynamics/paper/OUTLINE.md) already proposed combining the theory/validation draft with the empirical Semantic Dynamics draft. The [Embedding Dynamics specification](../../papers/embedding-dynamics/paper/SPEC.md) also describes an empirical companion paper, and its status notes retain contradictory “not run” language for an experiment that did run. The earlier plan is evidence of intent, not proof that the merger or validation was completed.

The current [Embedding Dynamics main file](../../papers/embedding-dynamics/paper/main.tex) contains an abstract and one substantial [validation section](../../papers/embedding-dynamics/paper/sections/04_validation.tex); other planned sections are commented out. Semantic Dynamics has a fuller [LaTeX manuscript](../../papers/semantic-dynamics/paper/paper.tex) and [Markdown draft](../../papers/semantic-dynamics/paper/paper.md). These formats differ substantively: LaTeX includes the force/state interpretation, user–assistant divergence, rhythm/role-transition results and continuation-marker analysis that are absent or less developed in Markdown. A merge must consult both formats.

The [October dynamics audit](../../future-ideas/iscs-2027/research-deep-dive/audits/dynamics.md) supersedes the old plan's inference that high prefix cosine validates the complete physics framework. It supplies source-level evidence, counterexamples and corrected uncertainty. The [completed public measurement pilot](../../papers/semantic-measurement/REPORT.md) supplies a separate task-specific result; it is not a general rejection of semantic dynamics.

## Shared and complementary material

| Component | Embedding Dynamics contribution | Semantic Dynamics contribution | Consolidation treatment |
|---|---|---|---|
| State, accumulation and lenses | Explicit input/state reinterpretation; lens composition, decay and proposed dynamical derivation | Same scan/aggregation model; LaTeX already contains force language | One definition section distinguishing observations, internal state and normalized readout |
| Prefix approximation | Actual message-versus-concatenated-prefix experiment; mass, recency and primacy variants | Earlier conceptual and synthetic motivation | Retain as a representation experiment with matched baselines and selection-aware uncertainty |
| Order and null models | Within-role shuffles, motion statistics and failed preregistered predictions | Mostly descriptive trajectory comparisons | Retain controlled evidence and negative findings; remove the false universal turning theorem |
| Reconstruction versus prediction | Different fitted decay values for two objectives | Adaptive next-message alpha and platform comparisons | A question about objective-dependent memory, not one established intrinsic cognitive timescale |
| Trajectory and graph summaries | Proposed invariance ladder and comparison framework | Graphs, communities and pairwise distances under several representations | Use correctly defined summaries; mean derivatives collapse to endpoint/boundary quantities |
| Episodes and transitions | Proposed derivative/smoothing applications | More developed segmentation analyses and continuation-marker proxy | Evaluate with independent labels; distinguish offline segmentation from prospective detection |
| User–assistant development | Role-controlled order tests and source-classification controls | Separate role trajectories and final-state divergence in LaTeX | Lead for studying back-and-forth development, with task, vocabulary, verbosity and tool-output controls |
| Continuation and revisitation | Brief trajectory-comparison applications | End-to-start versus mean-to-mean links and multi-lens neighborhoods | Hypotheses about relation types; similarity alone does not establish actual continuation |
| Software and figures | embflow, prefix and order figures | Empirical scripts and graph/episode/lens figures | One provenance description; reuse valid assets after API and measurement repairs |

The sources contain complementary **experiments and research questions**, but not two established independent theories. The empirical draft contributes more than background; the theory draft contains direct validation absent from a straightforward archive description.

## A coherent model to discuss

Distinguish three objects for text units \(u_k\) and a fixed encoder \(E\):

\[
e_k=E(u_k),\qquad
s_k=\alpha s_{k-1}+m_k e_k,\qquad
x_k=s_k/\|s_k\|.
\]

Here \(s_0=0\), \(m_k\) is a specified unit/token weight, and the normalized readout is defined when the state norm is nonzero. The raw message representation \(e_k\), accumulated state \(s_k\), and readout \(x_k\) support different questions. The independently encoded prefix

\[
p_k=E(u_1\mathbin\Vert\cdots\mathbin\Vert u_k)
\]

is a target representation with its own encoder and context-window limitations. It is not a ground-truth mental state.

A first-order decay process with impulsive inputs can produce this recurrence, with \(\alpha=\exp(-\lambda\Delta t)\) between regularly spaced events. This is exact exponential propagation under those assumptions, not the explicit Euler coefficient. For constant \(0<\alpha<1\), \(\log(0.5)/\log(\alpha)\) measures decay of filter coefficients in steps. An elapsed-time model requires timestamps and corresponding units. Inertia, nonlinear state and coupling models need their own definitions and predictions.

This preserves the useful mathematical intuition: incoming language updates a state, history can matter, and different readouts describe different properties. A physics-inspired model can be scientifically useful through its predictions. Calling an input a force does not by itself distinguish this recurrence from ordinary filtering.

## Evidence and claims to carry forward

| Item | Status for the consolidated project |
|---|---|
| June prefix experiment | Ran. Best-lens cosine is about .897 in 12 selected real conversations, supporting approximation under one observed encoder. |
| Increment over matched pooling | October leave-one-conversation-out gain is +.01766 cosine; selection-aware interval [−.00122, +.03481]. Positive point estimate, uncertain advantage; not an independent replication. |
| Within-archive order effects | Some statistics differ substantially from role-preserving shuffles. Preserve the failed speed-autocorrelation prediction, mixed-role gate failure and negative local-motif result too. |
| Prefix-position retrieval | Original exact-position target was missed. Conversation identification is easier than distinguishing neighboring prefixes. |
| Universal expected turning cosine of −1/2 | False as stated. The audit preserves explicit exchangeable and independent counterexamples; a moment ratio is not the expected normalized cosine. |
| Derivatives and shape | Mean derivatives retain endpoints/boundaries rather than full shape; second difference is not geometric curvature. The audited scalar cosine-DTW comparator degenerates on positive profiles. |
| Different clusters imply independent cognition | Unsupported. Use matched controls and independently defined targets. |
| Platform half-life / cross-model replication | Confounded observational estimates and incomplete model provenance. Recover original inputs/results before using replication language. |
| Episodes, role alignment and continuation | Interesting leads with incomplete construct validation. No universal episode density, learning effect or memory mechanism is established. |

These numerical findings are reused from the linked audit, not rerun for this planning document. [Verification and source hashes](verification-2026-10-17.json) identify the versions inspected.

## Three possible centers for the combined paper

**A. State reconstruction and context composition.** When can incremental weighted message embeddings approximate a context-sensitive prefix representation, at what cost, and where do they fail? Compare token-matched means, last-message and rolling-window representations, fixed/fitted exponential filters, and relevant contextual alternatives. Use difficult within-conversation comparisons and held-out selection. Downstream usefulness would strengthen a contribution based on cosine agreement. This is the most mature experimental starting point.

**B. Development through episodes and interaction.** Can trajectories help distinguish task switching, integration, revision and return, using independently defined events? The empirical draft's role-separated trajectories are a useful lead. Compare direct text, simple adjacent similarity, window segmentation and established dialogue codes. Measure added information rather than interpreting different clusters as different mental processes. This connects most naturally to ISCS, although the completed purpose-retrieval pilot did not establish a sequence benefit for its particular task.

**C. Falsifiable semantic dynamics.** Does a richer state model make predictions that simpler models miss? These are hypotheses for discussion, not existing findings or verified novelty claims:

- **History dependence:** after matching current content, which earlier relations remain necessary to predict a later observable? Order dependence built into a trajectory definition cannot establish this. Use held-out responses, relation/application labels or another separately measured target.
- **Persistence across scales:** does a persistence estimate remain interpretable when the same text is split into messages differently or measured by tokens/time? If it changes with arbitrary packaging, it needs a representation-specific interpretation. Test across tasks and encoders after controlling role composition.
- **Inertia or coupled development:** does lagged state or separate interacting user/assistant states improve held-out prediction beyond fixed EMA, comparable autoregressive/state-space models and complexity-matched alternatives? Improved prediction is distinct from identifying a cognitive mechanism. Shared content and alternating roles can produce apparent coupling.

The third route preserves ambitious physics ideas while specifying what would count as evidence. Further primary-literature screening is required before selecting any as a novelty claim. Existing work on [trajectories, recurrence, filtering and segmentation](../../future-ideas/iscs-2027/research-deep-dive/literature/semantic-timeseries.md) is already close.

## Proposed merged outline and next steps

Use a neutral working title, **Semantic Dynamics of Human–AI Conversations: State, Memory and Change**, until the central contribution is chosen.

1. Define the observable question and what a state representation should predict or reconstruct.
2. Specify raw embeddings, accumulated states, readouts, units and assumptions once; compare established alternatives.
3. Present calibration, counterexamples, provenance and valid order controls.
4. Present one primary validation task, with independent development/evaluation and uncertainty at the appropriate conversation/participant level.
5. Include one complementary empirical application if it answers the central question. Preserve other analyses as supporting or future work.
6. Discuss limits, failed predictions, computational cost and reproducible software/data boundaries.

Before new experiments, reconcile corpus/encoder/order provenance and define protocols for the relevant portfolio questions. Before drafting, select a coherent central contribution from the resulting evidence and map each retained claim to its source and correction. The `embedding-dynamics` tree is a plausible manuscript home because it already contains the merged outline and prefix validation. The destination remains undecided, and no repository move is implied.

A separate empirical companion could become worthwhile if it develops a distinct behavioral question and independent evidence. For now, consolidate the scientific planning and avoid treating the same experiment as two contributions.

## October 17 first-wave update: evidence constrains the proposed centers

The [executed wave](../../papers/portfolio-wave1/REPORT.md) adds public development
evidence and synthetic counterexamples to the historical comparison above. It
does not merge the two source manuscripts or change their Git repositories.

- **Reconstruction:** uniform pooling already matches short direct-prefix
  representations well. The selected exponential/token candidate loses to it
  by .00225 MiniLM and .00098 E5 cosine, with no .01 practical improvement or
  demonstrated matched-cost utility. Do not present approximation alone as new
  physical validation. Three spaced checkpoint discrimination is a different
  task from the earlier failed adjacent-prefix target.
- **Prediction and coupling:** fixed role-history recency features add less
  than .001 raw cosine and lose to the smaller content readout. Synthetic
  ordinary lags detect dependence the fixed EMA sometimes misses; common
  latent topics generate predictive coupling without a causal role effect.
  This supports a narrow readout result, not general independence.
- **Inertia and clocks:** the lag/inertial linear spans coincide and ridge
  coordinates matter. Exact token-clock subdivision holds for constant vectors;
  actual half-span re-encoding changes states and forecasts materially. Keep
  observation units separate from state propagation in the merged definitions.
- **Empirical lead:** uniform history improves next-user-turn forecasts over
  last-two pooling in both encoders. Nominate this fixed P05 comparison, with
  role interaction secondary, for falsification and later independent grouped
  confirmation. It is not an identified semantic half-life.
- **Episodes and development:** synthetic calibration shows gradual-change
  failures, role-offset false alarms and window over-detection. Independent
  natural-language annotations remain absent; 60 blinded development examples
  and rubrics are prepared, not validated labels.

These results narrow center A and C as presently formulated. Center B remains
an interesting construct-validation route with an annotation prerequisite. The
next manuscript thesis remains undecided pending a meaningful confirmed result.
Curved Contrastive Learning and Triple-Encoders are especially close precedents
for physics-inspired sequence geometry and efficient composition; see the
[follow-up novelty audit](../../papers/portfolio-wave1/audits/history-novelty-followup.md).
The [next goal](../../papers/portfolio-wave1/NEXT-GOAL.md) preserves ISCS priority
without making it depend on a broad theory of semantic physics.

## October 17 falsification update: retain measurement, narrow interpretation

The user mean-history advantage survives expanded visibility, restored chunk
weighting and a past-candidate exposure proxy in both instruments; matched
long/short role mixtures retain a benefit too. Initial-prompt pooling does not
explain it away. This is a useful development measurement, not a confirmed
scientific consequence beyond established context-modeling work.

The common-prefix comparison reverses the original role ordering in MiniLM and
brings it near zero in E5. It also changes the assistant horizon; do not convert
this diagnostic into a causal estimate of memory. Synthetic fixed-topic and
older-dependent processes both reproduce the qualitative pattern. The frozen
compressed ridge family fails to improve with longer history, but weak MiniLM
scores and upper-bound penalties prevent a claim that recent context is sufficient.

The packaging controls separate exact fixed-vector clock identities from real
re-encoding and weighting changes. Opposing changes can cancel in an aggregate
score. Keep observations, propagation, target and weighting explicit in any
consolidated definitions. Do not claim intrinsic half-life or validated physics.

One planned manuscript lineage still makes sense; its thesis remains unsettled.
The reserve stays closed because interpretation/contribution gates are unmet,
not because H1 disappeared. Center B's integration/revision measurement is the
next empirical priority if independent coders become available. The
[claim ledger](../../papers/history-falsification/CLAIMS.md),
[verification](../../papers/history-falsification/VERIFICATION.md) and
[current next goal](../../papers/history-falsification/NEXT-GOAL.md) preserve
the positive finding, failed predictions and remaining dependencies.
