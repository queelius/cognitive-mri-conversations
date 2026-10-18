# Full-paper lead: what earlier conversation predicts beyond topic

October 18, 2026. Alexander Richard Towell.

Current status: **development wave executed; no full-paper promotion**.
[HS01–HS04 results](../../papers/history-specificity/REPORT.md) fail the stronger
ranking and calibration gates while preserving positive cosine. Stop scaling
this operationalization; narrow the broader question toward independently
grounded dependencies. [Next proposed goal](NEXT-GOAL.md). All 248 reserve
components remain closed; privacy/network probe is deferred for valid relation
controls. Final review corrected prefix calibration and anchor sensitivity
denominators without changing the primary decision; [correction audit](../../papers/history-specificity/audits/correction-impact.json).
The original rationale below remains the proposal history.

Original status: **user-endorsed research priority; proposed experiment wave**.
The user confirms that a full paper is primary, supports continued
work on multiple questions and experiments, and regards the history-specificity
direction as a promising lead. This does not establish a positive finding for
the new question. An extended abstract is optional only alongside a separate
full-paper submission. Which venue hosts each submission remains undecided.

## Scientific question and starting evidence

**What information in earlier turns helps predict the next user contribution
after accounting for topic, recent context and stable participant/style cues?**

The [completed history study](../../papers/history-falsification/REPORT.md)
supplies a real starting point. Uniform history beats last-two-message pooling
on next-user target cosine in both fixed encoders and all four prescribed input
views. The native gains are .03440 and .01919. Initial-prompt and role-mixture
checks do not remove that descriptive benefit. It remains exposed development
evidence from one cohort, not independent confirmation.

The same report supplies the main competing explanation: a fixed-topic process
with conditionally independent user innovations can produce a history advantage.
Its weak compressed learned readouts do not establish recent-context sufficiency.
Changing the assistant's information boundary reverses or attenuates the role
ordering, so a human-versus-AI memory interpretation remains unsupported.

The new question is about conditional predictive information in the recorded
interaction. Even a successful result would not, by itself, demonstrate human
learning, causal assistant influence or a physical memory mechanism.
Here information means a specified predictive increment; a retrieval score is
not an estimate of conditional mutual information without additional justification.

## Proposed experiment families

These identifiers distinguish questions within this follow-up. They do not
replace P01–P12 or turn encoder/seed variants into independent experiments.
The implementation protocol must select the exact primary comparison, metrics,
practical thresholds and resource limits before computing new outcomes.

| ID | Question | Discriminating comparison | What a result could establish |
|---|---|---|---|
| HS01 | Is older information specific to the interaction? | Genuine older turns versus another conversation's older turns matched using available prefix information; strong initial/recent content baselines. | Conditional predictive value beyond the measured matching variables, subject to residual confounding and matching support. |
| HS02 | Does order add anything beyond the same content? | Keep the initial anchor and recent suffix fixed; reorder only eligible older material with declared role-preserving controls. Compare order-sensitive models with content pooling. | An order increment for a defined predictor and task; a null cannot rule out all sequence dependence. |
| HS03 | Where in the observed history is useful information? | Fixed, nested history budgets and past-only lag interventions, with message/token exposure accounted for. | A predictive horizon under the chosen representation and outcome. This is not an intrinsic semantic half-life or wall-clock memory measure. |
| HS04 | What do earlier user and assistant turns contribute? | Same next-user target/origin, fixed recent context, and matched history budgets or an explicit budget-response curve. Separate amount of text from role. | Predictive attribution to input subsets, not causal influence of one speaker on another. |

HS01 is the primary lead. HS02–HS04 are secondary questions whose null results
remain informative; they must not become substitute primary tests after HS01
fails. A material failure of the matching/outcome construction is a feasibility
result, not evidence that conversation-specific information is absent.

## Outcome and control design to settle before execution

Use an outcome tied to the recorded next user contribution, such as ranking it
among plausible alternatives. Keep the old raw-cosine contrast as a linked
reference, not the only new outcome. The actual next contribution is an observed
continuation, not necessarily the only valid response or a correct answer.
Report ambiguity and duplicate/near-duplicate alternatives explicitly.

Matching and model inputs must use information available before the target.
Topic models, transformations, donors and candidate-bank rules are fit/development
objects. Do not use the target to select an especially favorable historical donor,
choose a lag, or define a subgroup. A hard candidate construction should match
on pre-target context where feasible; any outcome-conditioned construction needs
an explicit different estimand and cannot be silently described as prospective.

Compare exact lexical/content baselines and full-dimensional initial/recent
representations with whole-history pooling and a small fixed set of sequence
methods. Do not repeat an inadequate recent baseline and interpret its failure
as proof of older-history value. Keep the already tested encoders fixed initially;
different encoders on the same cases are measurement checks, not replications.

Donor swapping changes discourse coherence as well as content. Report overlap,
matching balance, topic granularity, lengths, participant/style cues and the
availability of suitable donors. Within-participant donor checks may be useful
only if enough legitimate independent conversations exist; identifiers remain
linkage proxies. Compare conclusions under more than one prespecified reasonable
matching construction. A failed balance/support check cannot justify accepting
a weaker control after seeing the result.

Synthetic controls should include: fixed topic with independent innovations;
stable style/role offsets; sparse older information with a known predictive
effect; and an order-dependent positive control. Split scenario families, preserve
the earlier template-shortcut counterexamples, and check both false positives
and detection power. These controls calibrate instruments, not natural behavior.

Use the existing exposed development components for design. A deterministic
internal split can separate new fitting from checking, but cannot make previously
inspected data untouched again. Keep all 248 reserved PRISM components closed in
the next development goal. Any donor reuse or shared candidate bank induces
dependence: define the resampling unit for that construction, or state uncertainty
conditional on a fixed bank. Do not count target pairs as independent people.

## Broader portfolio remains open

Maintain a smaller independent line on **de-identification and the stability of
network conclusions**. It builds directly on the published static-network and
accepted hierarchy work. Its potential contribution is whether centrality,
bridge or community conclusions change materially, and whether a useful remedy
exists. Changes in embedding similarity alone are insufficient. The existing
[catalog](../../future-ideas/iscs-2027/research-deep-dive/PAPER-CATALOG-AND-OPTIONS.md)
already lists this candidate; it has not been executed here.

Semantic/embedding dynamics remain active through the order, horizon and role
questions, alongside the [consolidation assessment](DYNAMICS-CONSOLIDATION.md).
Richer state or physics claims require distinctive predictions and competent
simple baselines. Keep P01/P02 integration, artifact transformation and long-gap
reuse available under their existing annotation/access gates. The completed
revision/later-use no-go on current PRISM material remains unchanged.

Practical allocation: prioritize HS01 and the comparisons that share its valid
infrastructure, while allowing bounded independent probes that could alter the
ranking. Do not require every exploratory result to fit the eventual full paper.
Record every attempted comparison, including abandoned and negative results.

## Prior-work and publication boundaries

History-dependent dialogue prediction is already occupied by
[CCL](https://aclanthology.org/2023.findings-acl.319/),
[OnePred v2](https://arxiv.org/abs/2605.23668v2), and user-language-modeling work
audited in [the source study](../../papers/history-falsification/audits/literature.md).
The potential addition is a specific, adequately controlled empirical account
of interaction-specific information, its order and its predictive horizon.
A larger dataset or a more favorable encoder alone does not establish that addition.

For the independent privacy line, [name bias in embeddings](https://aclanthology.org/2025.findings-acl.914/)
and [structure-preserving de-identification](https://aclanthology.org/2026.findings-acl.963/)
are direct novelty constraints. They were identified or checked in the preceding
October 18 discussion; the name-bias abstract was read, while Anonpsy was located
through its primary-source search record. Full methods comparisons remain part
of the proposed audit. This document performs no new systematic literature review.

A full-paper recommendation requires a coherent contribution, adequate controls,
meaningful uncertainty, and a credible independent confirmation design. A precise
negative result can qualify only if it changes a substantive inference beyond
routine baseline comparisons. A development failure can justify another question;
it does not justify reclassifying exposed data as confirmation.

The optional abstract should present a distinct contribution from the earlier
CMRI work. If the full paper develops the history results, remove that empirical
panel from the companion abstract. Multiple-submission/presentation rules and
the specific 2027 publication/reuse terms remain to verify before submission.
The finished abstract package is preserved unchanged as a historical artifact.

## Next action

[The proposed next goal](NEXT-GOAL.md) calls for an executed, bounded development
wave centered on HS01 with several connected questions and a limited independent
privacy probe if its feasibility/novelty gates permit it. It does not authorize
confirmation, a manuscript, external contact or submission by its existence.
