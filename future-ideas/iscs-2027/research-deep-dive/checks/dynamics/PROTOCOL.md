# Bounded dynamics audit protocol

Registered 2026-10-15, before running the new checks below. This is a critical audit,
not a replacement manuscript. Source projects and their data remain read-only.
No network embedding calls, private message text, conversation titles, or individual
embedding vectors will be included in outputs. The parent audit handles literature.

## Questions and decision rules

1. **Turning theorem (exact falsification).** Enumerate all six permutations of
   three distinct unit vectors `(1,0), (0,1), (-1,0)`. Compute the expected normalized
   turning cosine and, separately, expected adjacent-velocity dot product divided
   by expected squared velocity norm. The former need not equal the latter.
   A difference from -1/2 larger than 1e-12 falsifies the stated universal lemma.
   A seeded i.i.d. unit-circle simulation (200,000 independent triples) checks the
   analytic value `-4/pi^2`; uncertainty is Monte Carlo standard error, not corpus
   uncertainty. Repeat the exact example after zero-padding to 256 dimensions.

2. **Calculus and invariance (exact identities and counterexamples).** Verify:
   - `Delta s_k = (alpha-1)s_(k-1) + e_k`; the pure-sum tower only holds at alpha=1.
   - normalization changes that tower;
   - mean first differences telescope to endpoint displacement, and mean second
     differences telescope to final-minus-initial velocity;
   - a straight path with changing speed has nonzero second difference but infinite
     three-point curvature radius;
   - velocity Gram matrices are invariant under one rigid transformation of an
     already constructed path; normalized accumulation does not commute with input
     translation;
   - uniform/length-weighted prefix pooling ignores within-prefix order, whereas
     exponential pooling can depend on it. This is an operator fact, not evidence
     that a language model's prefix representation encodes order.
   Use tiny constructed arrays and a fixed RNG. Record discrepancies numerically.

3. **Saved Exp04 reanalysis (no re-embedding).** Load only the existing score arrays
   and scalar metadata in `exp04_prefix.json`. Reproduce the 12-real-conversation
   macro averages, gate, and failed predictions. Compare recency against the
   matched uniform+mass baseline. Use 20,000 conversation-cluster bootstrap draws
   for a descriptive 95% interval of the mean paired gain; prefixes are dependent
   and must not be counted as independent observations. Select the best of all
   recorded lenses on 11 conversations, evaluate on the omitted conversation;
   report leave-one-conversation-out scores, chosen-lens counts, and paired gains.
   This controls lens selection only, not preselection of the archetypes, model,
   corpus, or study design. Also summarize gains by archetype, all per-case alpha
   differences, and correlation of reconstruction/prediction alphas. No causal
   interpretation or population generalization from these 12 selected cases.

4. **Bounded stored-vector prediction check.** If the local vector database and
   dependencies are available, use at most 80 conversations per observed platform
   with at least 40 user/assistant messages. Choose conversations deterministically
   by SHA-256 of the private ID plus the fixed seed (IDs stay local). Use at most
   the first 80 messages, preserve insertion order as in the newer motion script,
   and explicitly record that insertion order is not a proven event chronology.
   Fit alpha on the first half's next-message pairs; evaluate the selected alpha
   on second-half targets using only earlier messages to construct each state.
   Compare to fixed alpha=.85, the running mean (alpha=1), and last-message
   prediction. Record the in-sample oracle advantage as a diagnostic, not a valid
   forecast score. Report paired conversation bootstrap intervals (5,000 draws),
   platform sample sizes, alpha-grid boundary rates, and role proportions. Keep
   this a diagnostic convenience sample, not a platform-effect replication.
   If multiple embedding models exist, select one by the largest metadata row
   count and join using model_id; record model metadata. Query no text/title fields.

5. **Software verification.** Run the existing embflow unit/integration suite with
   coverage, with bytecode/cache/coverage outputs redirected outside source repos.
   Add focused local tests for the checks, including counterexamples and saved-score
   reanalysis shape/inclusion rules. Existing passing numerical tests do not prove
   scientific claims. Record missing coverage and API breakage without repairing
   original code.

## Reproducibility and budget

Fixed seed: 20261015. Record Python/dependency versions, configuration, source SHA-256
hashes, repo revisions/status, and elapsed time. Use a temporary environment if
dependencies are absent. The saved-score and analytic checks should take under one
minute; the stored-vector diagnostic is capped at 240 conversations x 80 messages
and should take under five minutes. Do not expand samples after seeing results.
Failures and deviations stay in the results and audit. Outputs are aggregate JSON
plus this protocol, executable Python, tests, coverage, and audit prose.

## Known limitations before execution

The published prefix result selects alpha on the same 12 chosen ChatGPT
conversations on which agreement is reported; its mismatched-conversation null
mainly tests content identity. No same-content reordered-prefix embeddings are
available in saved scores, so this audit cannot establish prefix order sensitivity.
The original adaptive-alpha objective has causal per-step predictors but selects
the parameter over the whole sequence. Its optimized training score is not
held-out predictive performance. A library fixture copied from the original code
checks consistency, not the validity of the original scientific assumptions.

## Addendum: scalar-cosine shape distance (before its new check)

After the primary run, code inspection identified another discriminating identity:
`compare._shape_distance` applies cosine-cost DTW to one-dimensional nonnegative
angular-velocity values. Any two strictly positive scalars have cosine one, so the
distance should be zero regardless of magnitude. Before numerical execution, fix
two unit-circle paths with angles `[0,.1,.2,.3]` and `[0,.6,1.8,2.0]`. Record both
angular-velocity profiles and their nonzero Euclidean difference, then evaluate the
actual library's shape distance. Add tests for the scalar identity and complete
library call. This bounded synthetic addendum is motivated by inspected code, not
selection among experimental results. It uses no corpus data.

## Post-review sensitivity: uncertainty after lens reselection

Registered before this sensitivity's local implementation and execution, after
independent review disclosed its approximate result. This is **not a blind
preregistration or confirmatory analysis**. The reviewer identified that the
original LOOCV bootstrap resamples fixed held-out losses and therefore conditions
on the lens selected in each original fold. It omits changes in those selections
when the conversation cohort changes. The reviewer reported an interval spanning
zero using the method below; that result is known before this reproduction.

Keep the existing `results.json` and its fixed-loss interval unchanged. Use only
the same saved 12-by-26 real-conversation mean-score matrix, k>=2, and the same
uniform+mass comparator. No API calls, new data, re-embedding or source-project
edits. Use seed 20261015 and 20,000 valid draws, matching the original draw count.

For each bootstrap draw, sample the 12 conversation multiplicities from
`Multinomial(12, [1/12]*12)` using NumPy `default_rng`. For each original identity
with positive multiplicity, remove **every copy of that identity** from the
training multiset. Choose the highest-mean-scoring lens on the remaining
multiset, retaining multiplicities and the original lens order to break ties.
Evaluate that selected lens minus uniform+mass on the held-out original case.
Average these gains weighted by the held-out multiplicities divided by 12.
Reject and redraw a draw containing only one unique identity; record the number
rejected. Do not silently train on a copy of a held-out case.

Report the original unresampled LOOCV point estimate alongside the new percentile
2.5%/97.5% limits, fraction of bootstrap gains <=0, valid/rejected counts, runtime,
configuration and input/implementation hashes. The fixed-loss interval remains
a conditional descriptive summary; it must not establish a robust positive
learning-procedure gain. Even the reselection-aware interval is a post-review
within-cohort sensitivity for 12 selected conversations from one user, not a
population confidence claim or a remedy for user/session/archetype selection.

Before execution, add tests with hand-computable small score matrices for complete
held-identity exclusion, training multiplicities, weighted held-out evaluation,
agreement with ordinary LOOCV when all counts equal one, and explicit handling of
singleton draws. Check the vectorized implementation against direct repeated-row
indexing on a small deterministic fixture. A bounded run should take seconds.
