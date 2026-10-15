# Dynamics audit: evidence survives, the strong interpretation does not

**Assessment, 2026-10-15:** the existing work supports useful recency-weighted
aggregation and within-archive order effects. It does not establish that embeddings
are forces, that an optimized decay rate measures cognitive tempo, or that the
reported derivative graphs reveal independent dynamical dimensions. The universal
turning-cosine proposition is false, and the package's shape-distance comparator
collapses on ordinary positive-valued profiles. These are concrete blockers to
submitting the current framing. A bounded calibration/benchmark paper is more
credible for ISCS2027; broad claims about intrinsic semantic dynamics belong in a
longer research program.

This audit inspected specifications, both manuscript formats, experiment code,
saved numerical results, and the current embflow implementation and tests. Original
projects were read-only. It ran the prewritten
[protocol](../checks/dynamics/PROTOCOL.md), including a separately registered final
shape-distance addendum. All conclusions below distinguish existing evidence from
new checks. External novelty assessment is handled by the parent literature audit.

## What actually exists, including stale status

- **Exp04 ran successfully on June 11.**
  `papers/embedding-dynamics/experiments/results/exp04_prefix.json` contains all
  lens-by-prefix scores; the results addendum in `experiments/EXPERIMENTS.md` and
  `paper/sections/04_validation.tex` agree with it. Best mean cosine is **0.896668**,
  mismatched-conversation null **0.314967**, cross-conversation prefix null
  **0.404902**. The numerical gate passed. The `.papermill/state.md` header says
  VALIDATED but retains multiple NOT RUN statements and an unchecked execute task.
  `CLAUDE.md` still says no manuscript exists, although `paper/main.tex` and its
  drafted validation section exist. Rerunning a paid API experiment is unnecessary
  to resolve this status contradiction.
- **The semantic-dynamics manuscript is further along but inconsistent with its
  code and later results.** `paper/paper.md` and `paper/paper.tex` differ in claims
  and episode-density wording. The LaTeX draft still reverses light/heavy smoothing
  labels at alpha .50/.95, although the later embedding-dynamics spec corrects
  them. Its corpus table attributes approximately 104k messages to ChatGPT and 42k
  to Claude Code; the current embedding metadata instead has 55,230 and 91,950
  rows, respectively. These may reflect different snapshots or definitions, but
  the table needs traceable denominators before reuse.
- **Current data provenance:** 3,144 conversation metadata rows, 149,623 embedded
  messages, one registered embedding model with observations,
  `text-embedding-3-small`, 256 dimensions. The old 3,097 count is not the current
  metadata count. Only **three distinct `embedded_at` values** exist across all
  message embeddings. Older semantic scripts sort by this batch timestamp; the
  newer motion scripts sort by insertion `rowid`. Neither field proves original
  event chronology, and incremental embedding can alter insertion order.
- **Reproduction friction:** semantic-dynamics scripts still use the removed
  `ef.Exponential`/`ef.Uniform` API and pre-reorganization database paths. The
  current package's tests pass, but these research scripts cannot simply be rerun
  unchanged against it. The newer standalone motion and prefix scripts avoid that
  API problem. No original code was repaired in this audit.

Exact file hashes, repository revisions and initial statuses are in
[source_hashes.json](../checks/dynamics/source_hashes.json). Every recorded source
hash was unchanged after the primary run. Key anchors are:

| Source | SHA-256 |
|---|---|
| `embflow/embflow/compare.py` | `e8c6c01dae3da7ae761b46909656fc6ce65ed2fb2428521492b8bfc75efdbc15` |

The manifest supplies the remaining full hashes, including the DB, manuscripts,
Exp04 result, experiment implementations, and library modules/tests; it avoids
copying private data into this report.

## Mathematical claims that require repair

**The -1/2 normalized turning-cosine theorem is false.** Its proof correctly obtains

\[
E\langle v_k,v_{k+1}\rangle=\mu-1,\qquad
E\|v_k\|^2=2-2\mu,
\]

but their ratio is not
\(E[\langle v_k,v_{k+1}\rangle/(\|v_k\|\|v_{k+1}\|)]\).
Uniformly permute the three distinct unit vectors `(1,0), (0,1), (-1,0)`.
This is exactly exchangeable, with no zero velocities. The six-permutation mean
turning cosine is **-sqrt(2)/3 = -0.471405**, while the ratio of expectations is
**-0.5**. Zero-padding the vectors to 256 dimensions changes neither answer.
For independent points uniformly distributed on the unit circle, the actual
expectation is **-4/pi² = -0.405285**: conditional on the middle point, the two
independent normalized chord directions have opposing conditional means of
magnitude 2/pi. A 200,000-triple simulation produced -0.404991, Monte Carlo 95%
interval [-0.407532, -0.402451]. This also defeats an attempted repair from
“exchangeable” to “i.i.d.” alone.

The observed shuffled corpus mean -0.493 is valid empirical evidence; it does not
prove a universal constant. Use each conversation's actual permutation null, or
state sufficient concentration/equal-chord conditions for an approximation. The
current false claim appears in the manuscript proposition, SPEC, experiment
summary, library documentation and tests. The tests check only 16-dimensional
Gaussian examples with loose .02/.03 tolerances.

**Accumulation is a model definition, not an identified physical mechanism.**
For `s_k = alpha*s_(k-1) + e_k`,

\[
\Delta s_k=(\alpha-1)s_{k-1}+e_k.
\]

Thus `Delta s = e` and the shifted derivative tower are exact only for the
unnormalized pure sum, alpha=1. Normalizing the state changes its derivatives
again. For continuous normalized readout `x=s/||s||`,
`dx/dt=(I-xx^T)(ds/dt)/||s||`; radial and tangential changes cannot be conflated.
The spec partly recognizes this, but the later language treats the tower as
general. In an overdamped ODE, `exp(-lambda*dt)` is exact homogeneous propagation,
not the explicit-Euler coefficient `1-lambda*dt`. Impulsive input at interval
boundaries makes the exponential recurrence exact; piecewise constant forcing
introduces an additional input coefficient. The distinction is straightforward
to state and removes the erroneous Euler derivation.

**“Curvature” currently means second difference.** A straight path `(0,0), (1,0),
(3,0)` has second-difference norm 1 and infinite three-point curvature radius.
It accelerates without bending. A second difference depends on sampling and speed;
geometric curvature requires normal acceleration divided by squared speed, or an
appropriate discrete geometric definition. The package has a circumradius
operator, but it is not what the manuscript's averaged “curvature” graph uses.

**The graph summaries telescope.** For any sampled trajectory,

\[
\overline{\Delta x}=\frac{x_{n-1}-x_0}{n-1},\qquad
\overline{\Delta^2 x}=\frac{(x_{n-1}-x_{n-2})-(x_1-x_0)}{n-2}.
\]

`semantic-dynamics/experiments/exp01_velocity_graph.py` normalizes these means.
Its mean-velocity graph is an endpoint-displacement-direction graph; its
mean-second-difference graph is a boundary-velocity-difference-direction graph.
Internal trajectory shape is lost. These graphs can be useful but cannot support
the advertised full “transition pattern” and “structural shape” interpretations.
The separate exponentially weighted velocity summary is not subject to the same
simple unweighted identity and should be evaluated separately.

**The raw “acceleration changepoint” is ordinary adjacent cosine novelty.** For
unit message embeddings, `||e_(k+1)-e_k||² = 2(1-cos(e_(k+1),e_k))` exactly.
It gives the same ranking as adjacent cosine distance. Calling it acceleration
under an integrator does not produce a new detector; thresholds must be mapped
correctly because distance and squared distance differ.

**Invariance is conditional and representation-specific.** Velocity Gram matrices
are invariant to a common rotation/reflection and translation of an already
constructed path. The test reproduces this to <1e-15. Translating inputs before
normalized accumulation changes the output geometry (maximum Gram discrepancy
1.464 in the fixed example); that is not the same operation. Gram comparison also
depends on temporal correspondence and scale. Scalar profiles and distributional
summaries discard further information; losing information is not proof of
topic-independent cognitive content. NMI between graph partitions is neither
statistical independence of the underlying representations nor proof that their
geometry is orthogonal. Stochastic clustering, density thresholds, and giant
component restrictions require matched partition/stability nulls.

**The implemented shape comparator is degenerate.**
`embflow.compare._shape_distance` computes the nonnegative scalar signal
`1-cos(x_k,x_(k+1))`, then sends it to DTW whose local cost is cosine distance.
For any positive scalars a,b, `cos([a],[b])=1`, so every local cost is zero.
The registered pair of unit-circle paths has distinct angular-velocity profiles
with Euclidean difference **0.655173**, yet the actual library returns **shape
distance 0.0**. This holds generically whenever all profile values are positive;
zeros merely introduce a largely stationary/nonstationary distinction. In
addition, the profile is point angular displacement, not turning angle between
velocities. The implementation must be repaired and recalibrated before any
shape-retrieval claim. This defect does not invalidate Exp03's separate scalar
motif-distance computation.

**Half-life is a filter-weight timescale.** `h=log(.5)/log(alpha)` is correct for
coefficient decay, not a measured topic-shift interval or cognitive forgetting
rate. Constant inputs can give a constant normalized trajectory while individual
weights decay. Message splitting, role frequency, token mass, embedding geometry,
candidate grid and scoring objective affect the estimate. Half-life also differs
from the exponential time constant by log(2). A Gaussian weighting window is a
smoother, not a bandpass filter merely because it has a focal position. A generic
fixed-alpha EMA is not automatically a fitted Kalman model with identified process
noise. Those connections need explicit assumptions.

The package documentation explicitly notes that a constant sequence selects
alpha=.999 because its epsilon-normalization favors larger accumulated norms in
near-ties. That is numerical tie behavior, not evidence for an identified
692.8-message memory timescale.

For numerical reporting, distinguish `h(mean alpha)`, `h(median alpha)` and
`mean h(alpha)`. The semantic draft's 2.4/4.3 values match its alpha medians
.75/.85, not its mean alphas .762/.827. The newer .782/.840 means transform to
2.82/3.98 messages; these are transforms of means, not mean individual half-lives.

## Existing evidence and honest failure accounting

| Existing result | What it supports | What it does not support |
|---|---|---|
| Exp01, 1,768 conversations: speed paired d=-2.03, tortuosity +1.39, adaptive alpha -1.83 versus own shuffle | Strong order effects within the archive | Cognitive independence, a universal turning law, population effects |
| Exp01 user-only turning: -.4400 real vs -.4932 shuffled, d=.958, n=1,269 | Empirical order effect after removing role alternation | Exact null -.5 |
| Exp01 preregistered gate **failed** | Useful disconfirmation recorded | A passed gate by retrospective reinterpretation |
| Exp02 first 16 messages: motion AUC .9732, length/role AUC .9966, topic .9997 | Source can be classified | Motion superiority or independence from topic/style |
| Same-browser comparison: motion .7563, length/role .7897, combined .8119, topic .8708 | Possible complementary signal worth testing | Better-than-baseline performance already established |
| Exp03 local motifs: best distance .594 vs shuffled .601 | Negative result for this representation/search | Established cross-topic procedures or useful motif retrieval |
| Exp04 cosine .8967 vs uniform .8695 | Approximation and modest recency advantage | Identified force/state semantics |
| Exp04 exact-position retrieval .4557, threshold .5 | Far above naive pooled chance | The preregistered exact-position target was met |

The Exp01 gate failed for **two** reasons: speed autocorrelation reversed
(real .0416 versus null .1609, d=-.479), and mixed-role turning reached only
d=.257, below the specified |d|>=.5. The addendum emphasizes the reversal but
does not make the second failure equally visible. Proceeding after these results
is legitimate if reported as a revised hypothesis, not as a passed preregistration.

The platform classifier puts imputation and standardization inside each fold, so
that common preprocessing leak is absent. Fixed message windows control window
length but leave topic, calendar period, project/session dependence, char counts,
assistant verbosity, role patterns and platform templates entangled. No grouped
user/project or calendar holdout was established. CV-fold standard deviations are
not confidence intervals for the difference between models.

The “cross-model replication” label lacks a traceable independent nomic run in the
audited chain. The earlier semantic manuscript states OpenAI-small-256, the
operational-memex adaptive-alpha script reads that analysis DB, and the newer
Exp02 merely stores SPEC values under `spec_nomic_values`. Current DB provenance
contains one OpenAI model. This is a provenance gap, not proof that no nomic run
exists elsewhere; recover its source results before citing replication.

Episode detection likewise needs external evaluation. Saved episode summaries
give approximately 16.6/21.7/17.6 messages per episode for Claude/Claude Code/ChatGPT,
not the old Markdown abstract's universal 12-15. The newer LaTeX text relaxes the
range. Window size, minimum segment size and sensitivity affect density. The
138/586 continuation markers within three messages of a boundary are a proxy
recall-like rate, labeled `precision` in the JSON without a detected-boundary
denominator. There is no matched chance/tolerance baseline. Low running-mean
increments can follow mechanically from the 1/k averaging update, so stabilization
around ten messages does not by itself establish a cognitive timescale. The
window detector looks ahead and ignores alpha; it is an offline segmenter, not
validation of the causal leaky-state forecast.

## New bounded checks: what changes after honest controls

All primary numerical outputs are in [results.json](../checks/dynamics/results.json),
the comparator counterexample is in
[shape-results.json](../checks/dynamics/shape-results.json), and the post-review
selection-uncertainty sensitivity is in
[selection-uncertainty-results.json](../checks/dynamics/selection-uncertainty-results.json).

**Prefix reconstruction remains promising but modest.** The saved scores contain
12 selected real ChatGPT conversations and 305 eligible prefixes, not independent
observations from three platforms. Selection is by embedding-derived archetypes,
not random sampling. The mean over conversations was reproduced to 2.3e-16.

| Comparison | Mean cosine gain | Descriptive 95% conversation bootstrap interval |
|---|---:|---:|
| Best lens vs uniform, fixed lens | .02713 | [.00772, .04774] |
| Best lens vs matched uniform+mass, fixed lens | .01935 | [.00479, .03904] |
| LOOCV selected lens vs uniform+mass, fixed fold losses | .01766 | [.00483, .03381] |
| Same LOOCV estimator, lens refitted in each bootstrap cohort | .01766 | **[-.00122, .03481]** |

Selecting one of 26 lenses on the same cases introduces optimism, but here the
LOOCV mean drops only from .89667 to .89498: 11 folds choose exp(.85)+mass and one
chooses exp(.90)+mass. This preserves a positive held-out **point estimate**.
However, the original interval resampled the 12 already-computed fold gains and
held their selected lenses fixed. It is a conditional descriptive interval; the
overlapping folds and changes in lens selection are not represented by treating
those losses as fixed observations.

Independent review prompted a separately recorded sensitivity: 20,000 multinomial
resamples of the 12 conversation identities, seed 20261015, refitting every lens
selection after excluding **all resampled copies of the held-out identity**.
Training and evaluation both retain the resample multiplicities. The resulting
percentile interval is **[-.001224, .034807]**, spanning zero; 3.31% of bootstrap
gains are nonpositive (a descriptive fraction, not a null-hypothesis p-value).
No singleton-cohort draws required rejection. The original point estimate and
fixed-loss results remain unchanged in their saved artifact. The reviewer's
approximate result was known before this reproduction, so this is a post-review
sensitivity, not a preregistered confirmation. **A robust positive gain is not
established by these intervals.** All versions condition on the selected one-user
cohort, embedding model and study design; none supplies a population claim or
controls dependence across sessions.

The gain is concentrated in drifting cases (+.06110); return cases gain .01785,
focused .00136, and **pivot cases lose .00292 against uniform+mass**. The claim
that every nonfocused archetype benefits needs qualification. Reconstruction and
prediction alpha **do correlate** (Spearman .660, Pearson .628, n=12), but
reconstruction alpha exceeds prediction alpha in all 12 cases, mean difference
.16083, interval [.1125,.2100]. Two best-fit alphas are .99. These results support
two objective-specific fitted scales, not a single intrinsic one.

The mismatched and cross-prefix nulls compare different conversations. They are
good content-identity checks but weak tests of order semantics: uniform+mass
already scores .8773 and clears the same numerical gate if compared with the
saved best-lens null (a uniform-specific null was not saved). Uniform/mass pooling
is invariant to reordering a fixed prefix multiset; exponential pooling is
order-sensitive by construction. A running-mean *path* still changes as prefix
membership changes, so calling the entire path order-free is also imprecise.
No re-embedded same-content reordered-prefix control was performed. Role tags
occur in concatenated prefixes but not isolated-message embedding inputs, despite
the manuscript saying preprocessing was identical. Perfect conversation retrieval
does not demonstrate precise chronology; the asserted adjacent-position nature
of errors cannot be verified from the saved aggregate retrieval scores.

**Prediction favors ordinary smoothing; per-conversation fitting adds no demonstrated
benefit here.** The bounded diagnostic used 80 ChatGPT, 80 Claude Code and all
17 eligible Claude conversations with >=40 messages, capped at 80. Alpha was
selected on the first half; only earlier messages formed each second-half
prediction. Selection was deterministic, and no sample expansion followed results.

| Source | Fitted alpha minus running mean, held-out cosine | Fitted alpha minus fixed .85 |
|---|---:|---:|
| ChatGPT, n=80 | +.04704 [.03986,.05479] | -.00120 [-.00387,.00127] |
| Claude Code, n=80 | +.03100 [.02612,.03631] | -.00226 [-.00330,-.00126] |
| Claude browser, n=17 | +.03576 [.02516,.04749] | -.00036 [-.00234,.00181] |

All intervals use conversations as bootstrap units. All three also beat the
last-message baseline. Full-sequence alpha selection scores higher on the same
held-out portion by .0033-.0048, illustrating evaluation optimism if test targets
select the parameter. The original per-step scoring is causal conditional on
alpha; the leakage is using all targets to select alpha and then treating the
optimized objective as forecast evidence. Full-sequence fitting remains valid
as a descriptive fit if labeled that way.

Training-half alpha means are .7719 ChatGPT versus .8175 Claude Code, but role
structures differ greatly: user proportions .480 versus .180; role-transition
proportions .926 versus .272. This is consistent with the earlier direction and
also demonstrates the confound. It is not a causal platform or cognitive-tempo
replication. The sample excludes short conversations and relies on insertion
order, whose chronology still needs validation.

## Recommendation and smallest evidence that would change it

**For ISCS2027:** do not lead with the force interpretation, universal -1/2 theorem,
orthogonal cognitive spaces, intrinsic half-life, or recovered cross-topic motifs.
Prefer a narrow, falsifiable contribution on calibrating embedding-trajectory
statistics and benchmarking incremental prefix/next-message representations. The
present strongest positives are order effects against role-preserving nulls and
a positive recency-gain point estimate under held-out lens selection, with an
interval spanning zero once selection uncertainty is included. The strongest
negative contribution is identifying how plausible metrics collapse to endpoint,
normalization, or composition effects. The parent audit's literature findings
should determine whether this calibration framing has enough novelty.

Before a submission decision, the minimum useful additions are: a frozen
chronology/provenance manifest; repair the false proposition and degenerate shape
cost; one held-out task with simple strong baselines; and a same-content ordering
control. Re-embedding a few selected examples would not by itself establish
generalization. No paid API work was needed for this audit.

**For a longer program:** retain state-space modeling as a modeling choice, then
test portable utility across embedding models, users, and tasks. Compare with
uniform/token-weighted pooling, a fixed EMA, sliding-window means, adjacent-cosine
novelty, standard offline/online changepoint methods, and actual geometric
speed/turning/Gram distances. Match role pattern, token mass, length, task and time;
use within-project/user or forward-time holdouts and assess human/task outcomes.
For prefix reconstruction, same-multiset shuffled/reversed text and last-message
or last-window baselines distinguish encoder order sensitivity from content
pooling. For segmentation, labeled boundaries plus Pk/WindowDiff or an explicitly
defined boundary metric are needed. For cognition claims, an external behavioral
criterion is necessary. A held-out gain over fixed EMA and task-matched controls
would materially improve the case; more in-sample alpha histograms would not.

## Verification and limits of this audit

The original library suite passed **191 tests**, with **95.1% statement coverage**.
The primary audit has **12 passing unit/integration/artifact tests**; the shape
counterexample adds **2 passing tests**. Primary focused-test coverage is 55.2%;
combining those tests with the complete bounded integration run covers **98.4%**
of the audit script. Uncovered lines are the deliberate >240-conversation guard
and stage-exception capture. Provider adapters and some segmentation edge paths
remain unexercised in the library suite. Its two warnings concern a future pytest
fixture deprecation, not a numerical failure.

The post-review selection sensitivity adds **11 passing tests** for complete
held-identity exclusion, training/evaluation multiplicities, ordinary-LOOCV
equivalence, tie handling, a direct repeated-row reference, and explicit rejection
and redrawing of degenerate cohorts. This sensitivity ran in .055 seconds, using
only the saved scalar score matrix. Both the original result JSON and child
Exp04 result retained their input hashes.

The primary run took 5.17 seconds; the final shape check took .006 seconds, excluding
environment setup and tests. Dependencies, seed, limits, source hashes, runtime,
test output and aggregate uncertainty are all saved under `checks/dynamics/`.
Private conversation text and titles were not queried by the checks; vectors
remained local and only aggregates were exported. The audit does not re-embed
prefixes, recover original event chronology, establish a population effect, or
replace external novelty review. Those are explicit limits, not hidden passes.
