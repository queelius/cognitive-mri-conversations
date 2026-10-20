# Reviewer 2 (methods and validity): "What the Whole Conversation Predicts"

Manuscript: `papers/iscs-2027/paper/main.tex` (12 pages, `main.pdf`). Review date 2026-10-19.
Scope: validity only. I did not assess prior art, framing or prose. Every finding cites a file
and line (or JSON/CSV field). All paths are relative to `/home/spinoza/github/cmri/papers/`
unless stated. I read the primary-contrast code rather than the prose description, and I
computed one descriptive breakdown (per-target-position gains) from the committed per-case
scores in `results/public-*.json`; nothing was re-encoded, re-drawn or re-scored.

Overall: the primary measurement is computed as described, the numbers in Sections 4 and 5
match the result files (I checked 44 of them; one cosmetic mismatch), and the two held-out
runs are genuine single-shot runs under a frozen rule. The problems are in what the paper says
the measurement shows. Three claims that the abstract foregrounds are not supported by the
evidence as presented: the saturation of the gain at four to six messages, the "reversal" of
the role asymmetry on PRISM, and "ranking does not follow cosine". The remaining concerns are
about verifiability of the freeze and the interpretive gap between the headline sentence and
what the development results already show about where the gain comes from.

## Findings by severity

### Blocking

**B1. The saturation claim is a property of the target-position mixture, not of conversations.**
Abstract (main.tex:30): the gain is "mostly reached by four messages of history". Sect. 4.1
(main.tex:88): "Most of it is reached quickly ... beyond six the curve changes by at most
.001". The budget curve pools trailing windows over all targets at positions 4, 6, 8, 10
(`iscs-2027/src/cmri_confirm/analysis.py:60-70`; `Recipe(WINDOW, n)` is `h[-n:].mean` in
`history-falsification/src/cmri_history/analysis.py:146`, so a window longer than the history
is the full mean). Targets are built at `k in range(4, len(x)-1, 2)`
(`iscs-2027/src/cmri_confirm/cases.py:55`), so a conversation of 6 messages contributes only
k=4, where the full history is four messages and windows 4, 6, 8, 10 and "all" are identical
by construction. From `iscs-2027/results/public-minilm.json` `per_case.keys`: of 1,482 PRISM
reserve user targets, 775 (52.3%) are at k=4, 405 (27.3%) at k=6, 204 at k=8 and 98 (6.6%) at
k=10. So for 80% of PRISM targets the window-6 and full-mean scores coincide, and only 98
targets (in 98 conversations) distinguish window 8 from window 10. The same holds for WildChat
(`wildchat-replication/results/public-minilm.json`: 801 of 1,998 targets at k=4, 534 at k=6).
A curve that is flat beyond six is therefore guaranteed by the sample structure, and "mostly
reached by four" is largely the statement that half the targets have exactly four prior
messages. The per-position gains computed from the committed per-case scores are, PRISM
MiniLM: +.026 (k=4), +.023 (k=6), +.034 (k=8), +.023 (k=10); E5: +.016, +.021, +.026, +.026;
WildChat MiniLM: +.012, +.036, +.046, +.056; E5: +.012, +.020, +.024, +.027. These do not show
saturation; on WildChat the gain keeps growing with available history. The budget-curve
sentences and the abstract clause must be either removed or replaced by a position-stratified
curve (which needs embeddings, so on development data: `history-falsification/cache/`,
`wildchat-replication/cache/development/`). The primary claim itself is unaffected: the gain is
positive at every position in both collections.

### Major

**M1. The headline is consistent with, and the development results already look like,
variance reduction over near-exchangeable messages; the paper reports neither the single-message
levels nor the stronger baseline it already has.** The primary quantity is
`cos(normalize(mean of all prior unit vectors), y) - cos(normalize(mean of last two), y)`
(`history-falsification/src/cmri_history/analysis.py:145-146,161`;
`portfolio-wave1/src/cmri_wave/modeling.py:156-160`). Under any process in which messages share
a conversation-level direction plus independent noise, this difference is positive because the
mean of n unit vectors estimates the shared direction better than the mean of two; process A
(`history-falsification/src/cmri_history/synthetic.py:128-131`) is exactly that process and
gives +.120 (main.tex:157). The paper concedes this in the discussion (main.tex:172, "a
statement about the geometry of what people write next, not about memory"). What it does not
report is that the committed development results show the signature of that null directly:
in `history-falsification/results/public-minilm.json`
(`results.views.native128.regimes.natural.user.pooling.models.*.cosine.mean`) the opening
prompt alone scores .343 and the latest message alone .339; the user's second-to-last own
message .343 and own latest .366. In E5 the opening prompt alone (.826) beats the latest
message alone (.810). Position carries almost no information beyond count. Further, the
stronger fixed baseline the authors already have, opening exchange plus latest exchange,
leaves a gain of +.015 [.006, .024] MiniLM and +.005 [.003, .007] E5 at k=6 against +.043 and
+.024 over the latest exchange alone on the same 117 cases
(`history-specificity/REPORT.md:131-134`), i.e. the opening exchange accounts for roughly two
thirds (MiniLM) to four fifths (E5) of the gain there. At k=4, which is 52% of the PRISM
targets, the "whole conversation" is literally the opening exchange plus the last exchange.
None of this contradicts the measurement, but the title and the conclusion sentence
(main.tex:186) invite the reading that information is distributed across the conversation,
and Sect. 4 withholds the two cheapest facts that bear on it. Report the single-message levels
and the opening-plus-latest baseline in Sect. 4 (development data; both already computed), and
state in Sect. 4.1 that the gain is of the form a static shared-direction model predicts.

**M2. "Reverses to a small positive value on PRISM" (abstract, main.tex:30) contradicts the
body, which declines the claim.** Sect. 4.2 (main.tex:123): "we do not claim PRISM's reversal,
which did not replicate"; `wildchat-replication/CLAIMS.md:17` lists the reversal as "not
claimed". The PRISM E5 "reversal" is +.0017 [.0006, .0029] (`iscs-2027/results/public-e5.json`
`roles.interaction_common`), a value with no practical meaning on a cosine level of .85. An
abstract should not foreground a result the paper disowns. Replace with "is not
distinguishable from zero on WildChat and small on PRISM".

**M3. "Ranking does not follow cosine" (main.tex:166) and "the cosine gain does not carry over
to ranking" (main.tex:30) overstate a small, near-ceiling, partly degenerate test, and test a
different contrast from the primary.** (a) The ranking contrast is long (opening + latest +
middle) against short (opening + latest), not mean against last two; the matched cosine
diagnostic for that population is +.015/+.005, not +.025/+.018. (b) In E5 the long and short
models are both ridge at penalty .1 (`history-specificity/results/public-corrected-01/result.json`
`encoders.e5.selection`), and the contrasts are degenerate: `HS01_long_short_top1` is 0 [0, 0],
`HS01_genuine_donor_mrr` is 0 [0, 0] and `HS01_long_short_mrr` has upper bound exactly .000
(`history-specificity/derived/contrasts.csv:5-10`). No bootstrap draw ever favoured long, and
donor substitution changed no rank: the E5 readout is insensitive to the middle, so it cannot
test whether the middle helps. The paper reports "-.002 [-.005, .000]" and "and nothing"
(main.tex:166) as if these were null results rather than a measurement with no resolution.
(c) In MiniLM, long is `pool:all` and short is `ridge:short:0.01` (same file,
`encoders.minilm.selection`), so the contrast compares model classes. (d) MRR is .82-.83 with
8 candidates and 79 queries in 47 components; the MiniLM interval [-.041, .018] excludes a +.03
gain but not a +.015 one. The paper's own "The negative is bounded" sentence is right; the
heading and the abstract are not. Weaken to: "on one small eight-way task the middle turns gave
no detectable ranking gain over opening-plus-latest; the test had no resolution in E5".

**M4. The length heterogeneity is confounded with the amount of history pooled, and is presented
as heterogeneity between conversations.** main.tex:134 reports the gain "rises with message
count" from +.012 to +.040 (MiniLM, WildChat). A 12-message conversation contributes targets at
k=4, 6, 8, 10, where the mean pools up to ten messages; a 6-message conversation contributes
only k=4. `subgroups()` keys on the conversation's length
(`iscs-2027/src/cmri_confirm/heterogeneity.py:31-32`), so the length axis is the budget curve
restratified. The per-position gains in B1 show this directly on WildChat. If the authors want
a conversation-level heterogeneity statement, they need the gain at a fixed position (k=4) by
conversation length; otherwise say the length effect is expected from the measurement.

**M5. The pre-registration cannot be verified by a reviewer.** The paper's methodological
contribution rests on "froze the protocol in a pushed commit" (main.tex:38) and on push-log
ordering (main.tex:180). Both repositories are private at review time; the paper says they
become public "with the camera-ready version" (main.tex:188). The ordering evidence is
`gh api repos/queelius/cmri-iscs-2027/activity` (`iscs-2027/VERIFICATION.md:140-147`) and
`wildchat-replication/audits/freeze-push.json`, readable only by the owner. A reviewer can
verify nothing: not the freeze hashes, not the ten-second and seven-minute margins, not the
post-freeze diffs. This is fixable without new data (public mirror, or a time-stamped Zenodo
deposit of both repositories at the freeze and result commits, made before submission), and it
should be fixed before the claim is made in print. Two further disclosure gaps: (i) the frozen
PRISM rule (lower bound > 0) replaced a stricter rule (lower bound > .01) one commit before the
freeze (`iscs-2027` commit `98f5ff1`, "Correct frozen success rule in protocol"; the predecessor
protocol `history-falsification/PROTOCOL.md:199-201` had planned "both encoders' primary lower
bounds >.01 with declared multiplicity treatment"). Both confirmations pass the stricter rule
too (lower bounds .0185/.0165 PRISM, .0136/.0139 WildChat), so nothing turns on it, but
"A reporting floor of .01 did not gate" (main.tex:69) reads differently once one knows the
floor had been the gate. (ii) The PRISM freeze hashed three files only
(`iscs-2027/FREEZE.json:2-6`); the paper says so (main.tex:71), but it should also say the
protocol's own sentence that the freeze "records the hashes of ... the code and the sample
list" (`iscs-2027/PROTOCOL.md:62-63`) is false as written (`iscs-2027/REPORT.md:231-236`).

**M6. The contrast was selected on the same development data that supplied the pre-registered
values, and the shrinkage is consistent with winner's curse; the paper names neither.** The
mean-minus-last-two user contrast was the surviving lead from a grid of 29 pooling recipes
(`history-falsification/results/public-minilm.json` `pooling.models`, plus the wave-1 grid,
`portfolio-wave1/REPORT.md:66-94`), chosen after seeing that it beat the practical floor in both
encoders. That is legitimate for a held-out test, but the development estimate is then an
optimistic selected maximum, and the observed shrinkage (26% and 21% in MiniLM, main.tex:79)
is what selection predicts. Say so; "the held-out values, not the pilot values, are the
result" is the right conclusion but the paper gives no reason for it.

### Minor

**m1. Synthetic null calibration does not calibrate the rule that was used.** Detection in the
synthetic study is `ci_low > practical_gain` with practical_gain = .01
(`history-falsification/src/cmri_history/synthetic.py:178-179,183`; main.tex:160 caption), but the
frozen empirical rule is `ci_low > 0` (`iscs-2027/PROTOCOL.md:19-21`). B's "0 of 100" is for the
stricter rule. Under the rule actually used, B would be expected to give about 2.5 false
positives per 100 per encoder; this is a one-line change on numeric data.

**m2. The revelation identity under-uses the authors' own diagnostics.** main.tex:155 says a
superlinear curve "needs no clustering" and that the denominator changes the exponent
"(1.56 against 1.41)". The identity E[M_k] = M k(k-1)/(N(N-1)) implies an exponent of 2 for the
all-arrived denominator under uniform revelation, and the pilot's null bands say exactly that:
`temporal-networks-calibration/pilot/results/diagnostics.csv:11` has observed gamma_all 1.559
against a uniform-null band [1.685, 2.329] (effect -.425, p = .02), and line 12 has
gamma_active 1.405 inside its band [1.252, 1.474]. So the author archive densifies less than
random revelation on one denominator and indistinguishably from it on the other. That is the
fact that bears on the earlier densification claim; "1.56 against 1.41" compares two observed
fits and only shows denominator sensitivity. The identities themselves are correct (edge
inclusion probability k(k-1)/(N(N-1)); vertex active iff arrived and at least one of d_i
neighbours among the other k-1 arrivals), and the caveat that expected slopes are not slopes
of expectations is present (main.tex:155).

**m3. Packaging identity: correct, but the "exact" result is about the EMA weight, and the
empirical sensitivity is a fit-sample result.** Splitting each vector into two identical pieces
doubles the sequence length, so an EMA with ratio alpha per piece has ratio alpha^2 per
message; sqrt(alpha) per piece restores the same normalized direction. Fine. The empirical
numbers (.054, .038, -.019, +.010; main.tex:147) come from 100 fit conversations, 81
components (`portfolio-wave1/REPORT.md:155-167`); say "development" there as elsewhere.

**m4. Two-encoder rule: state the implied error rate.** Each encoder's one-sided test has
nominal size .025; the two score the same conversations (main.tex:51, 176), so the conjunction
has size close to .025, not .025^2. The paper says the agreement "does not double the
evidence" but never states the rule's size.

**m5. Estimand is a component mean, not a conversation or target mean; on WildChat the two
differ.** `paired_summary` averages targets within conversation, conversations within
component, then components equally (`portfolio-wave1/src/cmri_wave/inference.py:26-38,41-58`).
From the per-case scores, the unweighted case mean gain is .026 (PRISM MiniLM) against the
component estimate .025, but .031 (WildChat MiniLM) against .021: longer conversations, which
carry more targets and larger gains, are down-weighted. State the estimand in Sect. 3 and
report the case-level mean alongside.

**m6. Per-component "fraction positive" hides the per-case picture.** main.tex:134 gives 72%
and 97% (PRISM) and 61% and 83% (WildChat) of components. At the case level the fractions are
60%/84% (PRISM MiniLM/E5) and 60%/77% (WildChat). Four in ten individual forecasts are worse
under the mean. Worth one clause.

**m7. Heterogeneity cells overlap and are unadjusted; stated, but the post hoc gpt-4o 2024 vs
2025 contrast (main.tex:134) is still presented with numbers.** 170+202+164 = 536 condition
memberships over 233 PRISM components (`iscs-2027/results/report/subgroups.csv:2-4`); 83
unadjusted intervals across both repositories including survey levels. The family-by-year
cells were computed at report time (`wildchat-replication/REPORT.md:368-374`). Either drop the
numbers or keep only the sentence that nothing is tested.

**m8. Common-prefix control confounds horizon with target role and target truncation.** Stated
for horizon (main.tex:121) but not for the fact that assistant targets are longer, more
heavily truncated (74.8% of assistant tokens retained, main.tex:176) and stylistically
homogeneous, which alone could make them profit more from averaging. The PRISM conjecture
(main.tex:174) is one of at least two.

**m9. Cosmetic.** Table 3 (`iscs-2027/paper/figures/synthetic-table.tex:6`) prints "$-$.000"
for B's interactions (-6.8e-06 and -3.1e-04); print "0". Table 2 mixes three and four decimal
places (main.tex:109-110). The paper counts "four samples" but the WildChat development sample
was drawn and scored before the freeze and supplied the centers; the caption says so
(main.tex:54), fine.

**m10. WildChat amendment: disclosure is complete and accurately worded** (main.tex:73 matches
`wildchat-replication/REPORT.md:306-322`; the "at least 95 of 828" figure is 68 `ip_changed` +
27 `too_short` from `wildchat-replication/results/prepare.json:71-84`). The runner edit
(main.tex:180) matches `git diff 0fdb54f 3fc9676 -- src` (a `LossHalt` class and a binding
re-run guard; tightening only). The 806-of-828 pre-freeze decode (main.tex:180) matches
limitation 20. No finding here; recorded so the editor knows it was checked.

## Rubric

### 1. Primary claim

What is computed. For each eligible conversation (6 to 12 alternating messages, user first,
each message its first 128 content tokens), for each even k in {4, 6, 8, 10} with k+1
present, the user message at k is the target; the history is the unit-normalized embeddings of
positions 0..k-1 (`iscs-2027/src/cmri_confirm/cases.py:55-60`). Pooling: `MEAN` is the
arithmetic mean of the unit vectors, renormalized; `WINDOW 2` is the mean of positions k-2
(user) and k-1 (assistant), renormalized (`history-falsification/src/cmri_history/analysis.py:145-146,161`).
Token masses are not used by either recipe. Score: raw cosine of the pooled vector with the
unit target (`modeling.py:158`). Gain per case: mean minus window-2 cosine; averaged within
conversation, then within component; 95% percentile interval from 2,000 component resamples
(`inference.py:41-58`). Dropping: conversations with fewer than six kept messages; in PRISM
the chosen path only, at most four conversations per component by salted rank; in WildChat
the first twelve entries cut at the first blank or role break.

Alternatives to "older messages carry predictive information", and their status:

(a) Variance reduction under a static shared direction (exchangeable noise around a
conversation-level vector). Not ruled out; process A shows it suffices; the development
single-message levels (M1) look like it. Decisive and cheap: on development data compare the
trailing-window curve with a "random n of the history" curve; if the two coincide, the budget
curve is a sample-size curve. Also run a synthetic own-AR(1) process (recency-only dependence)
through the same pipeline; if mean still beats window-2, the measurement cannot separate
recency-only from older dependence. Numeric only.

(b) The opening exchange. At k=4 (52% of PRISM targets) the contrast is exactly "add the
opening exchange to the last exchange". The opening prompt alone is as predictive as the
latest message alone on development data. Partially addressed by HS01's opening+latest
baseline, which removes two thirds to four fifths of the gain at k=6 (M1). Decisive and cheap:
add `initial pair + window 2` as a recipe on development data and report mean minus that, by
position.

(c) Role mix. Window-2 and the mean are both half assistant by count, so role mix is matched;
own-mean minus partner-mean (+.05 to +.10) shows the gain is not from adding assistant content.
Ruled out as a driver by the own-versus-partner secondary (main.tex:117).

(d) Degenerate last-two messages (very short user turns, refusals) making window-2 noisy.
Partly addressed: component medians (.0246 vs mean .0255 PRISM; .0203 vs .0213 WildChat,
`distribution.csv`) are close, so a few components do not drive it. Not addressed at the case
level; a trimmed mean over cases or a mass-stratified view would settle it (development data).

(e) Token truncation. Ruled out by the four development views (main.tex:69); the twelve-message
cap is not a confound for existence of the gain but caps what "whole" means.

(f) Component construction. WildChat IP components may split one person and merge several;
templated prompts across IPs are not merged (main.tex:178). Not ruled out; the bootstrap could
be mildly optimistic. Cheap check on development data: conversation-level bootstrap and an
opening-hash merge; both can be run without the confirmation sample.

(g) Centered cosine. Removes a role-specific mean direction fitted on development data
(`analysis.py:20-27`); gain survives (Table 2). Rules out one anisotropy artifact (a common
direction shared by all text of a role), not the within-conversation shared direction of (a).

(h) Averaging geometry. Normalized mean of n unit vectors: the only mechanism here is (a);
there is no separate artifact of normalization since both recipes normalize.

### 2. Baseline strength

"Last two messages" is the latest exchange, so the headline sentence is literally what was
measured. It is a weak comparator in two senses. First, the mean is itself the tune-selected
or near-selected user recipe on development data (`portfolio-wave1/REPORT.md:90-91`), so the
contrast is best-recipe-minus-fixed-baseline. Second, the authors' own stronger fixed baseline
(opening + latest exchange) cuts the gain to +.015/+.005 at k=6 (M1). The budget curves say
nothing about saturation that the position mixture does not force (B1); per position the gain
is flat in PRISM and increasing in WildChat. The headline "whole beats latest" survives at every
position in both collections; "mostly reached by four" does not.

### 3. Statistics

Unit: dependence component (PRISM: participants joined by shared id, opening-prompt hash or
candidate-set hash, `portfolio-wave1/src/cmri_wave/data.py:250-258`; WildChat: hashed IP).
Nesting handled by averaging first, then resampling components with replacement, equal
weights; same seed across contrasts (common random numbers). 2,000 draws; percentile interval;
233 and 482 components; adequate. The two-encoder rule's size is about .025 one-sided (m4).
The .01 floor is a point-estimate floor and passed everywhere; the lower bounds also exceed
.01, which the paper correctly labels post hoc in the repository (`iscs-2027/CLAIMS.md:13`)
but not in the text. Secondaries are unadjusted and labelled descriptive; fine. Rounding is to
three decimals except Table 2 lines 109-110.

Number check (text against files): primary 4 cells (gate.json both repos) match; centered 4
cells match; own-vs-partner 4, own-vs-own-window 4, interaction natural 4, interaction common
4, attenuation 4 (public-*.json `roles`) match; budget window 3 and 4 values and levels .38/.85/
.43/.87 (budget.csv both) match; fractions positive and q10/q90 (distribution.csv) match;
length subgroup end points, gpt-4.1 22, 2025 53, family-by-year gpt-4o cells (subgroups.csv,
family-by-year.csv) match; survey rho and intervals (survey-ordinal.csv) match; synthetic A, B,
C means and detections (synthetic-effects.csv) match; ranking numbers (contrasts.csv:32,35,8)
match; packaging numbers (portfolio-wave1/REPORT.md:160-167, history-falsification/REPORT.md:
180-184) match; development views (history-falsification/REPORT.md:67-72) match; 1.56/1.41
(diagnostics.csv:11-12) match; shrinkage percentages (26%, 6% to 21%) recompute correctly;
"at least 95 of 828" recomputes. Mismatch: synthetic table prints "-.000" (m9). No
substantive disagreement found.

### 4. Pre-registration credibility

Frozen, PRISM: PROTOCOL.md, config.json, audits/dependencies.json by sha256; code only via
`freeze_commit` f6ae4cd; sample only after the fact by `selected_ids_sha256`
(`iscs-2027/FREEZE.json`; `REPORT.md:231-236`). Pushed to a private GitHub repository ten
seconds before `prepare` started (`VERIFICATION.md:142-151`). Frozen, WildChat: 24 files
including seven analysis modules and their tests, with expected counts in config
(`wildchat-replication/FREEZE.json`); pushed seven minutes before the run. Changed after the
freeze: PRISM heterogeneity code (`e07c770`), survey join (post hoc, one run from an untracked
module, replayed to equality; `iscs-2027/REPORT.md:252-258,298-304`); WildChat `runner.py`
halt guard between freeze and run (tightening), reporting code after. Changed before the
freeze but after seeing development data: the WildChat text-check amendment (disclosed,
main.tex:73) and the PRISM success rule from >.01 to >0 (`98f5ff1`; not disclosed, M5).
Verifiable by a reviewer today: nothing, since both repositories are private (M5). Disclosures
in main.tex:71-73 and 180 are otherwise complete and match the repositories' own limitation
lists.

### 5. Forecast origin (Sect. 4.2)

The common-prefix control does what the text says (`cases.py:57`: assistant prefix stops at k,
so the request u_k is withheld; `role_panel` asserts user gains are identical across regimes,
`iscs-2027/src/cmri_confirm/analysis.py:91-92`). It supports "the large asymmetry depends on
what the assistant forecast sees". The natural-origin asymmetry is close to tautological: the
reply answers the request, and window-2 holds the request at weight 1/2 while the mean holds it
at 1/k. The control confounds horizon (two ahead vs one), target role and target truncation
(m8). "Reverses to a small positive value on PRISM" is not a fair abstract-level description
when the body declines to claim it and WildChat shows zero (M2).

### 6. Synthetic processes (Sect. 5)

A, B, C are generated as described (`synthetic.py:107-142`): A is topic plus independent
innovations with the assistant a near-copy of the current user vector (SD .05 against .25),
B is iid, C is own-second-lag AR with the same reactive assistant. They test what the text
says. The conclusion "the measurement does not identify a mechanism" is correct and, if
anything, understated: because A's assistant copies the request, A also reproduces the
natural-origin asymmetry trivially, so the simulations show that both headline patterns follow
from a static topic plus a reactive assistant. What is missing is the discriminating case: a
recency-only process (own-AR(1)); if mean still beats window-2 there, the measurement cannot
even distinguish "recent dependence" from "older dependence", which is the distinction the
title trades on. Magnitudes (16 dimensions, chosen SDs) are not comparable to the empirical
ones and the table should not print them to three decimals beside empirical values. Detection
rule mismatch: m1.

### 7. Packaging and revelation (Sect. 5)

Packaging algebra: correct (m3). Revelation algebra: both identities correct; the exhaustive
check is a check, not the proof. Empirical numbers: 1.56 and 1.41 are the observed all-arrived
and active-node exponents at threshold .9, block 0 (`diagnostics.csv:11-12`). The identity bears
on the earlier densification claim more strongly than the paper says: under uniform revelation
the all-arrived exponent is 2, and the archive's 1.56 lies below the null band (m2). As stated,
the sentence is valid but illustrates denominator sensitivity, not the identity.

### 8. Ranking section

HS01 supports: "on 79 supported k=6 queries in 47 components, with an eight-way near-ceiling
task, the selected long predictor gave no detectable MRR gain over opening-plus-latest, and the
test could exclude +.03 but not smaller gains; in E5 the long and short readouts rank
identically, so E5 provides no test; the donor control failed its calibration gate (18/100
against a ceiling of 10), so the genuine-versus-donor comparison has no validated null". That is
weaker than "ranking does not follow cosine" (M3).

### 9. Heterogeneity (Sect. 4.3)

Post hoc in code (heterogeneity after the reserve, survey after the report), declared in
protocol; cells overlap; 83 unadjusted intervals; survey join is a descriptive Spearman with a
component bootstrap (`survey.py:60-78`). Nothing is presented as more than exploratory except
the length trend, which is presented as a feature of conversations when it is mostly a feature
of the measurement (M4).

### 10. Strongest supported conclusion, and the weakest sentence

Strongest supportable conclusion: On two held-out samples from separately collected corpora,
scored once under a frozen rule, the normalized mean of all prior message embeddings has a
higher raw cosine with the next user message than the normalized mean of the last two, by
.016 to .025 with component-bootstrap intervals above zero in both encoders; the gain is
positive at every target position, is carried mostly by the user's own earlier messages and in
PRISM largely by the opening exchange, is of the size and form that a static shared-direction
model with independent noise produces, and on one small development ranking task did not
translate into a ranking gain over an opening-plus-latest baseline. The apparent user-assistant
asymmetry under the natural origin is explained by the assistant forecast seeing the request it
answers. Nothing here identifies memory, dynamics, or information specific to older turns.

Weakest sentence (main.tex:88): "Most of it is reached quickly (Fig. 1, lower): one message
alone is far below the last two; a third adds +.013 [.009,.017] on PRISM but only +.003
[-.002,.007] on WildChat in MiniLM (+.006 in E5 on both); four reach +.024 and +.018 in MiniLM
and +.016 and +.013 in E5; beyond six the curve changes by at most .001, and under the
twelve-message cap window 10 is the full mean." Every number is right and the inference is
wrong (B1).

### 11. Requested analyses, ranked

1. Position-stratified budget curve and primary gain, by k, on development data
(history-falsification cache, 1,191 PRISM conversations; wildchat-replication
`cache/development`, 599 conversations). Rules in or out: structural saturation (B1, M4).
Cost: minutes; cached embeddings; one loop over `budget_windows` restricted by `target_index`.
Respects spent samples. (The held-out per-position primary gains can additionally be reported
from the committed `per_case` scores without touching embeddings, as a declared post hoc
descriptive.)

2. Opening-exchange-plus-last-two baseline (and initial-only-plus-last-two) against the mean,
by position, on development data. Rules in or out: whether anything beyond the opening
exchange and the latest exchange is predictive (M1). Cost: one new `PoolKind` in
`cmri_history.analysis.pool_case`; minutes. Dev only.

3. Random-subset curve: mean of n randomly chosen history messages against trailing window n,
same n, development data. Rules in or out: recency information beyond sample count
(exchangeability). Cost: minutes. Dev only.

4. Synthetic own-AR(1) process (recency-only dependence) and B under the `ci_low > 0` rule
(m1). Rules in or out: whether the pipeline separates recency-only from older dependence, and
the false-positive rate of the rule actually used. Cost: numeric only, no data.

5. Case-level estimates and trimmed means for the primary (m5, m6), from the committed
per-case scores. Cost: trivial. No embeddings touched.

6. Ranking test that ranks by the same quantity as the primary (pooled cosine, no ridge) and a
count of queries where long and short rankings differ, development data
(history-specificity cache). Rules in or out: whether the E5 null is a resolution failure
(M3). Cost: small.

7. Conversation-level bootstrap and opening-hash merging of WildChat components, development
sample. Rules in or out: component-construction optimism (1f). Cost: small; needs development
text for the hash, which is already decoded in that repository's workflow.

8. Common-prefix interaction by position and with token-matched targets, development data
(m8). Cost: small.

9. A third encoder family on development data only (for example the program's
nomic-embed-text). Rules in or out: instrument dependence of the gain and of the levels.
Cost: re-encoding 1,191 + 599 conversations; hours at most. Must not be run on the held-out
samples as a confirmation.

10. Public (or time-stamped, deposited) repositories before submission (M5). Cost: none.

11. In the revelation paragraph, report the null band and the observed exponent's position in
it from `diagnostics.csv` (m2). Cost: none.
