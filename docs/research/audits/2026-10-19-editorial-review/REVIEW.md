# Editorial review of the ISCS 2027 long paper, the NetSci 2027 abstract and the F3 design

Date: 2026-10-19. Reviewed: `papers/iscs-2027/paper/main.tex` at commit 1756f23 (twelve pages),
`papers/iscs-2027/abstracts/netsci-2027/main.tex`, and
`docs/superpowers/specs/2026-10-19-f3-integration-coding-design.md`.

Method: four independent reads, each briefed without the others' conclusions, plus the
synthesizing session's own checks on development embeddings. The component files in this
directory are the evidence; this file is the verdict.

| File | Role |
|---|---|
| `referee-methods.md` | Reviewer 2: validity, statistics, freeze credibility; 44 numbers checked against result files |
| `editor-framing.md` | Editor: research questions, contribution sizing, venue fit, F3, programme direction |
| `prior-art-dialogue.md` | Prior art in dialogue systems, embedding geometry, human-LLM logs, pre-registration (30 verified entries) |
| `prior-art-networks.md` | Prior art in graph sampling, densification, temporal-network nulls, similarity networks (25 verified entries) |
| `pooling-diagnostics.md`, `pooling_diagnostics.py` | The synthesizing session's diagnostics on PRISM and WildChat development embeddings (held-out samples untouched) |

Verified by the synthesizing session, not taken on trust: the pre-freeze rule change
(commit 98f5ff1, `git show`), the degenerate E5 ranking intervals
(`history-specificity/derived/contrasts.csv`), the densification null band
(`temporal-networks-calibration/pilot/results/diagnostics.csv` rows 11 and 12), the synthetic
detection rule (`history-falsification/src/cmri_history/synthetic.py:178`), the two
load-bearing 2026 citations (arXiv 2605.07345 and 2602.24287, abstracts opened), the
Springer chapter metadata (Crossref), and all development-data numbers below (recomputed
with the pipeline's own pooling, aggregation and bootstrap; the replica reproduces the four
published development values to four decimals).

## Verdict

**Major revision before submission. The measurement and the discipline are sound; the story
told about them is wrong in three places, and the paper's methodological asset cannot be
inspected.** Nothing found here reverses the frozen result: the pre-registered contrast is
positive with intervals above zero on both held-out samples, under both encoders, and also
under the pre-registered centered-cosine secondary, which is the version that survives the
artifact described below. What changes is what the paper may say the gain is, which metric
carries the claim, how much of Section 5 is new, and what a reviewer can verify.

The reframed paper is a better paper. Its honest headline is not "the whole conversation
predicts the next message" but "the next user message is anchored by the person's previous
two turns, and nothing measurable beyond them; raw cosine overstates what history adds, and
the observation-process nulls that caught our own earlier claims are the field's, not ours".

## What holds

- The primary contrast is computed as described; every one of 44 checked numbers matches the
  result files (one cosmetic "$-$.000"). The two held-out runs are genuine single-shot runs.
- The gain is positive at every target position in both collections (referee B1, computed
  from the committed per-case scores: PRISM MiniLM +.026/+.023/+.034/+.023 at positions
  4/6/8/10; WildChat MiniLM +.012/+.036/+.046/+.056).
- The gain survives centering on both held-out samples: PRISM +.022 [.015, .029] and
  +.017 [.011, .023]; WildChat +.015 [.007, .023] and +.012 [.004, .019]. These were
  pre-registered secondaries, so promoting them costs no credibility if the reason is stated.
- The disclosure of the WildChat amendment, the 806-of-828 decode and the runner edit is
  complete and accurately worded (referee m10).
- The forecast-origin control does what the text says and the softened wording ("depends
  on what each forecast sees") is right.

## Findings by severity

### Blocking

**1. Raw cosine rewards pooling more vectors regardless of content, by the size of the
headline effect.** On development embeddings, pooling j messages drawn from OTHER
participants' conversations, against pooling two such messages, raises raw cosine with the
real target by +.020 (MiniLM) and +.031 (E5) on PRISM and +.031 and +.033 on WildChat.
Centering removes it (all four centered donor contrasts cover zero). In E5, adding unrelated
vectors to the real last-two window raises raw cosine (+.013 PRISM, +.010 WildChat), so an
E5 raw-cosine "history gain" cannot be read as information. The mechanism is anisotropy:
single E5 messages have cosine .87 with the user centroid, MiniLM .22, and a mean of more
vectors moves toward that centroid. The literature predicts exactly this (Mitra and Kumar,
arXiv 2605.07345, Prop. 1; Ethayarajh 2019; Timkey and van Schijndel 2021), and synthetic
process B, being isotropic, cannot reproduce it, so the paper's own null is miscalibrated to
its instruments. Two consequences: the "61% to 97% of components" sentence is an artifact of
the raw metric (under centering the held-out fractions are .67 and .67 on PRISM and .59 and
.59 on WildChat for MiniLM and E5 alike), and the paper must either make centered cosine
the reported primary, with the pre-freeze raw rule stated as the gate it was, or report the
raw primary beside the donor null in the abstract. Evidence: `pooling-diagnostics.md`
finding 2; `prior-art-dialogue.md` C-1, C-4, C-6, objection 1; `referee-methods.md` 1(g).

**2. The saturation sentences are forced by the target-position mixture, and the gain
reduces to one more user turn.** Half the PRISM targets (775 of 1,482) and 40% of the
WildChat targets have exactly four prior messages, where every window of four or more IS the
full mean, so "mostly reached by four messages" and "beyond six the curve changes by at most
.001" (main.tex:30, :88) are properties of the sample, not of conversations (referee B1).
Stratified by position on development data the picture is sharper than either the paper or
the referee states: adding only the user's previous message to the last two matches or beats
pooling everything (FIRST+WIN2 minus WIN2: PRISM +.031/+.018, WildChat +.030/+.015, against
MEAN minus WIN2 of +.027/+.019 and +.027/+.017), the mean minus that three-message recipe is
zero or negative in all four cases, and at positions 6 to 10 the user's previous turn does as
well as the opening, so the opening is not special. The mean minus a four-message window is
at most .016 and on WildChat not distinguishable from zero under centering. The last two do
beat a random prior pair of the same role mix by +.018 to +.024 centered (the
within-conversation shuffle null the editor asked for), so recency carries real information
and the result is not pure topic averaging. The supportable statement is: random pair <
last exchange < last exchange plus one earlier user turn, with nothing measurable beyond.
Evidence: `pooling-diagnostics.md` findings 1 and 3; `referee-methods.md` B1, M1, M4.

**3. Section 5's revelation null is prior work, uncited.** Pedarsani, Figueiredo and
Grossglauser (SIGMETRICS 2008, "Densification arising from sampling fixed graphs") state the
thesis in the same words; Leskovec and Faloutsos (KDD 2006) showed random-node samples of a
final graph reproduce the densification law; the identity E[M_k] = M k(k-1)/(N(N-1)) is the
pair-inclusion probability of simple random sampling (Frank 1978; Kolaczyk 2009, ch. 5).
Leskovec, Kleinberg and Faloutsos (2005, 2007), who defined the law, are also absent. The
manuscript calls this an "exact result" and checks it "by exhaustive enumeration". A
network-science reviewer will stop here. The one genuine point, that vertex revelation gives
asymptotic exponent 2 for every endpoint graph where edge sampling depends on the degree
tail, is not made. Evidence: `prior-art-networks.md` C6a, entries 1 to 3 and 7, objection 1.

**4. The freeze cannot be verified by anyone but the owner, and one pre-freeze change is
undisclosed.** Both study repositories are private (checked with `gh repo view`). The
success rule was weakened from "lower bound above .01 in both encoders" to "lower bound
above zero" in commit 98f5ff1 at 07:09 on 18 October, 24 minutes before the freeze commit
and the reserve opening; the paper says only that "a reporting floor of .01 did not gate".
Both confirmations pass the stricter rule (lower bounds .0185/.0165 and .0136/.0139), so
nothing turns on it, but a reviewer who finds it after the fact will not read the rest
charitably. "Pre-registered" also has a registry meaning that a private git push does not
meet. Evidence: `referee-methods.md` M5; `editor-framing.md` Sect. 7.

### Major

**5. Three abstract claims the body does not support.** "Reverses to a small positive value
on PRISM" (the body disowns the reversal); "the cosine gain does not carry over to ranking"
(the E5 ranking test is degenerate, every interval exactly [0, 0], and the MiniLM test
compares model classes on 79 queries near ceiling); "mostly reached by four messages"
(finding 2). Evidence: `referee-methods.md` M2, M3, B1.

**6. The densification sentence under-reports the authors' own diagnostic.** The archive's
all-arrived exponent 1.559 lies BELOW the uniform-revelation null band [1.685, 2.329]
(p = .02) and the active-node exponent 1.405 lies inside its band; the paper reports only
"1.56 against 1.41". The honest sentence is that the archive densifies less than random
revelation on one denominator and indistinguishably on the other, which is the fact that
bears on the earlier claim. Evidence: `referee-methods.md` m2; diagnostics.csv rows 11, 12.

**7. The framing leads with the programme's weakness and names no result.** The first
paragraph is a confession, the title names three nouns and no finding, the abstract opens
with a question and closes with a checklist, and the Related Work says twice that the result
is not new. The editor's strongest honest framing (field-level problem first, the programme
as one instance in one clause) and ten concrete edits are in `editor-framing.md` Sect. 5 and 10.

**8. Related work misses four literatures.** Embedding geometry and anisotropy (Ethayarajh
2019; Mu et al. 2018; Timkey and van Schijndel 2021; Steck et al. 2024; Mitra and Kumar
2026), human-LLM log analysis (Huang et al. 2026, arXiv 2602.24287: the most recent exchange
often matches full context for response quality, 36.4% of user turns self-contained; Laban
et al. ICLR 2026), chunking and compositionality (Günther et al. 2024; Guo et al. 2025), and
NLP pre-registration (van Miltenburg et al. NAACL 2021; Søgaard et al. EACL 2023; PMLR 148).
Two sentences are wrong as written: "their shared lesson" (main.tex:45; neither cited paper
states it) and "a common object of complex-systems study" (main.tex:36; the citations are
NLP). Evidence: `prior-art-dialogue.md` Sect. 3; `prior-art-networks.md` Sect. 3.

**9. The development estimate is a selected maximum and the paper never says so.** The
contrast was the surviving lead from a grid of 29 pooling recipes on the same development
data that supplied the pre-registered values; the 21% to 26% shrinkage on held-out data is
what selection predicts. Evidence: `referee-methods.md` M6.

**10. The "reporting requirement" is assembled from existing requirements.** Gauvin et al.'s
naming convention, Holme and Saramäki's 2012 vertex-entry warning, Kolaczyk's sampling
designs and Olteanu et al.'s bias catalogue each own one item. Present it as assembled and
cite per item, or cut it. Evidence: `prior-art-networks.md` C7, objection 3.

### Minor (fix in passing)

Synthetic detections used `ci_low > .01` while the frozen rule is `ci_low > 0`, so B's
"0 of 100" is for the wrong rule (referee m1). The two-encoder rule's size is about .025
one-sided, not .025 squared, since both encoders score the same conversations (m4). The
estimand is a component mean; on WildChat the case-level mean is .031 against .021 (m5).
Four in ten individual forecasts are worse under the mean (m6). The family-by-year cells are
post hoc and should lose their numbers (m7). The common-prefix control also confounds target
truncation (m8). Table 3 prints "$-$.000" (m9). The packaging numbers are from 100 fit
conversations and should say "development" (m3). NetSci abstract: delete the first sentence,
and its reference list has no densification or sampling reference at all.

## Research questions, experiments and prior art, in one paragraph each

**Questions.** RQ1 as posed ("does the whole prior exchange predict better than the latest")
is answerable, answered, and nearly trivial: the sign is predicted by pooling geometry and
the size by a static-topic process. The question the data actually answer is better: how
much of a person's past does their next message depend on, measured so that the metric cannot
manufacture the answer. The answer, about two user turns, replicated across two corpora, is
a statement about the memory length of the process and is unmistakably complex-systems
content. RQ2 (role asymmetry) is a necessary control, not a finding; its reach, that
reported human-versus-bot asymmetries may be visibility artifacts, is unstated. RQ3 (how the
gain is distributed across people) is the question with the most complex-systems content and
is the least asked: 33% to 41% of components show no gain under the valid metric, and the
paper stops at a percentile. RQ4 (observation nulls) is known in the networks literature.

**Experiments.** Well executed, with three design flaws. Raw cosine as the primary metric
(finding 1). A last-two baseline whose gap to the mean is closed by one more user turn, so
the budget curve measures sample composition, not history (finding 2). Synthetic nulls that
are isotropic where the instruments are not, and gated by a rule stricter than the frozen one.
Smaller: two 384-dimensional encoders scoring the same conversations are one instrument and a
half; the ranking test has no resolution in E5; the length axis of the heterogeneity figure is
the budget curve restated.

**Prior art.** The frozen two-corpus measurement itself has no precedent found. Its sign and
profile do. The revelation null is Pedarsani 2008 with vertices in place of edges. The
packaging identity is an identity, and chunk sensitivity is documented. Pre-registration in
NLP has a literature the paper does not cite. "Cognitive" collides with cognitive network
science, where nodes are concepts validated against behaviour; the programme's object is a
document-similarity network over conversations and should be called that.

## Recommended revision, in order

1. **Make centered cosine the reported primary, state why, and report the donor null beside
   it.** The centered secondaries were pre-registered and pass; the paper says the raw rule
   was the gate and that a development diagnostic run after the confirmations showed the raw
   metric rewards pool size. Cost: no new data; one paragraph; `pooling_diagnostics.py`
   already produces the numbers on development caches. (Findings 1, 5.)
2. **Replace the budget-curve claims with a position-stratified analysis and the
   three-message recipe.** Report, on development data by position and on held-out data from
   the committed per-case scores: mean minus last two; last two plus previous user turn minus
   last two; mean minus that; last two minus a random prior pair. State the result as random
   pair < last exchange < plus one earlier user turn < nothing beyond. Retitle around it.
   Cost: minutes of compute, half a page of text; the page is bought by cuts 4 and 5 of
   `editor-framing.md` Sect. 6. (Finding 2.)
3. **Rewrite Section 5's revelation paragraph as a cited special case and report the null
   exponent.** Cite Pedarsani 2008, Leskovec and Faloutsos 2006, Leskovec et al. 2007, Frank
   1978 or Kolaczyk 2009; derive the identity in two lines from exchangeability; say which
   denominator gives which exponent; give the null band and that the archive lies below it on
   one convention. Drop "exact result" and the enumeration sentence. Cut the packaging
   paragraph to two cited sentences. (Findings 3, 6, 10.)
4. **Disclose the rule change and make the freeze inspectable before submission.** One
   sentence at main.tex:69. Deposit PROTOCOL.md, FREEZE.json, the pins, both REPORT.md files,
   the results JSON, the sample manifests and the push-log evidence on Zenodo or OSF, cite
   the DOI, and make both study repositories public (or a frozen snapshot) at submission,
   not camera-ready. Use "frozen protocol" throughout and "pre-registered" only if a registry
   entry exists. Release the per-component gain tables as CSV. (Finding 4.)
5. **Reframe and retitle.** Apply the editor's ten edits, starting with the opening paragraph
   and the abstract's first sentence; move the programme's history to one clause; state the
   descriptive reading the secondaries support (a person's next message is anchored by their
   own earlier turns more than by the assistant's reply). (Finding 7.)
6. **Fix Related Work.** Add the four literatures; correct the two wrong sentences; narrow the
   pre-registration claim to "first frozen, replicated embedding-space contrast on human-AI
   logs to our knowledge". (Finding 8.)
7. **Weaken the ranking sentence, name the winner's curse, fix the minors.** (Findings 5, 9,
   minor list.)
8. **Spend recovered space on RQ3.** Shape of the per-component distribution under the
   centered metric, by collection, against the shuffle null; whether non-positive components
   are a tail or a mode. This is the complex-systems paragraph the paper lacks.

Decision point the owner must take: whether to submit the twelve-page paper with the three
observation nulls condensed to one cited Discussion paragraph (the editor's recommendation,
which I share), or to keep Section 5 and accept that it reads as a comment on the authors'
prior work attached to a prediction study. The NetSci abstract then has no network content
and should either lead with the growth null applied to real conversation networks (with the
null exponent and the calibration's return-lag result) or be expected to go to a poster.

## F3 and the programme

The integration study is the right next question for the programme's standing, with three
changes to the design: the cross-participant PRISM graph is the wrong construct for a bridge
score (a bridge was a conversation linking two of one person's communities; replace it with a
within-conversation, construct-matched score from cached message vectors, committed before
coding); the kappa gate as written cannot pass at plausible prevalence on 80 items (gate on
the point estimate with the interval reported, or go to 120 to 150, and add a non-author
third coder on a 30-item subset); and the design should say the middle outcome is a
workshop note, not a paper. Details in `editor-framing.md` Sect. 8.

Two cheaper experiments outrank F3 for return on effort. First, PRISM's own labelled outcome:
whether the pooled history predicts the reply the participant CHOSE over the rejected
candidates better than the last two do. The rejected candidates are already encoded in the
wave-1 development cache (`*__alternatives` arrays), so this is a day's work on development
data, labelled exploratory, and it is the first behavioural outcome the gain could have.
Second, the multi-archive network replication: build the conversation-similarity network for
twenty public hashed-IP archives and report modularity and bridge fraction against
configuration and revelation nulls. That is the paper that answers the PLOS objection on its
own terms, and the only one of the three that is unmistakably a complex-systems paper.

Stop lists and the P01 to P12 triage are in `editor-framing.md` Sect. 9; I agree with them.

## What this review did not do

No file outside this directory was edited. No held-out sample was read for a new analysis;
the only held-out numbers used are the committed result files. The dialogue sweep could not
open OpenReview, dblp, Semantic Scholar or several publisher pages, so venue confirmation for
a few entries rests on arXiv headers, as marked in that file. The Springer chapter's volume
number 1265 could not be confirmed from Crossref (title, series, pages 315 to 327, year 2026
and the 25 February online date were).
