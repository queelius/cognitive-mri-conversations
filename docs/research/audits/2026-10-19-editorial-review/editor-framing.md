# Editorial review: significance, framing and venue fit

Date: 2026-10-19. Role: editor assessing significance and framing, not validity. Statistics and
prior art are covered by other reviewers and are not duplicated here. Read-only review; no
paper or programme file was changed.

Files read: `papers/iscs-2027/paper/main.tex` (twelve pages, confirmed with `pdfinfo`; page 12
is half references) and its four rendered figures; `papers/iscs-2027/abstracts/netsci-2027/main.tex`;
the two READMEs; `future-ideas/iscs-2027/README.md`, `EVIDENCE.md`, `PILOTS.md`,
`research-deep-dive/{VENUE,PAPER-CATALOG-AND-OPTIONS,EMPIRICAL-PAPER-OPTIONS}.md`,
`extended-abstract/{README,CONTRIBUTION,main.tex,NEXT-GOAL}.md`;
`docs/research/{DECISIONS,EXPERIMENT-PORTFOLIO,README}.md`; the two superpowers specs; the root
`CLAUDE.md`. Calls checked on the web: ISCS 2027 home and submissions pages, NetSci 2027 site and
call for abstracts (links at the end). Line numbers below refer to `main.tex`.

## Verdict

**Major revision on framing. The evidence can stand; the paper built around it cannot yet.**

The two frozen confirmations are real and the disclosure is exemplary. But the manuscript
(a) opens by telling two conference reviewers that the authors' prior work was unsound, (b)
carries a title and an "exact results" claim that promise more than the content, (c) staples a
borrowed methods coda onto a measurement study, and (d) never confronts head-on the one
reduction every sceptic will reach for: that averaging more vectors drawn around a fixed
conversation topic must land closer to the next same-topic vector. The paper concedes this
reduction in Sect. 5 (process A) and in the Discussion, but it does not test it, and it does not
say what positive thing the data do support. The pre-registration story, which is the paper's
main methodological asset, cannot be inspected because both study repositories are private
until camera-ready.

The two biggest issues:

1. **The main result is, by the paper's own synthetic process A, compatible with pure
   topic-centroid variance reduction, and the paper neither tests the obvious null (a
   within-conversation shuffle of message order) nor states the positive reading the
   secondaries most directly support (the user's own earlier messages, not the assistant's
   latest reply, anchor the next message).** As written, a reader can reduce the paper to
   "the mean of n same-topic vectors is closer to the next same-topic vector than the mean of
   two", and the paper has pre-agreed.
2. **The framing puts the programme's weakness in front of the finding.** The first paragraph
   is a confession, the title names three unrelated nouns, the abstract's last sentence is a
   reporting checklist, the revelation identity is called an "exact result", and the
   repositories that would make "frozen protocol" mean something are closed. Each of these is
   a sentence-level fix; together they decide the disposition of a two-reviewer conference
   paper.

## 1. Research question quality

The manuscript poses one explicit question and three implicit ones.

**RQ1 (explicit, abstract l.30):** "Does the whole prior exchange predict what a person will say
next to an AI assistant better than the latest exchange does?"

- Needed by a complex-systems audience? Marginally. It calibrates what a conversation archive
  can be made to show, which matters because the ISCS and NetSci calls both list AI-mediated
  platforms and cognition as scope. It is not itself a complex-systems question.
- Surprising? No, and the paper says so twice: "That context beyond the latest utterance helps
  is not new" (l.43) and "We claim no priority for the idea that history matters" (l.43). The
  answer is confirmatory; the news is the size (.02 cosine), the saturation (four messages)
  and the replication.
- So what if true? The paper's own answer is a methods lesson: "the measurement is confirmed;
  its interpretation as memory is not" (l.186). That is honest and it is thin.
- Sceptic's reduction: "people stay on the topic they opened with, and a mean of more
  same-topic vectors is a better estimate of that topic than a mean of two." Does the paper
  pre-empt it? Only by concession. Process A "draws a fixed topic per conversation,
  independent user innovations around it and a reactive assistant, so conditional on the topic
  it has no dependence on older user messages; it nevertheless reproduces a large history gain
  ($+.120$)" (l.157), and the paragraph closes "The empirical pattern of Sects. 4.1 and 4.2 is
  compatible with both the absence and the presence of older-state dependence." The budget
  curve (Fig. 1, lower) saturating by window 4 to 6 is exactly the curve of a sample mean's
  error shrinking in n. The paper never says this in those words, and it never runs the
  cheapest test that would separate sequence from averaging: shuffle message order within
  each conversation and recompute the contrast. If the shuffled gain equals the observed gain,
  the result is variance reduction and should be presented as such; if it does not, the
  difference is the first sequence-specific number in the paper. The ISCS 2026 reviews of the
  hierarchy paper asked for null models; this is the null model for this paper.

**RQ2 (implicit, Sect. 4.2 title):** does the apparent role asymmetry reflect the roles or what
each forecast sees?

- Needed? Yes for anyone who compares user and assistant turns, and the related-work
  paragraph names a literature that does (ChatbotID, "with the human--bot asymmetry central to
  its motivation", l.45).
- Surprising? Once stated, no: the assistant's reply answers the request, so a predictor that
  contains the request wins. The paper's own first sentence of 4.2 gives the game away: "The
  pilot that fixed the primary contrast also found what looked like a role effect" (l.121). The
  reader is being shown the correction of a lead that was never published.
- So what? The real so-what is not stated: that human-versus-bot asymmetries reported
  elsewhere could be visibility artifacts of the same kind. Said that way, RQ2 has reach beyond
  the authors' pilot.

**RQ3 (not posed as a question; Sect. 4.3 opens "The mean gain is a population average over a
wide distribution", l.134):** how is the gain distributed across components?

- This is the one question with genuine complex-systems content, and the paper's own design
  document knew it: "A heavy tail here is a real complex-systems result; a tight unimodal is
  also worth one figure" (`2026-10-18-iscs-2027-paper-design.md`, F2). The manuscript reports
  "in MiniLM 39\% of WildChat components show no gain or a loss" (l.134) and the 10th/90th
  percentiles, then stops. No shape, no null for the spread, no attempt to say what predicts a
  component's gain beyond a survey join that "at this size is not evidence of none" (l.143).
  The question is asked by the figure and left unanswered by the text.

**RQ4 (implicit, Sect. 5 title):** do observation choices resemble dynamics?

- Needed? The temporal-networks community already knows this (Kobayashi, Gauvin, both cited).
  The authors' own planning file calls the revelation identity "an explanatory identity, not a
  novelty claim" (`future-ideas/iscs-2027/README.md`, proposal 1).
- So what? "a reporting requirement: state the unit retained, when each feature became
  available, the population denominator, the permitted prefix, and what each control
  preserves" (l.168). A checklist is a fine coda; it is not a contribution a programme
  committee will weigh.

Summary: RQ1 is answered and nearly trivial, RQ2 is a necessary control dressed as a finding,
RQ3 is the interesting question and is under-asked, RQ4 is familiar. The paper's strength is
rigour, not the questions. The fix is not new data; it is to pose RQ3 as the question, run the
shuffle null for RQ1, and state RQ2's reach.

## 2. Contribution sizing

The paper lists its contribution as "threefold: a confirmed and replicated measurement ...;
evidence that the companion claim ... depends on what each forecast is allowed to see ...; and
two exact results showing that message packaging and the order in which a fixed graph is
revealed manufacture apparent dynamics on their own" (l.40).

**Is this one paper or three?** It is one measurement paper with a borrowed coda.

- The replicated gain, its heterogeneity, the forecast-origin control and the synthetic
  processes are one study: they are the same contrast, the same samples, the same bootstrap,
  and the control and the processes exist to say what the gain does and does not mean. Keep
  them together.
- The ranking null (l.166) belongs to the same study (it bounds usefulness) but is
  development-only, calibration-failed ("the donor control failed its own calibration gate",
  l.166) and 79 conversations. It earns four sentences, not a third of a page.
- The packaging paragraph (l.147) and the revelation identity (l.149 to 155) are about two
  other measurements this paper does not otherwise make: token-clock pooling (the dynamics
  drafts) and conversation-network growth (the rejected temporal paper). They arrived from the
  extended abstract *When Observation Choices Resemble Dynamics*, whose own contribution memo
  says "It does not satisfy the earlier preference for a new independently supported
  behavioral discovery. That distinction is why the recommendation is an extended abstract,
  not an 8/12-page paper" (`extended-abstract/CONTRIBUTION.md`). They are stapled on. The only
  number in the paper from the private single-author archive is here ("1.56 against 1.41 on
  our author archive", l.155), which quietly reintroduces the data the paper was written to
  escape.

**Which is the real ISCS contribution?** The frozen two-corpus measurement with its component
distribution and the generative bracketing (processes A, B, C). The forecast-origin control is
a required secondary. The packaging and revelation material should be cut to one short
Discussion paragraph that points to the temporal revision and the dynamics paper, or moved
there entirely. The "reporting requirement" can survive as one sentence of the conclusion, but
it should not be the abstract's last sentence; as it stands, the paper's final word is a
checklist.

**Is the title honest and specific?** Honest, not specific. "What the Whole Conversation
Predicts: History, Forecast Origin and Observation in Human--AI Dialogue" (l.18) names no
result, no data, no design, and its three subtitle nouns advertise the staple. A reader cannot
tell from the title that there is a replication, a size, or two public corpora. A declarative
title that states the result and the design serves this paper better (edit 3 below).

**Is the abstract's first sentence the right opening?** No. A rhetorical question whose answer
is "yes, slightly, and it is not usable" reads as a setup. The abstract should open with the
result and its size, because that is what distinguishes this paper from the literature that
already says history helps.

## 3. Audience fit

**ISCS 2027.** The call lists, verbatim, "Complex Networks & Graph Machine Learning",
"Computational Social Science & Digital Platforms", "Data Science, AI & Simulation for Complex
Systems" and "Neuroscience, Cognition & Linguistics", welcomes "both methodological advances
and interdisciplinary applications", states no review criteria, and promises "at least two
reviewers". Long papers are twelve pages of unpublished work; proceedings are Springer SSDC
(`VENUE.md`). The bar is inclusive and the hierarchy paper was accepted here last year. A
programme committee would most plausibly file this paper under Computational Social Science
and Digital Platforms or Data Science and AI; a reviewer from Foundations or Networks would
say "NLP evaluation methodology with a research-methods coda". Sections 3 and 4 read as
encoder, cosine and MRR work; Section 5 reads as methods; only the ECDF and the synthetic
processes read as complex systems. Acceptance is likely regardless. The question is whether the
accepted paper is the one the programme wants on record.

**NetSci 2027.** The call (netsci2027.github.io/calls/abstracts) lists among its topics
"Multi-Agent and LLM Networks", "Temporal Networks", "Linguistic Networks", "Network Measures
And Metrics" and "Statistical Analysis Of Networks And Inference"; it requires one page with
"one mandatory descriptive figure", says it "especially welcomes work that breaks the field's
current boundaries", requires in-person presentation and limits each author to presenting one
abstract. The submitted abstract (`abstracts/netsci-2027/main.tex`) is two-thirds cosine
contrast and one-third the revelation identity. The cosine contrast is not a network result;
the revelation identity is a textbook one. A NetSci programme committee will ask where the
network is. Either reframe the abstract around network content (the fixed-graph growth null
applied to real conversation networks, with the denominator sensitivity and the better-powered
return-lag statistic from the temporal calibration, which the programme README records as
"24/25 vs 18/25"), or accept that it will be routed to a poster. Also note the one-abstract rule:
if F3 is to be a NetSci abstract later, the same presenter cannot carry both.

**What would make it unmistakably complex systems without faking it?**

- Treat the components as a population and ask RQ3 properly: the shape of the per-component
  gain distribution against a within-conversation shuffle null, by collection and by length,
  and whether the 39% non-positive components in WildChat are a tail or a second mode.
- Promote processes A, B and C from caveat to model. They are already small generative models.
  Fit process A's innovation scale to the observed last-two level, then show which observed
  quantities it does and does not reproduce (the budget curve, the ECDF spread, the length
  trend, the own-versus-partner gap). A memoryless model that reproduces everything is itself a
  complex-systems result; one that fails on the spread or the length trend points at where
  memory would have to live.
- Do not add a network figure to look the part. If the revelation identity stays, apply it to
  one observed growth curve so it is a null in use, not a formula cited.

## 4. Effect size and practical meaning

The paper is candid: "The gain is a gain in raw cosine, not in accuracy: cosine at the last-two
baseline is $.38$ (MiniLM) and $.85$ (E5) on PRISM and $.43$ and $.87$ on WildChat, so the gain is
small against the level and larger against the spread across components" (l.88); "A raw-cosine
advantage ... is a statement about the geometry of what people write next, not about memory,
and on this cohort it is not a ranking advantage either" (l.172).

**How to talk about a small, replicated, practically inert effect.** Three moves, none of which
the paper currently makes:

1. Standardise it once. From the reported percentiles (10th/90th at $-.033$/$+.085$ on PRISM
   MiniLM, $-.083$/$+.120$ on WildChat) the component standard deviation is roughly .045 and .08,
   so the mean gain is about half a component SD on PRISM and a quarter on WildChat (my rough
   reading of the percentiles, to be computed properly). Add the sign rate the paper already
   has (72%, 97%, 61%, 83% positive). "Small, consistent, present in most components" then has
   numbers behind it.
2. Size it against the paper's own ladder. Window 1 sits at $-.05$ to $-.085$ below the last
   two (Fig. 1, lower); the full mean sits $+.02$ above. Adding messages three through twelve is
   worth roughly a quarter to a third of what adding the second message was worth. That is the
   most honest single sentence available about practical magnitude.
3. Say what the gain most directly supports. The secondaries put it in the user's own
   messages: "pooling only the user's own messages beats pooling only the assistant's by
   $+.053$ and $+.096$" (l.117). The paper then refuses any reading: "Neither attributes
   influence" (l.117). Refusing to attribute influence is right; refusing to describe is not.
   The descriptive claim the data support is that a person's next message is anchored by their
   own earlier messages more than by the assistant's latest reply. Say it, hedge it, and the
   reader has a reason to care.

**Overclaiming or underclaiming?** Both. "predicted better by the whole prior exchange" (abstract
l.30, intro l.38, conclusion l.186) is literally true in cosine and will be heard as task
performance, which the paper disclaims three pages later. "History gain" is a usable label if it
is defined once as a raw-cosine difference and never as a benefit. Meanwhile the paper
underclaims by never stating the own-message anchoring, and "practically inert" is itself not
established: the ranking null is one small, development-only, calibration-failed test. The
right phrase is "no usable ranking advantage was found in a small development test;
usefulness is untested at scale", not "inert".

## 5. Negative framing

The paper opens: "Our own programme built semantic similarity networks from one author's 1,908
ChatGPT conversations ... Reviewers of the second paper and of a rejected temporal extension
raised the same objection: everything rested on a single user, and nothing had been confirmed
on unseen data" (l.36). The NetSci abstract opens even harder: "Our reported ``dynamics'' of
human--AI conversation rested on one archive we had already seen" (abstract l.44).

**Is self-correction a strength here?** In a journal with a rebuttal cycle, yes. At a conference
with two reviewers and no rebuttal, the first paragraph decides the disposition, and this one
tells the reviewer three things in order: the prior work was weak, a paper was rejected, and
(paragraph four) the new result "is not new". A reviewer who stops reading at Related Work has
been handed the rejection text. The self-correction also mis-sizes the answer: the objection was
to network-structure claims from one user; the paper confirms a different and much smaller
claim. "This paper answers that objection for one claim" (l.38) is carefully worded and will be
read as "they confirmed something else".

**Strongest honest framing.** Lead with the field's problem, not the programme's: archives of
human--AI conversations are increasingly used to make claims about development, memory and
learning; most such claims rest on one archive and on observation choices never tested on
unseen data; the smallest such claim that can be fixed in advance is whether the whole exchange
or the latest one is closer to what the person says next; here is what a frozen two-corpus
confirmation of that claim yields, and what it does and does not license. The programme's own
history then appears as one instance in a single clause with the two citations, which is all
the reviewer needs and all the authors owe. The point that reviewers objected is true and
belongs in the Discussion under "What would change the conclusion", where it reads as
scientific posture rather than apology. The NetSci first sentence should go entirely; a
one-page abstract has no room for a confession.

## 6. Structure and length

The manuscript is at the limit: twelve pages, with references taking half of page 12. Assume
the statistics reviewer wants one or two analyses at half a page each (the within-conversation
shuffle null is the one I would insist on). The following cuts buy roughly 1.2 pages.

**Cut or condense (in order of return):**

1. Lines 71 to 73 ("Freeze, samples and one amendment") and line 180 ("What the freezes did and
   did not fix") together run about 35 lines of procedural disclosure: "the PRISM freeze reached
   the server ten seconds before the run started, the WildChat freeze seven minutes before"; "a
   test parsed reserve rows structurally (counts only) before it"; "the runner ... was edited
   between freeze and run to make the loss halt binding". This belongs in the repositories'
   deviation logs, which the paper should cite. Keep the amendment (it decided whether a
   confirmation could happen) and the "806 of the 828 ... decoded into memory" admission (it is
   the one that matters), in one "Protocol deviations" paragraph of eight lines. Saves about
   half a page.
2. The packaging paragraph (l.147): eleven dense lines of development-only numbers whose
   role-signed detail ("$-.019$ for MiniLM assistant forecasts, $+.010$ for user forecasts") no
   reader will parse. Reduce to two sentences or move to the dynamics paper. Saves a third of a
   page.
3. Figure 3 (seventeen-row forest plot) with its own caveat "No between-subgroup test or
   multiplicity correction was applied" (l.134) and the confounding note "Family and year are
   confounded in WildChat". Replace with a four-row table (PRISM condition; WildChat family;
   WildChat year; length, both collections) or drop to supplementary, and keep the one sentence
   that matters: every declared subgroup point estimate is positive and the gain rises with
   length. Saves a third of a page.
4. The survey join (l.143): one sentence, or drop.
5. Table 2: the "Attenuation" rows are the difference of two rows above them; the centered
   cosine rows can be one sentence of text. Saves six lines.
6. The ranking null (l.166): keep the result and the bound in four sentences.

**Figures and tables that earn their space:** Fig. 1 (primary contrast plus budget curve) fully;
Fig. 2 upper (role interaction under two origins) fully; Fig. 2 lower (ECDF) fully, and it should
be the figure the text spends the most words on, not the fewest; Table 1 (samples) yes; Table 2
yes, trimmed; Table 3 (synthetic) yes if the processes are promoted to a model, otherwise it
can be four lines of text. Fig. 3 does not earn its space as drawn.

## 7. Reproducibility and openness as a contribution

The paper's methodological asset is the freeze, and the paper is honest that it is
self-administered: "The freezes are self-administered commits whose ordering is evidenced by the
hosting service's push log ..., not by an external registry" (l.180). The repositories are
private; the paper promises them "public with the camera-ready version" (l.188). The umbrella
`CLAUDE.md` confirms every `cmri-*` pilot and study repo is private.

**Does the "pre-registered, frozen" story carry weight when nobody can inspect the freeze?** At
review time, no. A reviewer sees the word "pre-registered" in the abstract, the keywords and
Table 1, and can verify none of it. Worse, "pre-registered" has a specific meaning (a
time-stamped registry entry) that a git commit on a private remote does not meet, and the paper
half-admits this in the Discussion. A pedantic reviewer will say so. The paper should say
"frozen protocol" throughout and reserve "pre-registered" for a registry entry, or earn the word.

**What to release, and when:**

- Before submission (March 1): deposit `PROTOCOL.md`, `FREEZE.json`, the dependency pins, the
  two `REPORT.md` files and the results JSON on OSF or Zenodo with the push-log evidence
  attached. A time-stamped DOI issued after the fact does not make the freeze prospective, but
  it makes the ordering publicly auditable, which is what the paper currently cannot offer.
  Cite the DOI in the paper.
- At submission, not camera-ready: make `cmri-iscs-2027` and `cmri-wildchat-replication`
  public, or at minimum a frozen anonymised snapshot of each. ISCS gives reviewers four weeks;
  a promise to open later is worth nothing to them.
- Release the sample manifests. Both PRISM and WildChat are public datasets; the conversation
  identifiers and hashed IPs that define the samples are already public and withholding them
  protects nothing while blocking reconstruction. The README's "pinning ... the sample only
  afterwards by a digest of its identifiers" lets a reader verify a manifest they already have,
  not obtain one. The WildChat salt, or the resulting identifier list, has to be published.
- Release the per-component gain tables (no text) as CSV so that RQ3 can be re-analysed by
  others. This is the single most useful artifact for a complex-systems reader.
- Keep the pilots they pin (`portfolio-wave1`, `history-falsification`, `semantic-measurement`)
  public too, or the third-party rerun in `README.md`, which needs them "at these commits",
  cannot be performed.

Openness could be a contribution of this paper; at present it is a promise.

## 8. The F3 design

**Is "integration" the right next question?** For the programme's standing, yes, with one
reservation. The programme's identity is the bridge taxonomy and the claim that some
conversations integrate ideas; no human has ever labelled any of it. The design says so:
"Every reviewer line of attack on the programme's behavioural vocabulary ('integration',
'bridge', 'revision') comes back to the same gap: no human has ever labelled any of it." Doing
this on public PRISM data also addresses the single-user objection. The reservation is base
rate: PRISM conversations are capped at twelve messages, participants have about five
conversations each, and the pilots found that "integration as written is satisfied by ordinary
explanatory answers" and that same-premise revision occurred in "0 clear cases in 100 windows".
Genuine integration of two distinct ideas in a six-to-twelve-message PRISM chat is likely rare.

**Kappa decision rule and sample size.** Not sensible as written, for a reason the design itself
records: "the component-clustered interval on a binary kappa at n=80 is wide (roughly plus or
minus .2 at kappa .6)" and "base-rate skew would depress kappa". With a plausible prevalence
of ten percent, eighty items yield about eight positives; the lower bound of a binary kappa
will not clear .4 even with good coders, so the rule "binary kappa lower bound above .4 and point
estimate at least .6" is set up to return the middle outcome ("the reliability result is the
paper") regardless of coder skill. And a kappa of .5 on eighty items is not a paper anywhere;
the design should say the middle outcome is a workshop note. Fixes, in order of value: (1)
stratify the draw, half from conversations with a high label-blind topic-count proxy and half
random, estimate prevalence from the random half only, and raise the positive count; (2) gate
on the point estimate with the interval reported, not on the lower bound, at n=80, or go to
n=120 to 150 if the lower-bound gate is kept; (3) add a third, non-author coder on a thirty-item
subset, which does more for the independence story than twenty extra items coded by two
authors who have both read the pilot reports. The design's disclosure of that exposure is
right, but a reviewer will still call the result "author-coded".

**Building the label-blind network score before reliability is known.** The pre-registration
instinct is right; the score is wrong. The design proposes "betweenness centrality in the giant
component and the number of distinct Louvain communities among its neighbours" on a graph of
"the 1,191 development conversations" from 1,500 different participants. A bridge in a
cross-participant similarity graph is a conversation resembling other people's conversations on
several topics; that is not the construct. The original bridge was a conversation linking two
of one person's knowledge communities. The programme's own catalogue says it: "A global
public-corpus graph is not a personal archive network." A null AUC would say nothing about the
bridge taxonomy, and a positive AUC would be uninterpretable. Replace it with a
within-conversation, construct-matched score that costs an hour: from the cached message
vectors, whether the conversation's messages form two or more distinct clusters, and whether
some later message lies close to both (a "bridging message" score). Commit its hash before
coding. That makes the pre-registration meaningful and answers the design's question 5: build
the cheap, matched score first; do not build the cross-participant graph at all.

**Two alternatives that would do more for standing after PLOS.**

1. *Replicate the network claims across many public archive proxies.* The PLOS objection was
   to network structure from one user; this paper confirms a non-network claim on public data,
   so the programme's defining claims (communities, modularity 0.75, bridge fraction) still rest
   on one archive. The temporal calibration already has the plumbing for public hashed-IP
   archives and a better-powered return-lag statistic; the ISCS paper excluded "IPs with more
   than 100 eligible conversations" (l.51), which is exactly the set that resembles personal
   archives. Build the Cognitive MRI network for twenty such archives, report the distribution
   of modularity and bridge fraction against configuration and revelation nulls, and the
   single-user objection is answered on its own terms. This is also the only option that is
   unmistakably a complex-systems paper (an ensemble of networks, null models, heterogeneity).
2. *Use PRISM's own labelled outcome.* PRISM records which candidate reply each participant
   chose, and per-turn ratings. The paper already conjectures that "the chosen reply may be
   the one that best continued the whole exchange" (l.174) and already has the rejected
   candidates in one of its four views (l.69). A frozen test of whether the whole-history mean
   predicts the chosen reply over the rejected ones better than the last two do turns a
   geometric gain into a behavioural one (choice) with thousands of labelled instances and no
   human coding. It is cheap and it directly answers "is the gain usable". If no unopened PRISM
   components remain (the paper used 233 plus 360 components of a 1,500-participant dataset;
   check what the eligibility filter left), it is exploratory on the development partition and
   must be labelled so.

Recommendation: run alternative 2 now (days, not weeks), run F3 with the fixes above, and plan
alternative 1 as the next full paper.

## 9. Programme direction

Reading `DECISIONS.md` (the owner decided the twelve-page paper, the NetSci abstract and F3 on
October 19) and `EXPERIMENT-PORTFOLIO.md` (P01 to P12), with the programme README's own
warning that "The campaign set its 'full paper' bar at journal-of-record level ... Treat those
verdicts as conservative":

**Worth pursuing after this paper:**

- **P01 (validate the integration distinction):** this is F3; pursue with the sample and score
  fixes in Sect. 8.
- **P02 (what the network adds):** only after P01 passes and only with a construct-matched
  within-conversation score; the cross-participant graph version is not worth building.
- **P08 (static versus temporal information):** worth pursuing in one form only, the
  multi-archive network replication (alternative 1 above, which is also F4). Dropping the
  beta estimand for the return-lag statistic is already decided; the paper to write is the
  ensemble one.
- **P09 (reconstruction at matched cost):** cheap, and the dynamics consolidation needs one
  empirical anchor; a methods note, not a finding. Low priority but not sunk.
- **Not in P01 to P12:** the PRISM choice-outcome test (alternative 2). Cheapest high-value
  experiment on the table.

**Sunk-cost continuations (stop):**

- **P04 (challenge, revision, reformulation sequences):** two pilots and an event assessment
  found "Zero clear same-premise AI-drafted revisions" in 100 windows and a no-go. Stop on this
  source.
- **P07 (independently defined episodes):** synthetic-only; the ThoughtTrace pilot found
  "sequence-trajectory features are reliably worse than pooling". Without an annotated episode
  set there is nothing to detect. Stop.
- **P10 (richer dynamics, inertial states):** "Added role-history/inertial features ... did not
  clear practical promotion gates". Stop; fold what survives into the dynamics consolidation as
  a negative.
- **P03, P05, P06:** completed by this paper (own-versus-partner, the history gain, packaging).
  Further runs are polishing, not research.
- **P11 (CoAuthor) and P12 (KITE, long-gap reuse):** data-gated with unresolved terms and
  linkage after two audits. Defer without investment; do not let them absorb a third audit.
- The "two-platform, one-person" idea in the design's brainstorm (Claude Code and Codex logs)
  does not fix single-user and should not be sold as if it did.

The strategic hole is unchanged by this paper: the claims that define the programme have never
been shown on more than one person's archive. Alternative 1 is the paper that closes it.

## 10. Ten edits, in priority order

1. **Replace the Introduction's opening (l.36, "Archives of conversations with AI assistants are
   now a common object of complex-systems study. Our own programme built ... nothing had been
   confirmed on unseen data.") with a field-level opening:** "Claims about how human--AI
   conversations develop, about memory, learning and the growth of knowledge, are increasingly
   drawn from conversation archives, usually one archive, with measurement choices that are
   rarely tested on unseen data. We take the smallest such claim that can be fixed in advance:
   whether the whole prior exchange or the latest one lies closer, in embedding space, to what
   the person says next. We froze the contrast, ran it once on held-out samples from two
   corpora collected by different groups, and report what the confirmation does and does not
   license." Move "Our own programme ... \cite{towell2026cmri,towell2026hierarchy}" to a single
   clause at the end of the paragraph, and move the reviewer-objection sentence to the
   Discussion paragraph "What would change the conclusion" (l.182).

2. **Add a paragraph after l.88 (the budget paragraph) that states and tests the sceptic's
   reduction:** "A reader may object that averaging more vectors drawn around a fixed topic must
   move the mean toward that topic, and that the saturation of Fig. 1 is the curve of a sample
   mean. That process is compatible with the result (process A, Table 3). To separate sequence
   from averaging we recomputed the contrast after shuffling message order within each
   conversation, which preserves topic, count and role mix: [shuffled gain with interval]. The
   observed-minus-shuffled difference, [value], is the sequence-specific part." Run the
   analysis; if the difference is zero, say so in the abstract and the paper is stronger for it.

3. **Replace the title** (l.18) with a declarative one that states result and design, for
   example: "The Whole Exchange Predicts the Next User Message Slightly Better Than the Latest:
   A Frozen-Protocol Replication on Two Public Corpora". Drop "Forecast Origin and Observation"
   from the title; they are secondaries.

4. **Replace the abstract's first two sentences** (l.30, "Does the whole prior exchange predict
   ... We pre-registered one contrast ... 482 hashed-IP components of WildChat.") with a
   result-first opening: "Across 233 PRISM and 482 WildChat components, the mean embedding of
   every earlier message lies closer to the next user message than the mean of the last two, by
   about .02 cosine, with 95% component-bootstrap intervals above zero in both corpora and
   both encoders, under a protocol frozen before either held-out sample was scored." Cut the
   abstract's last two sentences ("Two further observation choices ... state their observation
   process.") once Sect. 5 is condensed; end instead on what the gain does and does not show.

5. **Replace "two exact results showing that message packaging and the order in which a fixed
   graph is revealed manufacture apparent dynamics on their own" (l.40) with "two elementary
   observation-process nulls, for message packaging and for revealing a fixed graph in time
   order, that any report of conversational development should pass"; replace "two exact results
   show" (l.186) with "two elementary identities show"; and at l.155 add after Eq. (1): "The
   identities are elementary; the point is their consequence on real data: the choice of
   denominator alone moves a fitted densification exponent from 1.41 to 1.56 on the
   single-author archive of \cite{towell2026cmri}." Name the archive for what it is.

6. **Condense l.71 to 73 and l.180 into one "Protocol deviations" paragraph of about eight
   lines** that keeps the amendment and its consequence ("under the original rule at least 95 of
   the 828 confirmation conversations would have been dropped and the run halted"), the "806 of
   the 828" decoding admission, and "Every diff is recorded in the two repositories", and sends
   the push-log timings, the structural test parse, the runner edit and the family-by-year
   computation time to the repositories' deviation logs.

7. **Cut the packaging paragraph (l.147) to two sentences:** "Pooling recipes written as decaying
   weights over a token clock are invariant to subdividing fixed vectors once the weight is
   corrected, but not to re-encoding actual halves: on 774 PRISM development messages the
   pooled state moved by $.054$ cosine in MiniLM and $.010$ in E5. The packaging unit is therefore
   part of the measurement, not a detail of it [cite the dynamics line]."

8. **Replace Fig. 3 with a four-row table or move it to supplementary material**, keeping in the
   text: "Every declared subgroup has a positive point estimate; the gain rises with conversation
   length (from $+.012$ at six messages to $+.040$ at twelve in MiniLM on WildChat); family and
   year are confounded in WildChat and no multiplicity correction was applied." Spend the
   recovered space on the ECDF: distribution shape, the shuffle null for the spread, and
   whether the non-positive components are a tail or a mode.

9. **Replace "predicted better by the whole prior exchange than by the latest exchange"
   (l.30, l.38, l.186) with "lies closer in cosine to the whole prior exchange than to the latest
   exchange"**, and at first use of "history gain" (l.38 or l.79) add "(a difference in raw
   cosine, not in any task score)". Add, after l.117's "Neither attributes influence", the
   descriptive reading the data support: "Descriptively, a person's next message is anchored by
   their own earlier messages more than by the assistant's latest reply."

10. **Rewrite the Reproducibility paragraph (l.188):** replace "public with the camera-ready
    version" with "public at submission" and add a Zenodo or OSF DOI for the frozen protocol
    files, the results and the sample manifests; replace "pre-registered" (abstract, keywords,
    Table 1 caption, l.69) with "frozen" unless the registry deposit is made, in which case keep
    the word and cite the deposit. Release the per-component gain CSVs.

Two smaller items for the NetSci abstract: delete its first sentence, and decide whether it is a
network abstract (lead with the growth null in use and the return-lag result) or a cosine
abstract (accept a poster).

## Sources checked for venue fit

- ISCS 2027 home and call: https://iutdijon.u-bourgogne.fr/ccs-france/
- ISCS 2027 submissions: https://iutdijon.u-bourgogne.fr/ccs-france/submissions/
- NetSci 2027 site: https://netsci2027.github.io/
- NetSci 2027 call for abstracts: https://netsci2027.github.io/calls/abstracts
- NetSci 2027 Overleaf template (secondary, used only to confirm dates):
  https://www.overleaf.com/latex/templates/netsci2027-abstract-template/gqryjjnvrczn
