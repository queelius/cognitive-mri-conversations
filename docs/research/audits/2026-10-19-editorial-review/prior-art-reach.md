# Prior-art sweep: context reach, speaker signature, history-conditioned preference, memory and heterogeneity

Sweep date: 2026-10-19. Scope: the four literatures the editorial review (`REVIEW.md`, Findings 1
and 2) made load-bearing once the headline was reduced to "the next user message is anchored by
the last exchange plus the user's one previous turn, with nothing measurable beyond that", and the
five follow-up tests (a) long-conversation reach, (b) state versus signature, (c) role weighting,
(d) history-conditioned choice prediction on PRISM, (e) the per-person distribution of the gain.

This file does not repeat the 30 entries in `prior-art-dialogue.md`. Where one of those entries is
load-bearing here (Sankar 2019, Wu 2017, Huang 2026, Laban 2026, Dial2vec, Mitra and Kumar 2026,
Liu et al. 2025 on user feedback, the anisotropy group, Lost in the Middle), it is cited by name
with "(dialogue file)" and not re-described.

Verification rule: every entry in Sections 1 to 4 was opened on a primary page during this sweep
(ACL Anthology record, arXiv abstract or full text, a Crossref record for journal DOIs, the NIST
TREC proceedings PDF, or the publisher's PDF extracted locally). The status line on each entry says
which. "Crossref" means the bibliographic record was confirmed and the abstract was read from that
record if present; "abstract not opened" means only the record was confirmed and the one-line
description rests on the sweep's own knowledge of the paper. Nothing in this file is taken from a
search-engine snippet without that being marked.

## 0. The decisive question: has a reach of "about two user turns" been reported before?

Searched for explicitly (queries in Section 8). The answer in one paragraph:

**Yes for human-human information-seeking dialogue, yes at the token level for language models,
yes for assistant response quality on in-the-wild logs; no for the user's next message on human-AI
logs in embedding space.** The CoQA paper (Reddy, Chen and Manning 2019, entry R-4) states in so
many words that "most questions in a conversation have a limited dependency within a bound of two
turns", with human F1 of 19.9 given no history, 79.8 given the previous turn, 85.3 given the last
two turns and 86.4 given everything, and its best model used one previous turn. QuAC (Choi et al.
2018, R-5) finds saturation at two to three prior question-answer pairs. Khandelwal et al. (2018,
R-1) found that an LSTM language model uses about 200 tokens, with word order mattering only in
the most recent 50. Huang et al. (2026, dialogue file A-5) found that the most recent exchange
often matches the full context for assistant response quality on real logs. CAsT 2020 (R-7) found
that most curated search turns depend on previous user utterances and about a quarter on a system
response. What none of these reports is the quantity our paper measures: the reach of the next
*user* message, counted in the user's own turns, on in-the-wild human-AI logs, under a metric that
cannot reward pool size. That is the only novelty sentence the paper can keep, and it must be
written with CoQA and QuAC in the same paragraph, because a reviewer who knows conversational QA
will otherwise write the sentence for us. The two-turn number should be presented as a replication
of a known reach in a new population and a new measurement, not as a discovery.

One further caution that the sweep turned up and that bears on how the number is stated: Qu et
al. (2019, R-6) show that on the same QuAC data the measured reach depends on how history is
represented (one to two turns when history is prepended as text, five to six turns with a learned
history embedding). Our two-turn reach is a property of mean-pooled fixed sentence encoders as much
as of conversations, and the paper should say so.

## 1. Context reach and memory length in dialogue and language models

- **R-1. Khandelwal, He, Qi, Jurafsky (2018). Sharp Nearby, Fuzzy Far Away: How Neural Language
  Models Use Context.** ACL 2018, pp. 284-294. DOI 10.18653/v1/P18-1027. Shows: by dropping,
  shuffling and replacing prior context and measuring the loss increase, an LSTM LM uses about 200
  tokens on average, treats the most recent 50 tokens in fine detail (order matters) and the rest
  as a rough topic field (order does not). Means: the origin of the ablation method and of the
  two-regime reading (sharp nearby, topical far) that our last-exchange-plus-one-turn result
  instantiates at message granularity; cite as the token-level precedent. Status: verified at ACL
  Anthology.
- **R-2. Sun, Krishna, Mattarella-Micke, Iyyer (2021). Do Long-Range Language Models Actually Use
  Long-Range Context?** EMNLP 2021, pp. 807-822. DOI 10.18653/v1/2021.emnlp-main.62. Shows: two
  models that accept 8K tokens improve from context beyond the previous 2K tokens only on a small
  set of copyable tokens and not at all on sentence-level tasks; the benefit is concentrated in
  literary text. Means: even models built for long context have a short usable reach; our "nothing
  measurable beyond two user turns" is the dialogue-side instance of a general pattern and should
  be framed as expected. Status: verified at ACL Anthology.
- **R-3. O'Connor, Andreas (2021). What Context Features Can Transformer Language Models Use?**
  ACL-IJCNLP 2021, pp. 851-864. DOI 10.18653/v1/2021.acl-long.70. Shows: shuffling word order
  within sentences or keeping only nouns removes under 15% of the usable information in mid- and
  long-range context; the presence of long context matters for perplexity, its detailed syntactic
  and propositional content does not. Means: distant context is used as a bag of topical content,
  which is what a mean of earlier message vectors represents; this is why an isotropic static-topic
  process reproduces the pooled gain and why the gain is not evidence of memory. Status: verified
  at ACL Anthology.
- **R-4. Reddy, Chen, Manning (2019). CoQA: A Conversational Question Answering Challenge.** TACL
  7, pp. 249-266. DOI 10.1162/tacl_a_00266; arXiv:1808.07042. Shows: of sampled questions, 30.5%
  do not rely on the history, 49.7% carry explicit coreference and 19.8% implicit; a human
  experiment on 300 one- or two-word questions gives F1 19.9 with no history, 79.8 with the
  previous turn, 85.3 with the last two turns and 86.4 with the full history ("most questions in a
  conversation have a limited dependency within a bound of two turns"); the best DrQA setting used
  one previous turn (54.7 against 52.3 with full history, "the gains are little beyond one previous
  turn"). Means: **a prior two-turn reach estimate with the same shape as ours** (one previous turn
  carries most of it, two saturate), on crowdworker question chains over a fixed passage. Must be
  cited in the sentence that states our reach; our contribution is the same reach measured in a
  user's own turns on human-AI logs, under a content-neutral metric, with the state/signature
  decomposition. Status: verified at ACL Anthology (TACL record) and arXiv full text.
- **R-5. Choi, He, Iyyer, Yatskar, Yih, Choi, Liang, Zettlemoyer (2018). QuAC: Question Answering
  in Context.** EMNLP 2018, pp. 2174-2184. DOI 10.18653/v1/D18-1241; arXiv:1808.07036v3. Shows:
  BiDAF++ dev F1 rises from 51.8 with no context to 59.9 with one prior QA pair, 60.6 with two and
  60.6 with three ("we observe saturation using contexts of length 3"). Means: a second,
  independent two-to-three-turn reach estimate in information-seeking dialogue; with R-4 it fixes
  the pattern for that genre and must be cited alongside it. Status: verified at ACL Anthology and
  arXiv full text.
- **R-6. Qu, Yang, Qiu, Croft, Zhang, Iyyer (2019). BERT with History Answer Embedding for
  Conversational Question Answering.** SIGIR 2019; arXiv:1905.05412v2. Shows: on QuAC, prepending
  history as text performs best with one or two history turns, while a learned history-answer
  embedding keeps improving to five or six turns (Section 4.4, Figure 3). Means: the measured
  reach depends on the representation of history; our two-turn number is a statement about mean
  pooling in fixed encoders and the paper must bound it that way rather than as a property of
  conversations. Status: verified from arXiv full text (SIGIR venue from the paper header; ACM page
  not opened).
- **R-7. Dalton, Xiong, Callan (2020). CAsT 2020: The Conversational Assistance Track Overview.**
  TREC 2020 proceedings, NIST (PDF at trec.nist.gov/pubs/trec29/papers/OVERVIEW.C.pdf); and
  Dalton, Xiong, Callan (2020), TREC CAsT 2019: The Conversational Assistance Track Overview,
  arXiv:2003.13624. Shows: in the 2020 topics "the majority of turns have some dependence on
  previous utterances or responses" and "approximately a quarter of the turns depend on a previous
  system response", with fine-grained labels naming which earlier query or response each turn
  refers to; the 2019 overview reports that 86% of automatic runs used previous turns, that the
  topic title (first turn) was the most effective context, and that nDCG@3 falls 23% from turn one
  to turn eight. Means: the hand-labelled analogue of our role question (c): in curated search
  dialogues most dependence is on earlier user utterances and a minority on system responses. Cite
  for (c), with the caveat that CAsT dialogues were written by organisers. Status: 2020 overview
  PDF read locally; 2019 overview read at ar5iv.
- **R-8. Mele, Muntean, Nardini, Perego, Tonellotto, Frieder (2020). Topic Propagation in
  Conversational Search.** SIGIR 2020, pp. 2057-2060. DOI 10.1145/3397271.3401268;
  arXiv:2004.14054. Shows: "the conversation topic is typically enclosed in the first utterance"
  but "topic switches are also common"; their Context Query recipe enriches the current query with
  the first query and the previous turn's query, and a topic-shift-aware variant that propagates
  the current topic instead of the first is best in first-stage retrieval (P@1 .480 against .399
  for first-query-only). Means: "first turn plus previous turn" was a standard recipe in
  conversational search and its failure under topic shift is documented; our development finding
  that the opening is not special at later positions is the same observation and should be cited
  as such. Status: verified at Crossref and arXiv full text.
- **R-9. Xu, Szlam, Weston (2022). Beyond Goldfish Memory: Long-Term Open-Domain Conversation.**
  ACL 2022, pp. 5180-5197. DOI 10.18653/v1/2022.acl-long.356. Shows: the Multi-Session Chat
  dataset, built so that partners recall earlier sessions; models trained on short-context data do
  poorly, retrieval-augmented and summarise-and-recall models do better. Means: across sessions,
  engineered references reach back by construction; our within-conversation reach says nothing
  about cross-session reach, and the paper should mark that boundary explicitly. Status: verified
  at ACL Anthology.
- **R-10. Maharana, Lee, Tulyakov, Bansal, Barbieri, Fang (2024). Evaluating Very Long-Term
  Conversational Memory of LLM Agents (LoCoMo).** ACL 2024, pp. 13851-13870.
  DOI 10.18653/v1/2024.acl-long.747. Shows: conversations of about 600 turns over up to 32
  sessions generated from personas and event graphs; LLMs lag humans on long-range temporal and
  causal questions even with long context or RAG. Means: benchmarks that require long reach are
  built to require it; they test whether a model *can* use far history, not how far a user's next
  message *does* depend, which is our object. Status: verified at ACL Anthology.
- **R-11. Wu, Wang, Yu, Zhang, Chang, Yu (2025). LongMemEval: Benchmarking Chat Assistants on
  Long-Term Interactive Memory.** ICLR 2025; arXiv:2410.10813v2. Shows: 500 questions embedded in
  scalable histories; commercial assistants and long-context LLMs show a 30% accuracy drop on
  sustained-interaction memory. Means: same boundary as R-10, and the benchmark the
  selective-forgetting pilot found to contain no cross-topic cases. Status: verified at arXiv
  (ICLR 2025 in the journal-reference field).
- **R-12. Kwan, Zeng, Jiang, Wang, Li, Shang, Jiang, Liu, Wong (2024). MT-Eval: A Multi-Turn
  Capabilities Evaluation Benchmark for Large Language Models.** EMNLP 2024, pp. 20153-20177.
  DOI 10.18653/v1/2024.emnlp-main.1124. Shows: four interaction types (recollection, expansion,
  refinement, follow-up); most models degrade from single-turn to multi-turn; "the distance to
  relevant content and susceptibility to error propagation" are the key factors. Means: a typology
  of what a user turn depends on; test (a) should report reach by turn type, or at least say that
  follow-up and refinement turns depend on the immediate exchange while recollection turns reach
  further by construction, so a corpus's mix of types sets its average reach. Status: verified at
  ACL Anthology.

Gap in this group: no opened paper estimates the natural reach of the next user message on
human-AI logs. The nearest in-the-wild evidence remains Huang et al. 2026 (dialogue file A-5), which
measures assistant response quality, and ThoughtTrace (known to the authors), which does not vary
the history length.

## 2. Speaker signature and user identity in embedding space

- **S-1. Rivera-Soto, Miano, Ordonez, Chen, Khan, Bishop, Andrews (2021). Learning Universal
  Authorship Representations.** EMNLP 2021, pp. 913-919. DOI 10.18653/v1/2021.emnlp-main.70.
  Shows: neural authorship representations trained on one domain (Reddit, Amazon reviews,
  fanfiction) transfer zero-shot to some domains and poorly to others. Means: a person's signature
  is recoverable from text with learned embeddings; our condition (b) is an authorship-verification
  contrast (same-author versus different-author pairs) and should be described in that vocabulary,
  with the caveat that MiniLM and E5 are trained for content, not authorship, so a null is weak
  evidence against signature. Status: verified at ACL Anthology.
- **S-2. Wegmann, Schraagen, Nguyen (2022). Same Author or Just Same Topic? Towards
  Content-Independent Style Representations.** RepL4NLP 2022, pp. 249-268.
  DOI 10.18653/v1/2022.repl4nlp-1.26. Shows: authorship-verification training conflates style with
  topic because authors return to the same topics; controlling for content with conversation
  labels yields representations that capture style independent of content. Means: **the confound
  at the centre of (b)**: a person's turns from other conversations share topic as well as style,
  so a gain from "same person, other conversation" is not a signature result unless topic is
  matched; the design needs same-person-different-topic and other-person-same-topic cells. Status:
  verified at ACL Anthology.
- **S-3. Wang, Aggazzotti, Kotula, Rivera Soto, Bishop, Andrews (2023). Can Authorship
  Representation Learning Capture Stylistic Features?** TACL 11, pp. 1416-1431.
  DOI 10.1162/tacl_a_00610. Shows: probing experiments find authorship representations are
  sensitive to writing style and may be robust to topic drift over time. Means: supplies a
  diagnostic for (b): if the extra user turn carries signature, a style encoder should recover the
  gain and a content encoder should not; running both is a cheap way to label what the gain is.
  Status: verified at ACL Anthology.
- **S-4. Zhang, Dinan, Urbanek, Szlam, Kiela, Weston (2018). Personalizing Dialogue Agents: I have
  a dog, do you have pets too?** ACL 2018, pp. 2204-2213. DOI 10.18653/v1/P18-1205. Shows:
  conditioning on a speaker profile and on the partner's profile improves next-utterance
  prediction, and the dialogue can in turn predict the profiles. Means: the person-level signal in
  next-utterance prediction is established for human-human chit-chat with an explicit profile; (b)
  asks whether it is recoverable from the person's own prior turns with no profile. Status:
  verified at ACL Anthology.
- **S-5. Salemi, Mysore, Bendersky, Zamani (2024). LaMP: When Large Language Models Meet
  Personalization.** ACL 2024, pp. 7370-7392. DOI 10.18653/v1/2024.acl-long.399. Shows: seven
  tasks with multiple entries per user; retrieving items from the user's own profile (their prior
  texts) to augment the prompt improves zero-shot and fine-tuned models. Means: "the person's
  messages from other conversations" is LaMP's retrieval augmentation; cite it as the applied form
  of the signature condition, noting that LaMP reports task gains, not next-message prediction.
  Status: verified at ACL Anthology.
- **S-6. Soni, Matero, Balasubramanian, Schwartz (2022). Human Language Modeling.** Findings of
  ACL 2022, pp. 622-636. DOI 10.18653/v1/2022.findings-acl.52. Shows: a hierarchical language
  model with a human-level state linking a user's sequence of documents, pre-trained on about
  100,000 social-media users, improves perplexity and user-level tasks. Means: the state/signature
  decomposition already has a modelling counterpart in which the user state changes slowly across
  documents; (b) asks the same question with fixed encoders and should cite HuLM as the trained
  version. Status: verified at ACL Anthology.
- **S-7. Pickering, Garrod (2004). Toward a mechanistic psychology of dialogue.** Behavioral and
  Brain Sciences 27(2), pp. 169-190. DOI 10.1017/S0140525X04000056. Shows: the interactive
  alignment account: interlocutors' representations align largely automatically at many levels
  through priming. Means: the theoretical reason an assistant reply could predict the user's next
  words in (c): if users align to the assistant, assistant text carries user signal. Status:
  verified at Crossref (abstract in record).
- **S-8. Doyle, Frank (2016). Investigating the Sources of Linguistic Alignment in Conversation.**
  ACL 2016, pp. 526-536. DOI 10.18653/v1/P16-1050. Shows: with a probabilistic alignment model
  that separates word-level from category-level alignment, on telephone conversations and
  microblog replies, alignment is primarily lexical and is modulated substantially by discourse
  act. Means: alignment measures are confounded by baseline word frequency and by what kind of turn
  follows; the role-weighting test should control for lexical overlap between reply and user turn,
  since a reply that quotes the user raises both raw and centered similarity for reasons that are
  not memory. Status: verified from the ACL PDF extracted locally.
- **S-9. Blevins (2026). Accommodation Goes Both Ways: Studying Linguistic Convergence Between
  Humans and Language Models.** arXiv:2605.29278v1 (no venue listed). Shows: with an asymmetric
  convergence metric on WildChat across eight languages, LLMs strongly over-converge toward their
  users on function-word and open-class features, while humans converge toward the LLM about as
  much as toward another person. Means: **directly bears on (b) and (c)**: the assistant reply is
  partly an echo of the user, so an assistant-turn contribution in the role test can be reflected
  user signal, and the "same person versus other person" contrast must allow for the model having
  adapted to the person; the paper should cite this and add a reply-minus-user-overlap control.
  Status: verified at arXiv. [2025-2026]
- **S-10. Zhu, Rzeszotarski, Mimno (2026). Priming, Path-dependence, and Plasticity: Understanding
  the molding of user-LLM interaction and its implications from (many) chat logs in the wild.**
  arXiv:2605.05767v1 (cs.HC; no venue listed). Shows: in 140K sessions from 7,955 users,
  interaction patterns form and stabilise rapidly through each user's early trajectory; recurring
  text patterns and retention correlate strongly with early exploration; users are "not tabula
  rasa, nor constantly adapting". Means: the first large-scale per-user longitudinal analysis of
  chat logs found; per-user stability of phrasing is the signature hypothesis of (b) stated at the
  user level, and the per-user framing is the one (e) needs. Status: verified at arXiv.
  [2025-2026]

Gap in this group: no paper measures, with sentence embeddings on human-AI logs, how much of the
predictability of a user's next message is attributable to the person rather than to the
conversation. A 2026 style-embedding benchmark (STEB; Rivera Soto, Wegmann, Aggazzotti,
arXiv:2606.31741) is reported in a search summary to find that semantic embeddings "consistently
fail in stylistic tasks"; the abstract was not opened and the claim is not used above, but if it
holds it is the reason a MiniLM/E5 null in (b) cannot rule signature out.

## 3. History-conditioned preference prediction

- **P-1. Kirk, Whitefield, Röttger, Bean, Margatina, Ciro, Mosquera, Bartolo, Williams, He,
  Vidgen, Hale (2024). The PRISM Alignment Dataset.** NeurIPS 2024 Datasets and Benchmarks;
  arXiv:2404.16019v2. Shows (from the full text): an OLS of score on text features explains little
  (R² = 0.06; characters +0.03, ending in a question mark +2.6, enumeration +7.2, line breaks
  -10.8, de-anthropomorphic phrases -2.3, refusals -9.0); at T>1 the candidate responses come from
  the same model and are "much more similar in style and content" than the across-model T=1
  candidates, with narrower score deviations; in the US sample the model that maximises mean
  choice is chosen under 50% of the time; "implicit cues in subsequent prompts in the conversation"
  are named as future work; there is no analysis of conversation history or turn number as a
  predictor of scores or choices. Means: (d) is open in the PRISM paper itself, but the test has a
  low ceiling at T>1 because the candidates are near-duplicates from one model; the paper must
  report T=1 and T>1 separately and state the within-model similarity as the ceiling. Status:
  verified at arXiv abstract and HTML full text (Sections 3.2, 4.2.2, 4.3.2).
- **P-2. Siththaranjan, Laidlaw, Hadfield-Menell (2024). Distributional Preference Learning:
  Understanding and Accounting for Hidden Context in RLHF.** ICLR 2024; arXiv:2312.08358v2.
  Shows: standard preference learning implicitly aggregates over hidden context (such as annotator
  identity) by Borda count, which can produce counterintuitive results; distributional methods
  detect the hidden context. Means: the person is hidden context in pairwise choice; a
  history-conditioned predictor (d) is one way to expose it, and the zero-gain subpopulation of
  (e) is a hidden-context mixture question. Status: verified at arXiv.
- **P-3. Poddar, Wan, Ivison, Gupta, Jaques (2024). Personalizing Reinforcement Learning from
  Human Feedback with Variational Preference Learning.** arXiv:2408.10075v1 (no venue on the arXiv
  page). Shows: a single reward model averages over subgroups and misfits individuals; a
  user-specific latent inferred from a few comparisons conditions reward and policy on pluralistic
  language datasets. Means: our pooled history is a non-learned user latent; a test that it
  predicts the chosen reply better than the last exchange asks whether the latent can be read from
  the user's text alone. Status: verified at arXiv.
- **P-4. Li, Zhou, Lipton, Leqi (2024). Personalized Language Modeling from Personalized Human
  Feedback.** arXiv:2402.05133v3 (no venue on the arXiv page). Shows: a lightweight user model
  learned jointly with the LLM from feedback captures explicit and implicit preferences and beats
  non-personalised RLHF and prompting. Means: same frame as P-3; the user model is learned from
  feedback, not from the user's own text, so a text-only history predictor is the cheaper
  hypothesis and the one (d) tests. Status: verified at arXiv.
- **P-5. Chiang, Zheng, Sheng, Angelopoulos, Li, Li, Zhang, Zhu, Jordan, Gonzalez, Stoica (2024).
  Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.** arXiv:2403.04132v1
  (ICML 2024 venue not confirmed from the arXiv page); with Li, Angelopoulos, Chiang (2024), "Does
  style matter? Disentangling style and substance in Chatbot Arena", LMSYS blog, 29 August 2024.
  Shows: crowd votes agree with expert raters; adding style covariates to the Bradley-Terry model
  moves rankings substantially (GPT-4o-mini from 6th to 11th; length coefficient 0.249, markdown
  coefficients 0.02 to 0.03). Means: a measurable part of pairwise preference is predicted by
  surface features of the candidates; a history-conditioned predictor in (d) must beat a
  length-and-formatting baseline, which PRISM's R² of .06 suggests is itself weak on that data.
  Status: arXiv abstract and blog post opened.
- **P-6. Bai et al. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning
  from Human Feedback.** arXiv:2204.05862v1; data at github.com/anthropics/hh-rlhf. Shows:
  preference modelling and RLHF from human comparisons; the released data are multi-turn dialogues
  whose chosen and rejected versions share the full history and differ only in the final assistant
  turn. Means: a usable second collection for (d), with the structural caveat that a history-only
  predictor is at chance by construction (the history is shared); the admissible test is whether
  history improves the similarity of the chosen over the rejected candidate to the history, under
  centering. Status: arXiv abstract verified; the pair format is from the dataset documentation as
  reported in a search summary, not opened.
- **P-7. Jin, Xu, Liu, Tao, Golovneva, Shu, Zhao, Li, Weston (2025). The Era of Real-World Human
  Interaction: RL from User Conversations.** arXiv:2509.25137v1 (no venue listed). Shows: RLHI
  learns from users' natural follow-up messages (user-guided rewrites) and from a reward model
  conditioned on the user's long-term interaction history (persona), on WildChat-derived
  conversations, improving personalisation and instruction following. Means: the user's next turn
  is already in use as a reward signal on WildChat; our reach result is a statement about the
  information content of that signal (it is anchored by the person's last two turns), and the
  paper should cite RLHI as the application that depends on the answer. Status: verified at arXiv.
  [2025-2026]

Gap in this group: no paper tests whether the earlier turns of a conversation predict which of two
candidate replies a participant chose, on PRISM or elsewhere; the PRISM authors list it as future
work. The nearest is RLHI (P-7), which uses the next user turn as feedback without estimating how
much of it the history explains.

## 4. Complex-systems treatments of memory and heterogeneity in human communication

- **H-1. Goh, Barabási (2008). Burstiness and memory in complex systems.** EPL 81(4), 48002.
  DOI 10.1209/0295-5075/81/48002. Shows: a burstiness coefficient B (shape of the inter-event
  distribution) and a memory coefficient M (correlation between consecutive inter-event times)
  separate the two; human activity is bursty with weak memory, natural phenomena the reverse.
  Means: the field's word for memory length is the lag correlation between consecutive events; our
  reach is the content-space analogue of M, and the paper should report the lag-k gain profile
  (k = 1, 2, 3, ...) rather than only the two-versus-all contrast, so a complex-systems reader sees
  the correlation structure. Status: verified at Crossref.
- **H-2. Wu, Zhou, Xiao, Kurths, Schellnhuber (2010). Evidence for a bimodal distribution in human
  communication.** PNAS 107(44), pp. 18803-18808. DOI 10.1073/pnas.1013140107. Shows: SMS
  inter-event times are bimodal, a Poisson component from random initiations plus a power-law
  component from bursts of mutual exchange; a two-queue interaction model fits. Means: the
  precedent that per-user communication statistics are a mixture of two regimes; the template for
  (e), where the question is whether the per-person gain distribution has a mode at zero (two
  populations) or a single skewed shape (one population with a tail). Status: verified at Crossref
  (abstract in record).
- **H-3. Malmgren, Hofman, Amaral, Watts (2009). Characterizing individual communication
  patterns.** KDD 2009, pp. 607-616. DOI 10.1145/1557019.1557088. Shows: individual e-mail users
  are each well described by a cascading non-homogeneous Poisson process, and population
  heterogeneity is captured by the distribution of a few per-user parameters. Means: the standard
  for reporting a per-user quantity in platform logs is the distribution of per-user parameters
  against a per-user null, not a population mean; (e) should fit per-component gains and compare
  the spread to the shuffle null's own per-component spread before calling a mode. Status:
  verified at Crossref (abstract not in record; description from the sweep's knowledge of the
  paper).
- **H-4. Altmann, Cristadoro, Degli Esposti (2012). On the origin of long-range correlations in
  texts.** PNAS 109(29), pp. 11582-11587. DOI 10.1073/pnas.1117723109. Shows: long-range
  correlations in literary text originate at the topic level and propagate down to words and
  letters; near semantically relevant topics they appear as bursty event sequences. Means: content
  memory lives at the topic level; a short reach in message space is compatible with long-range
  topic correlation, which is the two-regime picture (sharp nearby, topical far) and the reason a
  static-topic null reproduces our gain. Status: verified at Crossref (abstract in record).
- **H-5. Lin, Tegmark (2017). Critical Behavior in Physics and Probabilistic Formal Languages.**
  Entropy 19(7), 299. DOI 10.3390/e19070299. Shows: mutual information between symbols decays
  exponentially under Markov (regular-grammar) processes and as a power law under hierarchical
  (context-free) ones, with power-law decay observed in text. Means: a correlation length exists
  only when decay is exponential; if the lag-k gain profile in (a) decays as a power law the reach
  is a scale, not a length, and the long-conversation test should fit both forms before quoting a
  number of turns. Status: verified at Crossref (abstract in record).
- **H-6. Brynjolfsson, Li, Raymond (2025). Generative AI at Work.** Quarterly Journal of Economics
  140(2), pp. 889-942. DOI 10.1093/qje/qjae044. Shows: a staggered rollout of a conversational
  assistant to 5,172 support agents raised issues resolved per hour by about 15% on average with
  large heterogeneity: less experienced agents gained most, the most experienced gained little in
  speed and declined slightly in quality. Means: the one human-AI setting found where the
  distribution of a per-person effect, not its mean, is the result; (e) belongs in that frame, and
  "a mean gain with a zero-gain subpopulation" is the usual shape of AI effects across people.
  Status: verified at Crossref (abstract in record).

Gap in this group: no prior work reports the distribution of a per-user history effect in human-AI
logs. Zhu et al. 2026 (S-10) is the closest in framing (per-user trajectories) but measures
phrasing stability, not predictability.

## 5. Opened, lower relevance (not counted above)

- Barabási (2005). The origin of bursts and heavy tails in human dynamics. Nature 435, pp.
  207-211, DOI 10.1038/nature03459. Priority-queue origin of bursty inter-event times. Crossref.
- Malmgren, Stouffer, Motter, Amaral (2008). A Poissonian explanation for heavy tails in e-mail
  communication. PNAS 105(47), pp. 18153-18158, DOI 10.1073/pnas.0800332105. Circadian and weekly
  cycles plus cascades explain the heavy tail. Crossref.
- Karsai, Kaski, Barabási, Kertész (2012). Universal features of correlated bursty behaviour.
  Scientific Reports 2, 397, DOI 10.1038/srep00397. Bursty trains with memory across datasets.
  Crossref; abstract not in record.
- Altmann, Pierrehumbert, Motter (2009). Beyond Word Frequency: Bursts, Lulls, and Scaling in
  the Temporal Distributions of Words. PLoS ONE 4(11), e7678, DOI 10.1371/journal.pone.0007678.
  Word burstiness beyond frequency, tied to semantic content. Crossref; abstract not in record.
- Chen, Mayné i Comas, Jin, Luo, Soljačić (2025). L²M: Mutual Information Scaling Law for
  Long-Context Language Modeling. arXiv:2503.04725v2. Bipartite mutual information in text follows
  a scaling law distinct from two-point MI. arXiv.
- Noy, Zhang (2023). Experimental evidence on the productivity effects of generative artificial
  intelligence. Science 381(6654), pp. 187-192, DOI 10.1126/science.adh2586. Preregistered, N =
  453; time down 40%, quality up 18%, inequality between workers decreased. Crossref.
- Futrell, Gibson, Levy (2020). Lossy-Context Surprisal. Cognitive Science 44(3), e12814,
  DOI 10.1111/cogs.12814. Processing difficulty given a lossy memory of context; information
  locality. The human-memory counterpart of a short reach. Crossref.
- Sordoni, Galley, Auli, Brockett, Ji, Mitchell, Nie, Gao, Dolan (2015). A Neural Network Approach
  to Context-Sensitive Generation of Conversational Responses. NAACL 2015, pp. 196-205,
  DOI 10.3115/v1/N15-1020. The context-message-response triple: the original assumption that one
  earlier message plus the current one is the context. Anthology record verified; abstract not
  opened.
- Jang, Boo, Kim (2023). Conversation Chronicles. EMNLP 2023, pp. 13584-13606,
  DOI 10.18653/v1/2023.emnlp-main.838. About one million LLM-generated multi-session dialogues with
  time intervals and relationships. Anthology.
- Bai et al. (2024). MT-Bench-101. ACL 2024, pp. 7421-7454, DOI 10.18653/v1/2024.acl-long.401.
  Three-tier taxonomy of multi-turn abilities over 1,388 dialogues; alignment techniques do not
  clearly improve multi-turn ability. Anthology (taxonomy items not listed on the page).
- Xu (2021). Global divergence and local convergence of utterance semantic representations in
  dialogue. SCiL 2021, pp. 116-124 (aclanthology.org/2021.scil-1.11). On Switchboard, BNC and Map
  Task with BERT/ELMo cosine: semantic distance between adjacent utterances grows with position in
  the dialogue, and distance between utterances grows with the number of turns between them.
  The human-human baseline for similarity decay with turn distance. PDF extracted locally.
- Li, Galley, Brockett, Spithourakis, Gao, Dolan (2016). A Persona-Based Neural Conversation
  Model. ACL 2016, pp. 994-1003, DOI 10.18653/v1/P16-1094. Speaker embeddings for response
  consistency. Anthology record; abstract not opened.
- Ning et al. (2024). User-LLM: Efficient LLM Contextualization with User Embeddings.
  arXiv:2402.13598v2. User embeddings from self-supervised encoding of interaction histories fed by
  cross-attention. arXiv.
- Brennan, Clark (1996). Conceptual pacts and lexical choice in conversation. JEP:LMC 22(6), pp.
  1482-1493, DOI 10.1037/0278-7393.22.6.1482. Partner-specific lexical entrainment. Crossref.
- Branigan, Pickering, Pearson, McLean (2010). Linguistic alignment between people and computers.
  Journal of Pragmatics 42(9), pp. 2355-2368, DOI 10.1016/j.pragma.2009.12.012. People align with
  computers, by some measures more than with people. Crossref; abstract not opened (publisher
  returned 403).
- Nenkova, Gravano, Hirschberg (2008). High Frequency Word Entrainment in Spoken Dialogue. ACL-08
  HLT short papers, pp. 169-172. Entrainment on high-frequency words predicts perceived naturalness
  and correlates with task success (Switchboard, Columbia Games). PDF extracted locally.
- Danescu-Niculescu-Mizil, Gamon, Dumais (2011). Mark my words! Linguistic style accommodation in
  social media. WWW 2011, pp. 745-754, DOI 10.1145/1963405.1963509. Style accommodation in Twitter
  replies. Crossref.
- Yakura et al. (2026). Empirical evidence of Large Language Model's influence on human spoken
  communication. arXiv:2409.01754v4. ChatGPT-typical words rose in podcast speech after release;
  a preregistered experiment (N = 496) shows chatbot words adopted and retained. arXiv.
- Kian et al. (2025). Using Linguistic Entrainment to Evaluate Large Language Models for Use in
  Cognitive Behavioral Therapy. Findings of NAACL 2025, pp. 7739-7758,
  DOI 10.18653/v1/2025.findings-naacl.430. GPT-3.5 entrains less than therapists or peers.
  Anthology.
- Shen et al. (2025). MiCRo: Mixture Modeling and Context-aware Routing for Personalized
  Preference Learning. arXiv:2505.24846v3. A single Bradley-Terry model has irreducible error under
  a mixture of subgroups. arXiv.
- Zollo, Siah, Ye, Li, Namkoong (2024). PersonalLLM. arXiv:2409.20296v2. Simulated heterogeneous
  users from reward models; persona prompting is too homogeneous. arXiv.
- Dubois, Galambosi, Liang, Hashimoto (2024). Length-Controlled AlpacaEval. COLM 2024;
  arXiv:2404.04475v2. Length control raises Spearman with Chatbot Arena from .94 to .98. arXiv.
- Shani et al. (2024). Multi-turn Reinforcement Learning from Preference Human Feedback.
  arXiv:2405.14655v2. Conversation-level rather than turn-level preference. arXiv.

## 6. Not verifiable or not used

- A 2025 survey (arXiv:2509.05716) is reported in a search summary to say that K = 2 history
  turns was best on both CoQA and QuAC in later BERT-based studies; not opened, not used, and the
  primary results in R-4 to R-6 are sufficient.
- SWRouter (arXiv:2609.11414v1) is reported in a search summary to find first-to-last-user-turn
  cosine above .91 for over 70% of long ShareGPT dialogues; the abstract is silent on this and the
  full text was not opened, so the claim is not used. If confirmed it would say that long
  conversations stay on topic, which is consistent with H-4 and with a static-topic null.
- Branigan et al. 2010 abstract (publisher 403) and Doyle and Frank 2016 via WebFetch (binary
  PDF; read instead by local text extraction, so it is verified).
- STEB (arXiv:2606.31741) and the Sordoni 2015 abstract, as noted in their sections.
- Semantic Scholar, dblp and OpenReview were not used (blocked in the companion sweep; not
  retried).

## 7. What our claims must say given this literature

For each test the sharpest honest statement, what it must cite, and what would contradict or
pre-empt it.

### (a) Reach of about two user turns in conversations of 20 to 40 messages

Sharpest honest statement: "In human-AI logs of up to twelve messages, the next user message is
predicted, under centered cosine with fixed sentence encoders, by the last exchange plus the user's
one previous turn; pooling more of the history adds nothing we can measure, and a random earlier
pair does worse than the last exchange. [If (a) passes:] The same holds in conversations of 20 to
40 messages. This reach is consistent with the two-turn dependency bound reported for conversational
question answering and with the short usable context of language models; what is new is the
population, the user's own turns as the unit, and a metric that cannot reward pool size." Must
cite: CoQA (R-4) and QuAC (R-5) in the same sentence as the number; Khandelwal 2018 (R-1) and Sun
2021 (R-2) for the general pattern; Huang 2026 (dialogue file) for the in-the-wild assistant-side
counterpart; Qu 2019 (R-6) for the dependence of reach on representation; MT-Eval (R-12) for the
turn-type caveat. Pre-empted by: CoQA and QuAC own the number "two" for information-seeking
dialogue, so "two turns" cannot be claimed as a discovery, only as a replication in a new
population and measurement. Contradicted by: nothing found; but R-6 means a learned history
representation could show a longer reach on the same data, and H-5 means the paper must fit the
lag profile (exponential versus power law) before calling the reach a length. If (a) fails in long
conversations, the honest statement is that the reach grows with length, which is what MSC, LoCoMo
and LongMemEval build into their benchmarks and what MT-Eval's recollection turns predict.

### (b) State versus signature

Sharpest honest statement: "The gain from the user's previous turn is [X] attributable to the
conversation and [Y] to the person: adding the same person's turn from a different conversation,
matched on [topic / not matched], yields [value]; adding another person's turn matched on topic
yields [value]." Must cite: Wegmann 2022 (S-2) for the topic confound of same-author pairs, which
dictates the matched cells; Rivera-Soto 2021 (S-1) and Wang 2023 (S-3) for authorship
representations and the style-versus-content diagnostic; PersonaChat (S-4) and LaMP (S-5) for the
established person-level signal; HuLM (S-6) and Zhu 2026 (S-10) for per-user state and stability;
Blevins 2026 (S-9) because the model has converged toward the person, so "same person" turns are
not independent of the assistant's style. Pre-empted by: LaMP and HuLM already show that a user's
own prior text improves prediction of their later text, so a positive signature result is expected
and its size, not its existence, is the contribution. Contradicted by: a positive "same person,
other conversation" result without topic matching is explained by Wegmann 2022 as topic reuse and
will be read that way; a null with MiniLM/E5 is not evidence against signature because these
encoders are content encoders (S-1, and the unverified STEB claim), so the paper must say the test
is for signature-as-content, not signature-as-style, unless a style encoder is added.

### (c) Do assistant replies contribute once two user turns are pooled?

Sharpest honest statement: "Once the last exchange and the user's previous turn are pooled, adding
earlier assistant replies changes the centered prediction of the next user message by [value]; the
assistant's last reply contributes [value] after removing its lexical overlap with the user's
turns." Must cite: Dial2vec (dialogue file) for interlocutor-level pooling; Huang 2026 (dialogue
file) for context pollution from the assistant's own words; Blevins 2026 (S-9) for the model's
convergence toward the user; Pickering and Garrod 2004 (S-7) and Doyle and Frank 2016 (S-8) for
why alignment inflates reply-to-user similarity and how to control it; CAsT 2020 (R-7) for the
hand-labelled finding that about a quarter of curated search turns depend on the system response;
Liu, Zhang and Choi 2025 (dialogue file) for the fact that later user turns are often feedback on
the reply. Pre-empted by: R-7 and Liu et al. 2025 already say that a minority of user turns depend
on the system response and that feedback turns do so by construction, so a small but non-zero
assistant contribution is the expected result and should be reported as a fraction of turns, not
only as a mean. Contradicted by: a positive assistant contribution can be reflected user signal
(S-9); the paper must show the contribution survives removing reply-user overlap, or state it as
"reply including echo".

### (d) Does history predict which candidate reply a PRISM participant chose?

Sharpest honest statement: "On PRISM development conversations, labelled exploratory, the
similarity of a candidate reply to the pooled history separates chosen from rejected candidates by
[value] at T = 1 and [value] at T > 1, against [value] for the last exchange and [value] for a
length-and-formatting baseline; at T > 1 the candidates come from one model and are near
duplicates, so the ceiling is low." Must cite: Kirk 2024 (P-1) for R² = .06 on text features, the
within-model similarity at T > 1, the under-50% winning-model fact and the explicit future-work
sentence; Chiang 2024 and the style-control post (P-5) for the surface-feature baseline;
Siththaranjan 2024 (P-2), Poddar 2024 (P-3) and Li 2024 (P-4) for the personalised-preference
frame; Jin 2025 (P-7) as the application that uses the next user turn as a signal; Bai 2022 (P-6)
if HH-RLHF is used as a second collection, with the shared-history caveat. Pre-empted by: nothing
found tests history as a predictor of choice; PRISM names it as future work. Contradicted by: a
history-similarity predictor that does not beat length and formatting is explained by P-5; a
positive result only at T = 1 is explained by P-1's candidate design (different models at T = 1,
one model after) and is a statement about model identity, not history. On HH-RLHF the history is
shared, so the paper may not report a history-only predictor there.

### (e) Is the zero-gain subpopulation a mode or a tail?

Sharpest honest statement: "Under centered cosine, [33 to 41]% of components show no gain; the
per-component gain distribution, compared with the shuffle null's own per-component distribution,
is [unimodal with a left tail / bimodal with a mode at zero]; the zero-gain components are
[characterised by length, turn type, or neither]." Must cite: Goh and Barabási 2008 (H-1) for the
memory vocabulary; Wu 2010 (H-2) for the bimodal-mixture precedent in human communication;
Malmgren 2009 (H-3) for per-user parameters against a per-user null; Brynjolfsson 2025 (H-6) for
heterogeneous per-person AI effects; Siththaranjan 2024 (P-2) and MiCRo (Section 5) for preference
mixtures; Zhu 2026 (S-10) for per-user stability in chat logs. Pre-empted by: no prior work reports
the distribution of a per-user history effect in human-AI logs, so the finding is new whichever way
it goes; H-6 means the shape "mean gain, zero-gain subgroup" is familiar and should be framed as
the expected shape of an AI effect across people. Contradicted by: a mode at zero that the shuffle
null also produces (few targets per component give a spike at zero by sampling), which H-3's
per-user-null discipline is designed to catch; and a subpopulation defined entirely by conversation
length, which would make (e) the budget curve restated, as the referee warned.

### The novelty sentence, rewritten

"To our knowledge this is the first frozen, replicated estimate on human-AI logs of how far back a
person's next message depends, counted in the person's own turns and measured so that pooling more
text cannot raise the score; the two-turn reach it finds agrees with the dependency bound long
reported for conversational question answering (Reddy et al. 2019; Choi et al. 2018) and with the
short usable context of language models (Khandelwal et al. 2018; Sun et al. 2021)."

## 8. Search record (2026-10-19)

Queries issued (WebSearch, standard or extended), in order: previous/last user turn context
ablation next user message prediction WildChat; CAsT 2019 overview turn dependence first/previous
turn; PRISM Kirk 2024 what predicts participant choices; lexical entrainment alignment users with
LLM chatbot 2024-2025; MT-Eval taxonomy follow-up recollection expansion refinement; topic drift
coherence decay long conversations similarity to first message by turn index; heterogeneity across
users human-LLM logs per-user distribution mixture; Chatbot Arena vote predictability style
control; authorship verification sentence embeddings style consistency idiolect Wegmann
Rivera-Soto; mutual information decay with turn distance correlation length conversation; HH-RLHF
multi-turn chosen rejected history; conversational search query rewriting first-turn previous-turn
dependency CAsT 2020 Mele; "last two turns" dialogue response generation context size ablation;
Siththaranjan / Poddar / Li personalised preference; PlatoLM user simulator; turn dependency
annotation human-LLM conversations context-dependent self-contained; Topic Propagation in
Conversational Search arXiv; conversational QA history turns ablation QuAC CoQA HAE; semantic
similarity decays with turn distance dialogue corpus; Xu 2021 global divergence local convergence;
chat logs follow-up queries depend only on previous turn "two turns".

Primary pages opened: ACL Anthology (P18-1027, 2021.emnlp-main.62, 2021.acl-long.70,
2022.acl-long.356, 2024.acl-long.747, 2024.emnlp-main.1124, Q19-1016, D18-1241, 2021.emnlp-main.70,
2022.repl4nlp-1.26, 2023.tacl-1.80, P18-1205, 2024.acl-long.399, P16-1094, 2022.findings-acl.52,
2023.emnlp-main.838, 2024.acl-long.401, N15-1020, P16-1050, P08-2043, 2025.findings-naacl.430,
2021.scil-1.11); arXiv (2410.10813, 2404.16019 abstract and HTML, 2605.05767, 2609.11414,
2409.01754, 2605.29278, 2509.25137, 2408.10075, 2402.05133, 2403.04132, 2312.08358, 2204.05862,
2405.14655, 2404.04475, 2409.20296, 2505.24846, 2402.13598, 2503.04725, 1808.07042 HTML,
1808.07036 HTML, 1905.05412 HTML, 2004.14054 HTML, 2003.13624 at ar5iv); Crossref records for
the twelve journal and conference DOIs in Sections 4 and 5; NIST TREC CAsT 2020 overview PDF; the
LMSYS style-control post. Blocked: sciencedirect.com (403), scholarworks.umass.edu (403; the
Anthology copy was used instead).
