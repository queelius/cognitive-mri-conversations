# ISCS 2027 research directions and decision history

## Current recommendation — October 15, 2026

**Pursue a bounded public measurement-validity pilot combining cross-sectional sampling with temporal controls; keep the corrected temporal-network extension as the fallback.** The broader [research deep dive](research-deep-dive/README.md) audited the nested projects, current primary literature and public datasets, and ran small falsification checks. It does not yet justify a full new paper. The temporal extension remains the most mature manuscript, while semantic-motion calibration offers the strongest new question if it can demonstrate a consequential public result.

The new evidence changes the earlier ranking: several motion summaries collapse to construction effects; a projection-aware hierarchy null changes the clustering interpretation; prefix gains have selection-sensitive uncertainty; and the corrected public role check is inconclusive. Cross-sectional validation is worth doing, but long IP-linked archives cannot be treated as verified people. The completed selective-forgetting stop/narrow decision stands.

See the [audited inventory](research-deep-dive/INVENTORY.md), [candidate comparison](research-deep-dive/CANDIDATES.md), [five briefs](research-deep-dive/BRIEFS.md), [bibliography](research-deep-dive/BIBLIOGRAPHY.md), and copyable [next goal plus fallback](research-deep-dive/NEXT-GOAL.md). A negative calibration result can justify a paper only with adequate controls, precision and a substantive contribution; an inconclusive pilot cannot.

## October 14 decision record — earlier three-direction comparison

The memo below is preserved as the historical recommendation after the forgetting pilot. The October 15 decision above supersedes its ranking; its completed evidence, proposed experiments and effort estimates remain available for comparison.

Research decision memo for Alexander Richard Towell and John Matta. Prepared and updated October 14, 2026. Assumes existing data and computational experiments, with no participant recruitment. The selective-forgetting pilot has now been executed; temporal and transit experiments remain proposed. Completed checks and results are documented in [EVIDENCE.md](EVIDENCE.md).

**Recommendation after the pilot: narrow the memory direction and prioritize validation of the temporal extension for ISCS 2027.** The completed memory experiment does not justify a full participation-based retention paper: no clear cross-topic benchmark cases, a synthetic ceiling/proxy failure, and no stable added value beyond the controls. It does not show that graph memory is universally unhelpful. The temporal paper has the strongest existing evidence and shortest path to submission, but its interpretation still needs correction. [Completed pilot report](../../papers/selective-forgetting/REPORT.md)

| Rank for a defensible submission by March | Direction | What makes it worth doing | Main obstacle | Estimated effort after pilot |
|---|---|---|---|---|
| **1** | **When apparent knowledge growth is a sampling effect** — temporal extension | Existing manuscript, embeddings, eleven archives, and permutation results | Validate the community tracker and replace unsupported behavioral labels; establish robustness beyond one null | 4–6 focused researcher-weeks, spread over the available calendar |
| **2, narrowed** | **Selective forgetting of connections between topics in AI memory** | Pilot now identifies measurement and benchmark-coverage limits | Validate necessary cross-topic evidence and its structural measurement before another policy study | Full-paper expansion not justified by this pilot |
| **3** | **Who benefits from transit restoration when the inputs are uncertain?** | Direct question about access to jobs for households without vehicles | New routing/data infrastructure and a crowded resilience/equity literature | 9–12 researcher-weeks; lowest feasibility for this deadline |

Effort estimates are planning estimates, not acceptance probabilities; a researcher-week means five focused workdays. With roughly two research days per week available across the team, the temporal option is the realistic default. The earlier estimate for a full memory study was seven to ten weeks, but the pilot does not support allocating that effort now. Its first-pass compression took about 15.8 summed elapsed minutes on the existing CPU workstation. Transit would require three to four research days per week including setup and writing.

The event is the **7th International Symposium on Complex Systems**, in **Porto, June 2–4, 2027**, with submission due **March 1, 2027**. Its stated scope includes complex networks, AI, cognition, and urban infrastructure, with societal impact emphasized. [Official call](https://iutdijon.u-bourgogne.fr/ccs-france/)

The current rules allow 12-page long papers and 8-page short papers containing unpublished work, or 2–4-page extended abstracts describing published or unpublished research. The pages checked do not settle the proceedings publisher or downstream journal treatment; verify those terms before deciding how much of the journal extension to publish. [Submission rules](https://iutdijon.u-bourgogne.fr/ccs-france/submissions/)

## What the existing work contributes

The published [Cognitive MRI manuscript](../../papers/knowledge-networks/paper/paper.tex) already establishes the conversion of conversation archives into semantic networks, including communities and bridge conversations. The [hierarchical-memory paper](../../papers/hierarchical-memory/paper/paper.tex), listed in the [ISCS 2026 program](https://iutdijon.u-bourgogne.fr/ccs-france/wp-content/uploads/2026/05/PROGRAM-ISCS-2026-V2.pdf), already studies abstraction across conversation, concept, theme, and domain scales. Repackaging either description would have limited novelty.

The [current temporal manuscript](../../papers/temporal-networks/paper/PLOS/paper.tex) is substantially more developed than its original single-archive version: it compares observed arrival order with permutations of a fixed final graph and includes ten additional WildChat archives. The rejection and June revisions are recorded in its [changelog](../../papers/temporal-networks/paper/PLOS/CHANGELOG.md). Its results are the starting point for proposal 1, rather than a proposed future accomplishment.

The embedding-dynamics, semantic-dynamics, and Operational Memex projects provide additional representations, experiments, and infrastructure. A read-only database count found 3,144 conversation metadata rows and 149,623 message-embedding rows in Operational Memex. Those counts do not establish how many complete, usable histories exist, and the corpus lacks a ready-made question-answer evaluation set. Public benchmarks should carry proposal 2's reproducible evaluation; CMRI can be a later case study.

## 1. When apparent knowledge growth is a sampling effect

**Working title:** *Geometry or Sequence? Calibrating Growth Claims in AI Conversation Networks*

**Research question.** Which apparent growth patterns in thresholded semantic networks survive controls for fixed geometry, archive size, and local topic persistence? Can any surviving order effect be connected to a directly measured feature of return behavior?

**Established starting evidence.** The saved 10,000-shuffle case-study output gives densification exponent γ = 1.405, inside the permutation interval [1.254, 1.463], with empirical p = .268. The attachment exponent β = .763 is below its null mean .880, with p = .024 before adjustment for the broader set of analyses. All eleven archive summaries have densification p > .05. These are verified readings of existing outputs, not fresh full-pipeline replications. Compatibility with a null does not prove geometry is the sole explanation. [Local evidence and diagnostics](EVIDENCE.md)

**Contribution beyond prior work.** Relative to the published static CMRI paper, the contribution is a calibrated analysis of temporal claims across archives. Relative to the existing unpublished extension, the needed advance is a validated measurement pipeline, stronger restricted nulls, sensitivity analysis, and a defensible interpretation of any residual signal. Temporal randomization itself is established methodology; the application and resulting empirical correction must carry the paper. [Gauvin et al., SIAM Review, 2022](https://arxiv.org/abs/1806.04032)

A useful elementary anchor is that, for a fixed graph with N nodes and M edges, revealing a uniformly chosen set of k nodes gives expected edge count M·k(k−1)/[N(N−1)]. Each edge is present exactly when both endpoints are selected. This is an explanatory identity, not a novelty claim. The current code fits against the number of nodes with positive degree, so the identity does **not** predict its fitted exponent directly.

**Societal relevance.** Systems that infer learning, expertise, or personal change from AI conversation histories need measurements that distinguish behavior from the way a graph was constructed. This paper could establish limits on those inferences. It would not establish effects of AI use on human learning or justify diagnosing individual users.

**Available data.** Existing case-study embeddings and edge lists, eleven archive summaries, and the temporal pipeline are already local. WildChat identifiers are archive-linkage proxies, not verified identities of eleven independent people. The local raw corpus is still undergoing sanitization; a paper should rely on releasable derived data and public-data reproductions rather than assuming raw text can be shared.

**Falsifiable hypotheses.**

- **T1:** Apparent densification often remains compatible with endpoint-preserving random order after changing thresholds and node denominators. Consistent, substantial departures would contradict this account; a high p-value alone would not establish equivalence.
- **T2:** An attachment deviation remains after preserving short topic runs through block-order randomization. If the deviation disappears, local persistence is a sufficient explanation under that control, weakening a claim about longer-term return behavior.
- **T3:** Attachment deviation increases with the age of the nearest prior semantic neighbor, after accounting for archive size and similarity strength. A weak, unstable, or opposite association would reject the proposed “persistent-hub versus recent-return” interpretation. Testing the same ten archives is exploratory; new archives are needed for confirmation.

**Baselines and smallest useful pilot.** Use the current uniform arrival shuffle, a contiguous-block shuffle that changes month membership while preserving within-block order, and synthetic sequences with known return-age mechanisms on the same fixed geometry. Compare all-arrived and positive-degree node counts, several thresholds, and community-detection seeds. Start with the author archive plus two public archives spanning the observed attachment deviations. Repair and validate the event detector first; then use approximately 200 shuffles per pilot condition to estimate effect stability and runtime. The full protocol and failure criteria are in [PILOTS.md](PILOTS.md#temporal-extension).

**Readiness judgment.** This is the safest paper if its main thesis becomes careful measurement rather than a universal law of cognition. Two current results cannot be used as written: simple merge/split events are missed by the tracker, and the reported “recency index” is actually maximum similarity to any previous conversation. The numerical correlation reproduces, but it does not validate recency. If robust sequence effects fail, a focused calibration/negative-result short paper may remain worthwhile; that still requires a clear contribution beyond applying a familiar null.

## 2. Selective forgetting of connections between topics

**Working title:** *What AI Memory Compression Forgets: Testing the Loss of Cross-Topic Connections*

**Research question.** At a fixed storage budget, do memory-selection policies disproportionately lose evidence connecting topics, beyond what can be explained by rarity, age, length, degree, and ordinary semantic diversity? Does retaining those connections improve questions requiring integration across contexts?

**Completed pilot evidence.** Six query-blind policies were compared at 2,048/4,096-token caps, with k=5/10/20, 32 synthetic evaluation worlds, and 40 history-disjoint LongMemEval evaluation cases (33 eligible after strict timestamp/abstention handling). Diversity, rarity, and the bridge policy retained all required synthetic facts; all planted support had participation zero at k=10. On the benchmark, diversity and bridge retained complete support in 0/33 cases at both budgets/all k, although every gold support set fit the budgets. There were no clear cross-topic evaluation questions and two uncertain cases. No reader run was warranted. [Results, uncertainty, and reproducible code](../../papers/selective-forgetting/REPORT.md)

**Contribution beyond prior work.** The candidate contribution is a controlled, structure-conditioned account of evidence loss, followed by a small intervention on retention. It would distinguish “this fact is rare” from “this fact connects otherwise separated contexts,” and test whether structural information adds value over inexpensive diversity-based selection. A broadly novel graph-memory architecture is neither necessary nor currently justified.

The literature already covers much of the surrounding territory:

- **Selective Forgetting: A Graph-Based Memory Framework for Long-Term LLM Agents** already prunes graph memory using recency, frequency, degree, and age. This August 2026 preprint materially reduces the novelty of a generic graph-forgetting proposal. Its modest graph-storage pruning differs from this pilot's severe extractive token compression. [Primary paper](https://arxiv.org/html/2608.28978v1)
- **WhenLoss** separates write/compression and retrieval bottlenecks and proposes predictive evidence retention. Its diagnostic logic is a baseline to adopt, not a new contribution here. It is a May 2026 preprint. [Primary paper](https://arxiv.org/abs/2605.24579)
- **MemORAI** combines selective compression, a graph with provenance, and query-adaptive retrieval for conversational agents. It is a direct comparator for a later systems claim. [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1408/)
- **StreamMeCo** already uses graph connectivity to guide compression in streaming video memory. Different modality does not justify claiming the first use of topology in compression. [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.647/)
- **MemRiskBench** shows why aggregate evaluation can conceal important memory failures. Its compression contribution selects benchmark episodes; that differs from compressing an agent's stored conversation history. It is a September 2026 preprint. [Primary paper](https://arxiv.org/html/2609.14976v1)

The completed pilot did not establish the proposed bridge-specific mechanism or stable predictive benefit. A method advance remains unsupported. The [focused primary-literature comparison](../../papers/selective-forgetting/LITERATURE.md) distinguishes this screening experiment from recent full memory systems; it is not a systematic priority review.

**Societal relevance.** Personal assistants may retain frequently repeated preferences while losing an infrequent constraint connecting work, accessibility, or another part of a person's life. The proposed experiment tests this failure pattern in controlled tasks. It cannot establish demographic fairness, clinical safety, or real-world harm rates from benchmark questions alone.

**Available data.** Use the public, cleaned LongMemEval release, with evidence-session identifiers and turn-level evidence annotations; its dataset card lists an MIT license. Freeze the release separately from the newer LongMemEval-V2. Add controlled synthetic histories that vary bridge role and frequency independently. Use local CMRI embeddings only for a secondary structural replication until usable histories and evaluation questions are validated. [Official repository](https://github.com/xiaowu0162/LongMemEval), [cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)

**Falsifiable hypotheses.**

- **M1:** Bridge role predicts evidence deletion beyond frequency, recency, token length, degree, and semantic diversity. No added predictive value, or dependence on one graph threshold, would weaken the structural explanation.
- **M2:** A retention policy using bridge information improves complete-evidence retention for cross-context questions over a strong diversity baseline at equal storage cost. A practical pilot target is an improvement of at least 5 percentage points without more than a 2-point loss on matched within-context questions. These are decision thresholds chosen in advance, not established effects.
- **M3:** The improvement produces a corresponding increase in answer accuracy with the reader and retriever held fixed. If evidence preservation improves but answers do not, the result supports a representational finding, not a useful agent-memory method.

**Baselines and smallest useful pilot.** Compare recency, random reservoir selection, semantic diversity, inverse-frequency selection, PageRank retention, and diversity plus bridge information. First run an exact evidence-preservation experiment on synthetic histories, then approximately 40 held-out LongMemEval question-histories at two token budgets. Compare questions with equal numbers of supporting facts; remove gold evidence annotations from everything visible to the memory writer. Evaluate complete evidence retained, complete evidence retrieved, and reader accuracy separately. [Detailed pilot](PILOTS.md#selective-forgetting)

**Readiness judgment: narrow; no full ISCS paper on current evidence.** Bridge-minus-diversity benchmark token recall was +1.56/+3.34 percentage points at k=10, but bridge-minus-shuffled differences were −0.20/+0.51 points with intervals spanning zero. Incremental evidence prediction was unstable across k. Before another memory-policy study, validate necessary cross-topic support with independent annotations and establish that the graph score measures that construct. Use new held-out histories after any generator/measurement redesign. A measurement-focused short paper would still need independent validation and a clearer contribution than this pilot alone.

## 3. Transit restoration under uncertain public data

**Working title:** *Uncertain Inputs, Unequal Recovery: Testing Transit Restoration Priorities*

**Research question.** How stable are recommendations for restoring disrupted transit service when accessibility thresholds, service conditions, and neighborhood estimates are uncertain? Does a restoration decision chosen for robustness protect access for households without vehicles better than an equity-aware decision optimized for one assumed world?

**Established starting evidence.** No transit experiment exists in CMRI. The transferable asset is network analysis, not a validated transport model. Public sources support a plausible study, but the schedule feed must pass an access and routing check before this direction is selected.

**Contribution beyond prior work.** Equity in transit resilience and robust disruption optimization are already established subjects. Zhang's 2021 doctoral thesis explicitly combines network resilience with transport equity. Mo et al.'s published disruption study optimizes route recommendations under uncertain demand; Xu et al. optimize rail disruption tolerance under downtime uncertainty. [UCL thesis abstract](https://discovery.ucl.ac.uk/id/eprint/10132133/), [Mo et al., Transportation Research B, 2023; revised manuscript 2025](https://arxiv.org/abs/2201.01437), [Xu et al., Transportation Science, 2021](https://pubsonline.informs.org/doi/10.1287/trsc.2021.1040)

A current author-hosted study also optimizes route headways for accessibility in areas of persistent poverty. Merely adding census weights to transit connectivity would be insufficient. [Kostopoulou et al., author-hosted abstract](https://people.umass.edu/jimi/publications/2026-kostopoulou-scenario-discovery-approach/)

The narrower candidate contribution is a reproducible audit of **decision instability and out-of-scenario regret** for restoration choices made from public schedules and demographic estimates, comparing nominal and robust versions of the same equity objective. Whether that combination is sufficiently new remains unverified; a St. Louis case study alone would not establish general novelty.

**Societal relevance.** Restoring a structurally central route may improve network connectivity while doing little for job access in places with many households lacking vehicles. Conversely, an apparently equitable repair could be fragile to plausible assumptions. The outcome is modeled accessibility, not observed rider welfare or an operational recommendation to the transit agency.

**Available data.** Metro St. Louis lists static GTFS and real-time feeds. The static download returned HTTP 403 in this environment, so usable schedule data are not yet verified. Census ACS table B08201 supplies household vehicle availability estimates and margins of error; LEHD LODES supplies workplace job counts. They are neighborhood and employment measures, not transit ridership. Geographic coverage must include the served Missouri/Illinois footprint and document excluded operators and mismatched data years. [Metro developer resources](https://www.metrostlouis.org/developer-resources/), [ACS B08201 schema](https://api.census.gov/data/2024/acs/acs5/groups/B08201.json), [LEHD data](https://lehd.ces.census.gov/data/)

**Falsifiable hypotheses.**

- **R1:** Connectivity-based restoration rankings disagree materially with rankings based on reachable jobs. Preregister a practical disagreement threshold; stable agreement would weaken the case for replacing the simpler metric.
- **R2:** Nominal equity-aware repair rankings change across plausible input scenarios and incur material regret. Stable rankings and negligible regret would undermine the need for robust selection.
- **R3:** Robust equity-aware selection reduces held-out worst-group accessibility loss compared with nominal equity-aware selection, with a reported aggregate-access tradeoff. It fails if the advantage disappears outside the scenarios used to select the repair, or if the tradeoff is unacceptable under the predeclared criterion.

**Baselines and smallest useful pilot.** Compare random restoration, degree/centrality priorities, giant-component restoration, aggregate job accessibility, nominal equity-aware accessibility, and a robust version of that same objective. Use one service weekday and one Saturday, about 30 origin areas, and 30 three-route outage scenarios. Enumerate the three possible single-route restorations rather than introducing an optimizer. Start with an explicit equal-cost scenario, then vary relative costs. Evaluate held-out service and accessibility assumptions. [Detailed pilot](PILOTS.md#transit-restoration)

**Readiness judgment.** Strongest direct societal story, but lowest readiness. A correct timetable and pedestrian-access model is essential: an undirected stop graph cannot substantiate claims about reachable jobs. Continue only if the feed and routing model are usable quickly and pilot regret is consequential. This direction would benefit most from a transport-domain collaborator; it does not require recruited participants.

## Decision and timetable

The memory pilot is complete and its continuation gate failed. Spend **two focused days validating the temporal measurements**, then decide whether the corrected extension supports a full or short ISCS paper. Keep the narrowed memory measurement question and transit as separate opportunities; neither currently warrants displacing the temporal submission work.

| Target | Deliverable and decision |
|---|---|
| By October 23 | Temporal measurement issues reproduced, repair scope estimated, key claim language corrected in a future manuscript revision |
| Completed October 14 | Memory pilot, matched controls, uncertainty, figures, and novelty comparison saved; recommendation: narrow, no reader expansion |
| By November 6 | Use the temporal measurement gate to settle the main submission and scope |
| November–December | Freeze protocol and data versions; run the main experiment and independent validation |
| By January 15 | Complete results, robustness analyses, and an explicit contribution relative to prior publications |
| By February 5 | Full manuscript draft, within the chosen 8- or 12-page format |
| By February 19 | Coauthor review, reproducibility check, and final format/archival-policy check |
| March 1 | Submission deadline; retain the final week as buffer |

Human–AI collective creativity remains interesting, but it is not in the top three under the no-recruitment preference. Human creative-diversity effects already have experimental evidence, and current work directly studies diversity collapse in interacting LLM systems. A simulation-only study would need to make claims about its simulated agents rather than people. [Doshi and Hauser, 2024](https://doi.org/10.1126/sciadv.adn5290), [Diversity Collapse in Multi-Agent LLM Systems, Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.13/)
