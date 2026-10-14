# Pilot plans and stopping rules

The selective-forgetting plan below was executed on October 14, 2026; temporal and transit remain plans. The source review, initial diagnostic audit, and completed memory results are indexed in [EVIDENCE.md](EVIDENCE.md). Effort estimates assume one researcher with focused time, a CPU workstation, and no participant recruitment. No GPU allocation or paid model budget is assumed.

The purpose of each pilot is to decide whether a larger study is justified. Pilot thresholds below are practical decision criteria; they are not power calculations, promises of statistical significance, or estimates of expected effects. Save configuration, dataset revision, hashes, seeds, model/tokenizer revision, failures, runtime, and peak memory with every run. Keep development choices separate from held-out evaluation.

## Temporal extension

**Smallest decision:** does a valid sequence effect remain after correcting measurement problems and preserving local persistence?

**Days 1–2: measurement gate.**

1. Run [audit_existing_results.py](audit_existing_results.py) against the current temporal checkout. The saved [audit-results.json](audit-results.json) records the examined source hashes and observed failures.
2. Define community events from the full bipartite overlap relation between consecutive partitions before assigning persistent identities. A continuation match must not consume a community before testing for a merge or split. Define how growth, partial overlaps, disappearances, and many-to-many transitions are handled; do not silently count a compound transition as a simple merge.
3. Add unit cases for unchanged partitions, label permutations, two-to-one merges, one-to-two splits, births, deaths, ties, and threshold boundaries. Add an integration case using a known sequence of partitions. These cases must establish the event definition before rerunning real data. The proposed paper can also omit lifecycle claims if they are unnecessary, but cannot retain the current zero-event inference without validation.
4. Keep the existing revisit-similarity statistic under its correct name. Add a separate return-age measure: for each arrival, find its closest earlier semantic neighbor, average normalized lag over exact ties, and summarize the distribution. Report match similarity alongside lag, since a distant unrelated neighbor is not evidence of revisiting. Inspect normalization by archive length and calendar time.

The audit has diagnosed these issues; it has not implemented these repairs.

**Days 3–5: scientific pilot.**

- Use the author archive and two nondegenerate public archives chosen in advance to span the existing attachment deviations. This is diagnostic sampling, not a population sample. Freeze inclusion criteria before any broader replication.
- At the existing threshold, compare observed order with 200 uniform timestamp-assignment permutations and 200 permutations of contiguous 20-conversation blocks. Remap the resulting sequence to the original timestamp slots. This preserves the monthly arrival counts, final graph, and most short-run adjacency within blocks, while changing where blocks occur in the history. Also examine block lengths 10 and 40 as sensitivity checks.
- Do not substitute within-month shuffling for the block control: membership in cumulative monthly snapshots would be unchanged, so most snapshot metrics could not test the intended question.
- Repeat the main comparison at three prespecified similarity thresholds around the existing 0.9 setting, such as 0.85/0.90/0.95 with the same encoder. Record fragmentation and unsupported fits rather than dropping inconvenient conditions. Show both all-arrived and positive-degree node denominators for densification.
- Construct synthetic fixed-geometry sequences with random order, short topic runs, and repeated returns to old hubs. Use them to check whether the attachment and lag measurements distinguish the mechanisms they are supposed to measure. Fit diagnostics must reject or flag degenerate attachment kernels.
- Canonicalize graph/node ordering and examine multiple community-detection seeds. Final-graph quantities are fixed in principle; variation caused by a heuristic or path-sampling estimator must not be presented as a temporal effect.

**Measurements.** Report observed-minus-null effects and Monte Carlo intervals for γ and β, graph coverage, fitted-point counts, and the directly measured return-age distribution. For trajectories, choose a single whole-trajectory statistic or simultaneous bands before testing; avoid treating each month as an independent discovery. The case-study p = .024 is exploratory evidence until the family of tests and sensitivities is accounted for.

**Continue if:** the measurement tests pass, core outputs are reproducible, and either a meaningful sequence effect survives the restricted null or a well-calibrated negative result demonstrably changes the interpretation of an important claim. The latter needs positive-control power checks; failure to reject alone is insufficient.

**Narrow or stop if:** β changes sign across reasonable thresholds, most archives lack estimable kernels, or local-run controls explain everything while the proposed paper offers only a routine shuffle analysis. Remove “recency-driven” labels if the direct lag measure does not support them. A failed behavioral interpretation is not repaired by renaming the axis again.

**If continued:** use all eligible existing archives, seek an independent archive set for confirmation, scale shuffles according to measured runtime and needed Monte Carlo precision, and correct across the prespecified primary tests. Bootstrap or summarize at the archive level; eleven archives are not thousands of independent behavioral observations. Deliver three core figures: null-calibrated growth, robustness of attachment effects, and direct sequence-measure validation. Estimated main-study effort: 4–6 researcher-weeks.

## Selective forgetting

**Completed pilot decision (October 14): NARROW.** All six policies, both budgets,
three graph settings, history-disjoint development/evaluation, shuffled controls,
incremental prediction, matched diagnostics, and uncertainty are implemented and
run. The benchmark has no clear cross-topic evaluation cases; the synthetic
generator has a retention ceiling and a failed bridge proxy. Benchmark complete
support is 0/33 for diversity and bridge at both budgets/all k despite every gold
support set fitting the cap. Small token gains do not clearly beat shuffled
scores. The original reader gate below is not met, so reader evaluation is
deliberately omitted. No positive result is claimed.
[Report and figures](../../papers/selective-forgetting/REPORT.md),
[reproduction](../../papers/selective-forgetting/README.md),
[frozen operational protocol](../../papers/selective-forgetting/PROTOCOL.md).

Retain the original plan and stopping criteria below as the decision record. Any
continuation should first validate cross-topic support and the structural proxy,
then use a new evaluation set; do not retune this completed sample for a gain.

**Smallest decision:** does graph structure explain a memory failure that rarity and semantic diversity do not?

**Days 1–2: data and controls.**

1. Pin the cleaned LongMemEval release and create a small development partition and roughly 40 held-out question-histories, including both multi-session and simpler questions. Group by shared history before splitting. Multi-session does not automatically mean cross-topic: classify query support separately, without consulting policy outcomes. If there are too few suitable cases, report that rather than forcing a balanced story.
2. Give the memory writer only the history available before the question. Remove question text, answers, `has_answer`, and `answer_session_ids` from writer/selector inputs. Retain gold annotations in a separate evaluator. Check this exclusion structurally, not by searching strings in a prompt. Prevent future turns from entering an earlier memory snapshot.
3. Construct about 32 synthetic history worlds with balanced probes crossing bridge role (between/within contexts) and evidence frequency (rare/common). Match support count, fact age, length, and approximate degree; counterbalance surface wording. Synthetic examples establish measurement behavior under controlled conditions, not prevalence in natural conversations.
4. Freeze one encoder, tokenizer, graph construction, and community algorithm on development data. A practical starting graph is symmetrized k-nearest-neighbor similarity with k = 10, checked at k = 5 and 20. Use the participation coefficient P(i) = 1 − Σc[k(i,c)/k(i)]² as one continuous structural score, defining P = 0 for isolates. It is a candidate proxy, not ground truth for semantic importance.

**Days 3–4: retention experiment, without a generative reader.**

Evaluate six query-blind policies at 2,048 and 4,096 stored-content tokens: latest items, uniform reservoir, semantic diversity, inverse-frequency weighting, PageRank, and diversity plus participation score. Ensure source histories are long enough that both budgets require compression. Token-count the stored records and their provenance headers; report unused capacity and handle oversized records consistently. Use a common storage format so comparisons are about selection rather than serialization. Charge any persistent graph, embeddings, or index separately in bytes, and report preprocessing cost.

Start with extractive retention. This keeps evidence survival directly observable. An abstractive summarizer is a later baseline, requiring validation of whether the summary actually preserves each fact. It must not be assumed equivalent to an extractive selector.

Primary outcomes are complete support-set retention and the difference between cross-context and matched within-context questions. Also report evidence recall and coverage by frequency, age, and structural role. Use repeated reservoir seeds for the retention analysis. Estimate the added predictive value of structural role over the other features using held-out histories; a larger graph score is not by itself a useful result.

**Days 5–7: downstream diagnostic, only if retention warrants it.**

- Fix one reader, prompt, tokenizer, and retriever across policies. Evaluate oracle evidence, all stored content, and retrieved stored content. Keep the reader's total context allowance sufficient for the largest stored-content budget, and fix the retrieval allowance across policies. These views adapt established memory diagnostics; score differences are conditional performance gaps, not automatically isolated causal mechanisms.
- A cap of approximately 1,000 reader calls covers 40 questions × 6 policies × 2 budgets × 2 memory views, plus 40 oracle calls. Choose an available local reader first; price any paid alternative before running it. Record runtime before expanding to multiple readers.
- Use the benchmark's scoring protocol, with a small blinded audit of judgments. Preserve abstention questions and report unsupported answers. An apparent improvement from answering everything is not a memory improvement.
- Ablate the structural term, shuffle structural scores within degree/frequency strata, and compare with semantic diversity under the same budget. A later diagnostic may restore a lost gold record to investigate an error, but that oracle intervention is not a deployable policy and cannot count toward the main score.

Use paired differences and uncertainty grouped by independent history, not by turns or multiple probes from the same synthetic world. The small pilot can reveal feasibility and effect direction; it cannot settle small population effects.

**Continue if:** structure adds stable predictive value beyond frequency/diversity controls, and the structural policy improves complete evidence retention by about 5 points on cross-context questions without losing more than 2 points on matched within-context questions. Check at both budgets and more than one graph setting. Reader accuracy should move in the same direction before making an agent-performance claim.

**Narrow or stop if:** the signal disappears after matching rarity or degree; query/evidence labels leaked into the writer; the effect is confined to planted examples; or simple diversity matches the proposed method. Do not count the obvious fact that deleting required evidence impairs an answer as evidence for a bridge-specific mechanism.

**If continued:** expand to the full eligible benchmark, add a second reader and an independent dataset, compare a reproducible implementation of a recent compression/graph-memory method, and test summarization as well as extraction. Distinguish batch compression from online memory updates; the initial pilot supports the former. Deliver a budget-performance curve, a matched structural-loss analysis, and an ablation table. Estimated main-study effort: 7–10 researcher-weeks.

## Transit restoration

**Smallest decision:** do plausible public-data assumptions change restoration choices enough to matter for accessibility?

**Days 1–2: data-access gate.**

Obtain a usable, dated Metro GTFS snapshot from the official feed or a documented, legitimately available archive. Record its license/terms, checksum, service dates, and operator coverage. The official endpoint returned HTTP 403 during this review; data availability is therefore a gate, not an accomplished step. If a suitable feed remains unavailable, stop this pilot or explicitly select another system before doing analysis.

Pin ACS 2024 five-year B08201 estimates/margins and an available LODES workplace vintage. Join on compatible geographies and document the mismatch between schedule year and demographic/job vintages. The relevant ACS fields are total households (`B08201_001E`) and households without vehicles (`B08201_002E`), with corresponding `M` fields. LODES provides job counts, not observed transit demand, and does not supply ACS-style sampling margins.

**Days 3–5: routing validation.**

Construct a timetable-aware routing model respecting service calendars and exceptions, direction, transfers, overnight times beyond 24:00, and valid walking paths. Validate a set of ordinary and disrupted journeys against published schedules. Euclidean walking distance may be a smoke test, but cannot be the final access model where rivers, highways, or station access matter. Add unit cases for missed transfers, service-day exceptions, midnight crossings, and unreachable destinations; add an integration case with a manually checked itinerary.

Use roughly 30 origin areas spanning the observed vehicle-availability distribution and a weekday/Saturday pair. Cover the transit-served area on both sides of the state line where data permit. Disclose missing operators and excluded areas, including places with zero baseline access; silently excluding those places would bias the equity analysis.

**Days 6–10: decision experiment.**

- Sample 30 disruptions, each removing three route services, and enumerate restoration of each one. This is an explicit service-restoration scenario, not a model of physical infrastructure repair costs. Begin with equal-cost choices; test alternative route-cost assumptions before generalizing the budget claim.
- Define A(i,r,s) as jobs reachable from origin i within a travel-time limit after repair r under scenario s. Use fixed departure windows and include walking, waiting, and transfers. Main limit: 45 minutes; sensitivity: 30 and 60 minutes. Vary service delays and walking access within documented plausible ranges.
- Define group accessibility retention Q(g,r,s) as the group-weighted sum of A after disruption/restoration divided by the same weighted sum under undisrupted service in scenario s. Use ACS counts for households with no vehicle and households with vehicles as two exposure weights. These are area-based exposure proxies, not observed individual journeys. Report raw access changes as well as ratios.
- Compare random, degree/centrality-based, giant-component, and aggregate-accessibility choices with **nominal equity** (maximize the minimum group retention at central assumptions) and **robust equity** (maximize the minimum group retention across the prespecified scenario set). Use identical candidates and budgets for every comparison. Include the nominal equity baseline so an improvement cannot merely reflect changing the objective.
- Evaluate repairs on held-out disruption/settings combinations. For each world, enumeration supplies the best feasible comparator. Report regret relative to that world-specific comparator, group losses, aggregate access, and how often the top choice changes. Show uncertainty-induced rank changes separately from changes caused by choosing a different social objective.

ACS margins can guide sensitivity bounds, but do not justify drawing every estimate independently from an unconstrained normal distribution. Keep household totals and subgroups consistent. Treat unspecified dependence and operating-cost variation as scenario assumptions, not as known probability distributions.

**Continue if:** validated routes and inputs are available, nominal choices change in at least 25% of the prespecified sensitivity comparisons, and those changes correspond to consequential regret—provisionally at least 5 percentage points of group access retention in a meaningful subset of disruptions. Prefer the robust method only if held-out worst-group loss improves and its aggregate-access cost is acceptable under a criterion fixed before evaluation.

**Narrow or stop if:** routing validation fails, ranking changes have negligible consequences, the result comes only from extreme assumptions, or robust selection merely wins on the same scenarios used to construct it. A centrality map alone is insufficient for the proposed claim.

**If continued:** enlarge origin/disruption coverage, include an independently held-out service period or system, examine cost and grouping choices, and obtain transport-domain review. Deliver a restoration-regret distribution, an accessibility tradeoff plot, and a map explaining representative failures. Estimated main-study effort: 9–12 researcher-weeks, with the highest setup risk of the three options.
