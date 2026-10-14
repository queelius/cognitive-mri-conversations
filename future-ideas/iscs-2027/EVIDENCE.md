# Evidence, limits, and reproducible checks

Checked October 14, 2026. This file distinguishes existing results from computations performed for the proposal comparison. It is a targeted research audit, not a complete review or reproduction of any manuscript.

## Evidence status

| Claim used in the comparison | Evidence | What it establishes |
|---|---|---|
| ISCS 2027 takes place in Porto, June 2–4; deadline March 1 | [Official call](https://iutdijon.u-bourgogne.fr/ccs-france/) | Current advertised dates and scope |
| 12-page long, 8-page short, and 2–4-page abstract formats | [Submission rules](https://iutdijon.u-bourgogne.fr/ccs-france/submissions/) | Current submission categories; not a verified journal-reuse policy |
| Existing CMRI publication and project inventory | [Compendium README](../../README.md), the linked current manuscript sources | What is already claimed or published; not independent replication of every numerical result |
| Hierarchical-memory work was on the ISCS 2026 program | [Official program](https://iutdijon.u-bourgogne.fr/ccs-france/wp-content/uploads/2026/05/PROGRAM-ISCS-2026-V2.pdf) | Prior conference presentation listing |
| Temporal case-study null values and eleven-archive summary | Local JSON/CSV files hashed in [audit-results.json](audit-results.json) | Existing stored results were inspected; the 10,000 permutations were not rerun |
| Community event detector misses simple transitions | Actual tracker function run on synthetic partitions | Reproducible measurement failure on these cases, not proof that all possible merges/splits are undetectable |
| “Recency” statistic does not measure return age | Actual index function, synthetic sequences, stored-index correlation recomputation | Numerical association reproduces, but the label/interpretation is unsupported by this statistic |
| Operational Memex contains 3,144 conversation metadata rows and 149,623 message-embedding rows | Read-only SQL counts in `papers/operational-memex/data/analysis.db` | Available infrastructure scale; not counts of complete evaluated histories or validated vectors |
| Public benchmark and transit inputs exist | Official LongMemEval, Metro, Census, and LEHD pages | Documented availability/schema; actual transit feed download did not succeed here |
| Participation-based retention adds value beyond simpler controls | Completed [selective-forgetting pilot](../../papers/selective-forgetting/REPORT.md), six policies/two budgets/three k settings | No stable added value established; synthetic measurement failure and no clear cross-topic benchmark coverage prevent the broader mechanism claim |
| Robust transit repair improves equity | No transit experiment performed | Proposed, falsifiable hypothesis |
| The precise proposed memory/transit contributions are new | Targeted primary-literature review only | Candidate gaps; priority and exhaustive novelty remain unverified |

## Temporal findings checked directly

The examined manuscript is [paper/PLOS/paper.tex](../../papers/temporal-networks/paper/PLOS/paper.tex), with its June revision history in [CHANGELOG.md](../../papers/temporal-networks/paper/PLOS/CHANGELOG.md). The audit JSON records the exact child-repository commit and SHA-256 hashes of the source/data files used. Repository summaries can be stale, so numerical statements here use the actual saved data and current functions.

### Existing permutation output

| Quantity | Observed | Null mean | Central 95% permutation interval | Stored empirical p |
|---|---:|---:|---:|---:|
| Densification γ | 1.4050 | 1.3480 | [1.2537, 1.4633] | .2682 |
| Attachment β | .7632 | .8798 | [.7803, .9781] | .0242 |

The case-study output specifies 10,000 shuffles, preserving the final graph and monthly arrival counts. All eleven rows in the multiarchive summary have γ p > .05. One archive has a degenerate β = 0 and is excluded from the stored index comparison, leaving ten rows. These are conditional comparisons for selected archives; the β p-value is not adjusted for the surrounding set of exploratory tests.

In particular, “inside the null interval” means compatible with this randomization. It does not prove there is no sequence effect, establish that geometry alone caused an observation, or validate a cognitive mechanism. Positive-control simulations and uncertainty about effect size are needed for a convincing negative finding.

### Merge/split detector: failing synthetic examples

The current [tracker](../../papers/temporal-networks/code/temporal_community_tracker.py) assigns mutual best overlaps as continuations before considering merges and splits. The later branches exclude communities already assigned. At the default Jaccard threshold of .3, two disjoint three-node communities each overlap their six-node union by .5, so both relationships satisfy the overlap criterion.

| Transition between consecutive months | Expected event | Observed events in second month |
|---|---|---|
| `{a,b,c}`, `{d,e,f}` → same two communities | Two continuations | Two continuations |
| `{a,b,c}`, `{d,e,f}` → `{a,b,c,d,e,f}` | Merge | One continuation; no merge |
| `{a,b,c,d,e,f}` → `{a,b,c}`, `{d,e,f}` | Split | One continuation and one birth; no split |

This is sufficient to invalidate reliance on zero detected merge/split counts without further validation. It does not determine the true event counts in the real data. The permutation pipeline reuses this tracker, so agreement between observed and shuffled zero counts is not independent validation of the detector. No tracker repair or real-data event recomputation was performed in this task.

### Revisit similarity is not recency

The current [index function](../../papers/temporal-networks/code/exploration_index.py) computes the mean, across arrivals, of the maximum cosine similarity to **any** earlier conversation. There is no age or time-distance term. Its companion step index uses only adjacent conversations.

For three orthogonal topic vectors A, B, and C:

| Sequence | Step index | Revisit similarity |
|---|---:|---:|
| A, A, B, B, C, C | .4 | .6 |
| A, B, C, A, B, C | 1.0 | .6 |

The same revisit value occurs with immediate versus delayed repetitions. This demonstrates the distinction in meaning; it does not imply revisit similarity could never correlate with a time-based measure in real data.

Recomputing Spearman correlations from the ten stored rows gives:

- Step index versus z(β): ρ = +.4061, p = .2443.
- Revisit similarity versus z(β): ρ = −.7356, p = .0153.

The numerical claim reproduces. Calling the second statistic a recency measure does not. A separately defined lag statistic and independent validation are needed before attributing the attachment deviation to recent returns versus old hubs.

### Construction and sampling limits

The current densification fit uses the number of positive-degree nodes, not the total number of conversations that have arrived. Those denominators answer different questions. The elementary fixed-graph expectation in the proposal memo uses all N nodes and k arrivals; it should not be substituted for a prediction of the current fitted γ.

The final graph is held fixed under the arrival shuffle. Exact endpoint graph properties therefore cannot establish a discovery about order. Louvain partitions and sampled path estimates can vary because of algorithmic details even on that same graph; this variation must be separated from order effects.

The manuscript describes WildChat linkage using hashed IP/request metadata. Such archives are not verified individual identities. Selection for substantial activity also limits generalization. Existing data support a multiarchive comparison, not a representative study of human exploration styles.

## Source review and data-access limits

The primary-literature comparisons are linked beside the relevant claims in [README.md](README.md). The review checked the current abstracts and available methods for the closest memory papers, the temporal randomized-reference-model literature, and transit resilience/accessibility work. It was a targeted search, not a systematic review.

Publication status matters: Gauvin et al. has a SIAM Review journal reference; MemORAI and StreamMeCo are in Findings of ACL 2026; WhenLoss and MemRiskBench were assessed as 2026 preprints. The cited UCL transit source is a doctoral thesis whose full text is unavailable on that repository. Mo et al.'s arXiv record links a 2023 Transportation Research B article and has a 2025 manuscript revision. The Kostopoulou study was checked through its author's abstract; no detailed algorithm comparison was inferred from that abstract alone.

For memory data, the [LongMemEval repository](https://github.com/xiaowu0162/LongMemEval) documents evidence annotations and the [cleaned dataset card](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) lists MIT licensing. The initial proposal review did not download it; the subsequent pilot successfully parsed the pinned S JSON and recorded its checksum. The viewer error did not prevent use. The newer V2 remains a separate candidate requiring its own protocol and data check.

## Completed selective-forgetting pilot

The [child repository](../../papers/selective-forgetting/README.md) contains pinned
data/model revisions, dependencies, the frozen protocol, 92 completed development
and evaluation histories/worlds, all selections and scores, runtime, tests, and
standalone figures. The [report](../../papers/selective-forgetting/REPORT.md)
distinguishes synthetic results from benchmark evidence and gives uncertainty.

The 500 benchmark histories share sessions transitively. An outcome-blind
independent-set sample yields 84 mutually disjoint histories; fixed 12/40
development/evaluation subsets share no session ID or identical session content.
This favors low-overlap histories and is not representative benchmark sampling.
Strict pre-question filtering leaves 33 annotated evaluation cases, five with
same-day later gold sessions, and two abstentions. None was replaced.

Diversity, rarity, bridge, and shuffled-bridge retention are perfect on all 32
synthetic evaluation worlds at both budgets/all k. Yet all planted support has
participation zero at primary k=10, including cross-topic wording. The generator
therefore fails to validate the intended structural mechanism. Nuisance controls
already predict its support perfectly.

On the benchmark, diversity and bridge retain complete annotated support in 0/33
eligible cases (individual exact 95% rate interval 0–10.6%) at both budgets/all k.
All support sets fit: median 179 tokens, maximum 645. At k=10 bridge-minus-diversity
gold-token recall is +1.56 points [0.00, 4.09] at 2,048 and +3.34 [0.00, 7.87] at
4,096. Against shuffled scores it is −0.20 [−3.60, 2.77] and +0.51 [−4.54, 5.80].
These are descriptive history-bootstrap intervals. Added predictive performance
varies by k and its intervals include zero. No evaluation question was clearly
cross-topic under the pre-outcome, single-analyst annotation; two remain uncertain.

The continuation gate fails, so no reader run was performed. **Recommendation:
narrow the memory work to validating evidence/measurement; do not expand this
pilot into a full ISCS paper.** This does not establish that bridges never matter.
The added [Selective Forgetting preprint](https://arxiv.org/html/2608.28978v1)
also weakens a generic graph-memory novelty claim; see the
[primary-literature comparison](../../papers/selective-forgetting/LITERATURE.md).

For transit, the [Metro developer page](https://www.metrostlouis.org/developer-resources/) links the static `google_transit.zip` feed and real-time feeds. A direct request to the static feed returned HTTP 403 here. No feed dates, row counts, or successful routing results are claimed. The [Census API schema](https://api.census.gov/data/2024/acs/acs5/groups/B08201.json) was successfully queried to verify total/no-vehicle household estimates and margins. The [LEHD data documentation](https://lehd.ces.census.gov/data/) supports workplace counts as a candidate destination measure; those counts are not actual transit trips.

## Reproduce the limited local audit

From the compendium root, with the temporal child repository present:

```bash
python3 -m venv /tmp/cmri-iscs-audit
/tmp/cmri-iscs-audit/bin/python -m pip install numpy==2.2.6 scipy==1.15.3 networkx==3.4.2 python-louvain==0.16
/tmp/cmri-iscs-audit/bin/python future-ideas/iscs-2027/audit_existing_results.py > /tmp/cmri-iscs-audit-results.json
```

The checked run used Python 3.10 and those package versions. The script reads aggregate outputs and imports existing functions, uses only synthetic conversation identifiers in its fixtures, disables bytecode writing for child imports, and prints strict JSON. It does not read raw conversation content, alter the temporal manuscript/code, or rerun the large permutation experiment. Compare the source hashes before interpreting a later run as a reproduction of these findings.

The continuation control passed; the merge and split expectations failed as recorded. A successful script exit means the diagnostics executed, not that the temporal pipeline passed its tests. No unit/integration suite for the child projects was run or certified by this research review. New experimental code and repairs are future work and have explicit validation steps in [PILOTS.md](PILOTS.md).
