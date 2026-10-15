# Bounded public-data check

Completed 2026-10-15. **Public conversation-level measurement is feasible at low cost; this check does not establish a reliable population of large personal archives or a precise role advantage.** The protocol was frozen before record acquisition. The test is exploratory and uses a preselected convenience slice.

## What was acquired

WildChat-4.8M revision `c827c6df8fcf008219ffaffa4d1dd77491099367`, source shard `00000`, and the matching domain/intent outputs from Adopt != Adapt revision `b4e77e44b62ba2ddd0f7b53667b218e345faba09`. The two label files have 37,208 rows each. We acquired only the first 2,000 source content rows through verified HTTP ranges. Total record transfer was **39,091,662 bytes (37.28 MiB)**, below the 60 MiB cap; source-content/footer transfers were 6,644,270 bytes. Acquisition took 50.24 seconds. No paid inference or full-corpus download was used. See [source-manifest.json](source-manifest.json) for LFS hashes, each byte-range hash, and downloaded-sample hash.

## Archive eligibility

The annotation shard spans April 9–27, 2023. These are exact counts for this shard, not an estimated number of archives in WildChat.

| Quantity | Observed |
|---|---:|
| Physical rows / distinct content hashes | 37,208 / 36,821 |
| Repeated-content excess rows / maximum multiplicity | 387 / 63 |
| Hashed-IP proxies | 9,645 |
| Proxies with one distinct conversation | 5,114 (53.0%) |
| Proxies with one active UTC date | 7,451 (77.3%) |
| Median / 90th / 99th percentile conversations per proxy | 1 / 7 / 41 |
| At least 20 conversations and 5 dates, before / after within-shard screen | 177 / 144 |
| At least 50 conversations and 5 dates, before / after screen | 51 / 30 |
| At least 100 conversations and 5 dates, before / after screen | 16 / 4 |

The screen rejects proxies with more than 161 rows or more than three countries, states, or detected languages. Eight proxies exceeded the volume threshold; 53 exceeded the language threshold. Country/state variation did not trigger rejection in this shard. Passing this screen does not verify an individual. A changing IP can fragment a person; a shared IP can merge people. A shard truncates histories and cannot support the full-release screen or extrapolation. The relevant uncertainty is linkage and coverage, so a binomial interval over conversation rows would be misleading.

## Role representation check

All 2,000 source records joined to their annotation keys without expansion; 1,994 had both substantive labels. Each 37,208-row annotation file had zero collisions on the eight-field composite key. The content slice covers April 9–10 only. Shared-IP or normalized first-user-prompt-prefix grouping across **all 2,000 acquired rows, before any eligibility filter**, produced 787 connected components. English, nonempty-role, complete-label and exact-content filters left 1,020 candidates occupying 406 of those components; 763 candidates belonged to components with multiple eligible records. One deterministic eligible representative per component yielded **245 development and 161 evaluation conversations**. An independent traversal of the full acquired graph verifies no component crosses the split and no component contributes more than one selected row. Links through unacquired source rows remain unknown.

An earlier implementation grouped only eligible rows, allowing excluded non-English records to sever dependency links. Review identified three full-source components crossing that earlier split. Those estimates and their independence claim are superseded; old outputs, failing regressions and the correction specified before rerunning are preserved in [DEVIATIONS.md](DEVIATIONS.md).

Cached `all-MiniLM-L6-v2` CPU embeddings use up to four 254-token chunks per role, with normalized chunk means. The cap bound 44/406 user representations and 115/406 assistant representations. This asymmetry limits the interpretation as a comparison of complete transcripts. Models are fixed nearest-centroid classifiers fit on development labels, not a tuned model benchmark. Classifier targets are the released whole-conversation GPT labels, not independently elicited user intentions.

| Released target | Evaluation N | Majority accuracy | User accuracy | Assistant accuracy | Equal role mean accuracy |
|---|---:|---:|---:|---:|---:|
| Domain/topic | 159 | 21.4% | 43.4% | 44.0% | 48.4% |
| Intent/purpose | 161 | 28.6% | 38.5% | 41.0% | 42.2% |

Two domain-label evaluation cases were unseen in development and are excluded from the domain estimates, as specified in the protocol. They remain in the manifest. Macro recalls (user / assistant / mixed) were 39.0 / 40.8 / 46.7% for domain and 24.6 / 46.4 / 32.1% for intent; rare-category counts are small, and these are descriptive.

| Paired accuracy difference | Estimate | 95% component bootstrap interval | Sign-flip p |
|---|---:|---:|---:|
| Assistant minus user: domain | +0.6 pp | [−7.5, +8.8] pp | 1.000 |
| Mixed minus user: domain | +5.0 pp | [−0.6, +10.7] pp | .150 |
| Assistant minus user: intent | +2.5 pp | [−5.0, +9.9] pp | .630 |
| Mixed minus user: intent | +3.7 pp | [−0.6, +8.1] pp | .212 |

Intervals use 2,000 paired bootstrap resamples; randomization tests use 5,000 sign flips. The estimand is the paired accuracy difference among selected held-out component representatives with labels seen in development, conditional on this convenience slice, grouping rule, fitted centroids and token budget. One selected conversation per observed component makes component and conversation resampling equivalent here; unobserved external links and training-set uncertainty are not included. These are exploratory, unadjusted comparisons; no interval excludes zero. The mixed-domain point estimate reaches 5 pp, but the predeclared joint requirement of at least 5 pp **and** an interval excluding zero was not met. That is **inconclusive evidence about relative roles**, not equivalence or a demonstration that roles do not matter. Representations descriptively outperform the majority baseline, so the release is useful for developing topic/purpose measurements. The check provides no test of incremental network structure beyond embeddings.

## Verification and cost

- **25 tests passed**, including source/annotation joins, excluded-row bridge regressions, an independent full-source split reconstruction, fixed-embedding normalization, missing-label handling, paired uncertainty and bounded HTTP reads. [Test output](tests.txt), [failures before the grouping correction](regression.before-fix.txt)
- Line coverage: **audit.py 99%, acquire.py 54%, combined 84%**. Acquisition orchestration was exercised by the live pinned download, but that first run was outside coverage; byte-range failure modes are mocked in unit tests. The excluded-bridge regressions exercise the insufficient-sample stop. Remaining uncovered analysis paths are defensive nonadvancing-chunk and empty-pair errors plus the rare token-budget boundary correction. [Coverage details](coverage.txt)
- Final analysis took **60.94 seconds** on four CPU threads. Package versions, model hashes and implementation fingerprints are recorded in [results.json](results.json). Earlier results are preserved separately; corrections handle a serialized missing-label sentinel, enforce the exact token budget at WordPiece boundaries, and retain dependency bridges through excluded rows. All **8,588 chunks** from the acquired content slice satisfy the final 254-token bound and preserve the original text prefix. [Corrections](DEVIATIONS.md), [chunk validation](chunk-validation.json), [analysis output](run.txt)

Reproduction from the repository root, using the existing model environment without modifying it. Install the extra dependency and acquire records only when their local caches are absent:

```bash
python -m pip install --target future-ideas/iscs-2027/research-deep-dive/checks/public-cross-section/.deps --python-version 3.11 --abi cp311 --platform manylinux_2_28_x86_64 --implementation cp --only-binary=:all: --no-deps pyarrow==21.0.0
PYTHONPATH=future-ideas/iscs-2027/research-deep-dive/checks/public-cross-section/.deps papers/selective-forgetting/.venv/bin/python future-ideas/iscs-2027/research-deep-dive/checks/public-cross-section/acquire.py
```

The cached analysis and tests run separately, without network access:

```bash
PYTHONPATH=future-ideas/iscs-2027/research-deep-dive/checks/public-cross-section/.deps papers/selective-forgetting/.venv/bin/python future-ideas/iscs-2027/research-deep-dive/checks/public-cross-section/audit.py
PYTHONPATH=future-ideas/iscs-2027/research-deep-dive/checks/public-cross-section/.deps papers/selective-forgetting/.venv/bin/python -m pytest future-ideas/iscs-2027/research-deep-dive/checks/public-cross-section/test_audit.py future-ideas/iscs-2027/research-deep-dive/checks/public-cross-section/test_acquire.py -q
```

The platform-specific dependency command matches this workstation; use a wheel matching the interpreter on another platform. Download caches are ignored by Git. Public text was processed locally and was not copied into these reports; no private CMRI corpus was used.
