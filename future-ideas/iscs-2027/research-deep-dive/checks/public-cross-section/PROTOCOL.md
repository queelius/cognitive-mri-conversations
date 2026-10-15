# Public cross-section feasibility and measurement protocol

Frozen 2026-10-15 before downloading or analysing conversation records. This is an exploratory feasibility check, not a preregistered confirmatory study. Owner: Alexander Richard Towell. Original CMRI repositories are read-only.

## Decision

Can an inexpensive, public release support (a) adequately linked and sufficiently rich archive comparisons or, failing that, (b) a narrower check of whether role-specific semantic representations recover released topic and purpose labels? A positive conversation-level check does not establish a personal archive, cognition, purpose causation, or publication novelty.

## Sources and resource bound

- `allenai/WildChat-4.8M`, HF revision `c827c6df8fcf008219ffaffa4d1dd77491099367`, ungated non-toxic subset; ODC-By. The advertised 4.8M family has 3,199,860 public rows at this revision.
- `microsoft/adopt-adapt-wildchat`, Git revision `b4e77e44b62ba2ddd0f7b53667b218e345faba09`; released domain and intent annotation shards, not newly purchased labels.
- Preselect source shard `data/train-00000-of-00086.parquet` and matching released labels `00000`. It is a convenience slice, not a probability sample of the full release. Do not search shards for favourable outcomes.
- Read the annotation shard (about 5 MB domain plus 27 MB intent) and source Parquet footer. Project source metadata and at most the first 2,000 rows of conversation content through HTTP byte ranges; stop text acquisition if it exceeds 30 MiB. Permit up to 60 MiB downloaded records in total. No paid model API. Only cached local CPU MiniLM if the role check is feasible.
- Retain no raw text or identifying request-header values in report outputs. Cache downloads locally, excluded from version control. Save pinned paths, source hashes, byte counts, row counts, and code/configuration fingerprints.

## Eligibility audit

Count physical rows, distinct content hashes, duplicate content-hash multiplicities, missing keys, distinct hashed-IP proxies, and proxy-size/activity distributions within this shard. A content hash is not a unique conversation ID. Treat exact repeats as one content unit; use a complete composite key when joining the labels and fail if a join expands the source rows. Record label coverage and collisions.

Screen archive feasibility at 20, 50, and 100 unique conversations and at least 5 active UTC dates, with a primary threshold of 50 conversations/5 dates. Record country/state/language variation and counts over 161 conversations as diagnostics inspired by Adopt != Adapt. Do not call a proxy validated because it passes these filters; do not apply global filters using incomplete shard histories and imply they are global. No extrapolated count of eligible archives from this selected shard. The eligibility count is a deterministic property of this slice; uncertainty is incomplete coverage and linkage, not a binomial CI over rows.

## Conditional conversation-level representation check

Proceed only if bounded content access yields at least 200 usable English conversations with user and assistant text and both labels. Restrict to dates before 2024-09-01. Remove exact duplicate content and form leakage groups from shared IP proxy or shared normalized first-user-prompt prefix (200 characters; whitespace normalization only). Use connected components, so a duplicate connects groups transitively. Select at most 600 conversations by deterministic hash order, at most one conversation per component. Split components deterministically 60% development / 40% evaluation before fitting. Save the split manifest without IPs/text. These components are independence controls, not verified people.

Represent each role independently using the cached MiniLM encoder. Split role text into at most four consecutive 254-token chunks; mean normalized chunk embeddings and renormalize. Compare user-only, assistant-only, and equal role mean. Report how often the token cap binds. Fixed nearest-centroid classification, using only development labels; no hyperparameter search. For each released label type (domain/topic and intent/purpose), evaluate accuracy and macro recall across evaluation classes represented in development. Include development-majority accuracy. Released GPT labels describe complete conversations; agreement with them is not independent validation of the user's latent intent.

Report paired assistant-minus-user and mixed-minus-user accuracy differences, percentile bootstrap 95% intervals across held-out components (2,000 resamples; seed 20261015), and a paired sign-flip randomization test (5,000 draws) for assistant-minus-user. Bootstrap at the conversation level is justified here only because one item is retained per component. Report unseen-label coverage and the effective held-out N. Do not call an inconclusive interval equivalence. A result is practically discriminating for planning only if the role difference is at least 5 percentage points with a CI excluding zero, or the majority baseline matches all representations within uncertainty. Otherwise treat the result as limited feasibility evidence.

## Validation and interpretation

Unit tests must check transitive grouping, no development/evaluation group leakage, duplicate join protection, chronological date extraction, embedding normalization/chunk bounds, and paired uncertainty against known inputs. An integration check must validate the bounded source/annotation join and finite representations. Check coverage and explain gaps. Preserve failures and deviations. One selected shard does not support platform prevalence, people-level conclusions, or causal differences across tasks/platforms. Role-weight ablation already exists in the original CMRI work: a stronger follow-up needs out-of-sample construct validation beyond topic/activity, not another weight sweep.
