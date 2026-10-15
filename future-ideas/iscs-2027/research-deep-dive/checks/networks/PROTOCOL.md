# Network and memory audit checks

Written 2026-10-15 before the new numerical checks below. This is a diagnostic
audit, not a preregistration or an independent empirical replication. Earlier
stored summaries and code have already been inspected. Originals remain read-only.

## Scope and decisions fixed before execution

1. Verify SHA-256 identity of the temporal files used in the 2026-10-14 audit.
   Reuse its tracker/recency fixtures and 10,000-shuffle summaries when unchanged;
   do not repeat expensive permutations or the completed selective-forgetting run.
2. Rebuild only the numeric episode–meta-concept incidence structure from the
   existing hierarchy extraction and membership JSON. Never emit titles, concepts,
   episode identifiers, or raw conversation text. Check node/edge counts and the
   fitted alphabetical growth exponent against saved results. Read the separate
   chronological summary without treating it as newly reproduced.
3. Derive the exact expected vocabulary curve for uniformly permuted episodes:
   E[V(k)] = sum_c {1 - choose(N-f_c,k)/choose(N,k)}, where f_c counts episodes
   containing concept c. Validate against exhaustive permutations on a small
   fixture. This null conditions on the final incidence matrix; it tests order,
   not cognitive consolidation or the validity of extracted concepts.
4. Diagnostic projection null: preserve every episode's concept count and every
   concept's episode count using standard uniform Curveball trades. For each of
   16 fixed independent RNG seeds 114001..114016, restart from the observed
   incidence sets; record at 10 and 40 trades per episode. Use the same random
   trajectory for these paired checkpoints. A trade allocates a uniformly chosen
   fixed-size subset of the union of the two rows' exclusive entries, allowing
   self-loops. Check both margins exactly after each checkpoint. These finite
   chains have no mixing proof; comparing checkpoints is a diagnostic only.
   Measure node/edge count, mean clustering, transitivity, and exact mean shortest
   path on the largest component of each projection. Report all draws, null means,
   ranges, and observed/null ratios. The primary statistic is mean clustering.
   With 16 draws, no publication-quality significance claim is permitted.
5. Verify the structural degeneracy of role-only alternating paths on synthetic
   lengths 2..20 and windows 3..5. A path has only one uncolored topology at any
   fixed window length; colored windows are determined by length and starting
   role. This says nothing about richer tool/repair/branch dialogue graphs.

## Validation and stopping

Before empirical execution, test Curveball margin preservation (including empty,
identical, and unequal rows), exact vocabulary expectation against exhaustive
enumeration, known clique projection, and deterministic alternating-role entropy.
Run those unit/integration checks with coverage. Attempt the existing hierarchy
suite using an available local environment without changing original sources or
writing caches there; report import or dependency failures separately. Stop after
the above fixed checks. Do not tune null parameters in response to outcomes.

Runtime, dependency versions, source hashes, and aggregate outputs live here.
No paid APIs, new embeddings, extraction calls, pipeline repairs, commits, or
publication actions are authorized by this protocol.
