# Research deep dive: progress and evidence trail

Started 2026-10-15. Owner: Alexander Richard Towell. Working directory: `/home/spinoza/github/cmri`.

## Objective

Critically assess the CMRI research program and select the strongest next research directions for ISCS 2027, separating longer-term opportunities. Audit published and unpublished local artifacts, current primary literature, datasets, and competing explanations. Revisit cross-sectional comparison, user/assistant attribution, semantic physics, embedding and time-series dynamics, temporal networks, and memory. Run bounded, reproducible checks that can falsify proposals. Produce an inventory, literature map, ranked candidates, three to five detailed briefs, evidence with uncertainty, a recommendation, and a concrete next goal plus fallback. Update the ISCS proposal while preserving prior decisions.

## Constraints

- Existing computational resources and public data; no participant recruitment.
- Private text and secrets stay local. Public-facing notes contain no private conversation excerpts.
- Papers are independent nested repositories. Existing root modifications to `.gitignore` and `CLAUDE.md` predate this task and must remain untouched.
- Freeze a protocol before each new pilot; distinguish exploratory checks from confirmatory evidence. Preserve negative results and report uncertainty at the correct sampling level.
- A physics metaphor, different clusters, or high embedding similarity is not proof of a mechanism, independence, or cognition.

## Execution plan

1. Audit local artifacts and reconcile status conflicts; maintain source-level evidence.
2. Search primary literature, recent work, datasets, and official venue rules; log coverage.
3. Compare a diverse longlist and develop three to five detailed briefs.
4. Run bounded analytical/empirical checks that could change the ranking; test reusable code and preserve runtime/configuration.
5. Synthesize decisions, limitations, next goal and fallback; update proposal notes and verify artifacts.

## Checkpoints

### 2026-10-15: initiation

- User explicitly authorized the full assessment and bounded pilots, following review of the research goal.
- Goal API remains attached to the previously paused, accidentally pasted Prove2Me objective. `create_goal` rejected the new objective because an unfinished goal exists. Do not mark the unrelated theorem objective complete. Continue the authorized research with this durable local record.
- Read the local umbrella instructions and relevant skills. Applying parallel research to independent audit domains; root coordinates literature and synthesis.
- Known status conflict requiring audit: embedding-dynamics `.papermill/state.md` lists a June 11 prefix experiment as passed while retaining older NOT RUN entries. Do not repeat either status without inspecting underlying artifacts.

## Completion checklist

### 2026-10-15: audits and literature underway

- Three independent auditors cover dynamics, network/memory work, and public cross-sectional data. Root inspected the Operational Memex outline, code and saved aggregate results and maintains the external semantic/time-series literature map.
- Direct competitors found: VECTOR (2025), cumulative-prefix semantic navigation (ICLR2026), semantic timescales (2026), recurrence-based communication analysis (2012/2018), SuperDialseg/Def-DTS, Embed-KCPD, and SeCom. Search coverage and access failures recorded.
- Root bounded analytical check of an external median-split entropy formula: exact length dependence and order invariance demonstrated, with valid unit-vector path realizations. Nine tests passed, 99% implementation statement coverage, ~1.05-second standalone run. Default Python lacked Matplotlib; reused the existing pinned selective-forgetting environment. No model/API calls.
- New public-data opportunity: ThoughtTrace provides participant-grouped conversations with reasons/reactions, potentially useful for construct validation; it does not supply rich longitudinal archives. Do not treat self-reports or model-derived labels as direct access to cognition.
- Preliminary audits already identify a false universal turning-angle derivation, endpoint-only derivative summaries, a degenerate shape-distance implementation, temporal tracker/label problems, and hierarchy/projection null concerns. Final numbers will be taken from completed artifacts, not interim messages.

### 2026-10-15: review, corrected evidence and decision

- Completed all domain audits, current primary-literature comparisons, public release/access audit, sixteen-candidate longlist and five detailed briefs. Corpus status/count inconsistencies are preserved rather than treated as an already verified release.
- Independent review identified actual public split leakage through excluded language records and incomplete selection uncertainty in the prefix result. Both were corrected with tests and saved pre-fix outputs. The public check now has 406 components/245 development/161 evaluation; all role intervals include zero. The prefix point gain remains .01766, but selection-sensitive interval [−.00122,+.03481] spans zero.
- Earlier missing-label and tokenizer-round-trip defects were also corrected and preserved in the public deviations log. No additional data acquisition or outcome-driven tuning was used to resolve them.
- Fresh final execution: **68 new-check tests passed**. Original embflow 191 and relevant hierarchy 48 tests passed during the audit; the full hierarchy suite's nine legacy-import errors remain documented. Per-file coverage gaps, source hashes and runtime are in VERIFICATION.md and linked machine-readable artifacts.
- Two scientific figures were inspected. All nested/shared repository revisions and pre-existing dirty states match the initial snapshot. Original research sources remain unchanged.
- Final recommendation: first pursue a small public semantic-measurement validity study with cross-sectional participant sampling and within-conversation temporal controls; corrected temporal-networks is the fallback. No full new paper is yet warranted. Semantic force/half-life interpretations, rich-personal-archive inference and the bridge-retention policy are not established.
- README.md, INVENTORY.md, BIBLIOGRAPHY.md, CANDIDATES.md, BRIEFS.md, NEXT-GOAL.md and VERIFICATION.md form the durable decision package. Existing ISCS README/PILOTS/EVIDENCE now have dated updates that preserve October 14 history.

- [x] Audited inventory and source provenance
- [x] Current literature map, bibliography, search log, venue terms
- [x] Public-data access/licensing/linkage/feasibility audit
- [x] Diverse longlist and three to five detailed candidate briefs
- [x] Bounded checks, meaningful tests, runtime and uncertainty
- [x] Recommendation, next goal and fallback
- [x] Proposal updates, limitations and final verification

The authorized research assessment is complete. The unrelated paused goal-tracker entry was not changed or falsely marked complete. The next experiments are proposed in NEXT-GOAL.md, not silently launched as an extension of this discovery task.

## 2026-10-15 — recommended pilot execution checkpoint (historical)

The user activated the Recommended goal in NEXT-GOAL.md. Execution lives in the new independent
repository [papers/semantic-measurement](../../../papers/semantic-measurement/README.md), branch
`pilot/public-validation`. The earlier deep-dive findings remain unchanged.

Public ThoughtTrace release is pinned and audited; development has195 dependency components
with >=4-user eligible reports, producing503 distinct anchor cases. Many reports are retrospective,
so the experiment is limited to reported-purpose recovery. Synthetic calibration and its review
are complete; public outcome evaluation has not begun. Protocol/config/source/model hashes were
frozen at child commit2cf3b1d after independent methodological review. Detailed progress and
verification are in the child's PROGRESS.md, PROTOCOL.md and audits/REVIEW-RESOLUTIONS.md.

Remaining: finish feature/inference implementation and review, run the frozen public pilot with
both CPU encoders, assess uncertainty and controls, then update the research recommendation.

## Public validation follow-up — 2026-10-15

The independent `papers/semantic-measurement` pilot has executed frozen public preparation,
calibration and scoring. Primary gain −4.8937 pp, selection-aware CI [−8.5845,+4.7630], 140
evaluation participants/28 blocks. All fixed-pair encoder/context point gains are negative;
order/donor increments lack support. The primary practical-negative criterion is narrowly met,
not robust across sensitivities. Recommendation: abandon full-paper expansion of this particular
retrieval pilot; preserve calibration assets and prioritize the bounded temporal fallback.
All three stages now reproduce exactly on the same machine after excluding documented timing,
cache and provenance fields; 106 checks and 148 passing tests are saved. Final broad review
passed, independently confirming 16 inference outputs and 553 candidate sets. The pilot is
complete. See its REPORT.md/VERIFICATION.md and committed ledger for evidence. Existing
deep-dive checks and earlier decisions remain preserved; the temporal fallback remains a
proposed next goal, not a silently launched continuation.

## October 16: temporal calibration follow-up

Separate authorized temporal goal executed in isolated child worktree `papers/temporal-networks-calibration`, branch `pilot/iscs-2027-calibration`, from 13e3a47. Source and literature audits, frozen protocol, 50-graph calibration, 72-cell archive diagnostic, exact replays and corrected claims saved under its `pilot/`. Decision NARROW; no full paper allocation. Original child PDF and umbrella dirty .gitignore/CLAUDE.md protected by hashes. Private author arrays remain ignored; numeric public arrays are releasable with attribution. No paper/submission or lifecycle repair. See [report](../../../papers/temporal-networks-calibration/pilot/RESULTS.md) and [durable execution record](../../../papers/temporal-networks-calibration/pilot/PROGRESS.md). The semantic pilot's negative result was not generalized to temporal networks.

Final review and one-pass correction completed: preparation now enforces all frozen source shard hashes. The same-size timestamp-change and missing-hash regressions passed after failing first. Full suite 52 passed, 92% coverage; all 86 shards were reverified and a fresh rebuild produced byte-identical public/private numeric inputs, with unchanged scientific results. The final review, resolution, runtime and verification limits are durable in the child pilot. The conditional recurrence goal remains proposed, not launched.

## October 16: user-directed empirical brainstorming and log search

- User wants a new empirical human–AI finding and prioritizes a defensible ISCS paper. Authorized searching local logs for examples instead of relying on recollection. Later clarified that useful connections emerge through back-and-forth; the candidate should study development through exchanges rather than equate first mention with authorship.
- Re-read substantive static/hierarchy manuscripts, Memex artifacts and completed pilot reports. Added descriptive queries on existing hierarchy and calibrated return inputs under `checks/exploratory-reuse/`. These are post hoc hypothesis generation, not a new evaluation. Source hashes and cross-checks are recorded; original runtime 0.646 seconds.
- Searched two local Memex databases read-only, inspected relevant turns and indexed newer raw-log headers. Identifier union 5,219; 472 overlapping conversation IDs. New raw session files are mostly subagents, so file counts cannot stand for human conversations. Six selected newer non-CMRI sessions received bounded content inspection. Private pointers/search details remain under `~/.local/share/cmri/research-mining/2026-10-16/` with restricted permissions, outside Git.
- Found evidence candidates for cross-context recurrence and revision, plus counterexamples: revised prompts, generated user-role summaries, intention mistaken for completed transfer, and disputed origin without verified provenance. A first raw-JSONL scan failed on a malformed line; the second recorded the malformed-record count and retained its incomplete-import limitation.
- A separate code-erasure research repository already has a substantial protocol, but its inspected readiness document remains pre-freeze/runner-blocked. It was read only, not modified or executed. Current deletion-avoidance literature constrains novelty.
- Primary literature follow-up found ThinkFlow and CoNode (CHI 2026), Choi/Nixon's idea-uptake preprint, message-level adoption work, and deletion avoidance. Methods-inspection limits and search coverage are in `EMPIRICAL-PAPER-OPTIONS.md`. Private content was not used in external queries.
- Proposed next step is a bounded public-data/annotation feasibility check for developed-and-reused connections. No paid calls, new embeddings, public-data acquisition, participant recruitment, paper writing, submission or new goal activation occurred. Earlier negative/inconclusive findings are preserved.
- Verification: fresh aggregate-probe execution passed all embedded consistency checks and exactly reproduced saved non-timing fields; Python syntax and all six local links in the new memo/query notes passed. `git diff --check` passed. Original root `.gitignore`/`CLAUDE.md` edits and nested repositories were not edited by this follow-up.

## October 16: complete paper catalog and renewed literature-led brainstorming

- User requested a fresh inventory of all papers before brainstorming extensions. Inspected actual manuscript entry points, partial sections, outlines, PDF/publication metadata and status histories across nested repositories. Seven manuscript lines, one extra outline and three completed pilot/report lines are distinguished; temporal format/worktree duplicates are not counted separately.
- Verified the static chapter's Springer DOI and February 25, 2026 publication date. Located the forthcoming ISCS 2026 proceedings listing, SSDC 709. Its header/eBook/print dates differ; the hierarchy chapter is accepted/program-listed, not verified as already published. No child citation metadata was silently rewritten.
- Additional primary research found CoAuthor's existing AI-entity-reuse analysis, temporal co-writing/ENA studies, September 2026 content-survival work, KITE's independent-solving design (NeurIPS 2025), and Bridging Talk and Thought (SIGDIAL 2026). This materially narrows the novelty of the preceding generic reuse proposal.
- Read KITE methods and inspected the complete 82-entry author repository tree via GitHub MCP. No clearly documented participant-linked trajectory/outcome export or dataset license was located; it remains a conditional lead. No Firebase/backend access or new human-data acquisition occurred.
- Saved `PAPER-CATALOG-AND-OPTIONS.md`, linked from the inventory and proposal indexes. Leading discussion candidate is public cross-sectional integration-versus-switching validation of the existing bridge taxonomy; five alternatives and their failure conditions are retained. No new pilot or full-paper decision was made.
- Verification: all 11 cataloged local entry-point hashes matched the observations made before writing; structured manifest counts are 7 manuscripts/1 outline/3 reports. Markdown parsing checked 100 local links across five documents with none missing. `git diff --check` passed. This was a documentation/literature review, so no implementation tests or new scientific experiments were run.

## October 17: durable program metadata and dynamics consolidation discussion

- User requested written records of existing work, ideas, plans and future work before further discussion. Confirmed conference priority while retaining semantic/embedding physics; suggested merging the related drafts if their complementary content warrants it.
- Created `docs/research/README.md`, `registry.yaml`, `DECISIONS.md` and `DYNAMICS-CONSOLIDATION.md`. The registry references the existing dated paper manifest rather than duplicating publication facts, and distinguishes user priorities, assistant recommendations and unresolved decisions.
- Re-read the substantive dynamics drafts, specification, validation section and results addendum. Located the June 13 merged outline: consolidation was already proposed but not completed. Semantic Dynamics LaTeX includes force/state material and role-divergence/rhythm analyses absent from its Markdown version. Recorded shared versus complementary material and preserved mathematical corrections and failed predictions from the October audit.
- Proposed one revised dynamics manuscript lineage with three possible centers: state reconstruction, observable conversational development, or falsifiable dynamical-model comparison. No canonical destination/thesis selected and no child manuscript, code, repository or dataset changed.
- Rechecked the official ISCS call/submission rules on October 17: Porto, Portugal, June 2–4, 2027; deadline March 1. The user's “France conference” wording is recorded as referring to the ongoing ISCS series, not a newly selected event.
- A combined documentation patch initially failed on an incorrect README context. Confirmed it made no partial addition, then applied the new document and index changes separately. No scientific result or source artifact was affected.
- Verification: registry YAML and catalog references passed; 95 local links across seven Markdown documents resolve. Eleven protected source hashes and both dynamics child Git states are unchanged. `git diff --check` passed. The evidence and scope limits are recorded in `docs/research/verification-2026-10-17.json`; this documentation-only work required no new implementation tests or scientific experiments.

## October 17 follow-up: investigate multiple questions before narrowing a paper

- User clarified interest in multiple research questions and many experiments. Replaced immediate single-question selection with portfolio planning; retained the conference priority and the proposed dynamics consolidation.
- Added `docs/research/EXPERIMENT-PORTFOLIO.md`: six question families, twelve proposed experiments, discriminating comparisons, data/annotation dependencies, counterexamples and staged exploration/confirmation. Updated the program index, registry, decision history, consolidation assessment and conference index. Specific protocols are not frozen and no experiment was executed.
- Rechecked four primary pages already in the literature map: inclusion/uptake, collaborative dialogue coding, CoAuthor traces and ICLR semantic navigation. This bounded page/abstract check is not a new systematic review; source links and limits are recorded in the portfolio.
- No child manuscript, code, private log, dataset or existing scientific result was modified. Previous negative and inconclusive findings retain their original scope.
- Verification: six registry families reference twelve unique experiment IDs, each present once in the portfolio. YAML, record paths and family/direction references pass; 97 local links across seven documents resolve. Eleven protected source hashes and both dynamics child Git states remain unchanged. `git diff --check` passes. Details and coverage limits are in `docs/research/verification-portfolio-2026-10-17.json`; no implementation tests were needed for these documentation changes.
