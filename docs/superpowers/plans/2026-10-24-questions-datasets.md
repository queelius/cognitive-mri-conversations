# Questions and datasets implementation plan

**Goal:** Complete the questions and datasets portion of the program map.

**Spec:** `docs/superpowers/specs/2026-10-23-cmri-research-program-design.md`, sections 2, 3, 5, 7, 8 and the migration table, narrowed by the owner's phase 1 task.

**Architecture:** Extend the existing standard-library record loaders and dataclasses. Questions own candidate dataset references; samples own dataset references. Exposure is a join over `spent.json`, with no stored totals or remaining-population calculation. Required nullable metadata fields use JSON null for unrecorded facts.

**Constraints:** All writes stay in this worktree. No network, real-repository commits, private archive reads or reads of other child repositories. Existing catalog prose and the five design lessons are preserved. New prose contains no em-dashes.

## Tasks

- [x] Add failing validation tests, then implement dataset records, source configuration, question candidates and lessons. Reject unknown fields, duplicate keys and ids, malformed metadata and unknown references. Missing source files retain the existing warning behavior.
- [x] Add failing exposure and command tests, then extend Sample with dataset, unit and recorded sizes. Share dataset-reference validation between program loading and `spent`. Render every question, study state, candidate exposure and the lessons. Write beside the questions source; check without mutation; refuse writes through symlinks or with missing required sources.
- [x] Migrate the seven datasets and all sample references from the permitted planning files. Add catalog anchors without rewriting prose. Move lessons verbatim into structured entries. Add candidate references only where the supplied planning records support them.
- [x] Update command documentation, regenerate tables and ledger, run the full unit/integration suite, inspect coverage and run board and spent verification on real files. Review the diff and leave changes uncommitted.

## Review focus

Tests cover missing versus empty sources, unknown versus zero exposure, mixed sampling units, malformed nullable metadata, stale/missing output, symlink refusal, reference validation in `spent`, and exact preservation of lesson text. Historical size labels are retained rather than summed or reinterpreted.

## Progress

Implementation authorized by the owner's detailed task; no additional approval gate. Existing worktrees are the intended isolation. Test git operations run only in disposable fixtures inside this worktree.


Verification: all 157 tests pass (104 existing cases plus 53 new cases). Standard-library
trace reports 99% executable-line coverage of `tools/workspace.py`, with all newly added
executable lines exercised. `render --check` and `questions --check` pass. Real-file
`spent --verify` reports 11 samples and 0 problems; missing child checkouts prevent checking
their evidence files. Board execution succeeds, with original-checkout publication locators
unmatched and 21 child checkouts absent in this worktree.

Independent read-only review found no important or critical issues. Catalog anchors were
verified during migration; future manual anchor edits are not checked against the catalog
at runtime. All five lessons were preserved byte for byte, and existing catalog prose is
unchanged. No changes were committed.
