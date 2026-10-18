# Cross-collection replication of the history-pooling gain on WildChat: design

Date: 2026-10-18. Owner: Alexander Richard Towell. Author of this document: the assistant,
from the owner's instruction to proceed on the recommendation of 2026-10-18 without a
question round. Every "decided" item below is the assistant's call unless marked as the
owner's; the owner can overturn any of them before the plan is written.

Parent design: `2026-10-18-iscs-2027-paper-design.md` (the ISCS 2027 paper). This study is
a new follow-up, numbered **F6** in that design's sequence, and it is the first
confirmatory experiment since the PRISM reserve was spent.

## Intent

The ISCS 2027 paper's central claim, that uniform history pooling beats last-two pooling
for predicting the next user turn, is confirmed on a held-out partition of PRISM
(`papers/iscs-2027/REPORT.md`). Limitation 10 of that report is the one a reviewer will
raise first: the reserve shares PRISM's platform, collection period, model pool and
participant recruitment with the development data. The confirmation shows generalization
across participants within one collection, not across collections.

This study runs the identical frozen primary contrast on a second, independently
collected public corpus of human–LLM conversations: Ai2's WildChat-4.8M (ODC-BY, not
gated, already mirrored locally by the temporal pilot). If the gain holds there, the paper
can say "replicated across two collections, two platforms, two and a half years and
twelve assistant models". If it does not, the paper reports that boundary. Either result
is written up; the protocol is frozen before any WildChat text is read.

The study also yields, at no extra cost, two heterogeneity panels PRISM could not give:
the gain by assistant model family (GPT-3.5, GPT-4, GPT-4o, GPT-4.1) and by calendar year
(2023, 2024, 2025). These are descriptive, declared in advance and not gating.

## Metadata-only audit (2026-10-18)

Performed on the local mirror at `papers/temporal-networks/data/wildchat/full_shards/`
(86 parquet shards) reading only `conversation_hash, model, timestamp, turn, language,
toxic, redacted, hashed_ip`. No conversation text was read. Script and output:
`docs/research/audits/2026-10-18-wildchat-metadata-audit.{py,json}`.

| Quantity | Value |
|---|---|
| Rows in mirror | 3,199,860 (equals the hub's row count for `allenai/WildChat-4.8M`, 86 shards) |
| Collection period | 2023-04-09 to 2025-07-31 |
| Eligible conversations: `language == 'English'`, `turn >= 3`, `toxic == False` | 152,730 (152,726 distinct hashes) |
| Hashed IPs with at least one eligible conversation | 52,203 |
| Hashed IPs with four or more eligible conversations | 6,761 |
| Eligible conversations by model | gpt-3.5-turbo-0613 35,492; gpt-3.5-turbo-0301 22,980; gpt-4o-mini-2024-07-18 18,628; gpt-4o-2024-08-06 16,679; gpt-4-1106-preview 12,261; gpt-4o-2024-05-13 11,337; gpt-4.1-mini-2025-04-14 9,259; gpt-4-0125-preview 9,160; gpt-3.5-turbo-0125 6,361; gpt-4-0314 4,905; gpt-4-turbo-2024-04-09 3,515; gpt-4o-2024-11-20 2,153 |
| Turn histogram | 1 round: 2,673,253; 2: 190,708; 3: 103,870; 4: 64,593; 5: 41,966; 6+: ~126,000 |

Notes from the audit. The mirror's bytes (14.86 GB) differ from the hub's parquet bytes
(15.28 GB) because the temporal pilot re-wrote shards through DuckDB `COPY`; row counts
match exactly, and the mirror's shard hashes are recorded in the temporal pilot's
`pilot/audits/DATA-SCHEMA.json`. Four eligible conversation hashes repeat once each
(152,730 rows, 152,726 hashes); duplicates are excluded below. `turn` counts rounds, so
`turn >= 3` means at least six messages, the same floor PRISM used (`first_target + 2`).

Other corpora considered and set aside. LMSYS-Chat-1M: gated, custom license forbidding
transfer of derived material to third parties; usable but awkward for a public artifact.
ShareChat (`tucnguyen/ShareChat`, 142,808 conversations shared from ChatGPT, Claude,
Gemini, Grok and Perplexity): gated, CC BY-NC 4.0, no user ids; attractive as a
cross-platform third collection and recorded as a possible F7, not scheduled, because
acceptance of its license needs the owner's Hugging Face account.

## Goals

1. A frozen, single-shot replication of the PRISM primary contrast on WildChat with the
   same estimand, success rule and bootstrap: uniform mean minus last-two window, raw
   cosine to the next user embedding at user indices 4, 6, 8, 10, native128 view, natural
   regime, both encoders (MiniLM-L6-v2 and E5-small-v2), 2,000 component-bootstrap draws,
   seed 104729, success iff the 95% lower bound is above zero in both encoders.
2. The same declared secondary panel as the PRISM confirmation (role interaction under the
   common prefix, budget curve, own versus partner mean, own mean versus own window two) so
   the two studies tabulate side by side.
3. Two new descriptive panels: gain by assistant model family and by collection year, plus
   the per-component distribution and the message-count subgroups already used on PRISM.
4. A report, claims ledger and verification file in the same form as `papers/iscs-2027/`,
   with the freeze repairs listed under "Constraints" below.

## Non-goals

- No new estimand, metric, encoder or view. Anything not in the PRISM protocol is
  descriptive here too.
- No ranking or retrieval outcome (HS01–HS04 are closed).
- No longitudinal or return-lag analysis on WildChat; that belongs to F4.
- No use of the toxic or redacted subsets, and no language other than English.
- No attempt to de-pseudonymize hashed IPs or to treat them as verified individuals.

## Anti-goals

- Opening WildChat text before the freeze, for any reason including "just to check the
  adapter". The adapter is checked on the development sample, which is drawn and declared
  before the confirmation sample is touched.
- Choosing the confirmation sample, its size or its stratification after seeing any
  development result. The sizes below are fixed now.
- Reporting only the passing encoder, or re-running with a different cap, floor or
  language filter after the confirmation sample is opened.
- Committing any conversation text, hashed IP, or per-message embedding. The repo
  carries code, hashes, counts and aggregate results.

## Constraints

- Platform: the pinned CPU pipeline. Encoders and tokenizers as pinned in
  `papers/semantic-measurement`; `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. The new
  repo gets its own Python 3.11 virtual environment with the same pinned numeric stack
  as `papers/selective-forgetting/.venv` plus `pyarrow` (the only new dependency), all
  pinned in `pyproject.toml` and recorded in `audits/dependencies.json`. The audit above
  ran under the temporal calibration worktree's `pyarrow`; the new repo re-runs it under
  its own pin as its first stage.
- Upstream pins: `papers/iscs-2027` at `ef845c2`, `papers/history-falsification` at
  `7a39114`, `papers/portfolio-wave1` at `e848465`, with file hashes recorded in
  `audits/dependencies.json` as before.
- Source pin: the 86 mirror shards by path, size and sha256 (the existing hashes live in
  the calibration worktree's `pilot/audits/DATA-SCHEMA.json`, not on the
  `temporal-networks` main branch; the new repo recomputes and records its own), plus
  the hub listing (dataset id, 86 parquet URLs, 3,199,860 rows) at audit time.
- License: ODC-BY. The attribution line used by the temporal pilot's
  `pilot/public-data/README.md` is reused verbatim in the new repo's README.
- Freeze repairs, each answering a numbered limitation in the PRISM report:
  1. `FREEZE.json` hashes `PROTOCOL.md`, `config.json`, `audits/dependencies.json`,
     `audits/samples.json`, and **every module on the path from parquet to numbers**:
     the adapter, selection, encoding, case-building, analysis and heterogeneity
     modules and their tests. Reporting, figure, `compare` and replay code is outside
     the frozen set and may change afterwards; the runner records such changes as
     `code_changed_since_freeze` exactly as PRISM did. A post-freeze change to a frozen
     file makes `prepare` and the `analyze-*` stages refuse to run; since those stages
     are single-shot, a bug found after the freeze is recorded in
     `results/amendments.json` and reported, not patched and re-run.
  2. The protocol says plainly that metadata columns (the eight listed in the audit)
     are read before the freeze to draw the sample, and that no message `content` is
     read, parsed or tokenized for the sampled rows before `FREEZE.json` exists. Tests
     use a synthetic parquet fixture, never the mirror.
  3. The expected confirmation counts (components, conversations) are written into
     `config.json` before the freeze, and the runner halts on a mismatch before reading
     the `conversation` column.
  4. The heterogeneity code is written and tested on the development sample before the
     freeze; the freeze hashes it.
- Soul-voice hook: no em-dashes in prose files. Commit trailer:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and the session line, on
  every commit including those made by subagents (the PRISM repo's mixed trailers are its
  Limitation 9; the dispatch brief for each task carries the trailer verbatim).
- Results under `results/` are write-once; regenerable tables and figures under
  `results/report/`.

## Decisions (the how, as decided)

### Home

New private repo `queelius/cmri-wildchat-replication`, cloned at
`papers/wildchat-replication/`, package `cmri_replicate`, default branch `main`, same
layout as `papers/iscs-2027/` (`PROTOCOL.md`, `config.json`, `FREEZE.json`, `audits/`,
`src/`, `tests/`, `results/`, `REPORT.md`, `CLAIMS.md`, `VERIFICATION.md`). The frozen
PRISM repo is not modified; its commit is pinned.

### Source adapter

One conversation row becomes a `ReserveConversation`-shaped object (reuse the dataclass
from `cmri_confirm.reserve`, or a copy named `ExternalConversation` if its PRISM-specific
guards get in the way) with:

- `id` = `conversation_hash`; `participant` = `component` = `hashed_ip` of the row
  (the row-level field; every message's `hashed_ip` must equal it or the row is
  excluded as `ip_inconsistent`);
- `condition` = model family, derived from the `model` string by its prefix:
  `gpt-3.5`, `gpt-4o` (includes mini), `gpt-4.1`, `gpt-4` (the remainder: `gpt-4-0314`,
  `-1106-preview`, `-0125-preview`, `-turbo`). The year is `timestamp.year`. Both are
  recorded on the record for the descriptive panels;
- `messages` = the first twelve entries of `conversation`, as `Message(role, content, (),
  model, source_index)` with `source_index` the list position; roles must strictly
  alternate starting with `user`, every `content` must be a non-empty string, and the
  row must have at least six messages after these checks. Exclusion reasons are counted
  and reported, never the text.

Eligibility, applied to metadata before the freeze: `language == 'English'`,
`toxic == False`, `redacted == False`, `turn >= 3`, `conversation_hash` unique in the
mirror. The `redacted` exclusion is new relative to the audit (which did not apply it)
and is the assistant's call: redaction substitutes placeholder tokens for names, which
would enter the embeddings. The protocol records the count it removes.

Two further metadata-only exclusions, both the assistant's call after the builder check:

- **Previously read WildChat rows and IPs are ineligible.** Earlier work in this program
  read WildChat text twice: the temporal pilot pulled full text and embeddings for the
  hashed IPs under `papers/temporal-networks/data/wildchat/users/` (and screened the
  candidates in its `selected_users.csv` and `candidates.csv`), and the 2026-10-15
  public cross-section check under
  `future-ideas/iscs-2027/research-deep-dive/checks/public-cross-section/` read and
  encoded the first 2,000 rows of shard 0000. Every hashed IP named in those pilot index
  files and every conversation among those 2,000 rows (by row position in the pinned
  shard, or by hash where the check recorded hashes) is removed from eligibility before
  either sample is drawn. The protocol states that "no text read before the freeze"
  holds for the sampled rows, not for the corpus, and lists these exclusions with their
  counts.
- **High-volume IPs are ineligible as components.** A hashed IP with more than 100
  eligible conversations is treated as a shared gateway or an automated client and is
  excluded. The threshold is fixed here from the audit's size histogram (the long tail
  past 20 holds few IPs) and is not tuned after any text is read.

The rule the text checks enforce after the freeze (role alternation, non-empty content,
IP consistency, at least six messages) is: a failing conversation is **dropped, not
replaced**; a component that loses all its conversations is dropped; the runner compares
the metadata-drawn counts against `config.json` before reading text and reports the
post-check counts alongside; the run halts, and the halt is reported as the outcome, if
more than 5% of the drawn conversations fail the text checks. The IP-consistency rule
applies to the non-null `hashed_ip` values among a conversation's messages (assistant
messages may carry null in this schema).

Grouping unit. PRISM's components merged participants who shared an opening prompt or a
candidate set; that linkage needs text and is not applied here. WildChat components are
bare hashed IPs, which is said plainly in the protocol as a difference from PRISM. The
primary contrast is parameter-free (nothing is fitted on development data except the
centers used by the centered-cosine secondary), so a templated prompt appearing in both
samples cannot inflate the primary; it can only make the component bootstrap slightly
optimistic, which the high-volume exclusion limits and the report discloses.

### Samples

Two disjoint samples, both drawn from the eligible set by salted hash so the draw is
reproducible from metadata alone:

| Sample | Components (hashed IPs) | Cap per component | Purpose |
|---|---|---|---|
| Development | 360 | 4 | Adapter and pipeline checks; role-specific centers for the centered-cosine secondary; the exposed pre-registration estimate; all heterogeneity code written against it |
| Confirmation | 500 | 4 | The frozen single run |

Salts: `cmri-wildchat-component-v1` ranks eligible hashed IPs; the first 360 are
development, the next 500 are confirmation, and the two lists are written to
`audits/samples.json` with their sha256 before the freeze. `cmri-wildchat-conversation-v1`
ranks conversations within an IP; the first four are kept. Expected conversation counts
are computed from metadata and written into `config.json`.

Why 500 and not 248: WildChat has an order of magnitude more eligible components, the
encoding cost is minutes per encoder, and a wider interval on the confirmation would
make a failure uninformative. Why not more: the bootstrap and figure code was sized for
hundreds of components, and 500 keeps the two studies' precision comparable rather than
letting WildChat dominate any pooled figure.

### Centers and the development estimate

The centered-cosine secondary needs role centers. They are fitted on the WildChat
development sample, not imported from PRISM, and this is said in the protocol (it mirrors
PRISM's Limitation 6 rather than hiding it). The development sample's primary gain in both
encoders is written into `PROTOCOL.md` as the pre-registered development value, exactly
as PRISM's +.03440 / +.01919 were.

### Primary, secondary and descriptive panels

Identical to `papers/iscs-2027/PROTOCOL.md` sections "Primary contrast and success rule",
"Declared secondary panel" and "Heterogeneity analyses", with `condition` now meaning
model family and one added subgroup axis, collection year. The protocol text is copied,
not paraphrased, with the PRISM-specific sample rule replaced by the one above.

Pre-registered expectations, stated as expectations and not as gates: both encoders
positive; magnitude at or below PRISM's reserve values (.025 MiniLM, .018 E5) because
WildChat conversations are longer and more task-switching; no prediction on model
family or year.

### Outcome mapping

- Pass in both encoders: the paper's confirmation section gains a second row and the
  phrase "replicated across collections"; the heterogeneity figure gains model-family and
  year panels.
- Fail in one or both: the paper reports the PRISM confirmation and the WildChat result
  with the same prominence, and the discussion treats the collection difference as the
  finding. No re-run.

### Tooling

Same stage runner pattern as `cmri_confirm.runner` (`audit`, `freeze`, `prepare-dev`,
`analyze-dev`, `freeze-confirmation`, `prepare`, `analyze-minilm`, `analyze-e5`,
`analyze-subgroups`, `report`, `compare`), write-once results, provenance with dirty-tree
flag, `--replay` and `--tolerance` on `compare`. Parquet is read with `pyarrow` restricted
to named columns; the `conversation` column is read only inside `prepare-dev` and
`prepare`.

## Builder check (2026-10-18)

A fresh reviewer read the document and asked five questions; all were settled above as the
assistant's calls rather than returned to the owner, because none changes what gets built,
only how the sample rule and freeze are written: (1) text-check failures after the
freeze: drop, do not replace, halt above 5%; (2) previously read WildChat rows and IPs:
excluded before sampling and disclosed; (3) grouping unit: bare hashed IP with a
high-volume exclusion, disclosed as a difference from PRISM; (4) freeze scope: the
parquet-to-numbers path only, with an amendment record; (5) trailer: Fable 5.1 on every
commit.

## What is left to the builder

- Whether to import `ReserveConversation` or copy it; the PRISM boundary module
  (`assert_reserve_only`) does not apply and must not be imported.
- The synthetic parquet fixture for tests (a dozen rows covering every exclusion reason).
- The figure set: the four PRISM figures with a WildChat column, plus one model-family by
  year panel.
- Wall-clock budget for `prepare` (PRISM used 3,600 s for 775 conversations; WildChat
  texts are longer but visibility truncates at 128 tokens; budget 7,200 s).

## Sequence

1. Owner reads this design; overturns anything above.
2. Plan (superpowers:writing-plans), tasks sized as in the PRISM plan.
3. Build the adapter and runner against the synthetic fixture; run `prepare-dev` and
   `analyze-dev`; write the heterogeneity code; record development values.
4. Freeze (protocol, config, code, sample list). Push. Run `prepare` and the analyses
   once. Push.
5. Report, claims, verification; umbrella records (DECISIONS.md, registry.yaml,
   CLAUDE.md pilot table, this design's parent).
6. Paper section drafted from the two reports together.
