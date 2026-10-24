# The Cognitive MRI research program: a map and an engine

Design, 2026-10-23, revised the same day after an independent Codex review (kept privately in
`papers/agenda/reviews/2026-10-23-program-design-codex.md`). Owner: Alexander Towell. The owner
skipped the design questions; every decision not attributed to him is the assistant's call and is
listed at the end for review.

## Why

The `cmri/` directory grew organically from one conference paper into a program of about twenty
repositories: published and in-press papers, a journal extension, a conference paper under way,
confirmed studies, nine pilots, retired drafts, infrastructure and early ideas. In the owner's
words, it is hard to tell "what papers are published, what has been abandoned, which are very
preliminary, which are drafts for extensions, which are being actively worked on for a conference
or journal", and the organic growth carries "a lot of cognitive load". Two further aims:

1. **A reader outside the work.** The owner wants to point his co-author, Dr. John Matta, at a URL
   (GitHub Pages) where he can see what the owner is working on, including work he already knows,
   so he can advise or join in.
2. **An engine.** The program should "quickly generate research questions, ideas, paper drafts,
   experiments, data sets" from what it already holds.

## Goals

- One command answers: where does each piece of work stand, what is due when, what needs
  attention.
- Every current fact has exactly one source; every view is generated from the sources.
- A public page for Dr. Matta built only from text the owner has approved word for word.
- A fast, repeatable path from a question to a designed study, its data, its controls, its reviews
  and a draft, reusing the program's existing conventions.

## Non-goals

- Moving directories (section 1).
- A monorepo, or taking down or merging any existing GitHub repository (owner's constraint).
- Replacing `pubs/` (the owner's global publication tracker) or `spent.json`.
- New heavy tooling: the umbrella's tool stays standard-library Python.

## Anti-goals (a failure even if it works)

- Speed that produces claims which later fall apart. The engine is fast because its parts are
  reusable, never because checks are skipped: a design before the first run, synthetic controls
  before real data, an independent review before interpretation, and the spent-sample ledger before
  any confirmation draw. An agent review is evidence of review, not approval of a claim.
- Anything public that the owner has not approved word for word: private-archive content, unfrozen
  designs, correspondence, funding or follow-up notes, local working-tree state, secrets.
- A second record of a fact that drifts from the first (as today: `pubs/` lists the temporal-networks
  paper as "seeking venue", while the program retired it on 2026-10-22).

## Decisions

### 1. Nothing moves; stage is data

The owner first imagined folders (`published/`, `unpub/`, `drafts/`). A paper's status changes over
its life, so a folder per status means moving repositories at every change, and moving breaks what
the program relies on (checked): relative sibling paths pinned in `sources.json` and resolved from
the child, the two worktrees' `.git` files that point to absolute git directories, and the shared
environment and corpus locations the children's Makefiles and READMEs name. (Absolute paths inside
virtual-environment scripts are a known property of such environments, not inspected here.) Every
repository stays where it is; a generated `by-stage/` view of relative symbolic links gives the
folder view the owner described (section 5).

### 2. Entities: repositories, manuscripts, studies, questions, datasets

A repository is not the unit of status: `cmri-iscs-2027` alone holds confirmatory studies, the ISCS
2027 paper, a NetSci abstract and the shared environment, each with its own state. So:

| Entity | Id | Authoritative source | State it carries |
|---|---|---|---|
| Repository | name | `workspace.json` (umbrella) | role (umbrella, corpus, planning, manuscript home, study, pilot, infrastructure), path, branch, visibility, data class, archived; no lifecycle prose |
| Manuscript (anything aimed at a venue) | `pubs` id | `~/github/pubs/papers/*.yaml`, read through its generated `pubs.jsonl` | `derived.status` exactly as `pubs` derives it; repository and path as locators |
| Study (pilot, study, analysis line) | `S-...` | new `studies.json` in the private agenda | research state: `idea`, `designed`, `running`, `confirmed`, `not-confirmed`, `stopped`, `deferred`; successor id when folded; repository and path; question ids |
| Question | `Q...` (the agenda's existing Q1 to Q6 kept) | new `questions.json` in the private agenda; `questions-ledger.md` generated from it | state, what is known, what remains, study ids |
| Dataset | `D-...` | new `datasets.json` in the private agenda; the narrative catalog keeps its prose sections and gains ids | provenance, version, access class, licence, location, documentation, sampling unit |

Joins are by id, never by parsing prose. Owner-specific paths (`~/github/...`) are locators, not
identifiers.

### 3. Statuses: one vocabulary per dimension, no re-mapping

Publication status is `pubs`' `derived.status`, consumed unchanged (today: in preparation, under
review, revising, seeking venue, in press, published, posted, withdrawn, retired). The umbrella
implements no event-to-status map of its own. Research state (studies, questions) is a separate
dimension with its own closed list, validated on load. Repository role is a third. A manuscript
whose study was stopped, for example, shows both facts rather than one merged label.

Disagreements are fixed at the source: temporal networks gets a `retired` event appended in `pubs`.
A known upstream limitation is recorded, not worked around: `pubs` derives status and venue from the
last event, so a Zenodo deposit after a submission reads "posted"; if venue and archive lifecycles
need separating, that change belongs in `pubs`.

### 4. Deadlines and time

Submission deadlines get one structured home: a proposed upstream `pubs` field for venue efforts
(deadline, source URL, date verified), separate from `follow_up.check_on`, which `pubs` defines as an
internal check date. Until it exists, the board shows the deadline as "not recorded" rather than
reading prose. Countdowns and working-tree state are computed when the board is run; committed
snapshots take an explicit `--as-of` date.

### 5. Views

- **The board (private):** `workspace.py board` prints manuscripts by publication status, studies by
  research state, and an attention list (deadlines within 45 days, `pubs` follow-ups due, sources
  that disagree, repositories with uncommitted or unpushed work). `--write` saves it to a gitignored
  `STATUS.local.md`. Nothing about private work is committed to the public umbrella.
- **`by-stage/` (private, disposable):** relative links only, gitignored before first creation, never
  read by any tool as input, excluded from scans and indexing; the umbrella is not linked into its own
  view (a cycle); a repository may appear under several groups when it holds several pieces of work;
  the generator refuses to overwrite real files, removes only links it owns, and reports missing
  targets. The board is the fallback where links are unavailable.
- **CLAUDE.md tables (public, as today):** generated from `workspace.json`, with repository notes
  reduced to role and pointers; current status lines come from the sources above.

### 6. The page for Dr. Matta

GitHub Pages from a public repository is public; granting Dr. Matta access to private repositories
does not make a page private to him. So the page is built only from a **public projection**:

- `public/program.json` (committed in the umbrella): the exact text and links to publish, one entry
  per item, each with an approval record (approved by, date, and a SHA-256 digest of the entry's
  content). Changing an approved entry voids its approval until the owner approves it again.
  Paragraphs drafted by an agent are proposals in the private agenda until approved.
- `workspace.py export` proposes projection entries from the private sources (titles, stages,
  DOIs, public dataset facts); it never copies notes, correspondence, follow-ups or local state.
- `workspace.py site` builds the page from the projection alone, into a clean directory: text is
  escaped, links are limited to `https` and relative pages, symbolic links and unexpected files are
  rejected, and a new artifact scanner (the existing secret patterns, applied to the exact bytes
  that will be deployed, which `scan` cannot do since it reads git objects) must pass.
- The GitHub Actions workflow builds from the committed projection without any private checkout or
  credential and deploys only the scanned artifact. The first version is previewed locally and goes
  live only after the owner approves.

### 7. Samples and data availability

`spent.json` stays the record of draws; dataset-to-draw links are derived from it, not stored again.
The ledger records draws and declared exclusions, not the remaining eligible population, and its
units differ (components, IPs, identities). So the tool reports each dataset's known exposure and
"remaining eligibility unverified". A verified availability statement needs a pinned dataset version,
eligibility rule, sampling unit and exclusion evidence produced by the child repository that owns the
sampler; the umbrella only reads such dated attestations.

### 8. The engine

- **Templates live in the children's world, versioned:** the hierarchy extension's conventions
  (design and dated amendments, synthetic controls first, claims ledger, pinned inputs, generated
  manuscript numbers, tests) become a versioned template. Studies and pilots get the research part
  only; manuscript machinery is added when a manuscript is started.
- `workspace.py new <kind> <name>` scaffolds from that template, registers the repository and a
  study (state `idea`; a design exists only when written), and links its question. Creating a private
  GitHub remote is a separate explicit flag.
- `workspace.py review <repo> <kind>` (design, controls, results, math) runs the Codex CLI on an
  explicit input packet: the named files are copied to a temporary directory and Codex runs there,
  read-only. A prompt prohibition is not an access boundary, and the read-only sandbox limits writes,
  not reads, so the packet narrows what Codex is pointed at but private data still must not sit in
  readable places it might open; the limitation is stated in each record. The run's record keeps the
  input hashes, the repository revision, the prompt, the model and settings, completion status and
  the verbatim reply, in the child's `notes/`.
- `workspace.py questions`: open questions, their studies, and candidate datasets with their known
  exposure (section 7).
- An ideas skill (Claude Code, umbrella `.claude/skills/`): reads questions, the withheld and failed
  claims across the claims ledgers, the dataset catalog and the novelty checks, and proposes new
  questions with a feasibility note as `idea` entries for the owner to accept or discard. Novelty
  checks and drafting are delegated to Codex where that saves tokens.

## Migration of existing records (Phase 1)

| Existing record | What happens |
|---|---|
| `workspace.json` repository notes | lifecycle prose removed; role and pointers stay; status comes from `pubs` or `studies.json` |
| Umbrella `README.md` status lines | generated from the sources, or pointed at the board and the public page |
| `docs/research/registry.yaml` | current facts (questions, statuses, priorities, deadlines, sample use) migrated to the records above; dated decisions and snapshots kept as history and marked as such; its duplicate `records.netsci_2027_abstract` key resolved (two different manuscripts) |
| `papers/agenda/questions-ledger.md` | becomes generated from `questions.json` |
| `papers/agenda/public-datasets.md` | keeps its narrative (controlled access, unavailable sets, gaps, design constraints); entries gain ids that `datasets.json` uses |
| `pubs` | a `retired` event for temporal networks; records for the ISCS 2027 paper and the NetSci 2027 abstracts; the proposed deadline field |

## Phases

1. **Map:** the entity records and their validation, the `pubs.jsonl` reader with its staleness
   check, the board, `by-stage/`, the migration above, then questions and datasets. Tests for each,
   including duplicate keys, unknown states, missing targets and refusal to overwrite.
2. **Page:** the projection and its approvals, `export`, `site`, the artifact scanner, a local
   preview, then the workflow after the owner approves. Tests for omitted fields, escaping, link
   schemes, stale output and a planted secret.
3. **Engine:** the versioned template exercised on one real new study first, then `new`, `review`,
   `questions`, then the ideas skill.

Each phase is reviewed by Codex before it is called done.

## Left to the builder

Exact JSON layouts, the template's text, the page's look (plain, readable, works on a phone, light
and dark), and ordering within a group.

## Calls made by the assistant (for the owner to check)

1. Folders stay where they are; a generated, private `by-stage/` link view replaces the proposed
   `published/`, `unpub/` and `drafts/` folders.
2. `pubs/` stays the single source for publication status, read unchanged.
3. Status is tracked per manuscript and per study, not per repository.
4. The board is private (local only); the public umbrella gets no status of private work.
5. The page for Dr. Matta publishes only text the owner approved word for word, and goes live only
   after he approves the preview.
6. Questions, studies and datasets become structured records in the private agenda.
7. Map first, page second, engine third.
