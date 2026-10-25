# Repository Guidelines

## Project Structure & Module Organization

CMRI is an umbrella for the Cognitive MRI research program. Read `README.md` for orientation and `CLAUDE.md` for detailed workspace rules.

- `tools/workspace.py` manages repositories, snapshots, and generated reports; `tools/tests/` contains its tests.
- `workspace.json` defines repositories and sources; `spent.json` records consumed research samples.
- `docs/research/` holds program decisions and audits; `future-ideas/` contains proposals, reproducibility checks, and supporting artifacts.
- `papers/` and `chatgpt-conversation-corpus/` contain independent, gitignored repositories. Their source, manuscripts, data, and assets must be changed and committed within the owning repository. Read its local guidance first.
- `dev/` holds private working data; `by-stage/` is a disposable generated view.

## Build, Test, and Development Commands

Run these from the repository root unless indicated:

- `python3 tools/workspace.py clone --dry-run`: preview workspace setup; omit `--dry-run` to clone missing repositories.
- `python3 tools/workspace.py status`: inspect child checkouts.
- `python3 tools/workspace.py render`: regenerate manifest-derived tables; use `render --check` to detect stale output.
- `python3 tools/workspace.py spent --verify`: validate sample exclusion records.
- `cd tools && uv run --no-project --with pytest pytest tests`: run tooling tests.

There is no umbrella application build. Follow each child repository's build instructions.

## Coding Style & Naming Conventions

Match existing Python: four-space indentation, `snake_case` functions, `PascalCase` dataclasses, and type annotations. Keep workspace tooling standard-library-only. Use structured JSON and validated records; derive reports from sources rather than duplicating mappings. JSON uses two-space indentation. No umbrella formatter or linter is configured. Edit sources, never generated tables or ledgers.

## Testing Guidelines

Use pytest with `test_*.py` files and `test_*` functions. Add regression tests for changed behavior, including validation failures and filesystem boundaries. Existing integration tests use temporary real Git repositories and fake only `gh`. For coverage, add `--with pytest-cov` to the uv options and `--cov=workspace --cov-report=term-missing` to pytest; inspect uncovered branches. No numeric coverage threshold is configured.

## Commit & Pull Request Guidelines

Recent commits use descriptive, action-led subjects, such as “Add guarded disposable stage views”; no mandatory prefix scheme is evident. Keep commits scoped to the owning repository. PRs should explain the change, link relevant issues or research decisions, and report validation commands and results.

## Data & Research Integrity

Keep credentials, private conversations, embeddings, and local reports out of public commits. Preserve frozen protocols and sample provenance; distinguish exploratory results from confirmation. Generate manuscript numbers from committed evidence and retain attribution.
