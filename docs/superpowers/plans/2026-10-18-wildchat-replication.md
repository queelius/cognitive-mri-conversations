# WildChat cross-collection replication (F6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the frozen PRISM primary contrast (uniform history mean minus last-two window, next-user target) once on a pre-drawn 500-component WildChat-4.8M confirmation sample, after a 360-component development sample has been used to check the adapter, fit the centers and record pre-registration values; report the result either way with model-family and year heterogeneity.

**Architecture:** A new child repo `papers/wildchat-replication/` (GitHub `queelius/cmri-wildchat-replication`, private) with package `cmri_replicate`. It reads the local WildChat parquet mirror with `pyarrow` (metadata columns before the freeze; the `conversation` column only inside the two `prepare` stages), converts rows to the same frozen conversation shape the PRISM confirmation used, and reuses the PRISM confirmation's scoring, bootstrap, secondary panel and distribution code by pinned import from `cmri_confirm` (`papers/iscs-2027`), `cmri_history` and `cmri_wave`. A staged runner with write-once outputs separates pre-freeze stages (audit, samples, prepare-dev, analyze-dev, freeze) from post-freeze single-shot stages (prepare, analyze-*, subgroups) and the regenerable report.

**Tech Stack:** Python 3.11 in the repo's own `.venv` (created with `uv venv --python 3.11`), numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, onnxruntime 1.23.2, tokenizers 0.22.2, matplotlib 3.10.7, pyarrow (pinned at install), pytest, pytest-cov. Encoders: the pinned ONNX MiniLM-L6-v2 and E5-small-v2 in `papers/semantic-measurement/cache/models/`.

**Spec:** `docs/superpowers/specs/2026-10-18-wildchat-replication-design.md`.

## Global Constraints

- Source: the local mirror `papers/temporal-networks/data/wildchat/full_shards/shard_0000.parquet` … `shard_0085.parquet` (86 shards, 3,199,860 rows, dataset `allenai/WildChat-4.8M`, ODC-BY). Pinned by per-shard bytes and sha256 in `audits/shards.json`. Never modified, never copied.
- Metadata columns read before the freeze: `conversation_hash, model, timestamp, turn, language, toxic, redacted, hashed_ip`. Message `content` is read only inside `prepare-dev` (development ids) and `prepare` (confirmation ids, after `FREEZE.json` exists). Tests never open the mirror; they use synthetic parquet fixtures.
- Eligibility (metadata): `language == 'English'`, `toxic == False`, `redacted == False`, `turn >= 3`, `conversation_hash` unique in the mirror (both copies of a duplicate are dropped), `model` matches a configured family prefix, `hashed_ip` not in `audits/exclusions.json`, `conversation_hash` not in `audits/exclusions.json`, and the IP has at most `max_eligible_per_component` = 100 eligible conversations.
- Model family: `config.json` `model_families` = `["gpt-4o", "gpt-4.1", "gpt-3.5", "gpt-4"]`; the family is the first list entry that is a prefix of the `model` string (so `gpt-4o-mini-…` is `gpt-4o`, `gpt-4-1106-preview` is `gpt-4`). No match is `unknown_model`, an exclusion reason, never a silent "other".
- Samples: eligible IPs ranked by `sha256(f'{component_salt}:{hashed_ip}')` with `component_salt = "cmri-wildchat-component-v1"`; first 360 are development, next 500 confirmation. Within an IP, conversations ranked by `sha256(f'{conversation_salt}:{conversation_hash}')`, `conversation_salt = "cmri-wildchat-conversation-v1"`, first 4 kept. Component id everywhere outside `cache/` is the first 16 hex characters of the component ranking digest; raw `hashed_ip` appears in no committed or `results/` file.
- Text checks (inside prepare, per sampled conversation): first 12 `conversation` entries; roles strictly alternate starting with `user`; every `content` a non-empty string after `strip()`; every non-null message `hashed_ip` equals the row's `hashed_ip`; at least 6 messages remain. A failing conversation is dropped (reason counted), not replaced. If dropped conversations exceed `max_text_check_loss` = 0.05 of the drawn count, the stage raises and the halt is the recorded outcome.
- Visibility, encoders and contrast identical to `papers/iscs-2027/config.json`: `message_tokens 128`, `max_messages 12`, `first_target 4`, `encoder_batch_sizes {"minilm": 32, "e5": 1}`, `threads 4`, `bootstrap_replicates 2000`, `bootstrap_seed 104729`, `practical_cosine_gain 0.01`, `budget_windows [1, 2, 3, 4, 6, 8, 10]`; `max_prepare_wall_seconds 7200`. `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
- Success rule (confirmation only): `ci_low > 0` in both encoders. Development values are pre-registration evidence written into `PROTOCOL.md` before the freeze.
- Centers for the centered-cosine secondary are fitted on the development sample's natural targets (row-weighted as `cmri_history.analysis.fixed_centers`) and stored in `results/development-{encoder}.json`, which the freeze hashes.
- Frozen set (`FREEZE.json`): `PROTOCOL.md`, `config.json`, `audits/dependencies.json`, `audits/shards.json`, `audits/exclusions.json`, `audits/samples.json`, `results/audit.json`, `results/prepare-development.json`, `results/development-minilm.json`, `results/development-e5.json`, and the modules `src/cmri_replicate/{source,eligibility,adapter,encode,cases,analysis,heterogeneity}.py` with their tests `tests/test_{source,eligibility,adapter,encode,cases,analysis,heterogeneity}.py`. `runner.py`, `__main__.py`, `reporting.py` and their tests are outside the frozen set and are recorded as `code_changed_since_freeze`.
- Upstream never modified; pinned commits: `papers/iscs-2027` `ef845c2`, `papers/history-falsification` `7a391141f7de3d06aeeca11434baadf6e744a3e0`, `papers/portfolio-wave1` `e84846590e228fc3457bfd7e8d411118fb6035ae`.
- JSON artifacts carry ids, digests, counts and numbers; never text, never raw hashed IPs. `cache/` is gitignored.
- No em-dashes in prose files. Every commit, including subagent commits, ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01ABMh8yCR6MXELxjhL2qx1W`.

## Review Focus

1. A sampled conversation whose text fails a check is dropped and counted; the stage halts only above 5% loss, and a halt leaves a failed attempt record and no output. Test in Task 7.
2. Assistant messages carry a null `hashed_ip` in this schema; null must not trigger `ip_inconsistent`, while a user message with a different non-null IP must. Test in Task 4.
3. An IP in the exclusion list that would have ranked first must change nothing except the counts: the remaining ranking is identical to a draw from a mirror that never contained it. Test in Task 3.
4. Both copies of a repeated `conversation_hash` are dropped, and the IP's eligible count is computed after that drop. Test in Task 3.
5. After the freeze, editing a frozen module makes `prepare` and `analyze-*` refuse, while editing `reporting.py` does not stop `report`. Test in Task 7.

---

### Task 1: Scaffold `papers/wildchat-replication` with environment, pins, exclusions and protocol draft

**Files:**
- Create: `papers/wildchat-replication/pyproject.toml`, `.gitignore`, `README.md`, `config.json`, `PROTOCOL.md` (draft; frozen in Task 8), `scripts/pin_dependencies.py`, `scripts/build_exclusions.py`, `audits/dependencies.json`, `audits/exclusions.json`, `src/cmri_replicate/__init__.py`, `tests/conftest.py`
- Test: `tests/test_pins.py`

**Interfaces:**
- Produces `config.json` keys: `mirror` = `{"dataset": "allenai/WildChat-4.8M", "directory": "papers/temporal-networks/data/wildchat/full_shards", "shards": 86, "rows": 3199860}`; `eligibility` = `{"language": "English", "min_turn": 3, "max_eligible_per_component": 100}`; `model_families` as in Global Constraints; `samples` = `{"development_components": 360, "confirmation_components": 500, "max_conversations_per_component": 4, "component_salt": "cmri-wildchat-component-v1", "conversation_salt": "cmri-wildchat-conversation-v1"}`; `expected_confirmation` = `{"components": null, "conversations": null}` (filled in Task 8 before the freeze); `max_text_check_loss: 0.05`; and verbatim from iscs-2027: `message_tokens, max_messages, first_target, encoder_batch_sizes, threads, bootstrap_replicates, bootstrap_seed, practical_cosine_gain, budget_windows`; `max_prepare_wall_seconds: 7200`.
- Produces `audits/dependencies.json`: `{"iscs_commit", "wave_commit", "history_commit", "workspace_relative_files": {...}}` hashing every file the iscs-2027 pin script lists plus `papers/iscs-2027/src/cmri_confirm/{boundary,cases,analysis,heterogeneity,runner}.py`, `papers/iscs-2027/config.json`, `papers/iscs-2027/FREEZE.json`.
- Produces `audits/exclusions.json`: `{"ip_sha256": [...sorted...], "conversation_hashes": [...sorted...], "sources": {"temporal_candidates": "papers/temporal-networks/data/wildchat/candidates.csv", "temporal_selected": ".../selected_users.csv", "temporal_users": ".../users/*/manifest.json", "cross_section_sample": "future-ideas/iscs-2027/research-deep-dive/checks/public-cross-section/cache/content-sample.parquet"}, "counts": {"ips": n, "conversations": 2000}}`. `ip_sha256` entries are `sha256(hashed_ip)` hex; conversation hashes are the public `conversation_hash` keys.
- `tests/conftest.py` defines `ROOT`, `WORKSPACE = ROOT.parents[1]`, `sha256(path)`, `git_head(path)`, a `config` session fixture, and `write_shard(path, rows)` that writes a parquet file with columns `conversation_hash (string), model (string), timestamp (timestamp[us]), turn (int64), language (string), toxic (bool), redacted (bool), hashed_ip (string), conversation (list<struct<content: string, role: string, hashed_ip: string, turn_identifier: int64>>)` from a list of dicts, plus `row(hash, ip, model='gpt-4o-2024-08-06', turns=3, language='English', toxic=False, redacted=False, year=2024, contents=None)` building one such dict with alternating user/assistant messages where assistant `hashed_ip` is `None`.

- [ ] **Step 1: Write `tests/test_pins.py`**

```python
def test_dependency_pins_match_workspace():
    pins = json.loads((ROOT/'audits/dependencies.json').read_text())
    for rel, digest in pins['workspace_relative_files'].items():
        assert sha256(WORKSPACE/rel) == digest, rel
    assert git_head(WORKSPACE/'papers/iscs-2027') == pins['iscs_commit'] == 'ef845c2' + pins['iscs_commit'][7:]
    assert git_head(WORKSPACE/'papers/portfolio-wave1') == pins['wave_commit'] == 'e84846590e228fc3457bfd7e8d411118fb6035ae'
    assert git_head(WORKSPACE/'papers/history-falsification') == pins['history_commit'] == '7a391141f7de3d06aeeca11434baadf6e744a3e0'

def test_config_matches_frozen_prism_visibility(config):
    prism = json.loads((WORKSPACE/'papers/iscs-2027/config.json').read_text())
    for key in ('message_tokens','max_messages','first_target','encoder_batch_sizes','threads','bootstrap_replicates','bootstrap_seed','practical_cosine_gain','budget_windows'):
        assert config[key] == prism[key], key
    assert config['model_families'] == ['gpt-4o','gpt-4.1','gpt-3.5','gpt-4']
    assert config['samples'] == {'development_components': 360, 'confirmation_components': 500, 'max_conversations_per_component': 4, 'component_salt': 'cmri-wildchat-component-v1', 'conversation_salt': 'cmri-wildchat-conversation-v1'}

def test_exclusions_are_digests_not_ips():
    ex = json.loads((ROOT/'audits/exclusions.json').read_text())
    assert ex['counts']['conversations'] == 2000 == len(ex['conversation_hashes'])
    assert 150 <= ex['counts']['ips'] == len(ex['ip_sha256']) <= 200
    assert all(len(d) == 64 for d in ex['ip_sha256']) and ex['ip_sha256'] == sorted(set(ex['ip_sha256']))
```

- [ ] **Step 2: Run to verify it fails**: `cd papers/wildchat-replication && .venv/bin/python -m pytest tests/test_pins.py -q` → FAIL (files missing). Before that, create the environment: `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e . pytest pytest-cov`; record the installed `pyarrow` version into `pyproject.toml` as an exact pin.
- [ ] **Step 3: Create the scaffold.** `pyproject.toml` with `name = "cmri-wildchat-replication"`, the dependency pins above plus `pyarrow==<installed>`, `pythonpath = ["src", "../iscs-2027/src", "../portfolio-wave1/src", "../history-falsification/src"]`, `[tool.coverage.run] branch = true, source = ["cmri_replicate"]`. `.gitignore`: `cache/`, `results/replay/`, `.venv/`, `__pycache__/`, `htmlcov/`, `.coverage*`, `*.log`. `README.md`: purpose, boundary (what is read when), the ODC-BY attribution paragraph copied from `papers/temporal-networks-calibration/pilot/public-data/README.md`, run commands (from Task 7). `scripts/pin_dependencies.py`: the iscs-2027 script extended with `ISCS = 'papers/iscs-2027'` files and `iscs_commit`. `scripts/build_exclusions.py`: reads the four sources with `csv`/`json`/`pyarrow` (column `conversation_hash` only from the parquet), writes `audits/exclusions.json`. `config.json` as in Interfaces.
- [ ] **Step 4: Write `PROTOCOL.md` draft**: copy iscs-2027 `PROTOCOL.md` sections "Primary contrast and success rule", "Declared secondary panel" and "Heterogeneity analyses" verbatim (add subgroup axes `model family` and `year`); replace "Sample rule" with the eligibility, exclusions, ranking and sizes above; replace "Embargo" with: metadata columns are read before the freeze to draw and declare both samples; development text is read and encoded before the freeze; confirmation text is read only after `FREEZE.json` exists; the expected confirmation counts live in `config.json`; dropped-conversation rule and 5% halt. Leave the "Pre-registered development values" section with the sentence "filled before the freeze from `results/development-*.json`".
- [ ] **Step 5: Run tests** → `tests/test_pins.py` PASS.
- [ ] **Step 6: Commit**: `git init -b main`, empty root commit `chore: root`, then `git add -A && git commit -m "Scaffold WildChat replication with pins, exclusions and protocol draft"`. `gh repo create queelius/cmri-wildchat-replication --private --source . --remote origin --push`.

---

### Task 2: Parquet source access

**Files:**
- Create: `src/cmri_replicate/source.py`
- Test: `tests/test_source.py`

**Interfaces:**
- Produces: `META_COLUMNS = ('conversation_hash','model','timestamp','turn','language','toxic','redacted','hashed_ip')`; `shard_paths(directory: Path, count: int) -> list[Path]` (exactly `shard_0000.parquet`…, raises if a name is missing or extra parquet files exist); `shard_manifest(paths) -> dict[str, dict]` (`{name: {'bytes': int, 'sha256': str}}`); `verify_shards(paths, manifest) -> None`; `scan_metadata(paths) -> list[dict]` (one dict per row with the eight columns, `timestamp` as a `datetime`); `read_rows(paths, ids: set[str]) -> dict[str, dict]` (full rows, including `conversation`, for exactly those `conversation_hash` values; reads `conversation_hash` first to locate rows, then takes only the matching row groups; raises if an id is missing or found twice).

- [ ] **Step 1: Write failing tests**

```python
def test_scan_metadata_reads_only_metadata_columns(tmp_path):
    write_shard(tmp_path/'shard_0000.parquet', [row('h1','ip1'), row('h2','ip2', language='Spanish')])
    rows = source.scan_metadata([tmp_path/'shard_0000.parquet'])
    assert [r['conversation_hash'] for r in rows] == ['h1','h2'] and set(rows[0]) == set(source.META_COLUMNS)

def test_read_rows_returns_exactly_requested_ids(tmp_path):
    p = tmp_path/'shard_0000.parquet'; write_shard(p, [row('h1','ip1'), row('h2','ip2')])
    got = source.read_rows([p], {'h2'})
    assert set(got) == {'h2'} and got['h2']['conversation'][0]['role'] == 'user'
    with pytest.raises(ValueError, match='missing'): source.read_rows([p], {'h9'})

def test_read_rows_rejects_duplicate_id(tmp_path): ...  # same hash in two shards -> ValueError 'duplicate'

def test_shard_paths_requires_exact_set(tmp_path): ...  # 2 shards expected, 1 present -> ValueError; extra parquet -> ValueError

def test_verify_shards_detects_change(tmp_path): ...  # manifest from shard_manifest, then append a byte -> ValueError
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** with `pyarrow.parquet.ParquetFile`; `scan_metadata` uses `read(columns=META_COLUMNS).to_pylist()` per shard; `read_rows` iterates row groups, reads `conversation_hash` for each, and reads the full row group only when it holds a requested id. **Step 4: Run** → PASS. **Step 5: Commit** `feat: parquet source access with shard pins`.

---

### Task 3: Eligibility, audit and sample draw

**Files:**
- Create: `src/cmri_replicate/eligibility.py`
- Test: `tests/test_eligibility.py`

**Interfaces:**
- Consumes: metadata dicts from `source.scan_metadata`.
- Produces: `model_family(model: str, families: Sequence[str]) -> str | None`; `component_id(hashed_ip: str, salt: str) -> str` (first 16 hex of `sha256(f'{salt}:{hashed_ip}')`); `classify(rows, config, exclusions) -> tuple[list[dict], Counter]` returning the eligible rows (each with added keys `family`, `year`, `component`, `component_rank`, `conversation_rank`) and a `Counter` of exclusion reasons in this order of precedence: `duplicate_hash`, `not_english`, `toxic`, `redacted`, `too_short`, `unknown_model`, `excluded_conversation`, `excluded_ip`, `high_volume_ip`; `audit(rows, config, exclusions) -> dict` (`rows`, `eligible`, `reasons`, `by_family`, `by_year`, `components`, `component_size_histogram` (capped at 20), `timestamp_min/max`); `draw_samples(eligible, config) -> dict` with `{'development': {component: {'ip_sha256': str, 'conversations': [hash,...]}}, 'confirmation': {...}, 'expected': {'development': {'components': n, 'conversations': n}, 'confirmation': {...}}}`.

- [ ] **Step 1: Write failing tests**

```python
def test_model_family_prefix_order():
    fams = ['gpt-4o','gpt-4.1','gpt-3.5','gpt-4']
    assert [eligibility.model_family(m, fams) for m in ('gpt-4o-mini-2024-07-18','gpt-4-1106-preview','gpt-4.1-mini-2025-04-14','gpt-3.5-turbo-0613','claude-3')] == ['gpt-4o','gpt-4','gpt-4.1','gpt-3.5',None]

def test_duplicate_hash_drops_both_copies_before_volume_count(config):
    rows = [meta('h1','ip1'), meta('h1','ip1'), meta('h2','ip1')]
    eligible, reasons = eligibility.classify(rows, config, no_exclusions)
    assert [r['conversation_hash'] for r in eligible] == ['h2'] and reasons['duplicate_hash'] == 2

def test_excluded_ip_does_not_shift_remaining_ranking(config):
    rows = [meta(f'h{i}', f'ip{i}') for i in range(50)]
    full, _ = eligibility.classify(rows, config, no_exclusions)
    first = sorted(full, key=lambda r: r['component_rank'])[0]['hashed_ip']
    ex = {'ip_sha256': [sha256(first)], 'conversation_hashes': []}
    without, _ = eligibility.classify([r for r in rows if r['hashed_ip'] != first], config, no_exclusions)
    excluded, reasons = eligibility.classify(rows, config, ex)
    assert reasons['excluded_ip'] == 1
    assert eligibility.draw_samples(excluded, small_config) == eligibility.draw_samples(without, small_config)

def test_high_volume_ip_excluded(config): ...  # 101 eligible rows on one ip -> all 'high_volume_ip'; 100 stay

def test_draw_samples_disjoint_and_capped(config): ...  # small_config: 3 dev, 2 confirmation components, cap 2; assert disjoint components, <=2 conversations each, expected counts match, component ids are 16 hex and ip_sha256 is 64 hex

def test_unknown_model_is_a_reason_not_other(config): ...
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Commit** `feat: eligibility, audit and salted sample draw`.

---

### Task 4: Adapter and encoding

**Files:**
- Create: `src/cmri_replicate/adapter.py`, `src/cmri_replicate/encode.py`
- Test: `tests/test_adapter.py`, `tests/test_encode.py`

**Interfaces:**
- Consumes: full rows from `source.read_rows`; `cmri_wave.data.Message, Role`; `cmri_wave.prepare._ENCODERS, _load_encoding_module, _sha256, _verified_dependencies`.
- Produces (`adapter.py`): frozen dataclass `ExternalConversation(id, participant, component, condition, sample, year, messages: tuple[Message, ...])` with the same `__post_init__` invariants as `cmri_confirm.reserve.ReserveConversation` (alternation, at least 6, strictly increasing `source_index`) plus `sample in ('development','confirmation')` and `year` an int; `convert(row: dict, component: str, ip_sha256: str, sample: str, config) -> ExternalConversation | str` returning the conversation or one reason string from `('ip_mismatch','ip_inconsistent','bad_roles','empty_content','too_short')`; `condition` is the model family, `Message.model` is the row's `model`, `source_index` the list position, `alternatives` empty.
- Produces (`encode.py`): `encode_sample(conversations: list[ExternalConversation], config, workspace: Path, output: Path) -> dict`, the body of `cmri_confirm.encode.encode_reserve` without the manifest guard and without `_check_source`; records carry `id, component, participant, split, sample, condition, year, roles, masses, original_tokens, model`, where `split` is `'fit'` for development and `'reserve'` for confirmation (the `cmri_history.Case` split type); artifacts `output/{minilm,e5}.{npz,json}` and `output/preparation.json` exactly as upstream.

- [ ] **Step 1: Write failing tests**

```python
def test_convert_accepts_null_assistant_ip_and_rejects_foreign_user_ip(config):
    ok = adapter.convert(fullrow('h1','ip1'), 'c1', sha256('ip1'), 'development', config)
    assert isinstance(ok, adapter.ExternalConversation) and ok.condition == 'gpt-4o' and ok.year == 2024
    bad = fullrow('h2','ip1'); bad['conversation'][2]['hashed_ip'] = 'ip9'
    assert adapter.convert(bad, 'c1', sha256('ip1'), 'development', config) == 'ip_inconsistent'
    assert adapter.convert(fullrow('h3','ip7'), 'c1', sha256('ip1'), 'development', config) == 'ip_mismatch'

def test_convert_truncates_to_twelve_and_requires_six(config): ...  # 20 messages -> 12; 5 messages -> 'too_short'; 7 messages -> 7
def test_convert_rejects_role_order_and_blank_content(config): ...  # assistant first -> 'bad_roles'; '   ' content -> 'empty_content'

def test_encode_sample_writes_both_encoders_and_records(tmp_path, monkeypatch):  # stub encoder as in iscs-2027 tests/test_encode.py
    ... assert set(result['encoders']) == {'minilm','e5'} and records[0]['split'] == 'fit' and 'hashed_ip' not in json.dumps(records)
def test_encode_sample_refuses_batched_e5(tmp_path): ...
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Commit** `feat: WildChat adapter and sample encoding`.

---

### Task 5: Cases and analysis

**Files:**
- Create: `src/cmri_replicate/cases.py`, `src/cmri_replicate/analysis.py`
- Test: `tests/test_cases.py`, `tests/test_analysis.py`

**Interfaces:**
- Consumes: `cmri_confirm.analysis.primary, budget_curve, role_panel`; `cmri_confirm.boundary.digest`; `cmri_wave.inference.row_weights`.
- Produces (`cases.py`): `SAMPLES = ('development','confirmation')`, `SPLIT_OF = {'development': Split.FIT, 'confirmation': Split.RESERVE}`; `load_sample(root, sample, encoder) -> tuple[list[dict], dict[str, np.ndarray]]` (reads `cache/{sample}/{encoder}.{json,npz}`, verifies bytes and sha256 and key set, and that every record's `sample` equals the argument); `build_cases(records, arrays, regime=Regime.NATURAL) -> list[Case]` (the body of `cmri_confirm.cases.build_reserve_cases` with `Split(r['split'])` taken from the record and no manifest guard).
- Produces (`analysis.py`): `fit_centers(cases) -> dict[Role, np.ndarray]` (natural, native, row-weighted mean of targets per role, as `fixed_centers` but over all given cases regardless of split); `load_centers(root, encoder) -> dict[Role, np.ndarray]` (from `results/development-{encoder}.json` key `centers`, only after `boundary.load_config(root)` has verified the freeze); `analyze(root, sample, encoder, config, centers) -> dict` with keys `encoder, sample, counts, primary, budget, roles, per_case` as in `cmri_confirm.analysis.analyze`, plus `centers` (lists) when `sample == 'development'`.

- [ ] **Step 1: Write failing tests**

```python
def test_six_and_twelve_message_targets(): ...  # 6 messages -> one user + one assistant target; 12 -> four each; indices never exceed 11
def test_build_cases_keeps_record_split(): ...  # development -> Split.FIT; confirmation -> Split.RESERVE
def test_load_sample_rejects_wrong_sample(tmp_path): ...  # records say 'development', ask for 'confirmation' -> ValueError
def test_fit_centers_row_weighted(): ...  # two components, one with 3 conversations, one with 1: center equals the equal-component-weight mean, not the plain mean
def test_analyze_primary_passes_on_synthetic_history_signal(): ...  # synthetic arrays where the mean predicts the target better than the last two -> primary['passed'] True, counts correct
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Commit** `feat: sample cases, centers and confirmation analysis`.

---

### Task 6: Heterogeneity

**Files:**
- Create: `src/cmri_replicate/heterogeneity.py`
- Test: `tests/test_heterogeneity.py`

**Interfaces:**
- Consumes: `cmri_confirm.heterogeneity.distribution`; `cmri_history.analysis.summarize`.
- Produces: `subgroups(cases, new, baseline, config, labels: Mapping[str, Mapping[str, object]]) -> dict` where `labels` maps axis name to `{conversation_id: value}` and the result maps axis to `{value: summary}` (re-bootstrapping components within each subset; values sorted); `analyze_subgroups(root, config, encoder, public: dict, development: dict) -> dict` with axes `family` (record `condition`), `year`, `message_count` (len of roles), and `distribution = {'development': ..., 'confirmation': ..., 'pooled': ..., 'development_ids_disjoint': True}` using `public['primary']` and `development['primary']` component differences, raising if the component sets intersect.

- [ ] **Step 1: Write failing tests**: `test_subgroups_bootstrap_within_subset_only` (a component present in two year groups is counted once per group it has cases in, with no conversation in two groups); `test_subgroups_axis_values_sorted_and_json_safe` (year keys are ints, family keys strings); `test_distribution_requires_disjoint_components`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Commit** `feat: model-family, year and length heterogeneity`.

---

### Task 7: Runner and CLI

**Files:**
- Create: `src/cmri_replicate/runner.py`, `src/cmri_replicate/__main__.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `cmri_confirm.runner.write_json, compare, source_state`; `cmri_confirm.boundary.digest, verify_paths, load_config`; everything above.
- Produces: `Stage` = `audit, samples, prepare-dev, analyze-dev, freeze, prepare, analyze-minilm, analyze-e5, analyze-subgroups, report, compare`; `FROZEN_FILES` and `FROZEN_CODE` tuples as in Global Constraints; `execute(root, stage, output='results', other=None, *, allow_dirty=False, replay=False, tolerance=None)`; CLI `python -m cmri_replicate <stage> [--output DIR] [--allow-dirty]` and `compare DIR_A DIR_B [--replay] [--tolerance F]` as in iscs-2027.
- Stage contracts: `audit` verifies `audits/shards.json` (creating it on first run from `shard_manifest`) and the configured row count, runs `eligibility.audit`, writes `results/audit.json` (write-once). `samples` reads `results/audit.json`'s hash, runs `classify` and `draw_samples`, writes `audits/samples.json` (write-once) and prints the expected counts; it refuses to run after `FREEZE.json` exists. `prepare-dev` reads `audits/samples.json`'s development ids, `source.read_rows`, `adapter.convert` each, drops failures with counted reasons, raises on loss above `max_text_check_loss`, calls `encode_sample` into `cache/development/`, writes `results/prepare-development.json` (selection counts, reasons, `selected_ids_sha256`, encoder fingerprints and npz hashes). `analyze-dev` writes `results/development-{minilm,e5}.json` (both in one stage; centers from `fit_centers`; subgroups included under `subgroups`). `freeze` refuses if any frozen file is uncommitted or `expected_confirmation` holds nulls, writes `FREEZE.json` with `freeze_commit` and all hashes. `prepare` requires `FREEZE.json`, verifies the hashes, checks the confirmation draw's component and conversation counts against `config['expected_confirmation']` before any `read_rows` call, then proceeds as `prepare-dev` into `cache/confirmation/` and `results/prepare.json`. `analyze-minilm`/`analyze-e5` write `results/public-{encoder}.json` using `load_centers`. `analyze-subgroups` writes `results/subgroups-{encoder}.json` for both encoders. `report` is regenerable (Task 9). Every post-freeze stage records provenance as upstream (`this_commit, freeze_commit, code_changed_since_freeze, dirty_src, iscs_commit, wave_commit, history_commit, started, runtime_seconds, python, thread_environment`) and an attempt record under `results/attempts/`.

- [ ] **Step 1: Write failing tests** (git and workspace pins stubbed as in iscs-2027 `tests/test_runner.py`; a two-shard synthetic mirror written by `write_shard` and pointed to through a tmp `config.json`):

```python
def test_samples_refuses_after_freeze(frozen_repo): ...
def test_prepare_refuses_without_freeze(repo): ...
def test_prepare_checks_expected_counts_before_reading_text(frozen_repo, monkeypatch):  # monkeypatch source.read_rows to raise AssertionError('text read'); wrong expected counts -> ValueError mentioning 'expected', not AssertionError
def test_prepare_drops_failing_conversations_and_halts_above_loss(frozen_repo, stub_encoder):  # 1 bad of 40 -> dropped, reasons recorded; 3 bad of 40 -> ValueError 'loss', failed attempt record, no results/prepare.json
def test_frozen_module_edit_blocks_prepare_but_not_report(frozen_repo, stub_stages):  # append a comment to src/cmri_replicate/adapter.py -> prepare raises 'frozen file changed'; append to reporting.py -> report runs
def test_outputs_are_write_once(frozen_repo, stub_stages): ...
def test_compare_replay_ignores_runtime_fields(tmp_path): ...  # reuse of cmri_confirm.compare: a copied result dir with changed runtime_seconds compares equal in replay mode only
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Commit** `feat: staged runner with pre-freeze development and post-freeze confirmation`.

---

### Task 8: Development run, protocol completion and freeze

**Files:**
- Modify: `config.json` (`expected_confirmation`), `PROTOCOL.md` (development values), `README.md` (counts)
- Create: `audits/shards.json`, `audits/samples.json`, `results/audit.json`, `results/prepare-development.json`, `results/development-{minilm,e5}.json`, `FREEZE.json`

- [ ] **Step 1: Run `audit`** with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m cmri_replicate audit`. Expected: `results/audit.json` with `rows == 3199860`; `eligible` near 150,000 (the exclusions and the `redacted` and `unknown_model` rules remove some of the 152,730 from the pre-design audit; record the exact number); reasons listed. Commit `audits/shards.json` and `results/audit.json`.
- [ ] **Step 2: Run `samples`.** Expected: 360 development and 500 confirmation components, up to 4 conversations each. Copy the printed `expected.confirmation` counts into `config.json` `expected_confirmation`. Commit `audits/samples.json` and `config.json`.
- [ ] **Step 3: Run `prepare-dev`** then `analyze-dev`. Expected: loss below 5%; both encoders analysed; `results/development-*.json` carry `primary.gain`, `centers`, `subgroups`. Record wall time.
- [ ] **Step 4: Complete `PROTOCOL.md`**: write the development primary gains (both encoders, five decimals, with intervals) under "Pre-registered development values" and the exact eligible, drawn and retained counts under "Sample rule". Commit everything with `git add -A` (`cache/` is ignored).
- [ ] **Step 5: Verify the tree is clean and the full suite passes**: `.venv/bin/python -m pytest -q` → all PASS. Push.
- [ ] **Step 6: Freeze**: `python -m cmri_replicate freeze` → `FREEZE.json`; commit `chore: freeze the WildChat confirmation protocol`; push; record `gh api repos/queelius/cmri-wildchat-replication/activity` timestamps of the freeze push in `audits/freeze-push.json` (committed later with the results).

---

### Task 9: Confirmation run

**Files:**
- Create: `results/prepare.json`, `results/public-{minilm,e5}.json`, `results/subgroups-{minilm,e5}.json`, `results/attempts/*`

- [ ] **Step 1: Run `prepare`** (single shot). Expected: counts match `expected_confirmation`; loss below 5%; `results/prepare.json` written; `provenance.started` later than the freeze push time.
- [ ] **Step 2: Run `analyze-minilm`, `analyze-e5`, `analyze-subgroups`.** Expected: three result files per encoder family; `primary.passed` recorded for each encoder.
- [ ] **Step 3: Commit** `results/` and `audits/freeze-push.json` as `results: WildChat confirmation run`; push; record the push timestamp in `audits/freeze-push.json` by a follow-up commit.
- [ ] **Step 4: Replay**: run the three analysis stages with `--output results/replay`, then `python -m cmri_replicate compare results results/replay` → `equal: true` (exact mode; nothing was re-encoded). Record the output.

---

### Task 10: Reporting and figures

**Files:**
- Create: `src/cmri_replicate/reporting.py`, `results/report/` (generated)
- Test: `tests/test_reporting.py`

**Interfaces:**
- Produces: `gate(results) -> dict` (reuse `cmri_confirm.reporting.gate`); `tables(results, development, subgroups, out)` writing `primary.csv` (encoder, sample ∈ {prism-development, prism-reserve, wildchat-development, wildchat-confirmation}, gain, ci_low, ci_high, components, conversations; the PRISM rows read from the pinned `papers/iscs-2027/results/public-*.json` and `papers/history-falsification/results/public-*.json`), `budget.csv`, `subgroups.csv` (encoder, axis ∈ {family, year, message_count}, label, gain, ci_low, ci_high, components, conversations), `distribution.csv`; `figures(...)` writing PNG+PDF pairs `primary-four-samples`, `component-gain-distribution` (ECDF: PRISM reserve vs WildChat confirmation, both encoders), `budget-curve`, `role-origin-confirmation`, `family-by-year` (gain with intervals per family, panels per year; cells with fewer than 10 components drawn hollow); `produce(root, output, provenance) -> gate` writing `results/report/{gate.json, README.md}` with numbers only.

- [ ] **Step 1: Write failing tests**: `test_tables_write_expected_columns`, `test_figures_produce_five_pairs`, `test_family_by_year_marks_small_cells_hollow` (inspect the artists' `fillstyle`).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (Agg backend; colours `minilm '#246b8e'`, `e5 '#a94c33'`). **Step 4: Run** → PASS. **Step 5: Run `python -m cmri_replicate report`**, open the five PNGs with the Read tool and check axes and labels. **Step 6: Commit** `feat: confirmation report, tables and figures` including `results/report/`.

---

### Task 11: Report, claims, verification and umbrella records

**Files:**
- Create: `papers/wildchat-replication/REPORT.md`, `CLAIMS.md`, `VERIFICATION.md`
- Modify: `cmri/docs/research/DECISIONS.md` (new top assessment), `cmri/docs/research/registry.yaml` (`wildchat_replication_{report,claims,verification,gate}` records), `cmri/CLAUDE.md` pilot table (add `papers/wildchat-replication/` row), `cmri/docs/research/README.md` (one bullet under "What the October 2026 pilots established"), `cmri/docs/superpowers/specs/2026-10-18-iscs-2027-paper-design.md` (add F6 line with the outcome)

- [ ] **Step 1: Write `REPORT.md`**: question; source and audit; exclusions; samples (counts drawn, retained, dropped by reason); primary table with four rows (PRISM development, PRISM reserve, WildChat development, WildChat confirmation) per encoder; pass or fail against the frozen rule; secondary panel beside the PRISM values; heterogeneity (family, year, length, distribution); limitations (hashed IPs are not participants; no prompt-hash linkage; families are prefixes of model strings; development data fitted the centers; same encoders and visibility as PRISM so instrument effects are shared; the freeze is self-administered with push times cited; anything else found). Every number copied from `results/`.
- [ ] **Step 2: Write `CLAIMS.md`** as a ledger (claim, evidence file, status).
- [ ] **Step 3: Write `VERIFICATION.md`**: test count and branch coverage (`pytest --cov=cmri_replicate --cov-branch`), compare output, hashes of `results/*.json`, interpreter and package pins, GitHub push timestamps for the freeze and results commits.
- [ ] **Step 4: Run the full suite one final time** → all PASS; record counts.
- [ ] **Step 5: Commit and push** the child repo. In the umbrella: DECISIONS assessment (outcome stated as pass or fail, consequence marked as the assistant's recommendation pending the owner), registry records, CLAUDE.md row, README bullet, design F6 line; commit `Record WildChat replication outcome`; push.

## Self-review notes

- Spec coverage: audit (Task 3, 8), exclusions (Task 1), eligibility and samples (Task 3), adapter and text checks (Task 4), identical primary and panel (Task 5 via pinned `cmri_confirm.analysis`), centers on development (Task 5), heterogeneity by family and year (Task 6), freeze repairs 1 to 4 (Task 7 frozen sets and count check; Task 8 ordering), outcome mapping and reporting (Tasks 10, 11), umbrella records (Task 11).
- Type consistency: `ExternalConversation.sample` and record `sample` are the strings `'development' | 'confirmation'`; `split` on records is `'fit' | 'reserve'` and is what `Case.split` carries; `labels` in Task 6 are keyed by conversation id (`Case.conversation`).
- Review Focus items 1 and 5 are tested in Task 7, item 2 in Task 4, items 3 and 4 in Task 3.
- Proportion: the plan names signatures and tests; bodies are left to the implementer except where the upstream function to copy is named.
