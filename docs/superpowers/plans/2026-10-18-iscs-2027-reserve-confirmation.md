# ISCS 2027 reserve confirmation (F1) and heterogeneity (F2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confirm the history-pooling gain (uniform mean minus last-two, next-user target) on the never-opened 248-component PRISM reserve under a protocol frozen before any reserve text is read, and describe how that gain is distributed across components, conditions and lengths.

**Architecture:** A new child repo `papers/iscs-2027/` (GitHub `queelius/cmri-iscs-2027`, private) with a small package `cmri_confirm` that imports the frozen pilots read-only (`cmri_wave` from portfolio-wave1, `cmri_history` from history-falsification) by pinned commit and file hash. The reserve gets its own selection, visibility and case builder because every upstream entry point rejects reserve rows by construction; the new code is guarded the opposite way (it rejects anything that is *not* reserve). Scoring, pooling recipes, component bootstrap and the role-interaction estimand are reused unchanged. Stages run through an immutable-output runner; the protocol and config are hash-frozen before the `prepare` stage can execute.

**Tech Stack:** Python 3.11 (`papers/selective-forgetting/.venv/bin/python`), numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, onnxruntime 1.23.2, tokenizers 0.22.2, matplotlib 3.10.7, pytest. Encoders: the pinned ONNX MiniLM-L6-v2 and E5-small-v2 in `papers/semantic-measurement/cache/models/`.

**Spec:** `docs/superpowers/specs/2026-10-18-iscs-2027-paper-design.md` (sections "F1" and "F2").

## Global Constraints

- Public data only: PRISM `conversations.jsonl` at HF revision `18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6`, sha256 `55ec67a194ca59d567b57acaeab813cb30abe801e73f9e495597a77ab674d638`, already at `papers/portfolio-wave1/cache/source/conversations.jsonl`.
- The reserve is the 248 components whose manifest `split == 'reserve'` in `papers/portfolio-wave1/cache/encoded/source-manifest.json`. Development components (the 360 selected plus the unselected "development" rows) never enter this repo's encoding or cases.
- Encoders fixed: `minilm` (batch 32) and `e5` (batch 1; the dynamic-quantized E5 leaks across batched inputs, see wave1 PROTOCOL amendment). Both 384-dim. `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`, `threads: 4`.
- Visibility identical to development: first 128 MiniLM content tokens per message (`message_tokens: 128`), `max_messages: 12`, `first_target: 4`, same `_truncate` from `cmri_validation.encoding`.
- Primary contrast, frozen: `Recipe(MEAN)` minus `Recipe(WINDOW, 2)`, next-**user** target, `native128` view, `natural` regime, raw cosine, component bootstrap 2,000 draws seed 104729. Success: `ci_low > 0` in **both** encoders. Practical floor `.01` is reported, not gating. Pre-registered development values: +.03440 MiniLM, +.01919 E5.
- Upstream pilots are never modified. Their commits are pinned: wave1 `e84846590e228fc3457bfd7e8d411118fb6035ae`; history-falsification HEAD at plan time (record in `audits/dependencies.json` at Task 1).
- JSON artifacts carry ids, counts and hashes only; never conversation text. `cache/` is gitignored.
- No em-dashes in prose files (repo voice hook).

## Review Focus

1. A reserve component with zero eligible conversations (15 of 248) must be skipped and counted, not raise. Test in Task 3.
2. A 6-message conversation yields exactly one user target (k=4) and one assistant target (k=5); a 12-message one yields four of each; nothing past index 11 is used. Test in Task 5.
3. `prepare` run twice must refuse to overwrite; `analyze` must refuse if `FREEZE.json` hashes no longer match `PROTOCOL.md`/`config.json`. Tests in Task 7.
4. A component whose conversations have different `condition` values must still be bootstrapped as one component in the condition subgroup summaries (no conversation double-counted across groups). Test in Task 8.
5. If the E5 batch size in `config.json` is not 1, preparation must refuse before loading any model. Test in Task 4.

---

### Task 1: Scaffold `papers/iscs-2027` with pins and protocol

**Files:**
- Create: `papers/iscs-2027/pyproject.toml`, `.gitignore`, `README.md`, `config.json`, `PROTOCOL.md`, `audits/dependencies.json`, `src/cmri_confirm/__init__.py`, `tests/conftest.py`
- Test: `papers/iscs-2027/tests/test_pins.py`

**Interfaces:**
- Produces: `config.json` keys `source` (copied verbatim from wave1 `config.json`), `message_tokens: 128`, `max_messages: 12`, `first_target: 4`, `max_conversations_per_component: 4`, `conversation_salt: "cmri-wave1-conversation-v1"`, `encoder_batch_sizes: {"minilm": 32, "e5": 1}`, `threads: 4`, `bootstrap_replicates: 2000`, `bootstrap_seed: 104729`, `practical_cosine_gain: 0.01`, `budget_windows: [1, 2, 3, 4, 6, 8, 10]`, `max_prepare_wall_seconds: 3600`.
- Produces: `audits/dependencies.json` with `wave_commit`, `history_commit`, `workspace_relative_files` (sha256 of: wave1 `config.json`, `audits/dependencies.json`, `cache/encoded/source-manifest.json`, `src/cmri_wave/{data,prepare,inference,representations,modeling}.py`; history-falsification `src/cmri_history/analysis.py`, `results/public-minilm.json`, `results/public-e5.json`; semantic-measurement `src/cmri_validation/{encoding,data}.py`, `audits/{minilm,e5}-model-manifest.json`).

- [ ] **Step 1: Write `tests/test_pins.py`**

```python
def test_dependency_pins_match_workspace():
    pins = json.loads((ROOT/'audits/dependencies.json').read_text())
    assert len(pins['workspace_relative_files']) >= 14
    for rel, digest in pins['workspace_relative_files'].items():
        assert sha256(WORKSPACE/rel) == digest, rel
    assert git_head(WORKSPACE/'papers/portfolio-wave1') == pins['wave_commit'] == 'e84846590e228fc3457bfd7e8d411118fb6035ae'
    assert git_head(WORKSPACE/'papers/history-falsification') == pins['history_commit']

def test_config_matches_development_visibility():
    cfg = json.loads((ROOT/'config.json').read_text()); wave = json.loads((WORKSPACE/'papers/portfolio-wave1/config.json').read_text())
    assert cfg['source'] == wave['source']
    assert (cfg['message_tokens'], cfg['max_messages'], cfg['first_target']) == (128, 12, 4)
    assert cfg['encoder_batch_sizes'] == {'minilm': 32, 'e5': 1}
```

`conftest.py` defines `ROOT = Path(__file__).resolve().parents[1]`, `WORKSPACE = ROOT.parents[1]`, `sha256(path)`, `git_head(path)`.

- [ ] **Step 2: Run to verify it fails**: `cd papers/iscs-2027 && ../selective-forgetting/.venv/bin/python -m pytest tests/test_pins.py -q` → FAIL (files missing).

- [ ] **Step 3: Create the scaffold.** `pyproject.toml` copies history-falsification's dependency pins with `name = "cmri-iscs-2027"`, `pythonpath = ["src", "../portfolio-wave1/src", "../history-falsification/src"]`. `.gitignore`: `cache/`, `results/replay/`, `*.log`, `__pycache__/`, `.venv/`, `htmlcov/`, `.coverage*`. Write `audits/dependencies.json` with a one-off script (`scripts/pin_dependencies.py`, kept) that hashes the listed files and records both commits. `config.json` as in Interfaces. `README.md`: three paragraphs (purpose, boundary, how to run; commands from Task 7).

- [ ] **Step 4: Write `PROTOCOL.md`** containing verbatim: the primary contrast, success rule, pre-registered development values, the declared secondary panel (role interaction natural vs common prefix; budget curve over `budget_windows` plus full mean; `own_mean − partner_mean`; `own_mean − own_window_2`), the heterogeneity analyses (per-component gain distribution; subgroups by `condition` and by message count 6/8/10/12), the sample rule (eligible reserve rows, 4 per component by `conversation_salt` ranking, expected 775 conversations / 233 components computed from manifest flags only), and the statement that no reserve text is read before `FREEZE.json` exists.

- [ ] **Step 5: Run tests** → `tests/test_pins.py` PASS.

- [ ] **Step 6: Commit**: `git init` on `main`, `git add -A`, `git commit -m "Scaffold reserve confirmation with pinned upstream dependencies and protocol"`. Create `gh repo create queelius/cmri-iscs-2027 --private`, add remote, push.

---

### Task 2: Boundary module

**Files:**
- Create: `src/cmri_confirm/boundary.py`
- Test: `tests/test_boundary.py`

**Interfaces:**
- Produces: `load_config(root: Path) -> dict` (verifies `FREEZE.json` file hashes when the file exists; returns config). `verify_dependencies(root: Path) -> dict` (checks `audits/dependencies.json` hashes and both commits; returns pins). `load_manifest(root: Path) -> dict` (wave1 source manifest, hash-verified). `reserve_components(manifest: dict) -> frozenset[str]` (exactly the ids with `split == 'reserve'`). `development_ids(manifest: dict) -> frozenset[str]` (conversation ids with `split != 'reserve'`). `assert_reserve_only(records: Sequence[dict], manifest: dict) -> None` (raises `ValueError` if any record's component is not in `reserve_components` or any id is in `development_ids`).

- [ ] **Step 1: Write failing tests**

```python
def test_reserve_components_count_is_248(manifest):
    assert len(boundary.reserve_components(manifest)) == 248

def test_assert_reserve_only_rejects_development_record(manifest):
    dev = next(r for r in manifest['conversations'] if r['selected'])
    with pytest.raises(ValueError, match='development'):
        boundary.assert_reserve_only([{'id': dev['id'], 'component': dev['component']}], manifest)

def test_assert_reserve_only_rejects_unknown_component(manifest):
    with pytest.raises(ValueError, match='reserve'):
        boundary.assert_reserve_only([{'id': 'x', 'component': 'nope'}], manifest)

def test_load_config_refuses_changed_frozen_file(tmp_root):   # tmp copy with FREEZE.json whose PROTOCOL.md hash is wrong
    with pytest.raises(ValueError, match='frozen'):
        boundary.load_config(tmp_root)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (reuse `digest`/`verify_paths` style from `cmri_history.boundary`; do not import them, they resolve paths against that repo). **Step 4: Run** → PASS. **Step 5: Commit** `feat: boundary pins and reserve-only guard`.

---

### Task 3: Reserve selection

**Files:**
- Create: `src/cmri_confirm/reserve.py`
- Test: `tests/test_reserve.py`

**Interfaces:**
- Consumes: `cmri_wave.data._parse(row, max_messages, minimum_messages) -> dict`, `cmri_wave.data._digest`, `cmri_wave.data.Role`.
- Produces: `@dataclass(frozen=True) ReserveConversation(id: str, participant: str, component: str, condition: str, messages: tuple[Message, ...])` where `Message` is `cmri_wave.data.Message`; `__post_init__` enforces alternation and ≥6 messages exactly as `Conversation` does. `select_reserve(source_path: Path, manifest: dict, config: dict) -> tuple[list[ReserveConversation], dict]`: parses only rows whose manifest split is reserve, keeps `not reasons`, ranks within component by `_digest({'domain': config['conversation_salt'], 'conversation': id})` then id, caps at `max_conversations_per_component`; returns conversations sorted by (component, id) and a `selection` dict with `components_total`, `components_with_eligible`, `conversations_eligible`, `conversations_selected`, `by_condition`, `by_message_count`.

- [ ] **Step 1: Write failing tests** (unit tests use a synthetic 3-row JSONL written to `tmp_path` and a matching mini-manifest; one integration test is marked `@pytest.mark.integration` and uses the real source)

```python
def test_select_reserve_skips_components_without_eligible_rows(tmp_source, mini_manifest, config):
    convs, sel = reserve.select_reserve(tmp_source, mini_manifest, config)
    assert sel['components_total'] == 2 and sel['components_with_eligible'] == 1 and len(convs) == 1

def test_select_reserve_caps_four_per_component_by_salted_rank(tmp_source_six, mini_manifest_one, config):
    convs, _ = reserve.select_reserve(tmp_source_six, mini_manifest_one, config)
    assert len(convs) == 4 and [c.id for c in convs] == expected_rank_order[:4]

def test_select_reserve_never_returns_development_rows(tmp_source_mixed, mini_manifest_mixed, config):
    convs, _ = reserve.select_reserve(...)
    assert all(mini_manifest_mixed_split[c.id] == 'reserve' for c in convs)

@pytest.mark.integration
def test_real_reserve_selection_counts(real_source, real_manifest, config):
    convs, sel = reserve.select_reserve(real_source, real_manifest, config)
    assert (sel['components_with_eligible'], sel['conversations_eligible'], len(convs)) == (233, 963, 775)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** Reading the source: decode the whole JSONL (same as `prepare_source`), verify bytes/sha against `config['source']`, then `_parse` **only** rows whose id is reserve per manifest; never construct `Conversation`. **Step 4: Run** → PASS (integration test included). **Step 5: Commit** `feat: reserve selection mirroring the development cap and ranking`.

---

### Task 4: Reserve visibility and encoding

**Files:**
- Create: `src/cmri_confirm/encode.py`
- Test: `tests/test_encode.py`

**Interfaces:**
- Consumes: `cmri_wave.prepare._load_encoding_module(workspace, pins)`, `_ENCODERS`, `_check_source`; from the loaded module `OnnxEncoder(model_dir, manifest, prefix=, threads=, batch_size=, cache_path=)` with `.tokenizer`, `.preflight(texts)`, `.encode(texts) -> (n,384)`, `.fingerprint`, `.stats`; `module.budget_tokenizer(tokenizer)`, `module._truncate(text, max_tokens, tokenizer) -> (text, count)`.
- Produces: `reserve_visibility(conversation: ReserveConversation, tokenizer, config: dict, truncate) -> tuple[tuple[str,...], tuple[int,...], tuple[int,...]]` (primary spans, retained masses, original token counts; raises `ValueError` on any zero-mass span or >12 messages). `encode_reserve(conversations: list[ReserveConversation], config: dict, workspace: Path, output: Path) -> dict` writing `output/{minilm,e5}.npz` with keys `f'{id}__messages'` only, `output/{minilm,e5}.json` with `records` (`id, component, participant, split: 'reserve', condition, roles, masses, original_tokens, model`), `fingerprint`, `npz_sha256`, `npz_bytes`, `stats`, `config_sha256`, and `output/preparation.json`. Refuses if `encoder_batch_sizes['e5'] != 1` before loading any model.

- [ ] **Step 1: Write failing tests**

```python
def test_reserve_visibility_truncates_to_128_and_keeps_original_counts(fake_tokenizer):
    spans, masses, original = encode.reserve_visibility(conv_with_200_token_message, fake_tokenizer, config, fake_truncate)
    assert masses[0] == 128 and original[0] == 200 and len(spans) == len(conv.messages)

def test_encode_refuses_batched_e5_before_model_load(monkeypatch, tmp_path):
    bad = {**config, 'encoder_batch_sizes': {'minilm': 32, 'e5': 8}}
    monkeypatch.setattr(encode, '_load_encoding_module', lambda *a, **k: pytest.fail('model loaded'))
    with pytest.raises(ValueError, match='singleton'):
        encode.encode_reserve([conv], bad, WORKSPACE, tmp_path)

def test_encode_records_carry_no_text(stub_encoder, tmp_path):   # stub returns unit vectors
    result = encode.encode_reserve([conv], config, WORKSPACE, tmp_path)
    meta = json.loads((tmp_path/'minilm.json').read_text())
    assert set(meta['records'][0]) == {'id','component','participant','split','condition','roles','masses','original_tokens','model'}
    assert meta['records'][0]['split'] == 'reserve'
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** Mirror `prepare_encoded`'s encoder construction (same model dirs/manifests under `papers/semantic-measurement`), sqlite cache under `output/cache/`. **Step 4: Run** → PASS. **Step 5: Commit** `feat: reserve-only visibility and pinned encoding`.

---

### Task 5: Reserve cases

**Files:**
- Create: `src/cmri_confirm/cases.py`
- Test: `tests/test_cases.py`

**Interfaces:**
- Consumes: `cmri_history.analysis.Case`, `Regime`, `View`, `cmri_wave.data.Role`, `Split`.
- Produces: `build_reserve_cases(records: Sequence[dict], arrays: Mapping[str, np.ndarray], regime: Regime = Regime.NATURAL) -> list[Case]` with `split=Split.RESERVE`, `view=View.NATIVE`, same target/prefix rule as `cmri_history.analysis.build_cases` (`for k in range(4, len(x)-1, 2): for j in (k, k+1): stop = j if natural else k`). Raises if any record's split is not `'reserve'`. `load_reserve(root: Path, encoder: str) -> tuple[list[dict], dict[str, np.ndarray]]` reading `cache/reserve/{encoder}.{json,npz}` and calling `boundary.assert_reserve_only`.

- [ ] **Step 1: Write failing tests** (reuse the `fixture()` shape from history-falsification's `tests/test_analysis.py` with `split='reserve'`)

```python
def test_six_message_conversation_yields_one_target_per_role():
    r, x = fixture(messages=6)
    cases = build_reserve_cases([r], x)
    assert [(c.target_index, c.role.value) for c in cases] == [(4,'user'), (5,'assistant')]

def test_twelve_message_conversation_yields_four_targets_per_role_and_nothing_beyond_index_11():
    r, x = fixture(messages=12)
    cases = build_reserve_cases([r], x)
    assert sorted(c.target_index for c in cases) == [4,5,6,7,8,9,10,11]

def test_common_prefix_withholds_current_user_request():
    r, x = fixture(messages=8); cases = build_reserve_cases([r], x, Regime.COMMON)
    user, asst = cases[0], cases[1]
    assert asst.prefix_end == user.prefix_end == 4 and np.array_equal(user.history, asst.history)

def test_rejects_non_reserve_record():
    r, x = fixture(messages=6); r['split'] = 'probe'
    with pytest.raises(ValueError, match='reserve'): build_reserve_cases([r], x)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Commit** `feat: reserve forecast cases under natural and common-prefix origins`.

---

### Task 6: Confirmation analysis

**Files:**
- Create: `src/cmri_confirm/analysis.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: `cmri_history.analysis.{Recipe, PoolKind, pool_case, summarize, role_interaction}`, `cmri_wave.modeling.score(predictions, targets, center) -> {'cosine','centered_cosine','mse'}`, development centers from `history-falsification/results/public-{encoder}.json['fit']['center_by_role']`.
- Produces: `development_centers(workspace: Path, encoder: str) -> dict[Role, np.ndarray]`. `primary(cases: list[Case], config: dict, center: np.ndarray) -> dict` = `summarize(score(mean), score(window_2))` for the `'cosine'` metric plus `'centered_cosine'`, keys `gain, ci_low, ci_high, components, conversations, cases, component_ids, component_differences, passed` (`ci_low > 0`), `practical` (`gain >= .01`). `budget_curve(cases, config, center) -> list[dict]` one row per `n in budget_windows + ['all']` with `window: n, cosine_mean, ci_low, ci_high, gain_vs_window2` (summarize vs window_2). `role_panel(user_natural, assistant_natural, user_common, assistant_common, config, centers) -> dict` with `interaction_natural`, `interaction_common`, `attenuation` (summarize natural minus common per paired key) and `own_mean_minus_partner_mean`, `own_mean_minus_own_window2` for user cases. `analyze(root: Path, encoder: str, config: dict) -> dict` assembling `{'encoder', 'counts', 'primary', 'budget', 'roles', 'per_case': {'new': [...], 'baseline': [...], 'keys': [...]}}`; the `per_case` arrays let Task 8 compute subgroups in its own stage without re-pooling.

- [ ] **Step 1: Write failing tests** (deterministic 2-D fixtures where mean and window_2 pools differ by construction)

```python
def test_primary_reports_component_bootstrap_and_pass_flag():
    cases = cases_with_known_gain(gain=.05, components=6)
    r = analysis.primary(cases, config, center=np.zeros(2))
    assert r['gain'] == pytest.approx(.05, abs=1e-9) and r['passed'] is True and r['components'] == 6

def test_primary_fails_when_interval_crosses_zero():
    r = analysis.primary(cases_with_known_gain(gain=0.0, noise=.1, components=6), config, np.zeros(2))
    assert r['passed'] is False

def test_budget_curve_last_row_equals_mean_recipe():
    rows = analysis.budget_curve(cases, config, np.zeros(2))
    assert rows[-1]['window'] == 'all' and rows[-1]['cosine_mean'] == pytest.approx(mean_recipe_cosine(cases))

def test_role_panel_pairs_user_and_assistant_by_round():
    panel = analysis.role_panel(u_nat, a_nat, u_com, a_com, config, centers)
    assert panel['interaction_natural']['cases'] == len(u_nat) == len(a_nat)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** Pool with `pool_case(c, Recipe(PoolKind.MEAN))` etc.; `summarize(new, baseline, cases, config['bootstrap_replicates'], config['bootstrap_seed'])`. **Step 4: Run** → PASS. **Step 5: Commit** `feat: primary contrast, budget curve and role panel on reserve cases`.

---

### Task 7: Runner with freeze and immutable outputs

**Files:**
- Create: `src/cmri_confirm/runner.py`, `src/cmri_confirm/__main__.py`, `FREEZE.json` (generated by `python -m cmri_confirm freeze`)
- Test: `tests/test_runner.py`

**Interfaces:**
- Produces: CLI `python -m cmri_confirm <stage>` with stages `freeze`, `prepare`, `analyze-minilm`, `analyze-e5`, `report`. `freeze` writes `FREEZE.json` = `{version: 1, freeze_commit: <git HEAD>, files: {PROTOCOL.md: sha, config.json: sha, audits/dependencies.json: sha}}` and refuses if it already exists. Every other stage calls `boundary.load_config` (which verifies FREEZE) and `boundary.verify_dependencies` first; `prepare` additionally refuses if `FREEZE.json` is missing. `prepare` runs `select_reserve` then `encode_reserve` into `cache/reserve/`, writes `results/prepare.json` (selection counts, encoder stats, runtime, pins; no text). `analyze-*` writes `results/public-{encoder}.json`. Each stage refuses an existing output file (`FileExistsError`). Each result embeds `provenance: {this_commit, wave_commit, history_commit, config_sha256, protocol_sha256, started, runtime_seconds}`.

- [ ] **Step 1: Write failing tests**

```python
def test_prepare_refuses_without_freeze(tmp_repo):           # tmp copy of repo without FREEZE.json
    with pytest.raises(ValueError, match='FREEZE'): runner.execute(tmp_repo, 'prepare')

def test_stage_refuses_existing_output(tmp_repo_frozen, stub_stage):
    runner.execute(tmp_repo_frozen, 'analyze-minilm'); 
    with pytest.raises(FileExistsError): runner.execute(tmp_repo_frozen, 'analyze-minilm')

def test_analyze_refuses_when_protocol_edited_after_freeze(tmp_repo_frozen):
    (tmp_repo_frozen/'PROTOCOL.md').write_text('changed')
    with pytest.raises(ValueError, match='frozen'): runner.execute(tmp_repo_frozen, 'analyze-minilm')

def test_freeze_refuses_second_time(tmp_repo_frozen):
    with pytest.raises(FileExistsError): runner.execute(tmp_repo_frozen, 'freeze')
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (stage functions receive `root`, read config, dispatch; stub-able via module-level functions for tests). **Step 4: Run** → PASS. **Step 5: Commit** `feat: staged runner with protocol freeze and immutable results`.

- [ ] **Step 6: Freeze and run the reserve.** From `papers/iscs-2027`:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:../portfolio-wave1/src:../history-falsification/src
PY=../selective-forgetting/.venv/bin/python
$PY -m pytest -q                                   # all green before freezing
$PY -m cmri_confirm freeze && git add FREEZE.json && git commit -m "Freeze reserve confirmation protocol"
$PY -m cmri_confirm prepare                        # ~15 min; 775 conversations encoded twice
$PY -m cmri_confirm analyze-minilm && $PY -m cmri_confirm analyze-e5
```

Expected: `results/prepare.json` shows `conversations_selected: 775`, `components_with_eligible: 233`; both `results/public-*.json` exist. Commit `results/` with `git commit -m "Record reserve confirmation results"`. Do not interpret yet; Task 9 reports.

---

### Task 8: Heterogeneity (F2)

**Files:**
- Create: `src/cmri_confirm/heterogeneity.py`; Modify: `src/cmri_confirm/runner.py` (new stage `analyze-subgroups` writing `results/subgroups-{encoder}.json` from `results/public-{encoder}.json['per_case']` and the reserve cases)
- Test: `tests/test_heterogeneity.py`

**Interfaces:**
- Consumes: `summarize`, the primary's per-case `new`/`baseline` cosine arrays and `cases`.
- Produces: `subgroups(cases, new, baseline, config, lengths: dict[str, int]) -> dict` with keys `condition` (one `summarize` result per distinct `case.condition`, cases masked by condition; components may appear in several conditions because grouping is by component within each masked subset) and `message_count` (per 6/8/10/12, derived from `len(case.roles)` of the conversation's full record, passed via a `lengths: dict[str,int]` argument). `distribution(component_differences: Sequence[float]) -> dict` with `n, mean, median, q10, q90, fraction_positive, skew` (scipy.stats.skew, bias=False). `pooled_with_development(workspace, encoder, reserve_primary) -> dict` reading the development `mean_minus_window2.cosine.component_differences` from the pinned `history-falsification/results/public-{encoder}.json` (native128, natural, user) and returning `{'development': distribution(dev), 'reserve': distribution(res), 'pooled': distribution(dev+res), 'development_ids_disjoint': bool}`.

- [ ] **Step 1: Write failing tests**

```python
def test_condition_subgroups_keep_component_bootstrap_unit():
    cases = mixed_condition_component()        # one component, two conversations, conditions A and B
    s = heterogeneity.subgroups(cases, new, baseline, config, lengths)
    assert s['condition']['A']['components'] == 1 and s['condition']['B']['components'] == 1
    assert s['condition']['A']['conversations'] + s['condition']['B']['conversations'] == 2

def test_distribution_fraction_positive():
    d = heterogeneity.distribution([.1, .2, -.05, .0])
    assert d['n'] == 4 and d['fraction_positive'] == .5

def test_pooled_requires_disjoint_component_ids():
    r = heterogeneity.pooled_with_development(WORKSPACE, 'minilm', reserve_primary_fixture)
    assert r['development_ids_disjoint'] is True
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5: Re-run analysis stages into `results/replay/`** (`--output results/replay`) and confirm `primary` is byte-identical to the Task 7 results via a `compare` subcommand (add to runner: `python -m cmri_confirm compare results results/replay` → exits 0 only if every non-provenance field matches). Because `analyze` outputs are immutable, the subgroup fields are added by a new stage `analyze-subgroups` writing `results/subgroups-{encoder}.json`. **Step 6: Commit** `feat: heterogeneity subgroups and development/reserve pooling`.

---

### Task 9: Reporting and figures

**Files:**
- Create: `src/cmri_confirm/reporting.py`, `results/report/` (generated)
- Test: `tests/test_reporting.py`

**Interfaces:**
- Produces: `gate(results: dict[str, dict]) -> dict` (`passed` iff both encoders' `primary.passed`; `cells` per encoder with gain/ci/practical). `tables(results, subgroups, out: Path)` writing `primary.csv` (encoder, sample ∈ {development, reserve}, gain, ci_low, ci_high, components, conversations), `budget.csv`, `subgroups.csv`, `distribution.csv`. `figures(results, subgroups, out)` writing four PNG+PDF pairs: `primary-dev-vs-reserve`, `component-gain-distribution` (ECDF, development vs reserve, both encoders), `budget-curve`, `role-origin-reserve` (same layout as history-falsification's information-boundary figure). `report` stage writes `results/report/gate.json` and `results/report/README.md` with the numbers filled from results (no interpretation beyond pass/fail).

- [ ] **Step 1: Write failing tests**

```python
def test_gate_requires_both_encoders():
    assert reporting.gate({'minilm': ok, 'e5': ok})['passed'] and not reporting.gate({'minilm': ok, 'e5': bad})['passed']

def test_tables_write_expected_columns(tmp_path):
    reporting.tables(fake_results, fake_subgroups, tmp_path)
    assert next(csv.reader(open(tmp_path/'primary.csv'))) == ['encoder','sample','gain','ci_low','ci_high','components','conversations']

def test_figures_produce_four_pairs(tmp_path):
    reporting.figures(fake_results, fake_subgroups, tmp_path)
    assert sorted(p.name for p in tmp_path.glob('*.png')) == ['budget-curve.png','component-gain-distribution.png','primary-dev-vs-reserve.png','role-origin-reserve.png']
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (matplotlib Agg; colours `minilm '#246b8e'`, `e5 '#a94c33'` as upstream). **Step 4: Run** → PASS. **Step 5: Run `python -m cmri_confirm report`**, open the four PNGs with the Read tool and check axes/labels. **Step 6: Commit** `feat: confirmation report, tables and figures` including `results/report/`.

---

### Task 10: Report, claims and umbrella links

**Files:**
- Create: `papers/iscs-2027/REPORT.md`, `papers/iscs-2027/CLAIMS.md`, `papers/iscs-2027/VERIFICATION.md`
- Modify: `cmri/docs/research/DECISIONS.md` (new dated row), `cmri/docs/research/registry.yaml` (`reserve_confirmation_*` records), `cmri/CLAUDE.md` pilot table (add `papers/iscs-2027/` row), `cmri/docs/research/README.md` (pilot section: one sentence with the reserve numbers)

- [ ] **Step 1: Write `REPORT.md`**: question, sample (233 components / 775 conversations; eligibility counts), primary table (development vs reserve, both encoders), pass/fail against the frozen rule, secondary panel, heterogeneity, limitations (exposed development used for centers and pre-registration only; PRISM capped at 12 messages; proxies not verified individuals), runtime. Every number copied from `results/`.
- [ ] **Step 2: Write `CLAIMS.md`** as a ledger: claim, evidence file, status (confirmed / not confirmed / descriptive).
- [ ] **Step 3: Write `VERIFICATION.md`**: test count and coverage (`pytest --cov=cmri_confirm`), replay comparison result, hashes of `results/*.json`, interpreter and env pins.
- [ ] **Step 4: Run the full suite one final time** → all PASS; record counts in VERIFICATION.md.
- [ ] **Step 5: Commit and push** the child repo. In the umbrella: add the DECISIONS row ("Reserve confirmation executed; primary passed/failed; F2 descriptive"), registry entries, CLAUDE.md row, README sentence; `git commit -m "Record reserve confirmation outcome"`; push.

---

### Task 11 (optional, after Task 10): participant survey join

Only if Task 9's distribution shows meaningful spread. Pin and download PRISM `survey.jsonl` at the same HF revision into `cache/source/` (record url, sha256, bytes in `audits/acquisition-manifest.json`); join `participant → {lm_familiarity, lm_frequency_use, stated_prefs}` to per-conversation gains; report Spearman correlations with component bootstrap. New stage `analyze-survey`; tests mirror Task 8's structure. Not part of the F1 gate.

## Self-review notes

- Spec coverage: F1 → Tasks 1–7, 9, 10; F2 → Tasks 8, 9, 11. "Budget-response curve" and "own vs partner middle" (spec rows 3.4) → Task 6. Pooling with development (spec "development numbers reported alongside") → Task 8.
- The reserve guard appears three times on purpose (selection, encoding records, case building); each is tested.
- Centered-cosine uses development FIT centers from the pinned results file so no reserve-derived center enters the score.
