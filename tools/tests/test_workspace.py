"""Tests for tools/workspace.py against real git repositories in a temporary directory.

Only `gh` is faked. Git runs with an empty global config so the user's settings (signing,
hooks, default branch) cannot leak in.
"""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import workspace as ws  # noqa: E402

FAKE_KEY = 'AI' + 'za' + 'Q' * 35   # assembled at runtime so no literal key sits in the repository


@pytest.fixture(autouse=True)
def isolated_git(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp('home')
    cfg = home / 'gitconfig'
    cfg.write_text('[user]\n\tname = Test\n\temail = test@example.com\n[init]\n\tdefaultBranch = main\n')
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(cfg))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.setenv('HOME', str(home))


def sh(*args, cwd=None):
    return subprocess.run(list(args), cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo: Path, files: dict, message='change'):
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            sh('git', 'rm', '-q', name, cwd=repo)
        else:
            path.write_text(content)
            sh('git', 'add', name, cwd=repo)
    sh('git', 'commit', '-q', '-m', message, cwd=repo)
    return sh('git', 'rev-parse', 'HEAD', cwd=repo)


def make_remote(remotes: Path, name: str, branches=('main',)):
    seed = remotes / f'{name}-seed'
    sh('git', 'init', '-q', '-b', branches[0], str(seed))
    commit(seed, {'README.md': f'{name}\n'}, 'initial')
    for extra in branches[1:]:
        sh('git', 'branch', extra, cwd=seed)
    bare = remotes / f'{name}.git'
    sh('git', 'clone', '-q', '--bare', str(seed), str(bare))
    return bare


def base_manifest(remotes: Path) -> dict:
    return {
        'sources': {'pubs': '../pubs/pubs.jsonl', 'studies': 'papers/agenda/studies.json',
                    'questions': 'papers/agenda/questions.json', 'datasets': 'records/datasets.json'},
        'version': 1, 'owner': 'tester', 'remote': str(remotes) + '/{name}.git',
        'repos': [
            {'name': 'umbrella', 'path': '.', 'branch': 'main', 'visibility': 'public', 'data': 'none',
             'group': 'umbrella', 'note': 'The umbrella.'},
            {'name': 'alpha', 'path': 'papers/alpha', 'branch': 'main', 'visibility': 'private', 'data': 'public',
             'group': 'study', 'note': 'Alpha study.'},
            {'name': 'beta', 'path': 'papers/beta', 'branch': 'dev', 'visibility': 'private',
             'data': 'private-archive', 'group': 'paper', 'note': 'Beta paper.'},
            {'name': 'gamma', 'path': 'papers/gamma', 'branch': 'main', 'visibility': 'private', 'data': 'public',
             'group': 'pilot', 'note': 'Gamma pilot.', 'archived': True},
        ],
        'worktrees': [{'path': 'papers/gamma-side', 'repo': 'gamma', 'branch': 'side'}],
        'snapshots': {'paper': {'description': 'Public study code.', 'publish': True,
                                'pins': {'alpha': '', 'gamma': ''}}},
        'publication_policy': {'forbidden_paths': ['*.npz', '*secret*'],
                               'secret_patterns': {'google_api_key': 'AIza[0-9A-Za-z_\\-]{35}'},
                               'max_blob_bytes': 1048576},
        'views': [{'file': 'CLAUDE.md', 'block': 'pilots', 'groups': ['pilot', 'study'],
                   'columns': ['path', 'repo_branch', 'note'], 'headers': ['Local path', 'Repo (branch)', 'Result']}],
    }


@pytest.fixture
def workspace(tmp_path):
    remotes = tmp_path / 'remotes'
    remotes.mkdir()
    make_remote(remotes, 'alpha')
    make_remote(remotes, 'beta', ('dev',))
    make_remote(remotes, 'gamma', ('main', 'side'))
    root = tmp_path / 'root'
    sh('git', 'init', '-q', '-b', 'main', str(root))
    sh('git', 'remote', 'add', 'origin', str(remotes / 'umbrella.git'), cwd=root)
    (root / 'CLAUDE.md').write_text('# Guide\n\n<!-- workspace:pilots:begin -->\nold\n<!-- workspace:pilots:end -->\n\nTail.\n')
    raw = base_manifest(remotes)
    (root / 'workspace.json').write_text(json.dumps(raw, indent=2) + '\n')
    (root / '.gitignore').write_text('papers/\n')
    dataset_path = write_datasets(root, raw)
    commit(root, {'.gitignore': 'papers/\n', 'CLAUDE.md': (root / 'CLAUDE.md').read_text(),
                  raw['sources']['datasets']: dataset_path.read_text(),
                  'workspace.json': (root / 'workspace.json').read_text()}, 'umbrella')
    return root, remotes, raw


def call(root, *argv):
    out = io.StringIO()
    manifest = ws.load(root / 'workspace.json')
    command, rest = argv[0], argv[1:]
    if command == 'status':
        return ws.cmd_status(manifest, root, out=out), out.getvalue()
    if command == 'clone':
        return ws.cmd_clone(manifest, root, dry_run='--dry-run' in rest, out=out), out.getvalue()
    if command == 'lock':
        return ws.cmd_lock(manifest, root, rest[0], path=root / 'workspace.json', out=out), out.getvalue()
    if command == 'verify':
        return ws.cmd_verify(manifest, root, list(rest), out=out), out.getvalue()
    if command == 'checkout':
        snap = None if rest[0] == '--branches' else rest[0]
        return ws.cmd_checkout(manifest, root, snap, out=out), out.getvalue()
    if command == 'scan':
        return ws.cmd_scan(manifest, root, rest[0], out=out), out.getvalue()
    if command == 'render':
        return ws.cmd_render(manifest, root, check='--check' in rest, out=out), out.getvalue()
    raise AssertionError(command)


# ----------------------------------------------------------------------------- manifest rules

def test_the_real_manifest_is_valid_and_its_tables_are_current():
    manifest = ws.load(ws.MANIFEST)
    assert manifest.repo('cmri-iscs-2027').data == 'public'
    assert all(manifest.repo(n).data == 'public' for n in manifest.snapshots['iscs-2027-paper'].pins)
    out = io.StringIO()
    assert ws.cmd_render(manifest, ws.ROOT, check=True, out=out) == 0, out.getvalue()


@pytest.mark.parametrize('mutate, message', [
    (lambda m: m['snapshots']['paper']['pins'].update({'beta': ''}), 'private-archive'),
    (lambda m: m['repos'][1].update({'group': 'misc'}), 'unknown group'),
    (lambda m: m['repos'][1].update({'data': 'secret'}), 'unknown data class'),
    (lambda m: m['repos'][2].update({'path': 'papers/alpha'}), 'paths must be unique'),
    (lambda m: m['repos'][2].update({'name': 'alpha'}), 'names must be unique'),
    (lambda m: m['repos'][1].update({'path': '../outside'}), 'inside the workspace'),
    (lambda m: m['snapshots']['paper']['pins'].update({'alpha': 'abc123'}), 'full commit id'),
    (lambda m: m['snapshots']['paper']['pins'].update({'delta': ''}), 'unknown repository'),
    (lambda m: m['worktrees'][0].update({'repo': 'delta'}), 'worktree of unknown'),
    (lambda m: m['views'][0].update({'columns': ['path', 'stars', 'note']}), 'unknown view column'),
    (lambda m: m['views'][0].update({'headers': ['only one']}), 'differ in length'),
    (lambda m: m.update({'remote': 'https://example.org/fixed.git'}), '{name}'),
    (lambda m: m['repos'][1].pop('note'), 'lacks note'),
    (lambda m: m['publication_policy'].update({'reviewed': [{'repo': 'delta', 'path': '*', 'reason': 'r',
                                                             'reviewed': '2026-10-21'}]}), 'unknown repository'),
    (lambda m: m['publication_policy'].update({'reviewed': [{'repo': 'alpha', 'path': '*.npz',
                                                             'reviewed': '2026-10-21'}]}), 'lacks reason'),
])
def test_manifest_rules(tmp_path, mutate, message):
    raw = base_manifest(tmp_path)
    mutate(raw)
    path = tmp_path / 'workspace.json'
    path.write_text(json.dumps(raw))
    with pytest.raises(ws.ManifestError, match=message):
        ws.load(path)


def test_main_reports_manifest_errors_with_exit_code_2(tmp_path, capsys):
    raw = base_manifest(tmp_path)
    raw['repos'][1]['group'] = 'misc'
    (tmp_path / 'workspace.json').write_text(json.dumps(raw))
    assert ws.main(['status'], root=tmp_path) == 2
    assert 'unknown group' in capsys.readouterr().err


# ----------------------------------------------------------------------------- local state

def test_clone_dry_run_changes_nothing(workspace):
    root, _, _ = workspace
    code, out = call(root, 'clone', '--dry-run')
    assert code == 0 and 'git clone --branch dev' in out and 'worktree add' in out
    assert not (root / 'papers').exists()


def test_clone_then_status_is_clean_and_idempotent(workspace):
    root, _, _ = workspace
    assert call(root, 'clone')[0] == 0
    assert (root / 'papers/beta/.git').is_dir()
    assert (root / 'papers/gamma-side/.git').is_file()           # a worktree, not a clone
    assert sh('git', 'branch', '--show-current', cwd=root / 'papers/gamma-side') == 'side'
    code, out = call(root, 'status')
    assert code == 0 and out.count('clean') == 5
    assert call(root, 'clone')[1] == ''                           # nothing left to do


def test_status_reports_missing_dirty_unpushed_and_wrong_remote(workspace):
    root, remotes, _ = workspace
    call(root, 'clone')
    alpha = root / 'papers/alpha'
    (alpha / 'scratch.txt').write_text('x')
    commit(root / 'papers/gamma', {'new.txt': 'y'})
    sh('git', 'remote', 'set-url', 'origin', str(remotes / 'elsewhere.git'), cwd=root / 'papers/beta')
    sh('git', 'switch', '-q', '-c', 'feature', cwd=alpha)
    code, out = call(root, 'status')
    assert code == 1                                               # the wrong remote is a problem
    assert '1 uncommitted' in out and '1 unpushed' in out
    assert 'on feature, manifest says main' in out and 'elsewhere.git' in out
    sh('rm', '-rf', str(root / 'papers/beta'))
    assert 'MISSING (beta)' in call(root, 'status')[1]


# ----------------------------------------------------------------------------- snapshots

def test_lock_refuses_dirty_or_unpushed_heads_then_records_pushed_ones(workspace):
    root, _, _ = workspace
    call(root, 'clone')
    alpha = root / 'papers/alpha'
    (alpha / 'wip.txt').write_text('w')
    code, out = call(root, 'lock', 'paper')
    assert code == 1 and 'alpha: 1 uncommitted' in out
    sha = commit(alpha, {'wip.txt': 'w'})
    code, out = call(root, 'lock', 'paper')
    assert code == 1 and 'not on any remote branch' in out
    assert json.loads((root / 'workspace.json').read_text())['snapshots']['paper']['pins']['alpha'] == ''
    sh('git', 'push', '-q', 'origin', 'main', cwd=alpha)
    assert call(root, 'lock', 'paper')[0] == 0
    pins = json.loads((root / 'workspace.json').read_text())['snapshots']['paper']['pins']
    assert pins['alpha'] == sha and len(pins['gamma']) == 40


def test_verify_flags_unpinned_missing_and_unpushed_commits(workspace):
    root, _, raw = workspace
    call(root, 'clone')
    code, out = call(root, 'verify', 'paper')
    assert code == 1 and out.count('not pinned') == 2
    call(root, 'lock', 'paper')
    assert call(root, 'verify')[0] == 0
    local = commit(root / 'papers/alpha', {'local.txt': 'l'})
    data = json.loads((root / 'workspace.json').read_text())
    data['snapshots']['paper']['pins']['alpha'] = local
    data['snapshots']['paper']['pins']['gamma'] = 'f' * 40
    (root / 'workspace.json').write_text(json.dumps(data))
    code, out = call(root, 'verify', 'paper')
    assert code == 1 and 'not on any remote branch' in out and 'commit not found' in out


def test_checkout_detaches_at_pins_refuses_dirty_and_returns_to_branches(workspace):
    root, _, _ = workspace
    call(root, 'clone')
    call(root, 'lock', 'paper')
    pins = json.loads((root / 'workspace.json').read_text())['snapshots']['paper']['pins']
    newer = commit(root / 'papers/alpha', {'later.txt': 'z'})
    sh('git', 'push', '-q', 'origin', 'main', cwd=root / 'papers/alpha')
    assert call(root, 'checkout', 'paper')[0] == 0
    alpha = root / 'papers/alpha'
    assert sh('git', 'rev-parse', 'HEAD', cwd=alpha) == pins['alpha'] != newer
    assert sh('git', 'branch', '--show-current', cwd=alpha) == ''
    (alpha / 'README.md').write_text('edited')
    code, out = call(root, 'checkout', '--branches')
    assert code == 1 and 'refusing' in out
    sh('git', 'checkout', '-q', '--', 'README.md', cwd=alpha)
    assert call(root, 'checkout', '--branches')[0] == 0
    assert sh('git', 'branch', '--show-current', cwd=alpha) == 'main'


def test_checkout_requires_a_locked_snapshot(workspace):
    root, _, _ = workspace
    call(root, 'clone')
    with pytest.raises(ws.ManifestError, match='run lock first'):
        call(root, 'checkout', 'paper')


# ----------------------------------------------------------------------------- publication checks

def test_scan_finds_forbidden_files_and_secrets_in_history_without_printing_them(workspace):
    root, _, _ = workspace
    call(root, 'clone')
    call(root, 'lock', 'paper')
    assert call(root, 'scan', 'paper')[0] == 0
    gamma = root / 'papers/gamma'
    commit(gamma, {'cache/emb.npz': 'vectors', 'notes.txt': f'key={FAKE_KEY}\n'}, 'oops')
    commit(gamma, {'cache/emb.npz': None, 'notes.txt': None}, 'remove')
    sh('git', 'push', '-q', 'origin', 'main', cwd=gamma)
    call(root, 'lock', 'paper')
    code, out = call(root, 'scan', 'paper')
    assert code == 1
    assert 'forbidden path in history: cache/emb.npz' in out
    assert 'secret pattern: notes.txt rule=google_api_key' in out
    assert 'forbidden path at pin' not in out                     # removed before the pin
    assert FAKE_KEY not in out


def test_scan_reports_forbidden_files_at_the_pin_and_oversized_blobs(workspace):
    root, _, raw = workspace
    call(root, 'clone')
    alpha = root / 'papers/alpha'
    commit(alpha, {'my_secret_notes.md': 'nothing', 'big.bin': 'x' * 2048})
    data = json.loads((root / 'workspace.json').read_text())
    data['publication_policy']['max_blob_bytes'] = 1024
    (root / 'workspace.json').write_text(json.dumps(data))
    code, out = call(root, 'scan', 'paper')                       # unpinned: scans HEAD
    assert code == 1 and 'unpinned, scanned HEAD' in out
    assert 'forbidden path at pin: my_secret_notes.md' in out and 'blob too large to scan: big.bin' in out


def test_remote_check_compares_visibility_default_branch_and_archive_flag(workspace, monkeypatch):
    root, _, _ = workspace
    github = {
        'umbrella': {'visibility': 'PUBLIC', 'defaultBranchRef': {'name': 'main'}, 'isArchived': False},
        'alpha': {'visibility': 'PRIVATE', 'defaultBranchRef': {'name': 'main'}, 'isArchived': False},
        'beta': {'visibility': 'PUBLIC', 'defaultBranchRef': {'name': 'main'}, 'isArchived': False},
    }
    real_run = ws.run

    def fake_run(cmd, cwd=None, check=True, input=None, text=True):
        if cmd[0] != 'gh':
            return real_run(cmd, cwd, check, input, text)
        name = cmd[3].split('/')[1]
        if name not in github:
            return subprocess.CompletedProcess(cmd, 1, '', 'not found')
        return subprocess.CompletedProcess(cmd, 0, json.dumps(github[name]), '')

    monkeypatch.setattr(ws, 'run', fake_run)
    out = io.StringIO()
    assert ws.cmd_remote_check(ws.load(root / 'workspace.json'), out=out) == 1
    text = out.getvalue()
    lines = {line.split()[0]: line for line in text.splitlines()}
    assert 'visibility is public, manifest says private' in lines['beta']
    assert 'default branch is main, manifest says dev' in lines['beta']
    assert 'unreachable' in lines['gamma']
    assert lines['umbrella'].endswith('ok') and lines['alpha'].endswith('ok')


# ----------------------------------------------------------------------------- rendering

def test_render_writes_the_table_and_check_detects_drift(workspace):
    root, _, _ = workspace
    code, out = call(root, 'render', '--check')
    assert code == 1 and 'stale: CLAUDE.md' in out
    assert call(root, 'render')[0] == 0
    text = (root / 'CLAUDE.md').read_text()
    assert '| `papers/alpha/` | alpha (`main`) | Alpha study. |' in text
    assert ("gamma (`main`); worktree `papers/gamma-side/` on `side`; archived on GitHub" in text)
    assert 'beta' not in text.split('<!-- workspace:pilots:begin -->')[1]   # papers are not in this view
    assert text.endswith('\n\nTail.\n') and 'old' not in text
    assert call(root, 'render', '--check') == (0, '')


def test_render_requires_markers(workspace):
    root, _, _ = workspace
    (root / 'CLAUDE.md').write_text('# no markers\n')
    with pytest.raises(ws.ManifestError, match='lacks the markers'):
        call(root, 'render')


def test_main_dispatches_commands(workspace, capsys):
    root, _, _ = workspace
    assert ws.main(['clone'], root=root) == 0
    assert ws.main(['status'], root=root) == 0
    assert ws.main(['lock', 'paper'], root=root) == 0
    assert ws.main(['verify'], root=root) == 0
    assert ws.main(['checkout', 'paper', '--dry-run'], root=root) == 0
    assert ws.main(['checkout', '--branches'], root=root) == 0
    assert ws.main(['scan', 'paper'], root=root) == 0
    assert ws.main(['render'], root=root) == 0
    assert ws.main(['render', '--check'], root=root) == 0
    with pytest.raises(SystemExit):
        ws.main(['checkout', 'paper', '--branches'], root=root)
    assert 'not both' in capsys.readouterr().err


def test_reviewed_exceptions_clear_forbidden_paths_but_never_secrets(workspace):
    root, _, _ = workspace
    call(root, 'clone')
    alpha = root / 'papers/alpha'
    commit(alpha, {'results/scores.npz': 'table', 'results/raw.npz': 'x', 'notes.txt': f'{FAKE_KEY}\n'})
    data = json.loads((root / 'workspace.json').read_text())
    data['publication_policy']['reviewed'] = [
        {'repo': 'alpha', 'path': 'results/scores.npz', 'reason': 'score table', 'reviewed': '2026-10-21'},
        {'repo': 'alpha', 'path': 'notes.txt', 'reason': 'cannot exempt a secret', 'reviewed': '2026-10-21'},
        {'repo': 'gamma', 'path': 'results/raw.npz', 'reason': 'other repository', 'reviewed': '2026-10-21'}]
    (root / 'workspace.json').write_text(json.dumps(data))
    code, out = call(root, 'scan', 'paper')
    assert code == 1
    assert 'alpha: 3 finding(s)' in out and '2 path(s) under reviewed exceptions' in out
    assert 'results/scores.npz' not in out                          # reviewed: summarised, not listed
    assert 'forbidden path at pin: results/raw.npz' in out           # exemption belongs to another repo
    assert 'secret pattern: notes.txt' in out and FAKE_KEY not in out


def test_checkout_fetches_a_pin_the_local_clone_lacks_and_status_shows_behind(workspace, tmp_path):
    root, remotes, _ = workspace
    call(root, 'clone')
    other = tmp_path / 'elsewhere'
    sh('git', 'clone', '-q', str(remotes / 'alpha.git'), str(other))
    pushed = commit(other, {'from-elsewhere.txt': 'e'})
    sh('git', 'push', '-q', 'origin', 'main', cwd=other)
    alpha = root / 'papers/alpha'
    assert not ws.has_commit(alpha, pushed)
    data = json.loads((root / 'workspace.json').read_text())
    data['snapshots']['paper']['pins'] = {'alpha': pushed, 'gamma': sh('git', 'rev-parse', 'HEAD', cwd=root / 'papers/gamma')}
    (root / 'workspace.json').write_text(json.dumps(data))
    assert call(root, 'checkout', 'paper')[0] == 0
    assert sh('git', 'rev-parse', 'HEAD', cwd=alpha) == pushed
    call(root, 'checkout', '--branches')
    assert '1 behind' in call(root, 'status')[1]


def test_snapshot_commands_report_missing_checkouts(workspace):
    root, _, _ = workspace
    code, out = call(root, 'lock', 'paper')
    assert code == 1 and 'alpha: missing' in out
    data = json.loads((root / 'workspace.json').read_text())
    data['snapshots']['paper']['pins'] = {'alpha': 'a' * 40, 'gamma': 'b' * 40}
    (root / 'workspace.json').write_text(json.dumps(data))
    code, out = call(root, 'verify', 'paper')
    assert code == 1 and out.count('missing locally') == 2
    code, out = call(root, 'checkout', 'paper')
    assert code == 1 and 'run clone first' in out
    with pytest.raises(ws.ManifestError, match='unknown repository'):
        ws.load(root / 'workspace.json').repo('delta')


# ----------------------------------------------------------------------------- spent samples


def ledger(samples):
    return {'version': 1, 'pools': {'p': 'pool p'}, 'samples': samples}


def sample(id_, spent_on, repo='alpha', record='README.md', excludes=(), siblings=(), evidence=(), **extra):
    return {'id': id_, 'pool': 'p', 'status': extra.pop('status', 'frozen'), 'spent_on': spent_on, 'repo': repo,
            'dataset': 'D-one', 'unit': 'component', 'size': {'components': 2},
            'record': record, 'excludes': list(excludes), 'siblings': list(siblings), 'evidence': list(evidence), **extra}


def write_ledger(root, samples):
    path = root / 'spent.json'
    path.write_text(json.dumps(ledger(samples)) + '\n')
    return path


def spent(root, *samples, verify=True):
    out = io.StringIO()
    code = ws.cmd_spent(ws.load(root / 'workspace.json'), root, verify, write_ledger(root, list(samples)), out=out)
    return code, out.getvalue()


def test_the_real_ledger_is_valid_and_its_chain_is_complete():
    samples = ws.load_ledger(ws.LEDGER)
    assert len(samples) >= 11
    problems = ws.ledger_problems(samples, ws.load(ws.MANIFEST), ws.ROOT)
    assert problems == []


@pytest.mark.parametrize('mutate, message', [
    (lambda s: s[0].update(pool='q'), 'undeclared pool'),
    (lambda s: s[0].update(status='spent'), 'unknown sample status'),
    (lambda s: s[0].update(spent_on='yesterday'), 'ISO date'),
    (lambda s: s[0].update(excludes=['nobody']), 'unknown sample'),
    (lambda s: s[1].update(siblings=['a']), 'siblings must be mutual'),
    (lambda s: s.append(dict(s[0])), 'duplicate sample ids'),
    (lambda s: s[0].update(evidence=[{'repo': 'alpha'}]), 'evidence needs'),
    (lambda s: s[1].update(excludes=[], reuses=['a'], reuse_decision='Alex, 2026-10-28'), 'only an exploratory'),
    (lambda s: s[1].update(excludes=[], reuses=['a'], status='exploratory'), 'reuse_decision'),
    (lambda s: s[1].update(reuses=['a'], status='exploratory', reuse_decision='Alex'), 'both excludes and reuses'),
    (lambda s: s[1].update(excludes=[], reuses=['nobody'], status='exploratory', reuse_decision='Alex'),
     'unknown sample'),
])
def test_ledger_rules(tmp_path, mutate, message):
    samples = [sample('a', '2026-10-01'), sample('b', '2026-10-02', excludes=['a'])]
    mutate(samples)
    path = tmp_path / 'spent.json'
    path.write_text(json.dumps(ledger(samples)))
    with pytest.raises(ws.ManifestError, match=message):
        ws.load_ledger(path)


def test_spent_verify_reports_a_draw_that_misses_an_earlier_one(workspace):
    root, _, _ = workspace
    call(root, 'clone')
    code, text = spent(root, sample('a', '2026-10-01'), sample('b', '2026-10-02'))
    assert code == 1 and 'b: does not exclude the earlier p sample a' in text
    code, text = spent(root, sample('a', '2026-10-01'), sample('b', '2026-10-02', excludes=['a']))
    assert code == 0 and '2 samples, 0 problems' in text


def test_spent_verify_accepts_a_recorded_exploratory_reuse_and_later_draws_must_exclude_it(workspace):
    root, _, _ = workspace
    call(root, 'clone')
    first = sample('screen', '2026-09-30', status='screen')
    reuse = sample('explore', '2026-10-01', status='exploratory', reuses=['screen'],
                   reuse_decision='Alex, 2026-10-01: the screen may be used for exploratory work')
    code, text = spent(root, first, reuse)
    assert code == 0, text
    later = sample('later', '2026-10-02', excludes=['screen'])
    code, text = spent(root, first, reuse, later)
    assert code == 1 and 'later: does not exclude the earlier p sample explore' in text


def test_spent_verify_accepts_same_day_siblings_and_still_needs_earlier_draws(workspace):
    root, _, _ = workspace
    call(root, 'clone')
    first = sample('screen', '2026-09-30', status='screen')
    dev = sample('dev', '2026-10-01', status='development', siblings=['conf'], excludes=['screen'])
    conf = sample('conf', '2026-10-01', siblings=['dev'], excludes=['screen'])
    code, text = spent(root, first, dev, conf)
    assert code == 0, text
    conf['excludes'] = []
    code, text = spent(root, first, dev, conf)
    assert code == 1 and 'conf: does not exclude the earlier p sample screen' in text


def test_spent_verify_checks_record_protocol_and_evidence_files(workspace):
    root, _, _ = workspace
    call(root, 'clone')
    commit(root / 'papers' / 'alpha', {'config.json': '{"spent_samples": ["a"]}\n', 'PROTOCOL.md': 'rules\n'})
    good = sample('b', '2026-10-02', excludes=['a'], protocol='PROTOCOL.md',
                  evidence=[{'repo': 'alpha', 'path': 'config.json', 'contains': '"spent_samples": ["a"]'}])
    code, text = spent(root, sample('a', '2026-10-01'), good)
    assert code == 0, text
    bad = dict(good, record='results/draw.json', freeze='FREEZE.json',
               evidence=[{'repo': 'alpha', 'path': 'config.json', 'contains': '"screened"'},
                         {'repo': 'alpha', 'path': 'missing.json', 'contains': 'x'},
                         {'repo': 'omega', 'path': 'config.json', 'contains': 'x'}])
    code, text = spent(root, sample('a', '2026-10-01'), bad)
    assert code == 1
    for expected in ('record results/draw.json is missing', 'freeze FREEZE.json is missing',
                     "lacks '\"screened\"'", 'missing.json is missing', 'unknown repository omega'):
        assert expected in text, (expected, text)
    code, text = spent(root, sample('a', '2026-10-01', repo='nowhere'))
    assert code == 1 and 'unknown repository nowhere' in text


def test_spent_lists_without_verifying_and_skips_absent_checkouts(workspace):
    root, _, _ = workspace
    code, text = spent(root, sample('a', '2026-10-01', record='nothing.json'), verify=False)
    assert code == 0 and 'a' in text and 'problems' not in text
    code, text = spent(root, sample('a', '2026-10-01', record='nothing.json'))
    assert code == 0 and '0 problems' in text   # alpha is not cloned, so its files are not checked


def test_main_runs_spent(workspace):
    root, _, _ = workspace
    write_ledger(root, [sample('a', '2026-10-01')])
    assert ws.main(['spent'], root=root) == 0
    assert ws.main(['spent', '--verify'], root=root) == 0
    write_ledger(root, [sample('a', '2026-10-01'), sample('b', '2026-10-02')])
    assert ws.main(['spent', '--verify'], root=root) == 1


# ----------------------------------------------------------------------------- program sources


def program_fixture(root):
    raw = base_manifest(root)
    raw['sources'] = {'pubs': 'records/pubs.jsonl', 'studies': 'records/studies.json',
                      'questions': 'records/questions.json', 'datasets': 'records/datasets.json'}
    studies = [{'id': 'S-one', 'title': 'Study one', 'state': 'stopped', 'repo': 'alpha',
                'questions': ['Q1'], 'successor': None, 'path': 'analysis'}]
    questions = [{'id': 'Q1', 'question': 'Does it work?', 'state': 'open', 'studies': ['S-one'],
                  'known': 'A control', 'remaining': 'A replication', 'datasets': ['D-one']}]
    pubs = [{'id': 'paper-one', 'program': 'cognitive-mri', 'title': 'Paper one',
             'repos': {'manuscript': str(root / 'papers/alpha')},
             'derived': {'status': 'under review', 'venue': 'Venue', 'last_event': 'submitted 2026-10-01',
                         'doi': '10.1234/example'},
             'deadline': {'date': '2026-11-01', 'source': 'https://example.org', 'verified': '2026-10-01'},
             'follow_up': {'check_on': '2026-10-20', 'action': 'Check', 'resume_at': 'notes'}}]
    write_program(root, raw, studies, questions, pubs)
    write_datasets(root, raw)
    write_ledger(root, [sample('draw-one', '2026-10-01')])
    return raw, studies, questions, pubs


def write_program(root, raw, studies, questions, pubs):
    (root / 'workspace.json').write_text(json.dumps(raw))
    for kind, payload in [('studies', {'version': 1, 'studies': studies}),
                          ('questions', {'version': 1, 'questions': questions, 'lessons': [
                              {'id': 'L-control', 'text': 'Keep **controls**.\n   Preserve their scope.'}]}), ('pubs', pubs)]:
        path = root / raw['sources'][kind]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(json.dumps(p) for p in payload) + '\n' if kind == 'pubs'
                        else json.dumps(payload))


def load_program(root):
    return ws.load_program(ws.load(root / 'workspace.json'), root)


@pytest.mark.parametrize('kind', ['workspace', 'spent', 'pubs', 'studies', 'questions', 'datasets'])
def test_duplicate_keys_rejected_in_every_file_kind(tmp_path, kind):
    raw, _, _, _ = program_fixture(tmp_path)
    path = tmp_path / (raw['sources'][kind] if kind in raw['sources'] else f'{kind}.json')
    path.write_text('{"outer": {"duplicate": 1, "duplicate": 2}}\n')
    with pytest.raises(ws.ManifestError, match="duplicate.*duplicate"):
        if kind == 'workspace':
            ws.load(path)
        elif kind == 'spent':
            ws.load_ledger(path)
        else:
            load_program(tmp_path)


@pytest.mark.parametrize('sources', [None, [], {}, {'pubs': '/absolute'},
                                    {'pubs': '../pubs/pubs.jsonl', 'studies': 42, 'questions': 'q.json'},
                                    {'pubs': 'by-stage/input.jsonl', 'studies': 's', 'questions': 'q'}])
def test_sources_configuration_is_validated(tmp_path, sources):
    raw, _, _, _ = program_fixture(tmp_path)
    raw['sources'] = sources
    (tmp_path / 'workspace.json').write_text(json.dumps(raw))
    with pytest.raises(ws.ManifestError, match='sources'):
        ws.load(tmp_path / 'workspace.json')


@pytest.mark.parametrize('kind, mutate, message', [
    ('studies', lambda s: s[0].update(state='finished'), 'unknown study state'),
    ('questions', lambda q: q[0].update(state='closed'), 'unknown question state'),
    ('studies', lambda s: s.append(dict(s[0])), 'duplicate study ids'),
    ('questions', lambda q: q.append(dict(q[0])), 'duplicate question ids'),
    ('studies', lambda s: s[0].update(repo='unknown'), 'unknown repository'),
    ('studies', lambda s: s[0].update(questions=['Q99']), 'unknown question'),
    ('questions', lambda q: q[0].update(studies=['S-missing']), 'unknown study'),
    ('studies', lambda s: s[0].update(state='folded'), 'folded.*successor'),
    ('studies', lambda s: s[0].pop('successor'), 'successor'),
    ('studies', lambda s: s[0].update(state='folded', successor='S-missing'), 'unknown successor'),
    ('studies', lambda s: s[0].update(successor='S-one'), 'successor.*null'),
    ('studies', lambda s: s[0].update(questions=[]), 'links must agree'),
    ('questions', lambda q: q[0].update(studies=[]), 'links must agree'),
    ('studies', lambda s: s[0].update(path='../outside'), 'subpath'),
    ('studies', lambda s: s[0].update(path='/outside'), 'subpath'),
    ('studies', lambda s: s[0].update(questions='Q1'), 'list'),
    ('questions', lambda q: q[0].update(known=42), 'known'),
])
def test_program_record_rules(tmp_path, kind, mutate, message):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    mutate(studies if kind == 'studies' else questions)
    write_program(tmp_path, raw, studies, questions, pubs)
    with pytest.raises(ws.ManifestError, match=message):
        load_program(tmp_path)


def test_pubs_join_filter_and_staleness(tmp_path, monkeypatch):
    import os
    raw, studies, questions, pubs = program_fixture(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    pubs[0]['repos']['manuscript'] = '~/papers/alpha/../alpha'
    pubs += [dict(pubs[0], id='unmatched', repos={'manuscript': str(tmp_path / 'elsewhere')}),
             dict(pubs[0], id='other-program', program='other')]
    write_program(tmp_path, raw, studies, questions, pubs)
    yaml = tmp_path / 'records/papers/paper.yaml'
    yaml.parent.mkdir()
    yaml.write_text('not parsed')
    source = tmp_path / raw['sources']['pubs']
    os.utime(source, (100, 100))
    os.utime(yaml, (200, 200))
    program = load_program(tmp_path)
    assert [p.id for p in program.manuscripts] == ['paper-one', 'unmatched']
    assert program.manuscripts[0].repo.name == 'alpha'
    assert program.manuscripts[1].repo is None
    assert program.manuscripts[0].status == 'under review'
    assert any('stale' in warning and 'paper.yaml' in warning for warning in program.warnings)
    assert any('unmatched' in warning for warning in program.warnings)
    os.utime(yaml, (50, 50))
    assert not any('stale' in warning for warning in load_program(tmp_path).warnings)


def test_missing_sources_are_reported_and_do_not_fail(tmp_path):
    raw, _, _, _ = program_fixture(tmp_path)
    for source in raw['sources'].values():
        (tmp_path / source).unlink()
    program = load_program(tmp_path)
    assert not program.manuscripts and not program.studies and not program.questions
    assert sum('absent' in warning for warning in program.warnings) == 4


def test_infrastructure_is_a_repository_role(tmp_path):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    raw['repos'][1]['group'] = 'infrastructure'
    write_program(tmp_path, raw, studies, questions, pubs)
    assert ws.load(tmp_path / 'workspace.json').repo('alpha').group == 'infrastructure'


# ----------------------------------------------------------------------------- board


def board(root, as_of='2026-10-23', write=False):
    out = io.StringIO()
    code = ws.cmd_board(ws.load(root / 'workspace.json'), root, ws.iso_date(as_of, 'as-of'), write, out)
    return code, out.getvalue()


def test_board_dates_order_and_all_disagreements(workspace):
    root, remotes, _ = workspace
    raw, studies, questions, pubs = program_fixture(root)
    raw['remote'] = str(remotes) + '/{name}.git'
    pubs += [dict(pubs[0], id='orphan', title='Unmatched paper', repos={}),
             dict(pubs[0], id='early', title='Early paper', derived=dict(pubs[0]['derived'], status='in preparation')),
             dict(pubs[0], id='custom', title='Custom paper', derived=dict(pubs[0]['derived'], status='custom status')),
             dict(pubs[0], id='published', title='Published paper', derived=dict(pubs[0]['derived'], status='published'))]
    write_program(root, raw, studies, questions, pubs)
    call(root, 'clone')
    commit(root / 'papers/alpha', {'new.txt': 'local commit'})
    (root / 'papers/alpha/new.txt').write_text('uncommitted')
    code, text = board(root)
    assert code == 0
    positions = [text.index(heading) for heading in ('## Manuscripts', '### in preparation', '### under review',
                 '### published', '### custom status', '## Studies', '### stopped', '## Attention')]
    assert positions == sorted(positions)
    assert '2026-11-01 (9 days left)' in text
    assert '2026-10-20 (overdue)' in text
    assert 'Venue' in text and 'submitted 2026-10-01' in text and '10.1234/example' in text
    assert 'S-one' in text and 'Q1' in text and 'alpha' in text
    attention = text.split('## Attention')[1]
    assert 'only stopped or folded studies' in attention
    assert 'orphan: pubs record unmatched' in attention
    assert 'beta: paper repository has neither a pubs record nor a study' in attention
    assert '1 uncommitted' in attention and '1 unpushed' in attention
    assert 'Skipped 0 missing checkouts' in text


def test_board_attention_boundaries_and_no_false_disagreements(tmp_path):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    pubs = [dict(pubs[0], id=f'p-{i}', title=f'Boundary {i}',
                 deadline=dict(pubs[0]['deadline'], date=deadline), follow_up={'check_on': check})
            for i, (deadline, check) in enumerate([
                ('2026-10-22', '2026-10-22'), ('2026-10-23', '2026-10-23'),
                ('2026-12-07', '2026-11-06'), ('2026-12-08', '2026-11-07')])]
    studies += [dict(studies[0], id='S-active', state='running', questions=[])]
    write_program(tmp_path, raw, studies, questions, pubs)
    _, text = board(tmp_path)
    attention = text.split('## Attention')[1]
    assert 'p-1: deadline 2026-10-23 (0 days left)' in attention
    assert 'p-2: deadline 2026-12-07 (45 days left)' in attention
    assert 'p-0: deadline' not in attention and 'p-3: deadline' not in attention
    assert 'p-0: follow-up 2026-10-22 (overdue)' in attention
    assert 'p-1: follow-up 2026-10-23\n' in attention
    assert 'p-2: follow-up 2026-11-06' in attention and 'p-3: follow-up' not in attention
    assert 'only stopped or folded' not in attention
    assert 'Skipped 5 missing checkouts' in text
    assert 'papers/gamma-side' not in text


def test_board_missing_data_write_and_cli(tmp_path, capfd):
    raw, _, _, _ = program_fixture(tmp_path)
    for source in raw['sources'].values():
        (tmp_path / source).unlink()
    code, text = board(tmp_path)
    assert code == 0 and 'pubs record absent' in text and 'studies record absent' in text
    assert board(tmp_path, write=True) == (0, '')
    assert (tmp_path / 'STATUS.local.md').read_text() == text
    assert ws.main(['board', '--as-of', '2026-10-23'], root=tmp_path) == 0
    assert capfd.readouterr().out == text
    assert ws.main(['board', '--as-of', '2026-10-23', '--write'], root=tmp_path) == 0
    assert capfd.readouterr().out == ''
    assert ws.main(['board', '--as-of', '2026-02-30'], root=tmp_path) == 2
    assert 'YYYY-MM-DD' in capfd.readouterr().err
    assert ws.main(['board'], root=tmp_path) == 0
    assert ws.date.today().isoformat() in capfd.readouterr().out


def test_board_missing_deadline_is_not_inferred_from_followup(tmp_path):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    pubs[0].pop('deadline')
    pubs[0]['title'] = 'Title | with\na newline'
    write_program(tmp_path, raw, studies, questions, pubs)
    _, text = board(tmp_path)
    assert 'not recorded' in text
    assert 'Title \\| with a newline' in text
    assert 'paper-one: deadline' not in text


# ----------------------------------------------------------------------------- disposable links


def by_stage(root):
    out = io.StringIO()
    code = ws.cmd_by_stage(ws.load(root / 'workspace.json'), root, out)
    return code, out.getvalue()


def test_by_stage_relative_links_subpaths_umbrella_and_missing_targets(tmp_path):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    pubs.append(dict(pubs[0], id='umbrella-paper', repos={'manuscript': str(tmp_path)}))
    studies.append(dict(studies[0], id='S-umbrella', repo='umbrella', questions=[], path='docs'))
    studies.append(dict(studies[0], id='S-unassigned', repo=None, questions=[]))
    studies[-1].pop('path')
    write_program(tmp_path, raw, studies, questions, pubs)
    (tmp_path / 'papers/alpha/analysis').mkdir(parents=True)
    code, output = by_stage(tmp_path)
    assert code == 0
    view = tmp_path / 'by-stage'
    assert (view / '.generated-by-workspace').is_file()
    expected = {'manuscripts/under-review/paper-one': 'papers/alpha',
                'studies/stopped/S-one': 'papers/alpha/analysis',
                'repos/study/alpha': 'papers/alpha', 'repos/paper/beta': 'papers/beta',
                'repos/pilot/gamma': 'papers/gamma'}
    links = [p for p in view.rglob('*') if p.is_symlink()]
    assert {str(p.relative_to(view)) for p in links} == set(expected)
    for link in links:
        assert not link.readlink().is_absolute()
        assert link.resolve() == (tmp_path / expected[str(link.relative_to(view))]).resolve()
    assert 'repos/paper/beta: missing target' in output
    assert 'repos/pilot/gamma: missing target' in output
    assert 'S-unassigned: no repository' in output
    assert 'umbrella-paper' not in [p.name for p in links]
    assert 'S-umbrella' not in [p.name for p in links]
    assert by_stage(tmp_path)[0] == 0


def test_by_stage_refuses_unmarked_directory_and_files_before_deleting(tmp_path):
    program_fixture(tmp_path)
    view = tmp_path / 'by-stage'
    view.mkdir()
    with pytest.raises(ws.ManifestError, match='marker'):
        by_stage(tmp_path)
    (view / '.generated-by-workspace').touch()
    stale = view / 'old-link'
    stale.symlink_to('../missing')
    protected = view / 'nested/real.txt'
    protected.parent.mkdir()
    protected.write_text('preserve me')
    with pytest.raises(ws.ManifestError, match='non-symlink'):
        by_stage(tmp_path)
    assert protected.read_text() == 'preserve me' and stale.is_symlink()


def test_by_stage_removes_stale_links_without_following_them(tmp_path):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    assert by_stage(tmp_path)[0] == 0
    view = tmp_path / 'by-stage'
    old = view / 'manuscripts/under-review/paper-one'
    outside = tmp_path / 'protected'
    outside.mkdir()
    (outside / 'real.txt').write_text('keep')
    (outside / 'link').symlink_to('real.txt')
    (view / 'stale-directory').symlink_to('../protected', target_is_directory=True)
    pubs[0]['derived']['status'] = 'in press'
    studies[0]['state'] = 'deferred'
    write_program(tmp_path, raw, studies, questions, pubs)
    assert by_stage(tmp_path)[0] == 0
    assert not old.is_symlink()
    assert not (view / 'studies/stopped/S-one').is_symlink()
    assert (view / 'studies/deferred/S-one').is_symlink()
    assert (view / 'manuscripts/in-press/paper-one').is_symlink()
    assert not (view / 'stale-directory').is_symlink()
    assert (outside / 'real.txt').read_text() == 'keep' and (outside / 'link').is_symlink()


@pytest.mark.parametrize('kind', ['root', 'marker', 'parent'])
def test_by_stage_symlink_boundaries(tmp_path, kind):
    program_fixture(tmp_path)
    view = tmp_path / 'by-stage'
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / '.generated-by-workspace').touch()
    if kind == 'root':
        view.symlink_to(outside, target_is_directory=True)
    else:
        view.mkdir()
        if kind == 'marker':
            (view / '.generated-by-workspace').symlink_to(outside / '.generated-by-workspace')
        else:
            (view / '.generated-by-workspace').touch()
            (view / 'repos').symlink_to(outside, target_is_directory=True)
    if kind == 'parent':
        assert by_stage(tmp_path)[0] == 0
        assert not (view / 'repos').is_symlink()
        assert not (outside / 'study').exists()
    else:
        with pytest.raises(ws.ManifestError, match='symlink|marker'):
            by_stage(tmp_path)


def test_by_stage_cli_and_missing_sources(tmp_path):
    raw, _, _, _ = program_fixture(tmp_path)
    for source in raw['sources'].values():
        (tmp_path / source).unlink()
    assert ws.main(['by-stage'], root=tmp_path) == 0
    assert (tmp_path / 'by-stage/repos/study/alpha').is_symlink()


def test_generated_view_is_never_a_source_even_through_aliases(tmp_path):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    view = tmp_path / 'by-stage'
    view.mkdir()
    (view / 'records').symlink_to(tmp_path / 'records', target_is_directory=True)
    (tmp_path / 'alias').symlink_to(view, target_is_directory=True)
    raw['sources']['studies'] = 'alias/records/studies.json'
    (tmp_path / 'workspace.json').write_text(json.dumps(raw))
    with pytest.raises(ws.ManifestError, match='by-stage'):
        load_program(tmp_path)


def test_by_stage_plan_failure_preserves_existing_links(tmp_path):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    assert by_stage(tmp_path)[0] == 0
    raw['repos'][1]['name'] = '../escape'
    studies[0]['repo'] = '../escape'
    raw['snapshots'] = {}
    write_program(tmp_path, raw, studies, questions, pubs)
    with pytest.raises(ws.ManifestError, match='invalid id'):
        by_stage(tmp_path)
    assert (tmp_path / 'by-stage/repos/study/alpha').is_symlink()


@pytest.mark.parametrize('collection, field', [('repos', 'path'), ('worktrees', 'path'), ('views', 'file')])
def test_manifest_inputs_cannot_traverse_generated_view(tmp_path, collection, field):
    raw, _, _, _ = program_fixture(tmp_path)
    raw[collection][0][field] = 'by-stage/records/input'
    (tmp_path / 'workspace.json').write_text(json.dumps(raw))
    with pytest.raises(ws.ManifestError, match='by-stage'):
        ws.load(tmp_path / 'workspace.json')


def test_spent_evidence_cannot_read_generated_view(workspace):
    root, _, _ = workspace
    call(root, 'clone')
    target = root / 'by-stage/evidence.txt'
    target.parent.mkdir()
    target.write_text('do not consume')
    with pytest.raises(ws.ManifestError, match='by-stage'):
        spent(root, sample('a', '2026-10-01', evidence=[
            {'repo': 'umbrella', 'path': 'by-stage/evidence.txt', 'contains': 'do not consume'}]))


def test_valid_fold_and_temporarily_absent_questions(tmp_path):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    studies[0].update(state='folded', successor='S-next', note='Merged into successor')
    studies.append(dict(studies[0], id='S-next', state='designed', successor=None, questions=[]))
    write_program(tmp_path, raw, studies, questions, pubs)
    assert load_program(tmp_path).studies[0].successor == 'S-next'
    (tmp_path / raw['sources']['questions']).unlink()
    program = load_program(tmp_path)
    assert len(program.studies) == 2 and not program.questions
    assert any('cross-references unverified' in warning for warning in program.warnings)


def test_board_all_statuses_in_order_and_folded_disagreement(tmp_path):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    studies[0].update(state='folded', successor='S-next')
    studies.append(dict(studies[0], id='S-next', state='idea', repo='gamma', successor=None, questions=[]))
    statuses = ['in preparation', 'revising', 'under review', 'seeking venue', 'in press',
                'published', 'posted', 'withdrawn', 'retired', 'upstream custom status']
    pubs = [dict(pubs[0], id=f'p-{i}', derived=dict(pubs[0]['derived'], status=status))
            for i, status in enumerate(reversed(statuses))]
    write_program(tmp_path, raw, studies, questions, pubs)
    text = board(tmp_path)[1]
    headings = [line.removeprefix('### ') for line in text.split('## Studies')[0].splitlines()
                if line.startswith('### ')]
    assert headings == statuses
    assert text.count('only stopped or folded studies') == 4


@pytest.mark.parametrize('date_', ['20261023', '2026-02-30', None, 42])
def test_invalid_structured_dates_are_manifest_errors(tmp_path, date_):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    pubs[0]['deadline']['date'] = date_
    write_program(tmp_path, raw, studies, questions, pubs)
    with pytest.raises(ws.ManifestError, match='YYYY-MM-DD'):
        load_program(tmp_path)


def test_malformed_json_reports_the_source_and_blank_jsonl_lines_are_ignored(tmp_path):
    raw, _, _, _ = program_fixture(tmp_path)
    source = tmp_path / raw['sources']['pubs']
    source.write_text('\n' + source.read_text() + '\n')
    assert len(load_program(tmp_path).manuscripts) == 1
    source.write_text('{')
    with pytest.raises(ws.ManifestError, match='pubs.jsonl:1'):
        load_program(tmp_path)


def test_by_stage_refuses_directory_at_planned_leaf_before_mutation(tmp_path):
    program_fixture(tmp_path)
    view = tmp_path / 'by-stage'
    (view / 'repos/study/alpha').mkdir(parents=True)
    (view / '.generated-by-workspace').touch()
    (view / 'stale').symlink_to('../missing')
    with pytest.raises(ws.ManifestError, match='replace a directory'):
        by_stage(tmp_path)
    assert (view / 'stale').is_symlink()


def test_board_write_refuses_symlink(tmp_path):
    program_fixture(tmp_path)
    protected = tmp_path / 'protected.md'
    protected.write_text('keep')
    (tmp_path / 'STATUS.local.md').symlink_to(protected)
    with pytest.raises(ws.ManifestError, match='symlink'):
        board(tmp_path, write=True)
    assert protected.read_text() == 'keep'


def test_generated_view_input_rejected_through_file_symlink(tmp_path):
    raw, _, _, _ = program_fixture(tmp_path)
    view = tmp_path / 'by-stage'
    view.mkdir()
    (view / 'records').symlink_to(tmp_path / 'records', target_is_directory=True)
    (tmp_path / 'indirect.json').symlink_to(view / 'records/studies.json')
    raw['sources']['studies'] = 'indirect.json'
    (tmp_path / 'workspace.json').write_text(json.dumps(raw))
    with pytest.raises(ws.ManifestError, match='by-stage'):
        load_program(tmp_path)


# ----------------------------------------------------------------------------- datasets and questions


def dataset(**extra):
    return {'id': 'D-one', 'name': 'Dataset one', 'provenance': 'Fixture collection',
            'version': None, 'access': 'public', 'licence': None, 'location': 'https://example.org/data',
            'documentation': 'public-datasets.md#fixture', 'sampling_unit': 'component',
            'catalog_sections': ['fixture'], **extra}


def write_datasets(root, raw, entries=None):
    path = root / raw['sources']['datasets']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'version': 1, 'datasets': [dataset()] if entries is None else entries}))
    return path


@pytest.mark.parametrize('mutate, message', [
    (lambda d: d[0].update(access='open-ish'), 'unknown access class'),
    (lambda d: d.append(dict(d[0])), 'duplicate dataset ids'),
    (lambda d: d[0].update(id='one'), 'invalid dataset id'),
    (lambda d: d[0].update(licence=42), 'licence'),
    (lambda d: d[0].pop('version'), 'version'),
    (lambda d: d[0].update(location=''), 'location'),
    (lambda d: d[0].update(sampling_unit='person-ish'), 'sampling unit'),
    (lambda d: d[0].update(catalog_sections=['a', 'a']), 'duplicate catalog_sections'),
    (lambda d: d[0].update(catalog_sections=['not an id']), 'invalid id'),
    (lambda d: d[0].update(extra='typo'), 'unknown fields'),
])
def test_dataset_record_rules(tmp_path, mutate, message):
    raw, _, _, _ = program_fixture(tmp_path)
    entries = [dataset()]
    mutate(entries)
    write_datasets(tmp_path, raw, entries)
    with pytest.raises(ws.ManifestError, match=message):
        load_program(tmp_path)


@pytest.mark.parametrize('access', ['public', 'gated', 'controlled', 'private-archive', 'unavailable'])
def test_dataset_access_and_explicit_unknown_metadata(tmp_path, access):
    raw, _, _, _ = program_fixture(tmp_path)
    entry = dataset(access=access)
    for key in ('provenance', 'version', 'licence', 'location', 'documentation', 'sampling_unit'):
        entry[key] = None
    write_datasets(tmp_path, raw, [entry])
    record = load_program(tmp_path).datasets[0]
    assert record.access == access and record.version is None and record.sampling_unit is None


@pytest.mark.parametrize('kind', ['studies', 'questions', 'datasets'])
def test_record_unknown_fields_rejected(tmp_path, kind):
    raw, _, _, _ = program_fixture(tmp_path)
    path = tmp_path / raw['sources'][kind]
    document = json.loads(path.read_text())
    document[kind][0]['typo'] = True
    path.write_text(json.dumps(document))
    with pytest.raises(ws.ManifestError, match='unknown fields'):
        load_program(tmp_path)


@pytest.mark.parametrize('kind', ['studies', 'datasets'])
def test_question_unknown_reference(tmp_path, kind):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    questions[0][kind] = ['S-missing' if kind == 'studies' else 'D-missing']
    write_program(tmp_path, raw, studies, questions, pubs)
    with pytest.raises(ws.ManifestError, match='unknown study|unknown dataset'):
        load_program(tmp_path)


@pytest.mark.parametrize('command', ['board', 'questions', 'spent'])
def test_sample_unknown_dataset_rejected_by_commands(tmp_path, capfd, command):
    program_fixture(tmp_path)
    write_ledger(tmp_path, [sample('bad-draw', '2026-10-01', dataset='D-missing')])
    assert ws.main([command], root=tmp_path) == 2
    assert 'bad-draw: unknown dataset D-missing' in capfd.readouterr().err


@pytest.mark.parametrize('extra, message', [
    ({'unit': 'people-ish'}, 'sampling unit'), ({'size': {'count': -1}}, 'size'),
    ({'size': {'count': True}}, 'size'), ({'size': []}, 'size'),
    ({'size': {'count': 1.5}}, 'size'), ({'dataset': None}, 'dataset'),
])
def test_sample_exposure_fields_validated(tmp_path, extra, message):
    path = write_ledger(tmp_path, [sample('draw', '2026-10-01', **extra)])
    with pytest.raises(ws.ManifestError, match=message):
        ws.load_ledger(path)


@pytest.mark.parametrize('constant', ['NaN', 'Infinity', '-Infinity'])
def test_non_json_constants_rejected(constant):
    with pytest.raises(ws.ManifestError, match='non-JSON constant'):
        ws.strict_json('{"value": ' + constant + '}', 'fixture')


def questions_output(root, **kwargs):
    out = io.StringIO()
    code = ws.cmd_questions(ws.load(root / 'workspace.json'), root, out=out, **kwargs)
    return code, out.getvalue()


def test_questions_round_trip_stale_check_and_cli(tmp_path, capfd):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    code, text = questions_output(tmp_path)
    assert code == 0
    assert 'generated' in text and 'must not be edited by hand' in text
    assert 'Q1' in text and 'Does it work?' in text and 'open' in text
    assert 'A control' in text and 'A replication' in text
    assert 'S-one' in text and 'stopped' in text
    assert '1. Keep **controls**.\n   Preserve their scope.' in text
    target = tmp_path / 'records/questions-ledger.md'
    assert not target.exists()
    assert questions_output(tmp_path, check=True)[0] == 1
    assert questions_output(tmp_path, write=True)[0] == 0
    assert target.read_text() == text
    assert questions_output(tmp_path, check=True)[0] == 0
    assert ws.main(['questions'], root=tmp_path) == 0
    assert capfd.readouterr().out == text
    questions[0]['remaining'] = 'A changed goal'
    write_program(tmp_path, raw, studies, questions, pubs)
    assert ws.main(['questions', '--check'], root=tmp_path) == 1
    assert 'stale' in capfd.readouterr().out
    assert target.read_text() == text
    assert ws.main(['questions', '--write'], root=tmp_path) == 0
    assert ws.main(['questions', '--check'], root=tmp_path) == 0
    assert 'A changed goal' in target.read_text()


def test_exposure_preserves_units_sizes_and_draw_counts(tmp_path):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    questions[0]['datasets'].append('D-unused')
    write_program(tmp_path, raw, studies, questions, pubs)
    write_datasets(tmp_path, raw, [dataset(), dataset(id='D-unused', name='Unused')])
    write_ledger(tmp_path, [sample('first', '2026-10-01', unit='ip', size={'ips_drawn': 9, 'ips_retained': 7}),
                            sample('second', '2026-10-02', unit='identity', size={'identities': 3}, excludes=['first'])])
    text = questions_output(tmp_path)[1]
    assert '2 known draws' in text and '0 known draws' in text
    assert 'first' in text and 'ip' in text and 'ips_drawn=9' in text and 'ips_retained=7' in text
    assert 'second' in text and 'identity' in text and 'identities=3' in text
    assert text.count('remaining eligibility unverified') == 2
    assert '| D-one | public | 2 |' in board(tmp_path)[1]
    assert '| D-unused | public | 0 |' in board(tmp_path)[1]


@pytest.mark.parametrize('source', ['studies', 'questions', 'datasets', 'spent'])
def test_questions_missing_inputs_reported_and_never_overwrite(tmp_path, source):
    raw, _, _, _ = program_fixture(tmp_path)
    target = tmp_path / 'records/questions-ledger.md'
    target.write_text('keep')
    (tmp_path / (raw['sources'][source] if source != 'spent' else 'spent.json')).unlink()
    code, text = questions_output(tmp_path)
    assert code == 0 and 'absent' in text
    if source == 'spent':
        assert 'draws unverified' in text and '0 known draws' not in text
        assert '| D-one | public | unverified |' in board(tmp_path)[1]
    with pytest.raises(ws.ManifestError, match='unavailable'):
        questions_output(tmp_path, write=True)
    assert questions_output(tmp_path, check=True)[0] != 0
    assert target.read_text() == 'keep'


def test_questions_write_refuses_symlink(tmp_path):
    program_fixture(tmp_path)
    target = tmp_path / 'protected.md'
    target.write_text('keep')
    (tmp_path / 'records/questions-ledger.md').symlink_to(target)
    with pytest.raises(ws.ManifestError, match='symlink'):
        questions_output(tmp_path, write=True)
    assert target.read_text() == 'keep'


@pytest.mark.parametrize('lessons', [None, ['text'], [{'id': 'L-one', 'text': ''}],
                                    [{'id': 'L-one', 'text': 'a'}, {'id': 'L-one', 'text': 'b'}],
                                    [{'id': 'L-one', 'text': 'a', 'extra': 1}]])
def test_lessons_validated(tmp_path, lessons):
    raw, _, _, _ = program_fixture(tmp_path)
    path = tmp_path / raw['sources']['questions']
    document = json.loads(path.read_text())
    document['lessons'] = lessons
    path.write_text(json.dumps(document))
    with pytest.raises(ws.ManifestError, match='lesson|unknown fields'):
        load_program(tmp_path)


def test_question_without_studies_or_candidates_is_not_incomplete(tmp_path):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    studies[0]['questions'] = []
    questions[0].update(studies=[], datasets=[])
    write_program(tmp_path, raw, studies, questions, pubs)
    text = questions_output(tmp_path)[1]
    assert 'No studies recorded.' in text and 'No candidate datasets recorded.' in text
    assert questions_output(tmp_path, write=True)[0] == 0
    assert questions_output(tmp_path, check=True)[0] == 0


def test_questions_output_cannot_escape_through_parent_symlink(tmp_path):
    root = tmp_path / 'workspace'
    root.mkdir()
    program_fixture(root)
    outside = tmp_path / 'outside'
    (root / 'records').rename(outside)
    (root / 'records').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ws.ManifestError, match='output must stay inside'):
        questions_output(root, write=True)
    assert not (outside / 'questions-ledger.md').exists()


def test_questions_cli_rejects_write_and_check_together(tmp_path):
    program_fixture(tmp_path)
    with pytest.raises(SystemExit) as error:
        ws.main(['questions', '--write', '--check'], root=tmp_path)
    assert error.value.code == 2
    assert not (tmp_path / 'records/questions-ledger.md').exists()


def test_questions_exposure_change_makes_ledger_stale(tmp_path):
    program_fixture(tmp_path)
    assert questions_output(tmp_path, write=True)[0] == 0
    write_ledger(tmp_path, [sample('draw-one', '2026-10-01', size={'components': 3})])
    assert questions_output(tmp_path, check=True)[0] == 1
    assert 'components=2' in (tmp_path / 'records/questions-ledger.md').read_text()
    assert questions_output(tmp_path, write=True)[0] == 0
    assert 'components=3' in (tmp_path / 'records/questions-ledger.md').read_text()


@pytest.mark.parametrize('kind', ['studies', 'questions', 'datasets'])
def test_unknown_document_fields_rejected(tmp_path, kind):
    raw, _, _, _ = program_fixture(tmp_path)
    path = tmp_path / raw['sources'][kind]
    document = json.loads(path.read_text())
    document['typo'] = []
    path.write_text(json.dumps(document))
    with pytest.raises(ws.ManifestError, match='unknown fields'):
        load_program(tmp_path)


def test_manuscript_locators_match_the_deepest_repository_and_the_umbrella_only_exactly(tmp_path):
    raw, studies, questions, pubs = program_fixture(tmp_path)
    pubs += [dict(pubs[0], id='in-subdir', repos={'manuscript': str(tmp_path / 'papers/alpha/paper/abstract')}),
             dict(pubs[0], id='umbrella-root', repos={'manuscript': str(tmp_path)}),
             dict(pubs[0], id='unlisted-checkout', repos={'manuscript': str(tmp_path / 'papers/unlisted')})]
    write_program(tmp_path, raw, studies, questions, pubs)
    repos = {p.id: (p.repo.name if p.repo else None) for p in load_program(tmp_path).manuscripts}
    assert repos == {'paper-one': 'alpha', 'in-subdir': 'alpha', 'umbrella-root': 'umbrella',
                     'unlisted-checkout': None}


def test_spent_missing_datasets_are_reported_but_do_not_fail_verification(tmp_path):
    # Like absent checkouts, absent private records are reported, not failed: a public clone of the
    # umbrella has spent.json but not the private agenda.
    raw, _, _, _ = program_fixture(tmp_path)
    (tmp_path / raw['sources']['datasets']).unlink()
    out = io.StringIO()
    assert ws.cmd_spent(ws.load(tmp_path / 'workspace.json'), tmp_path, verify=True,
                        path=tmp_path / 'spent.json', out=out) == 0
    text = out.getvalue()
    assert 'datasets record absent' in text and 'cross-references unverified' in text and '0 problems' in text


# ----------------------------------------------------------------------------- backup to Drive

def backup_manifest(tmp_path, root: Path, **over) -> 'ws.Manifest':
    raw = {'version': 1, 'owner': 'tester', 'remote': 'x/{name}.git', 'repos': [],
           'sources': base_manifest(tmp_path)['sources'],
           'backup': {'repository': str(tmp_path / 'restic-repo'), 'password_file': '~/.config/cmri/restic-password',
                      'paths': {'workspace': '.', 'outside': '~/outside', 'claude-memory': '{claude_memory}'},
                      'exclude': ['.venv', '__pycache__'], 'pack_size_mib': 4, **over}}
    path = root / 'workspace.json'
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw))
    return ws.load(path)


def test_backup_locations_follow_the_machine(tmp_path):
    root = tmp_path / 'some_place' / 'cmri'
    backup = ws.load_backup(backup_manifest(tmp_path, root))
    where = ws.backup_locations(backup, root)
    assert where['workspace'] == root.resolve()
    assert where['outside'] == Path.home() / 'outside'
    name = str(root.resolve()).replace('/', '-').replace('_', '-')
    assert where['claude-memory'] == Path.home() / '.claude' / 'projects' / name / 'memory'


@pytest.mark.parametrize('over, message', [
    ({'repository': ''}, 'backup lacks repository'),
    ({'paths': {}}, 'backup paths must map names'),
    ({'paths': {'Bad Name': '.'}}, 'kebab-case'),
    ({'exclude': '.venv'}, 'exclude must be a list'),
    ({'pack_size_mib': 512}, 'pack_size_mib'),
])
def test_backup_rules(tmp_path, over, message):
    with pytest.raises(ws.ManifestError, match=message):
        ws.load_backup(backup_manifest(tmp_path, tmp_path / 'root', **over))


def test_pull_needs_the_password_and_push_dry_run_creates_nothing(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    manifest = backup_manifest(tmp_path, root)
    calls = []
    monkeypatch.setattr(ws, 'restic', lambda backup, *args, capture=False: calls.append(args))
    out = io.StringIO()
    assert ws.cmd_backup(manifest, root, 'pull', out=out) == 1 and 'no restic password' in out.getvalue()
    assert ws.cmd_backup(manifest, root, 'push', dry_run=True, out=out) == 1
    assert not (Path.home() / '.config/cmri/restic-password').exists() and calls == []


@pytest.mark.skipif(subprocess.run(['which', 'restic'], capture_output=True).returncode != 0,
                    reason='restic is not installed')
def test_backup_push_then_pull_on_another_machine_and_back(tmp_path, monkeypatch):
    home1, first = Path.home(), tmp_path / 'machine1' / 'cmri'
    manifest = backup_manifest(tmp_path, first)
    (first / 'papers' / 'alpha' / '.venv').mkdir(parents=True)
    (first / 'papers' / 'alpha' / 'cache.json').write_text('{"a": 1}')
    (first / 'papers' / 'alpha' / '.venv' / 'big').write_text('regenerable')
    sh('git', 'init', '-q', str(first / 'papers' / 'beta'))
    commit(first / 'papers' / 'beta', {'a.txt': 'x'})
    sh('git', 'worktree', 'add', '-q', '-b', 'side', str(first / 'papers' / 'beta-wt'), cwd=first / 'papers' / 'beta')
    (Path.home() / 'outside').mkdir()
    (Path.home() / 'outside' / 'results.jsonl').write_text('paid\n')
    memory = ws.claude_memory_dir(first)
    memory.mkdir(parents=True)
    (memory / 'MEMORY.md').write_text('# index\n')
    out = io.StringIO()
    assert ws.cmd_backup(manifest, first, 'push', host='one', out=out) == 0, out.getvalue()
    password = (Path.home() / '.config/cmri/restic-password').read_text()

    home2 = tmp_path / 'home2'                                  # another machine: other home, other path
    (home2 / '.config/cmri').mkdir(parents=True)
    (home2 / '.config/cmri/restic-password').write_text(password)
    monkeypatch.setenv('HOME', str(home2))
    second = tmp_path / 'machine2' / 'repos' / 'cmri'
    manifest2 = backup_manifest(tmp_path, second)
    assert ws.cmd_backup(manifest2, second, 'pull', out=out) == 0, out.getvalue()
    assert (second / 'papers/alpha/cache.json').read_text() == '{"a": 1}'
    assert not (second / 'papers/alpha/.venv').exists()               # excluded
    assert (home2 / 'outside/results.jsonl').read_text() == 'paid\n'
    assert (ws.claude_memory_dir(second) / 'MEMORY.md').read_text() == '# index\n'
    assert sh('git', 'rev-parse', '--abbrev-ref', 'HEAD', cwd=second / 'papers/beta-wt') == 'side'   # repaired
    assert str(second) in sh('git', 'worktree', 'list', cwd=second / 'papers/beta')

    (second / 'papers/alpha/cache.json').write_text('{"a": 2}')     # work there, push, come back
    assert ws.cmd_backup(manifest2, second, 'push', host='two', out=out) == 0, out.getvalue()
    monkeypatch.setenv('HOME', str(home1))
    assert ws.cmd_backup(manifest, first, 'pull', host='two', out=out) == 0, out.getvalue()
    assert (first / 'papers/alpha/cache.json').read_text() == '{"a": 2}'
    assert 'of two' in out.getvalue()
