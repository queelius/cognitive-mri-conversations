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
    commit(root, {'.gitignore': 'papers/\n', 'CLAUDE.md': (root / 'CLAUDE.md').read_text(),
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
