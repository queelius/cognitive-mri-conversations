#!/usr/bin/env python3
"""One structured source for the CMRI child repositories.

`workspace.json` at the umbrella root lists every child repository and worktree (path, branch,
GitHub visibility, data class, one-line note) and named snapshots that pin the commits a paper
depends on. This tool derives everything else from it:

    python3 tools/workspace.py status                  # local state of every entry
    python3 tools/workspace.py clone [--dry-run]       # clone missing repos and worktrees into place
    python3 tools/workspace.py checkout SNAPSHOT       # detach pinned repos at their pins
    python3 tools/workspace.py checkout --branches     # switch every repo back to its branch
    python3 tools/workspace.py lock SNAPSHOT           # pin a snapshot to the current, pushed HEADs
    python3 tools/workspace.py verify [SNAPSHOT ...]   # every pin exists and is on the remote
    python3 tools/workspace.py remote-check            # GitHub visibility, default branch, archive flag
    python3 tools/workspace.py scan SNAPSHOT           # forbidden files and secrets, tree and history
    python3 tools/workspace.py render [--check]        # regenerate the CLAUDE.md tables

Standard library only; git and gh do the work. Secret scans report the pattern name, path and
blob id of a hit, never the matched text.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import fnmatch
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'workspace.json'
GROUPS = ('umbrella', 'corpus', 'paper', 'pilot', 'study')
DATA = ('none', 'public', 'private-archive')
VISIBILITY = ('public', 'private')
COLUMNS = ('path', 'repo', 'repo_branch', 'note')


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Repo:
    name: str
    path: str
    branch: str
    visibility: str
    data: str
    group: str
    note: str
    archived: bool = False
    default_branch: str | None = None

    @property
    def github_default(self) -> str:
        return self.default_branch or self.branch


@dataclass(frozen=True)
class Worktree:
    path: str
    repo: str
    branch: str


@dataclass(frozen=True)
class Snapshot:
    name: str
    description: str
    publish: bool
    pins: dict[str, str]


@dataclass(frozen=True)
class View:
    file: str
    block: str
    groups: tuple[str, ...]
    columns: tuple[str, ...]
    headers: tuple[str, ...]


@dataclass
class Manifest:
    owner: str
    remote: str
    repos: list[Repo]
    worktrees: list[Worktree]
    snapshots: dict[str, Snapshot]
    policy: dict
    views: list[View]
    raw: dict = field(repr=False)

    def repo(self, name: str) -> Repo:
        for repo in self.repos:
            if repo.name == name:
                return repo
        raise ManifestError(f'unknown repository: {name}')

    def url(self, repo: Repo) -> str:
        return self.remote.format(name=repo.name, owner=self.owner)


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ManifestError(message)


def load(path: Path = MANIFEST) -> Manifest:
    """Parse and validate the manifest; every structural rule lives here."""
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    _require(raw.get('version') == 1, 'manifest version must be 1')
    for key in ('owner', 'remote', 'repos'):
        _require(key in raw, f'manifest lacks {key}')
    _require('{name}' in raw['remote'], 'remote template must contain {name}')
    repos = []
    for entry in raw['repos']:
        for key in ('name', 'path', 'branch', 'visibility', 'data', 'group', 'note'):
            _require(isinstance(entry.get(key), str) and entry[key], f'repo entry lacks {key}: {entry}')
        _require(entry['group'] in GROUPS, f"unknown group {entry['group']!r}")
        _require(entry['data'] in DATA, f"unknown data class {entry['data']!r}")
        _require(entry['visibility'] in VISIBILITY, f"unknown visibility {entry['visibility']!r}")
        _require(not Path(entry['path']).is_absolute() and '..' not in Path(entry['path']).parts,
                 f"path must stay inside the workspace: {entry['path']}")
        repos.append(Repo(entry['name'], entry['path'], entry['branch'], entry['visibility'], entry['data'],
                          entry['group'], entry['note'], bool(entry.get('archived', False)),
                          entry.get('default_branch')))
    names = [r.name for r in repos]
    _require(len(set(names)) == len(names), 'repository names must be unique')
    worktrees = []
    for entry in raw.get('worktrees', []):
        for key in ('path', 'repo', 'branch'):
            _require(isinstance(entry.get(key), str) and entry[key], f'worktree entry lacks {key}: {entry}')
        _require(entry['repo'] in names, f"worktree of unknown repository {entry['repo']!r}")
        worktrees.append(Worktree(entry['path'], entry['repo'], entry['branch']))
    paths = [r.path for r in repos] + [w.path for w in worktrees]
    _require(len(set(paths)) == len(paths), 'paths must be unique')
    by_name = {r.name: r for r in repos}
    snapshots = {}
    for name, entry in raw.get('snapshots', {}).items():
        pins = entry.get('pins', {})
        _require(isinstance(pins, dict) and pins, f'snapshot {name} has no pins')
        for repo_name, sha in pins.items():
            _require(repo_name in by_name, f'snapshot {name} pins unknown repository {repo_name!r}')
            _require(sha == '' or re.fullmatch(r'[0-9a-f]{40}', sha) is not None,
                     f'snapshot {name}: pin for {repo_name} must be a full commit id or empty')
        publish = bool(entry.get('publish', False))
        if publish:
            leaking = [n for n in pins if by_name[n].data == 'private-archive']
            _require(not leaking, f'snapshot {name} is marked for publication but pins private-archive '
                                  f'repositories: {", ".join(leaking)}')
        snapshots[name] = Snapshot(name, entry.get('description', ''), publish, dict(pins))
    for entry in raw.get('publication_policy', {}).get('reviewed', []):
        for key in ('repo', 'path', 'reason', 'reviewed'):
            _require(isinstance(entry.get(key), str) and entry[key], f'reviewed exception lacks {key}: {entry}')
        _require(entry['repo'] in by_name, f"reviewed exception for unknown repository {entry['repo']!r}")
    views = []
    for entry in raw.get('views', []):
        columns, headers = tuple(entry['columns']), tuple(entry['headers'])
        _require(all(c in COLUMNS for c in columns), f'unknown view column in {columns}')
        _require(len(columns) == len(headers), 'view columns and headers differ in length')
        _require(all(g in GROUPS for g in entry['groups']), f"unknown view group in {entry['groups']}")
        views.append(View(entry['file'], entry['block'], tuple(entry['groups']), columns, headers))
    return Manifest(raw['owner'], raw['remote'], repos, worktrees, snapshots,
                    raw.get('publication_policy', {}), views, raw)


def save(manifest: Manifest, path: Path = MANIFEST) -> None:
    Path(path).write_text(json.dumps(manifest.raw, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


# ----------------------------------------------------------------------------- git and gh

def run(cmd: list[str], cwd: Path | None = None, check: bool = True, input: str | bytes | None = None,
        text: bool = True) -> subprocess.CompletedProcess:
    """The single process boundary (tests replace it to fake `gh`). Stdin is never inherited."""
    if input is None:
        input = '' if text else b''
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=text, input=input)


def git(path: Path, *args: str, check: bool = True) -> str:
    return run(['git', '-C', str(path), *args], check=check).stdout.strip()


def is_checkout(path: Path) -> bool:
    return (path / '.git').exists()


def same_url(a: str, b: str) -> bool:
    norm = lambda u: u.rstrip('/').removesuffix('.git')
    return norm(a) == norm(b)


def state(root: Path, path: str) -> dict:
    """Local facts about one checkout; `present` is False when the path is not a git checkout."""
    where = root / path
    if not is_checkout(where):
        return {'present': False}
    branch = git(where, 'branch', '--show-current')
    upstream = run(['git', '-C', str(where), 'rev-parse', '--abbrev-ref', '@{upstream}'], check=False)
    ahead = behind = None
    if upstream.returncode == 0:
        counts = git(where, 'rev-list', '--left-right', '--count', 'HEAD...@{upstream}').split()
        ahead, behind = int(counts[0]), int(counts[1])
    remote = run(['git', '-C', str(where), 'remote', 'get-url', 'origin'], check=False)
    return {'present': True, 'worktree': (where / '.git').is_file(), 'branch': branch or None,
            'head': git(where, 'rev-parse', 'HEAD'),
            'dirty': len([l for l in git(where, 'status', '--porcelain').splitlines() if l.strip()]),
            'ahead': ahead, 'behind': behind,
            'remote': remote.stdout.strip() if remote.returncode == 0 else None}


def on_remote(path: Path, sha: str) -> bool:
    out = run(['git', '-C', str(path), 'branch', '-r', '--contains', sha], check=False)
    return out.returncode == 0 and bool(out.stdout.strip())


def has_commit(path: Path, sha: str) -> bool:
    return run(['git', '-C', str(path), 'cat-file', '-e', f'{sha}^{{commit}}'], check=False).returncode == 0


# ----------------------------------------------------------------------------- commands

def cmd_status(manifest: Manifest, root: Path, out=sys.stdout) -> int:
    problems = 0
    rows = [(r.path, r.name, r.branch, manifest.url(r)) for r in manifest.repos]
    rows += [(w.path, w.repo, w.branch, manifest.url(manifest.repo(w.repo))) for w in manifest.worktrees]
    for path, name, branch, url in rows:
        s = state(root, path)
        if not s['present']:
            print(f'{path:36s} MISSING ({name})', file=out)
            continue
        notes = []
        if s['branch'] != branch:
            notes.append(f"on {s['branch'] or 'detached ' + s['head'][:7]}, manifest says {branch}")
        if s['dirty']:
            notes.append(f"{s['dirty']} uncommitted")
        if s['ahead']:
            notes.append(f"{s['ahead']} unpushed")
        if s['behind']:
            notes.append(f"{s['behind']} behind")
        if s['remote'] is None or not same_url(s['remote'], url):
            notes.append(f"origin is {s['remote']}, expected {url}")
            problems += 1
        print(f"{path:36s} {s['head'][:7]}  {'; '.join(notes) or 'clean'}", file=out)
    return 1 if problems else 0


def cmd_clone(manifest: Manifest, root: Path, dry_run: bool = False, out=sys.stdout) -> int:
    for repo in manifest.repos:
        where = root / repo.path
        if repo.path == '.' or is_checkout(where):
            continue
        cmd = ['git', 'clone', '--branch', repo.branch, manifest.url(repo), str(where)]
        print(' '.join(cmd), file=out)
        if not dry_run:
            run(cmd)
    for wt in manifest.worktrees:
        where = root / wt.path
        if is_checkout(where):
            continue
        parent = root / manifest.repo(wt.repo).path
        cmd = ['git', '-C', str(parent), 'worktree', 'add', str(where), wt.branch]
        print(' '.join(cmd), file=out)
        if not dry_run:
            run(cmd)
    return 0


def cmd_checkout(manifest: Manifest, root: Path, snapshot: str | None, dry_run: bool = False,
                 out=sys.stdout) -> int:
    """Detach every pinned repository at its pin, or switch every repository back to its branch."""
    if snapshot is None:
        targets = [(r.path, r.branch, False) for r in manifest.repos if r.path != '.']
    else:
        snap = _snapshot(manifest, snapshot)
        _require(all(snap.pins.values()), f'snapshot {snapshot} has empty pins; run lock first')
        targets = [(manifest.repo(n).path, sha, True) for n, sha in snap.pins.items()]
    dirty = [p for p, _, _ in targets if state(root, p).get('dirty')]
    if dirty:
        print('refusing: uncommitted changes in ' + ', '.join(dirty), file=out)
        return 1
    for path, target, detach in targets:
        where = root / path
        if not is_checkout(where):
            print(f'{path}: missing, run clone first', file=out)
            return 1
        if detach and not has_commit(where, target):
            run(['git', '-C', str(where), 'fetch', '--quiet', 'origin'])
        cmd = ['git', '-C', str(where), 'checkout', '--quiet'] + (['--detach', target] if detach else [target])
        print(' '.join(cmd), file=out)
        if not dry_run:
            run(cmd)
    return 0


def _snapshot(manifest: Manifest, name: str) -> Snapshot:
    _require(name in manifest.snapshots, f'unknown snapshot {name!r}')
    return manifest.snapshots[name]


def cmd_lock(manifest: Manifest, root: Path, snapshot: str, path: Path = MANIFEST, out=sys.stdout) -> int:
    """Pin a snapshot to each repository's current HEAD, refusing dirty or unpushed checkouts."""
    snap = _snapshot(manifest, snapshot)
    pins, problems = {}, []
    for name in snap.pins:
        where = root / manifest.repo(name).path
        s = state(root, manifest.repo(name).path)
        if not s['present']:
            problems.append(f'{name}: missing')
        elif s['dirty']:
            problems.append(f"{name}: {s['dirty']} uncommitted")
        elif not on_remote(where, s['head']):
            problems.append(f"{name}: HEAD {s['head'][:7]} is not on any remote branch")
        else:
            pins[name] = s['head']
    if problems:
        print('refusing to lock: ' + '; '.join(problems), file=out)
        return 1
    manifest.raw['snapshots'][snapshot]['pins'] = pins
    save(manifest, path)
    for name, sha in pins.items():
        print(f'{snapshot}: {name} {sha}', file=out)
    return 0


def cmd_verify(manifest: Manifest, root: Path, names: list[str], out=sys.stdout) -> int:
    failures = 0
    for name in names or list(manifest.snapshots):
        snap = _snapshot(manifest, name)
        for repo_name, sha in snap.pins.items():
            where = root / manifest.repo(repo_name).path
            if not sha:
                verdict = 'not pinned'
            elif not is_checkout(where):
                verdict = 'missing locally'
            elif not has_commit(where, sha):
                verdict = 'commit not found'
            elif not on_remote(where, sha):
                verdict = 'not on any remote branch'
            else:
                verdict = 'ok'
            failures += verdict != 'ok'
            print(f'{name:28s} {repo_name:30s} {sha[:7] or "-":7s} {verdict}', file=out)
    return 1 if failures else 0


def cmd_remote_check(manifest: Manifest, out=sys.stdout) -> int:
    """Compare each repository's GitHub visibility, default branch and archive flag with the manifest."""
    mismatches = 0
    for repo in manifest.repos:
        res = run(['gh', 'repo', 'view', f'{manifest.owner}/{repo.name}', '--json',
                   'visibility,defaultBranchRef,isArchived'], check=False)
        if res.returncode != 0:
            print(f'{repo.name:30s} unreachable on GitHub', file=out)
            mismatches += 1
            continue
        info = json.loads(res.stdout)
        found = {'visibility': info['visibility'].lower(),
                 'default branch': (info.get('defaultBranchRef') or {}).get('name'),
                 'archived': bool(info['isArchived'])}
        expected = {'visibility': repo.visibility, 'default branch': repo.github_default, 'archived': repo.archived}
        diffs = [f'{k} is {found[k]}, manifest says {expected[k]}' for k in expected if found[k] != expected[k]]
        mismatches += bool(diffs)
        print(f"{repo.name:30s} {'; '.join(diffs) or 'ok'}", file=out)
    return 1 if mismatches else 0


def _forbidden(path: str, globs) -> str | None:
    for pattern in globs:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern):
            return pattern
    return None


def scan_repo(where: Path, pin: str, policy: dict, exempt=()) -> list[dict]:
    """Forbidden paths in the pinned tree and anywhere in history, and secret patterns in every blob.

    `exempt` holds path globs a person reviewed for this repository; a forbidden path matching one
    is reported with kind `reviewed` instead of as a finding. Secret patterns are never exempt.
    """
    globs = policy.get('forbidden_paths', [])

    def forbidden(path):
        rule = _forbidden(path, globs)
        if rule and any(fnmatch.fnmatch(path, e) for e in exempt):
            return rule, True
        return rule, False

    patterns = {k: re.compile(v.encode()) for k, v in policy.get('secret_patterns', {}).items()}
    limit = int(policy.get('max_blob_bytes', 50 * 2 ** 20))
    findings = []
    for path in git(where, 'ls-tree', '-r', '--name-only', pin).splitlines():
        hit, reviewed = forbidden(path)
        if hit:
            findings.append({'kind': 'reviewed' if reviewed else 'forbidden path at pin', 'path': path, 'rule': hit})
    listing = git(where, 'rev-list', '--objects', '--all').splitlines()
    named = {}
    for line in listing:
        sha, _, path = line.partition(' ')
        if path and sha not in named:
            named[sha] = path
    if not named:
        return findings
    check = run(['git', '-C', str(where), 'cat-file', '--batch-check=%(objectname) %(objecttype) %(objectsize)'],
                input='\n'.join(named) + '\n')
    blobs = []
    for line in check.stdout.splitlines():
        sha, kind, size = line.split()
        if kind != 'blob':
            continue
        path = named[sha]
        hit, reviewed = forbidden(path)
        if hit:
            findings.append({'kind': 'reviewed' if reviewed else 'forbidden path in history', 'path': path,
                             'rule': hit, 'blob': sha[:10]})
        if int(size) > limit:
            findings.append({'kind': 'blob too large to scan', 'path': path, 'blob': sha[:10]})
            continue
        blobs.append(sha)
    if blobs and patterns:
        data = run(['git', '-C', str(where), 'cat-file', '--batch'], input=('\n'.join(blobs) + '\n').encode(),
                   text=False).stdout
        pos = 0
        while pos < len(data):
            header_end = data.index(b'\n', pos)
            sha, _, size = data[pos:header_end].decode().split()
            body = data[header_end + 1: header_end + 1 + int(size)]
            pos = header_end + 1 + int(size) + 1
            for name, rx in patterns.items():
                if rx.search(body):
                    findings.append({'kind': 'secret pattern', 'path': named[sha], 'rule': name, 'blob': sha[:10]})
    return findings


def cmd_scan(manifest: Manifest, root: Path, snapshot: str, out=sys.stdout) -> int:
    snap = _snapshot(manifest, snapshot)
    total = 0
    for name, sha in snap.pins.items():
        repo = manifest.repo(name)
        where = root / repo.path
        pin = sha or git(where, 'rev-parse', 'HEAD')
        exempt = [e['path'] for e in manifest.policy.get('reviewed', []) if e['repo'] == name]
        results = scan_repo(where, pin, manifest.policy, exempt)
        findings = [f for f in results if f['kind'] != 'reviewed']
        reviewed = len(results) - len(findings)
        total += len(findings)
        print(f"{name}: {len(findings)} finding(s) at {pin[:7]}{'' if sha else ' (unpinned, scanned HEAD)'}"
              + (f'; {reviewed} path(s) under reviewed exceptions' if reviewed else ''), file=out)
        for f in findings:
            extra = ' '.join(f'{k}={v}' for k, v in f.items() if k not in ('kind', 'path'))
            print(f"  {f['kind']}: {f['path']} {extra}", file=out)
    return 1 if total else 0


def render_view(manifest: Manifest, view: View) -> str:
    worktrees = {}
    for wt in manifest.worktrees:
        worktrees.setdefault(wt.repo, []).append(wt)

    def cell(repo: Repo, column: str) -> str:
        if column == 'path':
            return f'`{repo.path}/`'
        if column == 'note':
            return repo.note
        text = repo.name if column == 'repo' else f'{repo.name} (`{repo.branch}`)'
        extras = [f'worktree `{w.path}/` on `{w.branch}`' for w in worktrees.get(repo.name, [])]
        if repo.archived:
            extras.append('archived on GitHub')
        return text + (f"; {'; '.join(extras)}" if extras else '')

    lines = ['<!-- Generated from workspace.json by tools/workspace.py render; edit the manifest. -->',
             '| ' + ' | '.join(view.headers) + ' |',
             '|' + '|'.join('-' * (len(h) + 2) for h in view.headers) + '|']
    for repo in manifest.repos:
        if repo.group in view.groups:
            lines.append('| ' + ' | '.join(cell(repo, c) for c in view.columns) + ' |')
    return '\n'.join(lines)


def cmd_render(manifest: Manifest, root: Path, check: bool = False, out=sys.stdout) -> int:
    stale = []
    texts = {}
    for view in manifest.views:
        path = root / view.file
        text = texts.get(path, path.read_text(encoding='utf-8'))
        begin, end = f'<!-- workspace:{view.block}:begin -->', f'<!-- workspace:{view.block}:end -->'
        _require(begin in text and end in text, f'{view.file} lacks the markers for block {view.block!r}')
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        texts[path] = f'{head}{begin}\n{render_view(manifest, view)}\n{end}{tail}'
    for path, new in texts.items():
        if path.read_text(encoding='utf-8') != new:
            stale.append(path)
            if not check:
                path.write_text(new, encoding='utf-8')
    for path in stale:
        print(f"{'stale' if check else 'rendered'}: {path.relative_to(root)}", file=out)
    return 1 if (check and stale) else 0


def main(argv: list[str] | None = None, root: Path = ROOT, manifest_path: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    p = sub.add_parser('clone'); p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('checkout'); p.add_argument('snapshot', nargs='?')
    p.add_argument('--branches', action='store_true'); p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('lock'); p.add_argument('snapshot')
    p = sub.add_parser('verify'); p.add_argument('snapshots', nargs='*')
    sub.add_parser('remote-check')
    p = sub.add_parser('scan'); p.add_argument('snapshot')
    p = sub.add_parser('render'); p.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    manifest_path = manifest_path or root / 'workspace.json'
    try:
        manifest = load(manifest_path)
        if args.command == 'status':
            return cmd_status(manifest, root)
        if args.command == 'clone':
            return cmd_clone(manifest, root, args.dry_run)
        if args.command == 'checkout':
            if args.branches == (args.snapshot is not None):
                parser.error('give a snapshot or --branches, not both')
            return cmd_checkout(manifest, root, None if args.branches else args.snapshot, args.dry_run)
        if args.command == 'lock':
            return cmd_lock(manifest, root, args.snapshot, manifest_path)
        if args.command == 'verify':
            return cmd_verify(manifest, root, args.snapshots)
        if args.command == 'remote-check':
            return cmd_remote_check(manifest)
        if args.command == 'scan':
            return cmd_scan(manifest, root, args.snapshot)
        return cmd_render(manifest, root, args.check)
    except ManifestError as error:
        print(f'workspace.json: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
