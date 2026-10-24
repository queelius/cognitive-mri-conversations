#!/usr/bin/env python3
"""One structured source for the CMRI child repositories.

`workspace.json` at the umbrella root lists every child repository and worktree (path, branch,
GitHub visibility, data class, one-line note) and named snapshots that pin the commits a paper
depends on. This tool derives everything else from it:

    python3 tools/workspace.py by-stage                # rebuild the private relative-link view
    python3 tools/workspace.py board [--as-of YYYY-MM-DD] [--write]  # private research board
    python3 tools/workspace.py questions [--write | --check]  # questions, studies and dataset exposure
    python3 tools/workspace.py status                  # local state of every entry
    python3 tools/workspace.py clone [--dry-run]       # clone missing repos and worktrees into place
    python3 tools/workspace.py checkout SNAPSHOT       # detach pinned repos at their pins
    python3 tools/workspace.py checkout --branches     # switch every repo back to its branch
    python3 tools/workspace.py lock SNAPSHOT           # pin a snapshot to the current, pushed HEADs
    python3 tools/workspace.py verify [SNAPSHOT ...]   # every pin exists and is on the remote
    python3 tools/workspace.py remote-check            # GitHub visibility, default branch, archive flag
    python3 tools/workspace.py scan SNAPSHOT           # forbidden files and secrets, tree and history
    python3 tools/workspace.py render [--check]        # regenerate the CLAUDE.md tables
    python3 tools/workspace.py spent [--verify]        # the ledger of spent samples; --verify checks the exclusion chain

Standard library only; git and gh do the work. Secret scans report the pattern name, path and
blob id of a hit, never the matched text.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, timedelta
import fnmatch
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'workspace.json'
GROUPS = ('umbrella', 'corpus', 'planning', 'paper', 'pilot', 'study', 'infrastructure')
DATA = ('none', 'public', 'private-archive')
VISIBILITY = ('public', 'private')
COLUMNS = ('path', 'repo', 'repo_branch', 'note')
LEDGER = ROOT / 'spent.json'
SAMPLE_STATUS = ('development', 'frozen', 'exploratory', 'screen')


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
    sources: Sources

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
    raw = read_json(path)
    _require(isinstance(raw, dict) and raw.get('version') == 1, 'manifest version must be 1')
    sources = load_sources(raw.get('sources'))
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
        no_generated_input(Path(path).parent / entry['path'])
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
        no_generated_input(Path(path).parent / entry['path'])
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
        no_generated_input(Path(path).parent / entry['file'])
        columns, headers = tuple(entry['columns']), tuple(entry['headers'])
        _require(all(c in COLUMNS for c in columns), f'unknown view column in {columns}')
        _require(len(columns) == len(headers), 'view columns and headers differ in length')
        _require(all(g in GROUPS for g in entry['groups']), f"unknown view group in {entry['groups']}")
        views.append(View(entry['file'], entry['block'], tuple(entry['groups']), columns, headers))
    return Manifest(raw['owner'], raw['remote'], repos, worktrees, snapshots,
                    raw.get('publication_policy', {}), views, raw, sources)


def save(manifest: Manifest, path: Path = MANIFEST) -> None:
    Path(path).write_text(json.dumps(manifest.raw, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


# ----------------------------------------------------------------------------- program records

STUDY_STATES = ('idea', 'designed', 'running', 'explored', 'confirmed', 'not-confirmed', 'stopped', 'deferred', 'folded')
QUESTION_STATES = ('open', 'answered', 'narrowed', 'deferred', 'stopped')
MANUSCRIPT_STATUSES = ('in preparation', 'revising', 'under review', 'seeking venue', 'in press',
                       'published', 'posted', 'withdrawn', 'retired')
ACCESS_CLASSES = ('public', 'gated', 'controlled', 'private-archive', 'unavailable')
SAMPLING_UNITS = ('component', 'ip', 'identity', 'conversation')


def no_generated_input(path: Path) -> None:
    """Reject generated inputs, including symlinks that traverse the view then leave it."""
    pending, seen = [path], set()
    while pending:
        candidate = pending.pop()
        if candidate in seen:
            continue
        seen.add(candidate)
        _require('by-stage' not in candidate.parts, f'input must not read from by-stage: {path}')
        try:
            resolved = candidate.resolve()
        except RuntimeError as error:
            raise ManifestError(f'invalid input path {path}: {error}') from error
        _require('by-stage' not in resolved.parts, f'input must not read from by-stage: {path}')
        # resolve() alone hides intermediate link destinations. Check those explicitly.
        for parent in (candidate, *candidate.parents):
            if parent.is_symlink():
                pending.append(parent.parent / parent.readlink())


def strict_json(text: str, source: str):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f'{source}: duplicate object key {key!r}')
            result[key] = value
        return result
    def non_json(value):
        raise ManifestError(f'{source}: non-JSON constant {value}')
    try:
        return json.loads(text, object_pairs_hook=unique, parse_constant=non_json)
    except json.JSONDecodeError as error:
        raise ManifestError(f'{source}: {error}') from error


def read_json(path: Path):
    path = Path(path)
    no_generated_input(path)
    return strict_json(path.read_text(encoding='utf-8'), str(path))


def required_text(entry: dict, key: str, label: str, empty: bool = False) -> str:
    value = entry.get(key)
    _require(isinstance(value, str) and (empty or bool(value.strip())), f'{label}: invalid {key}')
    return value


def known_fields(entry: dict, fields: tuple[str, ...], label: str) -> None:
    unknown = set(entry) - set(fields)
    _require(not unknown, f'{label}: unknown fields: {", ".join(sorted(unknown))}')


def nullable_text(entry: dict, key: str, label: str) -> str | None:
    _require(key in entry, f'{label}: {key} is required (null means unknown)')
    return None if entry[key] is None else required_text(entry, key, label)


def dataset_id(value: str, label: str) -> str:
    value = identifier(value, label)
    _require(value.startswith('D-') and len(value) > 2, f'{label}: invalid dataset id: {value}')
    return value


def identifier(value: str, label: str) -> str:
    _require(isinstance(value, str) and bool(value) and value not in ('.', '..')
             and all(c.isalnum() or c in '-_.' for c in value), f'{label}: invalid id {value!r}')
    return value


def iso_date(value: str, label: str) -> date:
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError(value)
        return parsed
    except (TypeError, ValueError) as error:
        raise ManifestError(f'{label}: expected YYYY-MM-DD, got {value!r}') from error


@dataclass(frozen=True)
class Sources:
    pubs: str
    studies: str
    questions: str
    datasets: str


def load_sources(raw) -> Sources:
    _require(isinstance(raw, dict), 'sources must be an object')
    known_fields(raw, ('pubs', 'studies', 'questions', 'datasets'), 'sources')
    for key in ('pubs', 'studies', 'questions', 'datasets'):
        value = required_text(raw, key, 'sources')
        _require(not Path(value).is_absolute() and not value.startswith('~')
                 and 'by-stage' not in Path(value).parts, f'sources.{key}: expected a relative input path')
    return Sources(raw['pubs'], raw['studies'], raw['questions'], raw['datasets'])


@dataclass(frozen=True)
class Deadline:
    date: date
    source: str
    verified: date


@dataclass(frozen=True)
class Manuscript:
    id: str
    title: str
    status: str
    venue: str
    last_event: str
    doi: str
    repo: Repo | None
    deadline: Deadline | None
    check_on: date | None


@dataclass(frozen=True)
class Study:
    id: str
    title: str
    state: str
    repo: str | None
    path: str | None
    questions: tuple[str, ...]
    successor: str | None
    note: str | None


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    state: str
    studies: tuple[str, ...]
    known: str
    remaining: str
    datasets: tuple[str, ...]


@dataclass(frozen=True)
class Dataset:
    id: str
    name: str
    provenance: str | None
    version: str | None
    access: str
    licence: str | None
    location: str | None
    documentation: str | None
    sampling_unit: str | None
    catalog_sections: tuple[str, ...]


@dataclass(frozen=True)
class Lesson:
    id: str
    text: str


@dataclass
class Program:
    manuscripts: list[Manuscript]
    studies: list[Study]
    questions: list[Question]
    warnings: list[str]
    datasets: list[Dataset]
    lessons: list[Lesson]
    samples: list[Sample] | None
    unavailable: tuple[str, ...]


def read_records(path: Path, collection: str, warnings: list[str]) -> list[dict] | None:
    raw = read_record_document(path, collection, warnings)
    return raw[collection] if raw is not None else None


def read_record_document(path: Path, collection: str, warnings: list[str],
                         extra: tuple[str, ...] = ()) -> dict | None:
    no_generated_input(path)
    if not path.exists():
        warnings.append(f'{collection} record absent: {path}')
        return None
    raw = read_json(path)
    _require(isinstance(raw, dict) and type(raw.get('version')) is int and raw['version'] == 1,
             f'{path}: version must be 1')
    known_fields(raw, ('version', collection, *extra), str(path))
    entries = raw.get(collection)
    _require(isinstance(entries, list), f'{path}: {collection} must be a list')
    _require(all(isinstance(e, dict) for e in entries), f'{path}: records must be objects')
    return raw


def reference_list(entry: dict, key: str, label: str) -> tuple[str, ...]:
    values = entry.get(key)
    _require(isinstance(values, list) and all(isinstance(v, str) and v for v in values),
             f'{label}: {key} must be a list of ids')
    _require(len(values) == len(set(values)), f'{label}: duplicate {key} references')
    return tuple(values)


def unique_records(records, label: str) -> dict:
    result = {r.id: r for r in records}
    _require(len(result) == len(records), f'duplicate {label} ids')
    return result


def load_datasets(path: Path, warnings: list[str]) -> list[Dataset] | None:
    entries = read_records(path, 'datasets', warnings)
    if entries is None:
        return None
    datasets = []
    for entry in entries:
        id_ = dataset_id(entry.get('id'), 'dataset')
        known_fields(entry, tuple(Dataset.__dataclass_fields__), id_)
        name = required_text(entry, 'name', id_)
        access = required_text(entry, 'access', id_)
        _require(access in ACCESS_CLASSES, f'{id_}: unknown access class {access!r}')
        metadata = {key: nullable_text(entry, key, id_) for key in
                    ('provenance', 'version', 'licence', 'location', 'documentation', 'sampling_unit')}
        unit = metadata['sampling_unit']
        _require(unit is None or unit in SAMPLING_UNITS, f'{id_}: unknown sampling unit {unit!r}')
        sections = reference_list(entry, 'catalog_sections', id_)
        for section in sections:
            identifier(section, f'{id_} catalog section')
        datasets.append(Dataset(id_, name, access=access, catalog_sections=sections, **metadata))
    unique_records(datasets, 'dataset')
    return datasets


def validate_sample_datasets(samples: list[Sample], datasets: list[Dataset] | None,
                             warnings: list[str]) -> None:
    if datasets is None:
        warnings.append('sample/dataset cross-references unverified while a source is unavailable')
        return
    ids = {d.id for d in datasets}
    for sample in samples:
        _require(sample.dataset in ids, f'{sample.id}: unknown dataset {sample.dataset}')


def containing_repo(location: Path, repos: dict[Path, Repo]) -> Repo | None:
    """The deepest manifest repository holding a manuscript locator (a paper may live in a
    subdirectory). The umbrella holds every nested checkout, so it matches only exactly."""
    holders = [(len(path.parts), repo) for path, repo in repos.items()
               if location == path or (repo.group != 'umbrella' and location.is_relative_to(path))]
    return max(holders, key=lambda h: h[0])[1] if holders else None


def load_pubs(path: Path, manifest: Manifest, root: Path, warnings: list[str]) -> list[Manuscript]:
    no_generated_input(path)
    if not path.exists():
        warnings.append(f'pubs record absent: {path}')
        return []
    repos = {}
    for repo in manifest.repos:
        location = (root / repo.path).resolve()
        _require(location not in repos, f'ambiguous repository path: {location}')
        repos[location] = repo
    manuscripts = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        label = f'{path}:{number}'
        raw = strict_json(line, label)
        _require(isinstance(raw, dict), f'{label}: expected an object')
        if raw.get('program') != 'cognitive-mri':
            continue
        id_ = identifier(raw.get('id'), label)
        title = required_text(raw, 'title', label)
        derived = raw.get('derived')
        _require(isinstance(derived, dict), f'{label}: derived must be an object')
        status = required_text(derived, 'status', label)
        fields = [required_text(derived, key, label, empty=True) for key in ('venue', 'last_event', 'doi')]
        locations = raw.get('repos', {})
        _require(isinstance(locations, dict), f'{label}: repos must be an object')
        location = locations.get('manuscript')
        _require(location is None or isinstance(location, str), f'{label}: invalid manuscript locator')
        # Relative locators are anchored to the umbrella, never the caller's cwd.
        repo = containing_repo((root / Path(location).expanduser()).resolve(), repos) if location else None
        if repo is None:
            warnings.append(f'{id_}: pubs record unmatched to a manifest repository')
        deadline = None
        if raw.get('deadline') is not None:
            value = raw['deadline']
            _require(isinstance(value, dict), f'{label}: deadline must be an object')
            deadline = Deadline(iso_date(value.get('date'), f'{label} deadline.date'),
                                required_text(value, 'source', label),
                                iso_date(value.get('verified'), f'{label} deadline.verified'))
        check_on = None
        if raw.get('follow_up') is not None:
            value = raw['follow_up']
            _require(isinstance(value, dict), f'{label}: follow_up must be an object')
            if value.get('check_on') is not None:
                check_on = iso_date(value['check_on'], f'{label} follow_up.check_on')
        manuscripts.append(Manuscript(id_, title, status, *fields, repo, deadline, check_on))
    unique_records(manuscripts, 'manuscript')
    yaml_dir = path.parent / 'papers'
    no_generated_input(yaml_dir)
    for yaml in sorted(yaml_dir.glob('*.yaml')):
        no_generated_input(yaml)
        if yaml.stat().st_mtime_ns > path.stat().st_mtime_ns:
            warnings.append(f'pubs index stale: {yaml} is newer than {path}')
    return manuscripts


def load_program(manifest: Manifest, root: Path, include_pubs: bool = True) -> Program:
    warnings = []
    manuscripts = load_pubs(root / manifest.sources.pubs, manifest, root, warnings) if include_pubs else []
    raw_studies = read_records(root / manifest.sources.studies, 'studies', warnings)
    document = read_record_document(root / manifest.sources.questions, 'questions', warnings, ('lessons',))
    raw_questions = document['questions'] if document is not None else None
    datasets = load_datasets(root / manifest.sources.datasets, warnings)
    lessons = []
    if document is not None:
        entries = document.get('lessons')
        _require(isinstance(entries, list) and all(isinstance(e, dict) for e in entries),
                 'lessons must be a list of objects')
        for entry in entries:
            known_fields(entry, ('id', 'text'), 'lesson')
            lessons.append(Lesson(identifier(entry.get('id'), 'lesson'), required_text(entry, 'text', 'lesson')))
        unique_records(lessons, 'lesson')
    studies, questions = [], []
    for entry in raw_studies or []:
        id_ = identifier(entry.get('id'), 'study')
        known_fields(entry, tuple(Study.__dataclass_fields__), id_)
        _require(id_.startswith('S-') and len(id_) > 2, f'invalid study id: {id_}')
        title = required_text(entry, 'title', id_)
        state_ = required_text(entry, 'state', id_)
        _require(state_ in STUDY_STATES, f'{id_}: unknown study state {state_!r}')
        _require('repo' in entry and (entry['repo'] is None or isinstance(entry['repo'], str)),
                 f'{id_}: repo must be a repository name or null')
        repo = entry['repo']
        if repo is not None:
            manifest.repo(repo)
        _require('successor' in entry, f'{id_}: successor is required')
        successor = entry['successor']
        if state_ == 'folded':
            _require(isinstance(successor, str) and bool(successor), f'{id_}: folded study needs a successor')
        else:
            _require(successor is None, f'{id_}: successor must be null unless folded')
        path = entry.get('path')
        if 'path' in entry:
            _require(isinstance(path, str) and bool(path) and not Path(path).is_absolute()
                     and '..' not in Path(path).parts and 'by-stage' not in Path(path).parts,
                     f'{id_}: path must be a relative subpath')
            _require(repo is not None, f'{id_}: subpath requires a repository')
        note = required_text(entry, 'note', id_, empty=True) if 'note' in entry else None
        studies.append(Study(id_, title, state_, repo, path, reference_list(entry, 'questions', id_), successor, note))
    for entry in raw_questions or []:
        id_ = identifier(entry.get('id'), 'question')
        known_fields(entry, tuple(Question.__dataclass_fields__), id_)
        _require(id_.startswith('Q') and len(id_) > 1, f'invalid question id: {id_}')
        state_ = required_text(entry, 'state', id_)
        _require(state_ in QUESTION_STATES, f'{id_}: unknown question state {state_!r}')
        questions.append(Question(id_, required_text(entry, 'question', id_), state_,
                                  reference_list(entry, 'studies', id_),
                                  required_text(entry, 'known', id_, empty=True),
                                  required_text(entry, 'remaining', id_, empty=True),
                                  reference_list(entry, 'datasets', id_)))
    by_study = unique_records(studies, 'study')
    by_question = unique_records(questions, 'question')
    for study in studies:
        if study.successor is not None:
            _require(study.successor in by_study, f'{study.id}: unknown successor {study.successor}')
        if raw_questions is not None:
            for qid in study.questions:
                _require(qid in by_question, f'{study.id}: unknown question {qid}')
    if raw_studies is not None:
        for question in questions:
            for sid in question.studies:
                _require(sid in by_study, f'{question.id}: unknown study {sid}')
    if raw_studies is not None and raw_questions is not None:
        study_links = {(s.id, q) for s in studies for q in s.questions}
        question_links = {(s, q.id) for q in questions for s in q.studies}
        _require(study_links == question_links, 'study/question links must agree in both directions')
    if raw_studies is None or raw_questions is None:
        warnings.append('study/question cross-references unverified while a source is unavailable')
    by_dataset = {d.id: d for d in datasets or []}
    for question in questions:
        for did in question.datasets:
            dataset_id(did, question.id)
            if datasets is not None:
                _require(did in by_dataset, f'{question.id}: unknown dataset {did}')
    if datasets is None or raw_questions is None:
        warnings.append('question/dataset cross-references unverified while a source is unavailable')
    ledger = root / 'spent.json'
    no_generated_input(ledger)
    samples = load_ledger(ledger) if ledger.exists() else None
    if samples is None:
        warnings.append(f'spent record absent: {ledger}')
    else:
        validate_sample_datasets(samples, datasets, warnings)
    unavailable = tuple(name for name, records in [('studies', raw_studies), ('questions', raw_questions),
                                                  ('datasets', datasets), ('spent', samples)] if records is None)
    return Program(manuscripts, studies, questions, warnings, datasets or [], lessons, samples, unavailable)


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
    no_generated_input(path)
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


# ----------------------------------------------------------------------------- private board


def markdown_cell(value) -> str:
    return str(value).replace('\\', '\\\\').replace('|', '\\|').replace('\r', ' ').replace('\n', ' ')


def markdown_table(headers: tuple[str, ...], rows) -> list[str]:
    return ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|'] + [
        '| ' + ' | '.join(markdown_cell(value) for value in row) + ' |' for row in rows]


def deadline_text(deadline: Deadline | None, as_of: date) -> str:
    if deadline is None:
        return 'not recorded'
    return f'{deadline.date} ({(deadline.date - as_of).days} days left)'


def cmd_board(manifest: Manifest, root: Path, as_of: date | None = None,
              write: bool = False, out=sys.stdout) -> int:
    as_of = as_of or date.today()
    program = load_program(manifest, root)
    lines = [f'# Research program board ({as_of})', '', '## Manuscripts', '']
    groups = {}
    for paper in program.manuscripts:
        groups.setdefault(paper.status, []).append(paper)
    order = [s for s in MANUSCRIPT_STATUSES if s in groups] + sorted(set(groups) - set(MANUSCRIPT_STATUSES))
    for status_ in order:
        lines += [f'### {markdown_cell(status_)}', '']
        rows = [(p.title, p.venue, p.last_event, p.doi or 'not recorded',
                 p.repo.name if p.repo else 'unmatched', deadline_text(p.deadline, as_of),
                 str(p.check_on) if p.check_on else 'not recorded')
                for p in sorted(groups[status_], key=lambda p: p.id)]
        lines += markdown_table(('Title', 'Venue', 'Last event', 'DOI', 'Repository', 'Deadline', 'Follow-up check'), rows)
        lines.append('')
    if not groups:
        lines += ['No manuscript records available.', '']
    lines += ['## Studies', '']
    for state_ in STUDY_STATES:
        studies = sorted((s for s in program.studies if s.state == state_), key=lambda s: s.id)
        if studies:
            lines += [f'### {state_}', '']
            lines += markdown_table(('Id', 'Title', 'Repository', 'Questions'),
                                    [(s.id, s.title, s.repo or 'not recorded', ', '.join(s.questions) or 'none')
                                     for s in studies])
            lines.append('')
    if not program.studies:
        lines += ['No study records available.', '']
    lines += ['## Datasets', '']
    draws = dataset_draws(program)
    lines += markdown_table(('Id', 'Access class', 'Known draws'),
                            [(d.id, d.access, len(draws[d.id]) if program.samples is not None else 'unverified')
                             for d in program.datasets])
    if not program.datasets:
        lines += ['No dataset records available.']
    lines += ['', 'Draw counts describe the spent ledger only; remaining eligibility unverified.', '']
    lines += ['## Attention', '']
    attention = list(program.warnings)
    for paper in sorted(program.manuscripts, key=lambda p: p.id):
        if paper.deadline and 0 <= (paper.deadline.date - as_of).days <= 45:
            attention.append(f'{paper.id}: deadline {deadline_text(paper.deadline, as_of)}')
        if paper.check_on and paper.check_on <= as_of + timedelta(days=14):
            attention.append(f'{paper.id}: follow-up {paper.check_on}' + (' (overdue)' if paper.check_on < as_of else ''))
        if paper.repo and paper.status in MANUSCRIPT_STATUSES[:4]:
            hosted = [s for s in program.studies if s.repo == paper.repo.name]
            if hosted and all(s.state in ('stopped', 'folded') for s in hosted):
                attention.append(f'{paper.id}: {paper.repo.name} hosts only stopped or folded studies '
                                 f'while pubs status is {paper.status}')
    recorded = {p.repo.name for p in program.manuscripts if p.repo} | {s.repo for s in program.studies if s.repo}
    for repo in manifest.repos:
        if repo.group == 'paper' and repo.name not in recorded:
            attention.append(f'{repo.name}: paper repository has neither a pubs record nor a study')
    skipped = 0
    for entry in [*manifest.repos, *manifest.worktrees]:
        no_generated_input(root / entry.path)
        info = state(root, entry.path)
        if not info['present']:
            skipped += 1
            continue
        changes = []
        if info['dirty']:
            changes.append(f"{info['dirty']} uncommitted")
        if info['ahead']:
            changes.append(f"{info['ahead']} unpushed")
        if changes:
            attention.append(f'{entry.path}: ' + ', '.join(changes))
    lines += ['- ' + markdown_cell(item) for item in attention] or ['No attention items.']
    lines += ['', f'Skipped {skipped} missing checkouts.', '']
    text = '\n'.join(lines)
    if write:
        target = root / 'STATUS.local.md'
        _require(not target.is_symlink(), 'refusing to write STATUS.local.md through a symlink')
        target.write_text(text, encoding='utf-8')
    else:
        print(text, end='', file=out)
    return 0


# ----------------------------------------------------------------------------- questions ledger


def dataset_draws(program: Program) -> dict[str, list[Sample]]:
    draws = {d.id: [] for d in program.datasets}
    for sample in sorted(program.samples or [], key=lambda s: (s.spent_on, s.id)):
        if sample.dataset in draws:
            draws[sample.dataset].append(sample)
    return draws


def render_questions(program: Program) -> str:
    lines = ['# Where each programme question stands', '',
             'This ledger is generated by `python3 tools/workspace.py questions --write` and must not be edited by hand.',
             'Edit `questions.json`, `studies.json`, `datasets.json` and the umbrella `spent.json` instead.', '',
             '"Answered" means answered at the scope stated, not settled for all data.', '']
    if program.warnings:
        lines += ['## Source warnings', ''] + ['- ' + warning for warning in program.warnings] + ['']
    studies = {s.id: s for s in program.studies}
    datasets = {d.id: d for d in program.datasets}
    draws = dataset_draws(program)
    for question in program.questions:
        lines += [f'## {question.id}: {question.question}', '', f'State: {question.state}.', '',
                  f'What is known: {question.known}', '', f'What remains: {question.remaining}', '',
                  '### Studies', '']
        lines += markdown_table(('Id', 'Title', 'State'),
                                [(sid, studies[sid].title, studies[sid].state) if sid in studies
                                 else (sid, 'record unavailable', 'unverified') for sid in question.studies])
        if not question.studies:
            lines.append('No studies recorded.')
        lines += ['', '### Candidate datasets', '']
        for did in question.datasets:
            if did in datasets:
                exposure = f'{len(draws[did])} known draws' if program.samples is not None else 'draws unverified'
                lines.append(f'- [{did}](#{did}): {datasets[did].name} ({datasets[did].access}), {exposure}.')
            else:
                lines.append(f'- {did}: record unavailable; draws unverified.')
        if not question.datasets:
            lines.append('No candidate datasets recorded.')
        lines.append('')
    if not program.questions:
        lines += ['No question records available.', '']
    lines += ['## Dataset exposure', '',
              'Known draws come only from `spent.json`. Zero recorded draws does not mean unused data.',
              'Sizes retain the ledger labels at the time of each draw; they are not added across draws or units.', '']
    for dataset in program.datasets:
        lines += [f'<a id="{dataset.id}"></a>', '', f'### {dataset.id}: {dataset.name}', '']
        exposure = f'{len(draws[dataset.id])} known draws' if program.samples is not None else 'draws unverified'
        lines += [f'Access: {dataset.access}; {exposure}; remaining eligibility unverified.', '']
        if dataset.catalog_sections:
            lines += ['Catalog: ' + ', '.join(f'[{section}](public-datasets.md#{section})'
                                            for section in dataset.catalog_sections) + '.', '']
        if draws[dataset.id]:
            lines += markdown_table(('Draw', 'Spent on', 'State', 'Unit', 'Recorded sizes (at draw)'),
                                    [(s.id, s.spent_on, s.status, s.unit,
                                      ', '.join(f'{key}={value}' for key, value in s.size.items()) or 'unknown')
                                     for s in draws[dataset.id]]) + ['']
    lines += ['## Lessons that now bind every new design', '']
    lines += [f'{i}. {lesson.text}' for i, lesson in enumerate(program.lessons, 1)]
    return '\n'.join(lines) + '\n'


def cmd_questions(manifest: Manifest, root: Path, write: bool = False, check: bool = False, out=sys.stdout) -> int:
    _require(not (write and check), 'questions: --write and --check are mutually exclusive')
    program = load_program(manifest, root, include_pubs=False)
    target = (root / manifest.sources.questions).with_name('questions-ledger.md')
    no_generated_input(target)
    if write or check:
        _require(not target.is_symlink(), 'refusing questions-ledger.md through a symlink')
        _require(target.resolve().is_relative_to(root.resolve()), 'questions output must stay inside the workspace')
        if program.unavailable:
            message = 'questions: sources unavailable: ' + ', '.join(program.unavailable)
            if check:
                print(message, file=out)
                return 1
            raise ManifestError(message)
    text = render_questions(program)
    if check:
        if not target.exists() or target.read_text(encoding='utf-8') != text:
            print(f'stale: {target.relative_to(root)}', file=out)
            return 1
    elif write:
        target.write_text(text, encoding='utf-8')
    else:
        print(text, end='', file=out)
    return 0


# ----------------------------------------------------------------------------- disposable view

VIEW_MARKER = '.generated-by-workspace'


def status_slug(status: str) -> str:
    """A path component only; publication status itself is never rewritten."""
    return '-'.join(''.join(c.lower() if c.isalnum() else ' ' for c in status).split()) or 'unknown'


def view_links(manifest: Manifest, program: Program, root: Path, warnings: list[str]) -> dict[Path, Path]:
    links = {}

    def add(relative: Path, repo: Repo, subpath: str | None = None):
        if repo.path == '.' or (root / repo.path).resolve() == root.resolve():
            return
        target = root / repo.path
        if subpath:
            target /= subpath
        no_generated_input(target)
        _require(relative not in links, f'duplicate view link: {relative}')
        links[relative] = target

    for paper in program.manuscripts:
        if paper.repo:
            add(Path('manuscripts') / status_slug(paper.status) / paper.id, paper.repo)
    for study in program.studies:
        if study.repo:
            add(Path('studies') / study.state / study.id, manifest.repo(study.repo), study.path)
        else:
            warnings.append(f'{study.id}: no repository; no link created')
    for repo in manifest.repos:
        name = identifier(repo.name, 'repository')
        add(Path('repos') / repo.group / name, repo)
    return links


def inspect_view(view: Path) -> tuple[list[Path], set[Path]]:
    """The marker owns the subtree's links. Inventory it without following any link.

    Refuse every regular/special file except the root marker before removing anything.
    Empty directories are left intact. Link targets, including directory targets, are untouched.
    """
    _require(not view.is_symlink(), f'refusing symlink view root: {view}')
    if not view.exists():
        return [], set()
    marker = view / VIEW_MARKER
    _require(view.is_dir() and not marker.is_symlink() and marker.is_file(),
             f'{view}: missing regular {VIEW_MARKER} marker')
    links, directories = [], {view}

    def visit(directory: Path):
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if path == marker:
                    continue
                if entry.is_symlink():
                    links.append(path)
                elif entry.is_dir(follow_symlinks=False):
                    directories.add(path)
                    visit(path)
                else:
                    raise ManifestError(f'refusing non-symlink file inside by-stage: {path}')
    visit(view)
    return links, directories


def cmd_by_stage(manifest: Manifest, root: Path, out=sys.stdout) -> int:
    root = root.absolute()
    view = root / 'by-stage'
    old_links, directories = inspect_view(view)
    program = load_program(manifest, root)
    warnings = list(program.warnings)
    links = view_links(manifest, program, root, warnings)
    for relative in links:
        _require(view / relative not in directories, f'view link would replace a directory: {relative}')
    # Validation and the complete no-follow inventory precede all mutations.
    view.mkdir(exist_ok=True)
    (view / VIEW_MARKER).touch()
    for link in old_links:
        link.unlink()
    for relative, target in sorted(links.items()):
        link = view / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
        if not target.exists():
            warnings.append(f'{relative}: missing target {target}')
    for warning in warnings:
        print(warning, file=out)
    print(f'Created {len(links)} relative links in by-stage/.', file=out)
    return 0


# ------------------------------------------------------------------ spent samples


@dataclass(frozen=True)
class Sample:
    id: str
    pool: str
    status: str
    spent_on: str
    repo: str
    record: str
    excludes: tuple[str, ...]
    siblings: tuple[str, ...]
    evidence: tuple[dict, ...]
    dataset: str
    unit: str
    size: dict[str, int]
    protocol: str | None = None
    freeze: str | None = None


def load_ledger(path: Path = LEDGER) -> list[Sample]:
    """Parse and validate `spent.json`; every structural rule of the ledger lives here."""
    raw = read_json(path)
    _require(raw.get('version') == 1, 'ledger version must be 1')
    _require(isinstance(raw.get('samples'), list) and raw['samples'], 'ledger lacks samples')
    pools = raw.get('pools', {})
    samples = []
    for entry in raw['samples']:
        for key in ('id', 'pool', 'status', 'spent_on', 'repo', 'record'):
            _require(isinstance(entry.get(key), str) and entry[key], f'sample entry lacks {key}: {entry}')
        _require(entry['status'] in SAMPLE_STATUS, f"unknown sample status {entry['status']!r}")
        _require(entry['pool'] in pools, f"sample {entry['id']} names an undeclared pool {entry['pool']!r}")
        _require(re.fullmatch(r'\d{4}-\d{2}-\d{2}', entry['spent_on']) is not None,
                 f"sample {entry['id']}: spent_on must be an ISO date")
        for key in ('excludes', 'siblings', 'evidence'):
            _require(isinstance(entry.get(key, []), list), f"sample {entry['id']}: {key} must be a list")
        for item in entry.get('evidence', []):
            _require(all(isinstance(item.get(k), str) and item[k] for k in ('repo', 'path', 'contains')),
                     f"sample {entry['id']}: evidence needs repo, path and contains")
        did = dataset_id(entry.get('dataset'), f"sample {entry['id']} dataset")
        unit = required_text(entry, 'unit', entry['id'])
        _require(unit in SAMPLING_UNITS, f"sample {entry['id']}: unknown sampling unit {unit!r}")
        size = entry.get('size')
        _require(isinstance(size, dict) and all(isinstance(k, str) and k.strip() and type(v) is int and v >= 0
                                               for k, v in size.items()),
                 f"sample {entry['id']}: size must be an object of nonnegative integer counts")
        samples.append(Sample(entry['id'], entry['pool'], entry['status'], entry['spent_on'], entry['repo'],
                              entry['record'], tuple(entry.get('excludes', [])), tuple(entry.get('siblings', [])),
                              tuple(entry.get('evidence', [])), did, unit, size,
                              entry.get('protocol'), entry.get('freeze')))
    ids = [s.id for s in samples]
    _require(len(ids) == len(set(ids)), 'duplicate sample ids')
    for sample in samples:
        for other in sample.excludes + sample.siblings:
            _require(other in ids, f'sample {sample.id} names unknown sample {other}')
        for other in sample.siblings:
            _require(sample.id in next(s for s in samples if s.id == other).siblings,
                     f'siblings must be mutual: {sample.id} and {other}')
    return samples


def ledger_problems(samples: list[Sample], manifest: Manifest, root: Path) -> list[str]:
    """Every way the ledger can fail: an unknown repository, a missing record or protocol file, missing
    evidence text, or a draw that does not name every earlier draw from the same pool."""
    problems = []
    by_pool: dict[str, list[Sample]] = {}
    for sample in samples:
        by_pool.setdefault(sample.pool, []).append(sample)
    for pool, entries in by_pool.items():
        ordered = sorted(entries, key=lambda s: s.spent_on)
        for index, sample in enumerate(ordered):
            for earlier in ordered[:index]:
                if earlier.spent_on == sample.spent_on and earlier.id in sample.siblings:
                    continue
                if earlier.id not in sample.excludes and earlier.id not in sample.siblings:
                    problems.append(f'{sample.id}: does not exclude the earlier {pool} sample {earlier.id}')
    for sample in samples:
        try:
            where = root / manifest.repo(sample.repo).path
        except ManifestError:
            problems.append(f'{sample.id}: unknown repository {sample.repo}')
            continue
        if not is_checkout(where):
            continue  # a missing checkout is reported by `status`, not here
        for label, rel in (('record', sample.record), ('protocol', sample.protocol), ('freeze', sample.freeze)):
            if rel:
                no_generated_input(where / rel)
            if rel and not (where / rel).is_file():
                problems.append(f'{sample.id}: {label} {rel} is missing in {sample.repo}')
        for item in sample.evidence:
            try:
                target = root / manifest.repo(item['repo']).path / item['path']
            except ManifestError:
                problems.append(f"{sample.id}: evidence names unknown repository {item['repo']}")
                continue
            no_generated_input(target)
            if not target.is_file():
                problems.append(f"{sample.id}: evidence file {item['path']} is missing in {item['repo']}")
            elif item['contains'] not in target.read_text(encoding='utf-8', errors='replace'):
                problems.append(f"{sample.id}: evidence file {item['path']} in {item['repo']} lacks {item['contains']!r}")
    return problems


def cmd_spent(manifest: Manifest, root: Path, verify: bool = False, path: Path = LEDGER, out=sys.stdout) -> int:
    """List the ledger of spent samples; with --verify, check the exclusion chain and the evidence files."""
    samples = load_ledger(path)
    warnings = []
    datasets = load_datasets(root / manifest.sources.datasets, warnings)
    validate_sample_datasets(samples, datasets, warnings)
    for warning in warnings:
        print(warning, file=out)
    for sample in sorted(samples, key=lambda s: (s.pool, s.spent_on, s.id)):
        gate = sample.protocol or '-'
        print(f'{sample.pool:9s} {sample.spent_on} {sample.status:11s} {sample.id:30s} {sample.repo:28s} {gate}', file=out)
    if not verify:
        return 0
    problems = ledger_problems(samples, manifest, root)
    for problem in problems:
        print(f'problem: {problem}', file=out)
    print(f'{len(samples)} samples, {len(problems)} problems', file=out)
    return 1 if problems else 0


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
        no_generated_input(path)
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
    p = sub.add_parser('spent'); p.add_argument('--verify', action='store_true')
    sub.add_parser('by-stage')
    p = sub.add_parser('board'); p.add_argument('--as-of'); p.add_argument('--write', action='store_true')
    p = sub.add_parser('questions')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--write', action='store_true'); mode.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    manifest_path = manifest_path or root / 'workspace.json'
    try:
        manifest = load(manifest_path)
        if args.command == 'questions':
            return cmd_questions(manifest, root, args.write, args.check, out=sys.stdout)
        if args.command == 'by-stage':
            return cmd_by_stage(manifest, root, out=sys.stdout)
        if args.command == 'board':
            return cmd_board(manifest, root, iso_date(args.as_of, '--as-of') if args.as_of else None,
                             args.write, out=sys.stdout)
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
        if args.command == 'spent':
            return cmd_spent(manifest, root, args.verify, root / 'spent.json')
        return cmd_render(manifest, root, args.check)
    except ManifestError as error:
        print(f'workspace.json: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
