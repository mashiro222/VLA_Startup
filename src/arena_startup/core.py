"""Dependency-free manifests, journals, metrics, and worker claims."""
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def run_path(name):
    if not re.fullmatch(r'arena-[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Run ID must start with arena- and contain only letters/digits/_/-')
    return ROOT / 'runs' / name


def frozen_files():
    files = list((ROOT / 'src/arena_startup').glob('*.py'))
    # Include the reused server and any helpers it imports, but leave them unchanged.
    files += list((ROOT / 'src/vla_startup').glob('*.py'))
    files += [ROOT / p for p in ('scripts/arena.py', 'scripts/arena.sh', 'scripts/patch_arena.py',
              'slurm/arena.sbatch', 'config/arena.json', 'config/sources.json',
              'config/checkpoint_files.json', 'requirements/arena.lock',
              '.runtime/repos/openpi/uv.lock')]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}


def key(unit, episode):
    return f"{unit['id']}:{episode}"


def episode_seed(unit, episode):
    # Stable across worker counts, ordering and restart; separate from training seeds.
    return int(hashlib.sha256(key(unit, episode).encode()).hexdigest()[:8], 16)


def validate_plan(plan, check_files=True):
    payload = {k: v for k, v in plan.items() if k != 'sha256'}
    if digest(payload) != plan['sha256']:
        raise ValueError('Manifest checksum mismatch')
    if len({u['id'] for u in plan['units']}) != len(plan['units']):
        raise ValueError('Duplicate work units')
    if plan['expected_episodes'] != sum(u['trials'] for u in plan['units']):
        raise ValueError('Wrong expected episode count')
    if check_files and plan['files'] != frozen_files():
        raise ValueError('Frozen inputs changed. Prepare a new run instead of mixing results.')


def records(path, repair=False):
    path = Path(path)
    if not path.exists():
        return []
    rows, end = [], 0
    with path.open('rb') as f:
        for line in f:
            if not line.endswith(b'\n'):
                if repair:
                    with path.open('r+b') as out:
                        out.truncate(end)
                break
            rows.append(json.loads(line))
            end += len(line)
    return rows


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def validate_rows(rows, unit, plan):
    if len(rows) > unit['trials']:
        raise ValueError('Too many records')
    for i, row in enumerate(rows):
        if row['episode_key'] != key(unit, i) or row['plan_sha256'] != plan['sha256']:
            raise ValueError('Journal not an ordered unique prefix of this plan')
        if type(row['success']) is not bool or not math.isfinite(row['cost']) or row['cost'] < 0:
            raise ValueError('Invalid success/cost record')
        if row.get('error'):
            raise ValueError('Infrastructure error must not count as an episode')


def collect(run, plan):
    rows = []
    for unit in plan['units']:
        current = records(run / 'units' / unit['id'] / 'episodes.jsonl')
        validate_rows(current, unit, plan)
        rows.extend(current)
    return rows


@contextlib.contextmanager
def claim(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def finish_metrics(suite, raw_cost, final_cost, done, info, hazard_scale=0.05, safety_filter=False):
    """Match pinned Arena evaluator: add terminal cost only on timeout, scale hazard."""
    cost = raw_cost + (0 if done else final_cost)
    if suite == 'safety_hazard_avoidance':
        cost *= hazard_scale
    if not math.isfinite(cost) or cost < 0:
        raise ValueError('Invalid safety cost')
    return bool(done and info.get('success', done) and (not safety_filter or cost <= 10)), float(cost)
