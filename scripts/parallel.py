#!/usr/bin/env python3
"""Prepare, inspect, execute, and merge independent one-GPU evaluation shards."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vla_startup.journal import episode_keys, read_records, validate_prefix
from vla_startup.manifest import build, digest, validate

STRATEGY = 'whole tasks, greedy horizon balance, canonical order within each shard'
RNG_NOTE = ('Each shard starts the policy RNG at [0, 0], as in the serial evaluator. '
            'Its stream is then saved and resumed independently. This is not the '
            'same continuous random stream as a single serial full run.')


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError('Invalid run/batch ID')
    return value


def signed(value, field='manifest_sha256'):
    value = copy.deepcopy(value)
    value.pop(field, None)
    value[field] = digest(value)
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing plan differs; use a new batch ID: {path}')
    else:
        write_json(path, value)


def split_manifest(base, count):
    validate(base)
    if not 1 <= count <= len(base['tasks']):
        raise ValueError('Shard count must be between 1 and the number of tasks')
    aggregate = signed({**base, 'parallel_aggregate': {
        'shards': count, 'strategy': STRATEGY, 'rng_note': RNG_NOTE,
        'report_only': True}})
    buckets, loads = [[] for _ in range(count)], [0] * count
    # Retain every trial of a task on one worker: no repeated environment setup.
    for index in sorted(range(len(base['tasks'])),
                        key=lambda i: (-base['tasks'][i]['max_steps'], i)):
        worker = min(range(count), key=lambda i: (loads[i], i))
        buckets[worker].append(index)
        loads[worker] += base['tasks'][index]['max_steps'] * base['trials_per_task']
    shards = []
    for index, indices in enumerate(buckets):
        tasks = [base['tasks'][i] for i in sorted(indices)]
        shard = signed({**base, 'profile': 'shard-' + base['profile'],
                        'selection': 'parallel shard', 'tasks': tasks,
                        'expected_episodes': len(tasks) * base['trials_per_task'],
                        'parallel_shard': {'index': index, 'count': count,
                            'parent_manifest_sha256': aggregate['manifest_sha256'],
                            'rng_note': RNG_NOTE}})
        validate(shard)
        shards.append(shard)
    keys = [key for shard in shards for key in episode_keys(shard)]
    if len(keys) != len(set(keys)) or set(keys) != set(episode_keys(base)):
        raise ValueError('Partition lost or duplicated episodes')
    return aggregate, shards


def frozen_inputs():
    paths = list((ROOT / 'src/vla_startup').glob('*.py')) + [
        ROOT / name for name in (
            'scripts/parallel.py', 'scripts/run_node.py', 'scripts/patch_sim.py',
            'scripts/report.py', 'slurm/parallel.sbatch', 'scripts/common.sh',
            'config/evaluation.json', 'config/sources.json', 'config/checkpoint_files.json',
            'config/imagemagick.json', 'requirements/sim.lock',
            '.runtime/repos/openpi/uv.lock')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def prepare(batch_id, count):
    identifier(batch_id)
    if not 1 <= count <= 4:
        raise ValueError('This overnight workflow is capped at 4 simultaneous GPUs')
    plan = {'schema_version': 1, 'batch_id': batch_id, 'workers': count,
            'strategy': STRATEGY, 'rng_note': RNG_NOTE,
            'frozen_inputs': frozen_inputs(), 'benchmarks': {}}
    for mode in ('clean', 'plus'):
        base = build(ROOT, 'full-' + mode)
        parent, shards = split_manifest(base, count)
        parent_id = f'{batch_id}-full-{mode}'
        immutable_json(ROOT / 'runs' / parent_id / 'manifest.json', parent)
        children = []
        for i, shard in enumerate(shards):
            run_id = f'{parent_id}-shard-{i:02d}'
            immutable_json(ROOT / 'runs' / run_id / 'manifest.json', shard)
            children.append(run_id)
        plan['benchmarks'][mode] = {'parent': parent_id, 'children': children,
                                    'manifest_sha256': parent['manifest_sha256']}
    plan = signed(plan, 'plan_sha256')
    immutable_json(ROOT / 'runs' / batch_id / 'batch.json', plan)
    print(f'Prepared {count} workers, each using one GPU: clean shard then plus shard.')
    status(plan)


def load_plan(batch_id, check_inputs=False):
    identifier(batch_id)
    plan = json.loads((ROOT / 'runs' / batch_id / 'batch.json').read_text())
    if signed(plan, 'plan_sha256') != plan or plan['batch_id'] != batch_id:
        raise ValueError('Batch checksum or ID mismatch')
    if not 1 <= plan['workers'] <= 4:
        raise ValueError('Invalid worker count')
    if check_inputs and plan['frozen_inputs'] != frozen_inputs():
        raise ValueError('Code/configuration changed since preparation; restore it or use a new batch ID')
    if set(plan['benchmarks']) != {'clean', 'plus'}:
        raise ValueError('Batch must contain clean and plus')
    for mode, entry in plan['benchmarks'].items():
        expected_parent = f'{batch_id}-full-{mode}'
        expected_children = [f'{expected_parent}-shard-{i:02d}' for i in range(plan['workers'])]
        if entry['parent'] != expected_parent or entry['children'] != expected_children:
            raise ValueError('Unexpected run paths in batch')
        parent = json.loads((ROOT / 'runs' / entry['parent'] / 'manifest.json').read_text())
        validate(parent)
        if parent['manifest_sha256'] != entry['manifest_sha256']:
            raise ValueError('Parent manifest changed')
        base = signed({k: v for k, v in parent.items() if k != 'parallel_aggregate'})
        regenerated_parent, children = split_manifest(base, plan['workers'])
        if regenerated_parent != parent:
            raise ValueError('Unexpected aggregate metadata')
        for name, expected in zip(entry['children'], children):
            if json.loads((ROOT / 'runs' / name / 'manifest.json').read_text()) != expected:
                raise ValueError(f'Shard manifest changed: {name}')
    return plan


def inspect_run(run_id):
    run = ROOT / 'runs' / identifier(run_id)
    manifest = json.loads((run / 'manifest.json').read_text())
    validate(manifest)
    records = read_records(run / 'episodes.jsonl')
    settings_path = run / 'run_config.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else None
    if records and settings is None:
        raise ValueError(f'Missing run configuration: {run_id}')
    if settings and settings['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError(f'Run configuration does not match manifest: {run_id}')
    validate_prefix(records, manifest, digest(settings) if settings else None)
    return manifest, records, settings


def require_smoke(run_id, profile):
    manifest, records, _ = inspect_run(run_id)
    if manifest['profile'] != profile or len(records) != manifest['expected_episodes']:
        raise ValueError(f'{profile} must have all episodes completed first: {run_id}')
    if manifest != build(ROOT, profile, manifest['seed'], manifest['prompt_mode']):
        raise ValueError(f'{profile} configuration differs from the current experiment')
    errors = ROOT / 'runs' / run_id / 'errors.jsonl'
    if errors.exists() and errors.stat().st_size:
        raise ValueError(f'Review infrastructure errors before full submission: {errors}')
    print(f'{profile}: {sum(r["success"] for r in records)}/{len(records)} successes; '
          'infrastructure gate passed (task failure is a valid evaluation outcome).')


def status(plan):
    print('run_id | completed/expected | successes')
    for entry in plan['benchmarks'].values():
        completed = expected = successes = 0
        for run_id in entry['children']:
            manifest, records, _ = inspect_run(run_id)
            passed = sum(r['success'] for r in records)
            print(f'{run_id} | {len(records)}/{manifest["expected_episodes"]} | {passed}')
            completed += len(records)
            expected += manifest['expected_episodes']
            successes += passed
        print(f'TOTAL {entry["parent"]} | {completed}/{expected} | {successes}')


def merge_records(parent, shards):
    """Validate full coverage and configurations before constructing an aggregate."""
    validate(parent)
    by_key, configs, protocol = {}, {}, None
    expected = set(episode_keys(parent))
    for run_id, manifest, records, settings in shards:
        validate(manifest)
        if manifest['parallel_shard']['parent_manifest_sha256'] != parent['manifest_sha256']:
            raise ValueError('Shard belongs to a different aggregate')
        if len(records) != manifest['expected_episodes'] or settings is None:
            raise ValueError(f'Incomplete shard: {run_id}; resume it before merging')
        if settings['manifest_sha256'] != manifest['manifest_sha256']:
            raise ValueError('Run config belongs to a different shard')
        validate_prefix(records, manifest, digest(settings))
        common = {k: v for k, v in settings.items() if k != 'manifest_sha256'}
        if protocol is not None and common != protocol:
            raise ValueError('Shards used different model/code/protocol settings')
        protocol = common
        configs[run_id] = settings
        rows = {(t['suite'], t['task_id']): t for t in manifest['tasks']}
        for record in records:
            key = record['episode_key']
            if key not in expected or key in by_key:
                raise ValueError('Duplicate or unexpected episode in merge')
            row = rows[(record['suite'], record['task_id'])]
            if any(record.get(k) != v for k, v in row.items()):
                raise ValueError('Episode task metadata differs from manifest')
            by_key[key] = {**record, 'source_run_id': run_id,
                           'source_manifest_sha256': record['manifest_sha256'],
                           'source_run_fingerprint': record['run_fingerprint']}
    if set(by_key) != expected:
        raise ValueError('Merged coverage differs from the full benchmark')
    settings = {'execution': 'parallel_aggregate_report_only',
                'manifest_sha256': parent['manifest_sha256'], 'rng_note': RNG_NOTE,
                'shard_run_configs': configs}
    fingerprint = digest(settings)
    ordered = [{**by_key[key], 'manifest_sha256': parent['manifest_sha256'],
                'run_fingerprint': fingerprint} for key in episode_keys(parent)]
    validate_prefix(ordered, parent, fingerprint)
    return ordered, settings


def merge(plan):
    for entry in plan['benchmarks'].values():
        run = ROOT / 'runs' / entry['parent']
        parent = json.loads((run / 'manifest.json').read_text())
        shard_data = [(name, *inspect_run(name)) for name in entry['children']]
        records, settings = merge_records(parent, shard_data)
        with (run / '.job.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write_json(run / 'run_config.json', settings)
            temporary = run / 'episodes.jsonl.tmp'
            with temporary.open('w') as output:
                for record in records:
                    output.write(json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n')
            temporary.replace(run / 'episodes.jsonl')
            write_json(run / 'parallel_execution.json', {
                'batch': plan['batch_id'], 'rng_note': RNG_NOTE,
                'shards': entry['children'], 'plan_sha256': plan['plan_sha256'],
                'frozen_inputs': plan['frozen_inputs']})
            # Keep original per-shard GPU/job provenance with unambiguous names.
            for name in entry['children']:
                for source in (ROOT / 'runs' / name).glob('provenance-*.json'):
                    (run / f'provenance-{name}-{source.name[11:]}').write_bytes(source.read_bytes())
            subprocess.run(['bash', str(ROOT / 'scripts/report.sh'), run.name], check=True)
        print(f'Merged {len(records)} episodes: reports/{run.name}/summary.md')


def worker(plan, index):
    if not os.environ.get('SLURM_JOB_ID') or not 0 <= index < plan['workers']:
        raise ValueError('Worker requires a Slurm allocation and a valid array index')
    active = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for mode in ('clean', 'plus'):
        if stopping:
            return 143
        run_id = plan['benchmarks'][mode]['children'][index]
        print(f'Worker {index}: starting {run_id}', flush=True)
        active = subprocess.Popen([sys.executable, str(ROOT / 'scripts/run_node.py'), run_id])
        code = active.wait()
        if code or stopping:
            return code if code > 0 else 143
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check', 'status', 'merge', 'worker', 'remaining'])
    parser.add_argument('batch_id')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--index', type=int)
    parser.add_argument('--require-smoke', action='store_true')
    parser.add_argument('--clean-smoke', default='smoke-clean-001')
    parser.add_argument('--plus-smoke', default='smoke-plus-001')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.batch_id, args.workers)
        return 0
    plan = load_plan(args.batch_id, check_inputs=args.command in ('check', 'worker'))
    if args.require_smoke:
        require_smoke(args.clean_smoke, 'smoke-clean')
        require_smoke(args.plus_smoke, 'smoke-plus')
    if args.command == 'check':
        print(f'Validated {plan["workers"]} workers and locked code/configuration.')
    elif args.command == 'status':
        status(plan)
    elif args.command == 'merge':
        merge(plan)
    elif args.command == 'remaining':
        pending = []
        for index in range(plan['workers']):
            for entry in plan['benchmarks'].values():
                manifest, records, _ = inspect_run(entry['children'][index])
                if len(records) < manifest['expected_episodes']:
                    pending.append(str(index))
                    break
        print(','.join(pending))
    else:
        if args.index is None:
            parser.error('--index is required for worker')
        return worker(plan, args.index)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
