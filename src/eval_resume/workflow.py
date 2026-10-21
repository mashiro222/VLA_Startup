"""Freeze completed journals and submit one allocation for the remaining episodes."""
import argparse
import hashlib
import json
from pathlib import Path

from arena_startup.core import ROOT, collect, file_hash, key, records, save, validate_rows
from sf_followup import evaluation
from sf_followup.control import command, submit_once


def code_files():
    paths = list((ROOT / 'src/eval_resume').glob('*.py'))
    paths += [ROOT / 'slurm/eval_resume.sbatch', ROOT / 'src/sf_recovery/stage.py']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def snapshot(run, plan):
    prefixes, pending = {}, []
    for unit in plan['units']:
        path = run / 'units' / unit['id'] / 'episodes.jsonl'
        rows = records(path)
        validate_rows(rows, unit, plan)
        data = path.read_bytes() if path.exists() else b''
        # Preserve complete durable records; an interrupted final write is not scored.
        size = sum(len(line) for line in data.splitlines(keepends=True)[:len(rows)])
        prefixes[unit['id']] = dict(count=len(rows), bytes=size,
                                    sha256=hashlib.sha256(data[:size]).hexdigest())
        pending.extend(key(unit, i) for i in range(len(rows), unit['trials']))
    return prefixes, pending


def verify_prefixes(run, plan, spec):
    """Allow appends but reject any change to a previously scored episode."""
    for unit in plan['units']:
        before = spec['prefixes'][unit['id']]
        path = run / 'units' / unit['id'] / 'episodes.jsonl'
        if path.exists():
            with path.open('rb') as f:
                data = f.read(before['bytes'])
        else:
            data = b''
        if len(data) != before['bytes'] or hashlib.sha256(data).hexdigest() != before['sha256']:
            raise ValueError('Completed journal prefix changed: ' + unit['id'])
    rows = collect(run, plan)
    expected = sum(x['count'] for x in spec['prefixes'].values())
    if not expected <= len(rows) <= plan['expected_episodes']:
        raise ValueError('Unexpected completed episode count')
    return len(rows)


def prepare(run, directory, expected_remaining):
    run, directory = Path(run).resolve(), Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    plan = json.loads((run / 'plan.json').read_text())
    evaluation.validate(plan)
    prefixes, pending = snapshot(run, plan)
    if len(pending) != expected_remaining or not pending:
        raise ValueError('Remaining episode count differs from requested retry')
    metadata = json.loads((run / 'workers/policy-14199187.json').read_text())
    if any(metadata[k] != v for k, v in evaluation.expected_metadata(plan).items()):
        raise ValueError('Previous policy identity differs')
    conversion = ROOT / '.runtime/training/pi05_base_torch/conversion.json'
    spec = dict(schema=1, run=str(run), plan_file_sha256=file_hash(run / 'plan.json'),
                plan_sha256=plan['sha256'], checkpoint_sha256=metadata['checkpoint_sha256'],
                base_sha256=json.loads(conversion.read_text())['sha256'],
                completed_before=plan['expected_episodes']-len(pending), remaining=len(pending),
                prefixes=prefixes, pending_keys=pending, max_attempts=3,
                automatic_slurm_resubmission=False, files=code_files())
    save(directory / 'resume.json', spec)
    return spec


def check(path):
    spec = json.loads(Path(path).read_text())
    run = Path(spec['run'])
    if spec['files'] != code_files() or file_hash(run / 'plan.json') != spec['plan_file_sha256']:
        raise ValueError('Frozen recovery code/plan changed')
    plan = json.loads((run / 'plan.json').read_text())
    evaluation.validate(plan)
    verify_prefixes(run, plan, spec)
    return spec


def submit(path, execute=False):
    path = Path(path).resolve()
    spec = check(path)
    plan = json.loads((Path(spec['run']) / 'plan.json').read_text())
    if verify_prefixes(Path(spec['run']), plan, spec) == plan['expected_episodes']:
        raise ValueError('Already complete; no job needed')
    cmd = command('baseline-5952-eval-resume', 'slurm/eval_resume.sbatch', [path], gpus=1, hours=24)
    import shlex
    print(shlex.join(cmd), flush=True)
    if execute:
        return submit_once(path.parent, 'eval', cmd)
    print('Preview only; no job submitted')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='action', required=True)
    q = s.add_parser('prepare'); q.add_argument('run'); q.add_argument('directory')
    q.add_argument('--expected-remaining', type=int, required=True)
    q = s.add_parser('check'); q.add_argument('spec')
    q = s.add_parser('submit'); q.add_argument('spec'); q.add_argument('--submit', action='store_true')
    a = p.parse_args()
    if a.action == 'prepare':
        result = prepare(a.run, a.directory, a.expected_remaining)
        print(f"Preserved {result['completed_before']}; remaining {result['remaining']}")
    elif a.action == 'submit': submit(a.spec, a.submit)
    else: check(a.spec); print('Recovery inputs and completed prefixes verified')
