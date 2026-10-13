#!/usr/bin/env python3
"""Explicitly recorded full evaluation without waiting for a separate smoke job.

The original frozen evaluator and smoke job remain unchanged. This entry point
retains plan/file validation, submission locking and duplicate-job detection.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arena
from arena_startup.core import ROOT, claim, collect, digest, file_hash, run_path, save
from arena_startup.runtime import supervise

LAUNCH_FILES = ('scripts/arena_direct.py', 'slurm/arena-direct.sbatch')


def launch_files():
    return {name: file_hash(ROOT / name) for name in LAUNCH_FILES}


def load(run):
    plan = arena.load(run)
    record = json.loads((run / 'direct-launch.json').read_text())
    if record['sha256'] != digest({k: v for k, v in record.items() if k != 'sha256'}):
        raise ValueError('Direct-launch checksum mismatch')
    if (plan['profile'] != 'full' or record['run_id'] != plan['run_id']
            or record['plan_sha256'] != plan['sha256'] or record['files'] != launch_files()
            or record.get('smoke_requirement') != 'waived_by_user' or not record.get('reason')):
        raise ValueError('Direct-launch record does not match the reviewed full plan and launcher')
    return plan, record


def prepare(args):
    arena.preflight()
    run = run_path(args.run_id)
    plan = arena.load(run)
    if plan['profile'] != 'full':
        raise ValueError('This separate entry point is only for the full profile')
    if (run / 'direct-launch.json').exists():
        load(run)
        print('Existing matching direct-launch record retained; no job submitted')
        return
    if collect(run, plan):
        raise ValueError('Cannot introduce a smoke waiver after evaluation has begun')
    record = {'schema': 1, 'run_id': plan['run_id'], 'plan_sha256': plan['sha256'],
              'smoke_requirement': 'waived_by_user', 'reason': args.reason,
              'gpu_inference_and_egl_verified': False, 'files': launch_files(),
              'prepared_unix': time.time()}
    record['sha256'] = digest(record)
    save(run / 'direct-launch.json', record)
    print(f'Prepared direct full launch: {plan["expected_episodes"]} episodes, '
          f'up to {plan["workers"]} independent GPUs. No job submitted.')


def command(args, plan):
    limit = args.time_limit or plan['time_limit']
    match = re.fullmatch(r'(\d{1,2}):([0-5]\d):([0-5]\d)', limit)
    seconds = sum(int(v) * factor for v, factor in zip(match.groups(), (3600, 60, 1))) if match else 0
    if not 0 < seconds <= 72 * 3600:
        raise ValueError('Use a positive HH:MM:SS time limit, at most 72:00:00')
    if args.gpu not in ('a100', 'h100') or not 1 <= plan['workers'] <= 4:
        raise ValueError('Use 1–4 single-GPU A100 or H100 workers')
    return ['sbatch', '--parsable', '--account=gts-jli741-paid', f'--partition=gpu-{args.gpu}',
            '--qos=inferno', f'--gres=gpu:{args.gpu}:1', '--cpus-per-task=8', '--mem=64G',
            f'--time={limit}', f'--array=0-{plan["workers"]-1}%{plan["workers"]}',
            f'--job-name=pi05-{args.run_id}', f'--chdir={ROOT}',
            f'--output={ROOT}/logs/{args.run_id}-%A_%a.out',
            str(ROOT / 'slurm/arena-direct.sbatch'), args.run_id]


def submit(args):
    arena.preflight()
    run = run_path(args.run_id)
    plan, record = load(run)
    if len(collect(run, plan)) == plan['expected_episodes']:
        print('Already complete; no job to submit')
        return
    cmd = command(args, plan)
    if args.test_only:
        cmd.insert(1, '--test-only')
    print(shlex.join(cmd), flush=True)
    print('Separate smoke completion was waived by the user; GPU/EGL execution is unverified.', flush=True)
    if args.test_only:
        subprocess.run(cmd, check=True, timeout=60)
        return
    if not args.submit:
        print('Dry run only; no Slurm request sent. Choose a GPU before adding --submit.')
        return
    with claim(run / 'submission') as owned:
        if not owned:
            raise RuntimeError('Another submission command is active')
        unknown = run / 'submission/unknown.json'
        if unknown.exists():
            raise RuntimeError('Prior submission result unknown; inspect squeue before clearing unknown.json')
        active = subprocess.check_output(
            ['squeue', '-h', '-u', str(os.getuid()), '-n', f'pi05-{args.run_id}', '-o', '%i'],
            text=True, timeout=30)
        if active.strip():
            raise RuntimeError(f'Already queued/running: {active.strip()}')
        (ROOT / 'logs').mkdir(exist_ok=True)
        metadata = {'command': cmd, 'time': time.time(), 'direct_launch_sha256': record['sha256'],
                    'smoke_requirement': record['smoke_requirement']}
        save(unknown, metadata)
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
        print(result.stdout, result.stderr, end='', flush=True)
        result.check_returncode()
        job_id = result.stdout.strip().split(';')[0]
        if not job_id.isdigit():
            raise RuntimeError('Unrecognized sbatch reply; inspect squeue before retrying')
        save(run / 'submission' / f'{job_id}.json', dict(metadata, job_id=job_id))
        unknown.unlink()
        print(f'Submitted {job_id}; logs/{args.run_id}-{job_id}_<worker>.out')


def worker(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU evaluation requires a Slurm allocation')
    arena.preflight()
    run = run_path(args.run_id)
    plan, record = load(run)
    worker_id = f'{os.environ["SLURM_JOB_ID"]}-{os.environ.get("SLURM_ARRAY_TASK_ID", "0")}'
    # The existing report builder copies workers/*.json into report provenance.
    save(run / 'workers' / f'{worker_id}-direct-launch.json', record)
    print(f'Direct full worker: {plan["expected_episodes"]} episodes; smoke waived: {record["reason"]}', flush=True)
    return supervise(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('prepare'); p.add_argument('run_id'); p.add_argument('--reason', required=True)
    p = subs.add_parser('submit'); p.add_argument('run_id'); p.add_argument('--gpu', choices=['a100', 'h100'], required=True)
    p.add_argument('--time-limit')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--submit', action='store_true'); mode.add_argument('--test-only', action='store_true')
    p = subs.add_parser('worker'); p.add_argument('run_id')
    args = parser.parse_args()
    return {'prepare': prepare, 'submit': submit, 'worker': worker}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
