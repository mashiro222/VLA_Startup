#!/usr/bin/env python3
"""Prepare/inspect/submit Arena evaluation. Submission is opt-in via --submit."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from arena_startup.core import collect, digest, file_hash, frozen_files, run_path, save, validate_plan, claim
from arena_startup.runtime import environment, supervise


def catalog():
    with tempfile.TemporaryDirectory(prefix='arena-catalog-') as d:
        path = Path(d) / 'catalog.json'
        subprocess.run([str(ROOT / '.runtime/arena/bin/python'), '-m', 'arena_startup.catalog', str(path)],
                       env=environment(), check=True)
        return json.loads(path.read_text())


def preflight(imports=False):
    cfg = json.loads((ROOT / 'config/arena.json').read_text())
    sources = json.loads((ROOT / 'config/sources.json').read_text())
    for name, commit in [('vla_arena', cfg['upstream']['commit']), ('openpi', sources['openpi']['commit'])]:
        repo = ROOT / '.runtime/repos' / name
        actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != commit:
            raise ValueError(f'Wrong upstream commit for {name}')
        subprocess.run(['git', '-C', str(repo), 'diff', '--quiet', 'HEAD', '--'], check=True)
    for rel in ['.runtime/arena/bin/python', '.runtime/server/bin/python', 'requirements/arena.lock',
                '.runtime/config/arena/config.yaml',
                '.runtime/cache/openpi/big_vision/paligemma_tokenizer.model',
                '.runtime/native-7.1.1/squashfs-root/usr/lib/libMagickWand-7.Q16HDRI.so']:
        if not (ROOT / rel).is_file():
            raise FileNotFoundError(rel)
    checkpoint = ROOT / cfg['checkpoint']
    if cfg['checkpoint'] != '.runtime/checkpoints/pi05_libero' or cfg['policy_config'] != 'pi05_libero':
        raise ValueError('Only the existing pi05_libero checkpoint is authorized by this plan')
    metadata = json.loads((ROOT / 'config/checkpoint_files.json').read_text())
    if json.loads((checkpoint / '.verified.json').read_text()) != metadata:
        raise ValueError('Checkpoint verification marker does not match')
    for row in metadata:
        p = checkpoint / row['name'].split('checkpoints/pi05_libero/', 1)[1]
        if p.stat().st_size != int(row['size']):
            raise ValueError(f'Checkpoint file size mismatch: {p}')
    if imports:
        rows = catalog()
        if len(rows) != 75:
            raise ValueError('Safety catalog must contain 75 tasks')
        subprocess.run([str(ROOT / '.runtime/arena/bin/python'), '-c',
                        'import mujoco; import robosuite; from vla_arena.vla_arena.envs import OffScreenRenderEnv; '
                        'from openpi_client import image_tools; from wand.image import Image; '
                        'from vla_arena.vla_arena.utils.eval_cost import get_timeout_final_cost; '
                        'from vla_arena.vla_arena.utils.eval_init_state import select_init_state_index; '
                        'print("Arena imports OK; simulator and policy were not constructed")'],
                       check=True, env=environment())
        print('All 75 task definitions and initial-state files loaded')
    print('PASS: local preparation checks; GPU policy/EGL smoke is still required')


def load(run):
    plan = json.loads((run / 'plan.json').read_text())
    validate_plan(plan)
    return plan


def prepare(args):
    preflight()
    cfg = json.loads((ROOT / 'config/arena.json').read_text())
    profile = cfg['profiles'][args.profile]
    proto = cfg['evaluation']
    if proto['seed_mode'] != 'sha256-per-episode-v1':
        raise ValueError('Only the recorded per-episode seed protocol is implemented')
    if not 1 <= proto['replan_steps'] <= 10 or min(proto['num_steps'], proto['max_steps'], profile['trials']) < 1:
        raise ValueError('Invalid inference horizon/steps or trial count')
    workers = args.workers or profile['workers']
    if not 1 <= workers <= 4:
        raise ValueError('Use 1 to 4 independent one-GPU workers')
    run = run_path(args.run_id)
    tasks = catalog()
    units = []
    for task in tasks:
        if task['task_id'] not in profile['task_ids']:
            continue
        for seed in profile['seeds']:
            units.append({**task, 'seed': seed, 'trials': profile['trials'],
                          'id': f"{task['suite']}-L{task['level']}-t{task['task_id']}-s{seed}"})
    plan = {'schema': 1, 'run_id': args.run_id, 'profile': args.profile, 'workers': workers,
            'time_limit': profile['time_limit'], 'videos_per_worker': profile['videos_per_worker'],
            'config': cfg, 'units': units, 'expected_episodes': sum(u['trials'] for u in units),
            'training_condition': 'zero-shot target transfer from fixed LIBERO-finetuned pi05_libero',
            'files': frozen_files()}
    plan['sha256'] = digest(plan)
    validate_plan(plan)
    if (run / 'plan.json').exists():
        if json.loads((run / 'plan.json').read_text()) != plan:
            raise ValueError('Existing run differs; use a new run ID')
    else:
        save(run / 'plan.json', plan)
    print(f"Prepared {args.run_id}: {len(units)} units, {plan['expected_episodes']} episodes, up to {workers} GPUs; no job submitted")


def require_smoke(plan, smoke_id):
    smoke_run = run_path(smoke_id)
    smoke = load(smoke_run)
    if smoke['profile'] != 'smoke' or smoke['config'] != plan['config']:
        raise ValueError('Smoke must match this model and evaluation configuration')
    if len(collect(smoke_run, smoke)) != smoke['expected_episodes']:
        raise ValueError('Arena smoke has not completed all 15 episodes')
    if any(p.stat().st_size for p in (smoke_run / 'units').glob('*/errors.jsonl')):
        raise ValueError('Smoke recorded infrastructure errors; investigate before submitting full')


def submit(args):
    preflight()
    run = run_path(args.run_id)
    plan = load(run)
    if len(collect(run, plan)) == plan['expected_episodes']:
        print('Already complete; no job to submit')
        return
    gpu = args.gpu
    time_limit = args.time_limit or plan['time_limit']
    cmd = ['sbatch', '--parsable', '--account=gts-jli741-paid', f'--partition=gpu-{gpu}', '--qos=inferno',
           f'--gres=gpu:{gpu}:1', '--cpus-per-task=8', '--mem=64G', f'--time={time_limit}',
           f"--array=0-{plan['workers']-1}%{plan['workers']}", f'--job-name=pi05-{args.run_id}',
           f'--chdir={ROOT}', f'--output={ROOT}/logs/{args.run_id}-%A_%a.out',
           str(ROOT / 'slurm/arena.sbatch'), args.run_id, args.smoke_run]
    print(shlex.join(cmd), flush=True)
    if not args.submit:
        print('Dry run only; no Slurm request was sent. Add --submit when ready.')
        if plan['profile'] != 'smoke':
            print(f'Actual submission requires completed Arena smoke: {args.smoke_run}')
        return
    if plan['profile'] != 'smoke':
        require_smoke(plan, args.smoke_run)
    with claim(run / 'submission') as locked:
        if not locked:
            raise RuntimeError('Another submission command is active')
        if (run / 'submission/unknown.json').exists():
            raise RuntimeError('Prior submission result unknown; inspect squeue before clearing submission/unknown.json')
        active = subprocess.check_output(['squeue', '-h', '-u', str(os.getuid()), '-n', f'pi05-{args.run_id}', '-o', '%i'], text=True, timeout=30)
        if active.strip():
            raise RuntimeError(f'Already queued/running: {active.strip()}')
        (ROOT / 'logs').mkdir(exist_ok=True)
        save(run / 'submission/unknown.json', {'command': cmd, 'time': time.time()})
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
        print(result.stdout, result.stderr, end='', flush=True)
        result.check_returncode()
        job_id = result.stdout.strip().split(';')[0]
        if not job_id.isdigit():
            raise RuntimeError('Unrecognized sbatch reply; check squeue before retrying')
        save(run / 'submission' / f'{job_id}.json', {'job_id': job_id, 'command': cmd, 'time': time.time()})
        (run / 'submission/unknown.json').unlink()
        print(f'Submitted {job_id}; logs/{args.run_id}-{job_id}_<worker>.out')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('preflight'); q.add_argument('--imports', action='store_true')
    q = sub.add_parser('prepare'); q.add_argument('profile', choices=['smoke', 'pilot', 'full']); q.add_argument('run_id'); q.add_argument('--workers', type=int)
    q = sub.add_parser('submit'); q.add_argument('run_id'); q.add_argument('--submit', action='store_true'); q.add_argument('--gpu', choices=['a100', 'h100'], default='a100'); q.add_argument('--time-limit'); q.add_argument('--smoke-run', default='arena-smoke-001')
    for name in ['status', 'check', 'report', 'worker']:
        q = sub.add_parser(name); q.add_argument('run_id')
        if name == 'report': q.add_argument('--partial', action='store_true')
        if name == 'worker': q.add_argument('--smoke-run', default='arena-smoke-001')
    args = p.parse_args()
    if args.command == 'preflight': preflight(args.imports)
    elif args.command == 'prepare': prepare(args)
    elif args.command == 'submit': submit(args)
    else:
        run = run_path(args.run_id)
        plan = load(run)
        rows = collect(run, plan)
        if args.command in ['status', 'check']:
            print(f"{len(rows)}/{plan['expected_episodes']} episodes, {sum(r['success'] for r in rows)} successes")
            errors = [str(p.relative_to(run)) for p in (run / 'units').glob('*/errors.jsonl') if p.stat().st_size]
            print('Error logs:', errors)
        elif args.command == 'worker':
            preflight()
            if plan['profile'] != 'smoke': require_smoke(plan, args.smoke_run)
            return supervise(run)
        elif args.command == 'report':
            cmd = [str(ROOT / '.runtime/arena/bin/python'), '-m', 'arena_startup.report', str(run)]
            if args.partial: cmd.append('--partial')
            subprocess.run(cmd, env=environment(), check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
