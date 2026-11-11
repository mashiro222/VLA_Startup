#!/usr/bin/env python3
"""Prepare, preview submission, inspect and summarize the CBF experiment."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cbf_static.common import environment, record_path


def main():
    p = argparse.ArgumentParser(description=__doc__); commands = p.add_subparsers(dest='cmd', required=True)
    prep = commands.add_parser('prepare'); prep.add_argument('profile', choices=['calibration', 'smoke', 'main', 'sanity', 'full', 'gamma']); prep.add_argument('run', type=Path); prep.add_argument('--calibration', type=Path)
    sub = commands.add_parser('submit'); sub.add_argument('run', type=Path); sub.add_argument('--gpu', required=True, choices=['a100', 'h100']); sub.add_argument('--parallel', required=True, type=int); sub.add_argument('--hours', type=int, default=12); sub.add_argument('--submit', action='store_true')
    sub.add_argument('--indices', help='Explicit incomplete array indices, e.g. 2,5,8-10. Default: all units')
    st = commands.add_parser('status'); st.add_argument('run', type=Path)
    rp = commands.add_parser('report'); rp.add_argument('run', type=Path); rp.add_argument('--calibrate', action='store_true'); rp.add_argument('--extra', nargs='*', type=Path, default=[])
    te = commands.add_parser('test'); te.add_argument('--scene-audit', action='store_true')
    args = p.parse_args()
    # Use existing uv simulator Python plus an isolated QP package directory.
    if Path(sys.prefix) != ROOT / '.runtime/arena':
        return subprocess.run([str(ROOT / '.runtime/arena/bin/python'), str(Path(__file__).resolve()), *sys.argv[1:]], env=environment()).returncode
    if args.cmd == 'prepare':
        from cbf_static.plan import prepare
        prepare(args.profile, args.run.resolve(), args.calibration); return 0
    if args.cmd == 'test':
        if args.scene_audit:
            return subprocess.run([sys.executable, '-m', 'cbf_static.audit'], env=environment()).returncode
        return subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests/cbf_static'), '-v'], env=environment()).returncode
    if args.cmd == 'report':
        from cbf_static.report import report
        report(args.run, args.calibrate, args.extra); return 0
    plan = json.loads((args.run / 'plan.json').read_text())
    if args.cmd == 'status':
        for i, unit in enumerate(plan['units']):
            done = [json.loads((record_path(args.run, e) / 'episode.json').read_text()) for e in unit['episodes'] if (record_path(args.run, e) / 'episode.json').exists()]
            errors = sum((record_path(args.run, e) / 'infrastructure-error.json').exists() and not (record_path(args.run, e) / 'episode.json').exists() for e in unit['episodes'])
            print(i, unit['id'], f'{len(done)}/{len(unit["episodes"])}', 'successes', sum(e['success'] for e in done), 'unfinished_errors', errors)
        return 0
    if args.parallel < 1 or args.hours < 1: p.error('Positive parallel count and wall time required')
    indices = args.indices or f'0-{len(plan["units"])-1}'
    import re
    if not re.fullmatch(r'\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*', indices): p.error('Invalid array indices')
    for part in indices.split(','):
        bounds = [int(x) for x in part.split('-')]
        if min(bounds) < 0 or max(bounds) >= len(plan['units']) or bounds[-1] < bounds[0]: p.error('Array index outside prepared plan')
    cmd = ['sbatch', '--account=gts-jli741-paid', f'--partition=gpu-{args.gpu}', '--qos=inferno',
        f'--gres=gpu:{args.gpu}:1', '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--mem=64G',
        f'--time={args.hours}:00:00', f'--array={indices}%{args.parallel}',
        f'--job-name=cbf-{plan["profile"]}', f'--chdir={ROOT}',
        f'--output={args.run.resolve()}/slurm-%A_%a.out', str(ROOT / 'slurm/cbf-static.sbatch'), str(args.run.resolve())]
    print(shlex.join(cmd), flush=True)
    if not args.submit:
        print('Preview only. No job submitted. GPU choice and --submit are explicit.'); return 0
    from cbf_static.plan import validate
    validate(args.run)
    return subprocess.run(cmd).returncode


if __name__ == '__main__': raise SystemExit(main())
