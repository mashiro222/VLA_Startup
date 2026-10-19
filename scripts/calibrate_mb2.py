"""Finish only micro-batch 2 calibration, reusing completed micro-batch 1 evidence."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from sf_followup.calibration import summarize


def validate_case(directory, micro):
    directory = Path(directory)
    status = json.loads((directory / 'status.json').read_text())
    runtime = json.loads((directory / 'runtime.json').read_text())
    inference = json.loads((directory / 'inference_smoke.json').read_text())
    metrics = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    if not (status['completed'] and status['step'] == 12 and inference['finite']
            and runtime['global_batch'] == 32 and runtime['world'] == 2
            and runtime['micro_batch'] == micro and len(metrics) == 12
            and [row['step'] for row in metrics] == list(range(1, 13))):
        raise ValueError(f'Incomplete calibration evidence: {directory}')
    if any(not 0 <= row['flow_mse'] <= 1000 or row['seconds'] <= 0
           or any(not math.isfinite(row[key]) for key in ('loss', 'grad_norm', 'seconds'))
           for row in metrics):
        raise ValueError(f'Invalid calibration metrics: {directory}')


def pending_arms(directory):
    directory = Path(directory).resolve()
    for micro in (1, 2):
        plan = json.loads((directory / f'mb{micro}/plan.json').read_text())
        cfg = plan['recipe']
        if (plan['profile'], plan['gpus'], cfg['global_batch'], cfg['micro_batch'], cfg['steps']) != (
                'smoke', 2, 32, micro, 12):
            raise ValueError('Expected the existing two-GPU, twelve-update calibration plans')
    for arm in ('baseline', 'sf'):
        validate_case(directory / 'mb1' / arm, 1)
    pending = []
    for arm in ('baseline', 'sf'):
        case = directory / 'mb2' / arm
        if (case / 'status.json').exists():
            validate_case(case, 2)
        elif case.exists() and any(case.iterdir()):
            raise ValueError(f'Inspect partial batch-2 output before retrying: {case}')
        else:
            pending.append(arm)
    return pending


def run(directory):
    directory = Path(directory).resolve()
    pending = pending_arms(directory)
    print(f'Reusing batch-1 evidence; batch-2 cases to run: {pending}', flush=True)
    for arm in pending:
        subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                        '--nproc_per_node=2', '-m', 'sf_training.train',
                        '--plan', str(directory / 'mb2/plan.json'), '--arm', arm], check=True)
        validate_case(directory / 'mb2' / arm, 2)
    summarize(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps({'pending_batch2_arms': pending_arms(args.directory)}))
    else:
        run(args.directory)
