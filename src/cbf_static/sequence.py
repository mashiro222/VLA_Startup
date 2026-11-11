"""Two explicitly submitted allocations; no scheduler calls from a running job."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from .common import ROOT, digest, environment, record_path, save, source_files
from .plan import config_check, prepare, validate


def prepare_sequence(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError('Choose a new sequence directory')
    path.mkdir(parents=True)
    for profile, name in [('calibration', 'calibration'), ('calibration', 'full-calibration'),
                          ('smoke', 'smoke-template'), ('full', 'full-template')]:
        prepare(profile, path / name)
    save(path / 'sequence.json', dict(schema='cbf-sequence-v1', source_sha256=source_files(),
         templates={name: digest(path / name / 'plan.json') for name in ('calibration', 'full-calibration', 'smoke-template', 'full-template')},
         validation_episodes=50, formal_episodes=4000, account='gts-jli741-paid'))
    return path


def check_sequence(path):
    manifest = json.loads((path / 'sequence.json').read_text())
    for name, checksum in manifest['source_sha256'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f'Source changed: {name}; prepare a new sequence')
    for name, checksum in manifest['templates'].items():
        if digest(path / name / 'plan.json') != checksum:
            raise ValueError(f'Frozen template changed: {name}')
        validate(path / name, require_ready=name in ('calibration', 'full-calibration'))
    return manifest


def materialize(path, profile, calibration_file=None):
    """Insert the measured scale into an already frozen episode manifest."""
    calibration_file = calibration_file or path / 'calibration/calibration.json'
    calibration = json.loads(calibration_file.read_text())
    original = json.loads((path / f'{profile}-template/plan.json').read_text())
    if not calibration.get('passed') or calibration['source_plan_sha256'] != digest(path / 'calibration/plan.json'):
        raise ValueError('Accepted calibration for this sequence is required')
    revision = json.loads((ROOT / 'config/cbf_static/checkpoint.json').read_text())['revision']
    if calibration['checkpoint_revision'] != revision:
        raise ValueError('Calibration checkpoint differs')
    plan = copy.deepcopy(original)
    plan['calibration'] = calibration
    plan['config']['filter'].update(action_scale_m=calibration['action_scale_m'], scale_source='held_out_L0_empirical_fit')
    config_check(plan['config'])
    run = path / profile
    if run.exists():
        if json.loads((run / 'plan.json').read_text()) != plan:
            raise ValueError('Existing run differs from calibrated frozen plan')
    else:
        import yaml
        save(run / 'plan.json', plan)
        (run / 'config.yaml').write_text(yaml.safe_dump(plan['config'], sort_keys=False))
    validate(run)
    return run


def smoke_gate(run):
    """Validate execution integrity; never impose a minimum task success rate."""
    import numpy as np
    import imageio.v2 as imageio
    from .report import collect
    plan, rows = collect(run)
    if plan['profile'] != 'smoke' or len(rows) != 40 or plan['expected_episodes'] != 40:
        raise ValueError('All 40 smoke episodes must finish')
    checks = json.loads((run / 'summary/sanity-checks.json').read_text())
    if len(checks) != 10 or not all(x['identical_actions'] and x['identical_positions'] for x in checks):
        raise ValueError('L0 C0/C2 or L1 C1/C2 paired consistency check failed')
    intervention_steps = 0
    for row in rows:
        folder = Path(row['artifact_dir'])
        selected = len(row['selected_obstacles'])
        expected = 0 if row['condition'] == 'C0' else min(row['level'], 1) if row['condition'] == 'C1' else row['level']
        if selected != expected:
            raise ValueError('Unexpected number of selected obstacles')
        cc = sum(x['cc'] for x in row['official_diagnostics']['cost_components'])
        if not np.isclose(cc, row['CC'], rtol=1e-8, atol=1e-8):
            raise ValueError('Official CC does not match recorded predicates')
        for camera in ('front', 'wrist'):
            frame = imageio.imread(folder / f'{camera}-initial.png')
            if frame.shape != (224, 224, 3) or np.ptp(frame) == 0:
                raise ValueError('Missing, constant or incorrectly shaped real camera image')
        with np.load(folder / 'trace.npz') as trace:
            for field in ('action_nominal', 'action_executed', 'eef_after', 'h', 'h_after', 'intervention_m'):
                if not np.isfinite(trace[field]).all():
                    raise ValueError(f'Nonfinite trace: {field}')
            nominal, action = trace['action_nominal'], trace['action_executed']
            if not np.array_equal(nominal[:, 3:], action[:, 3:]):
                raise ValueError('CBF changed rotation or gripper')
            if not selected and not np.array_equal(nominal, action):
                raise ValueError('Bypass did not preserve the nominal action')
            if selected:
                intervention_steps += int(np.count_nonzero(trace['intervention_m'] > 1e-6))
                if row['solver_status_counts'].get('bypass', 0):
                    raise ValueError('Obstacle condition bypassed the filter')
    if intervention_steps == 0:
        raise ValueError('No actual CBF intervention observed; inspect before the full experiment')
    groups = {}
    for row in rows:
        groups.setdefault((row['level'], row['condition']), []).append(row['wall_seconds'])
    # Smoke covers Apple only. This is a conservative projection, not a runtime guarantee.
    seconds = sum(float(np.mean(values)) * 500 for values in groups.values())
    return dict(passed=True, completed=40, paired_checks=len(checks), intervention_steps=intervention_steps,
                solver_failures=sum(x['solver_failures'] for x in rows),
                raw_4000_projection_hours=seconds / 3600,
                projection_with_25percent_and_startup_hours=(1.25 * seconds + 1800) / 3600,
                projection_note='Apple smoke only; task variation, I/O and compilation remain uncertain. No SR threshold.',
                smoke_plan_sha256=digest(run / 'plan.json'))


def child(command):
    """Forward Slurm termination to the complete server/simulator process group."""
    proc = subprocess.Popen(command, env=environment(), start_new_session=True)
    def interrupted(signum, unused):
        raise InterruptedError(f'Slurm signal {signum}; completed episodes remain saved')
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        code = proc.wait()
        if code:
            raise RuntimeError(f'Child process exited {code}: {shlex.join(map(str, command))}')
    finally:
        for s in previous: signal.signal(s, signal.SIG_IGN)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        for s, handler in previous.items(): signal.signal(s, handler)


def execute(path, stage):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU execution requires a submitted allocation')
    check_sequence(path)
    with (path / f'{stage}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        python = str(ROOT / '.runtime/arena/bin/python')
        if stage == 'validation':
            child([python, '-m', 'cbf_static.runtime', str(path / 'calibration'), '--index', '0'])
            smoke = materialize(path, 'smoke')
            child([python, '-m', 'cbf_static.runtime', str(smoke), '--index', '0'])
            child([python, '-m', 'cbf_static.report', str(smoke)])
            result = smoke_gate(smoke)
            result.update(calibration_sha256=digest(path / 'calibration/calibration.json'),
                          sequence_sha256=digest(path / 'sequence.json'), job_id=os.environ['SLURM_JOB_ID'])
            save(path / 'validation-passed.json', result)
            print(json.dumps(result, indent=2), flush=True)
        else:
            full = path / 'full'
            if full.exists():
                validate(full)  # A resumed run always keeps its original calibration.
            else:
                calibration_file = path / 'calibration/calibration.json'
                available = calibration_file.exists() and json.loads(calibration_file.read_text()).get('passed')
                if not available:
                    # No waiting or scheduler dependency: a formal allocation is self-contained.
                    child([python, '-m', 'cbf_static.runtime', str(path / 'full-calibration'), '--index', '0'])
                    calibration_file = path / 'full-calibration/calibration.json'
                full = materialize(path, 'full', calibration_file)
            child([python, '-m', 'cbf_static.runtime', str(full), '--index', '0'])
            child([python, '-m', 'cbf_static.report', str(full)])
            print('Completed all 4,000 formal episodes; report:', full / 'summary/README.md', flush=True)


def submission_command(path, stage, gpu='a100'):
    if gpu not in ('a100', 'h200'):
        raise ValueError('Supported sequence GPU: a100 (80 GB) or h200')
    cmd = ['sbatch', '--parsable', '--account=gts-jli741-paid', f'--partition=gpu-{gpu}', '--qos=inferno',
           f'--gres=gpu:{gpu}:1', '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--mem=64G',
           '--time=02:00:00' if stage == 'validation' else '--time=24:00:00',
           f'--job-name=cbf-{stage}-single', f'--chdir={ROOT}', f'--output={path}/{stage}-%j.out']
    if gpu == 'a100': cmd.append('--constraint=A100-80GB')
    return cmd + [str(ROOT / 'slurm/cbf-sequence.sbatch'), str(path), stage]


def submit(path, gpu, confirmed=False):
    check_sequence(path)
    if not confirmed:
        print(shlex.join(submission_command(path, 'validation', gpu=gpu)))
        print(shlex.join(submission_command(path, 'full', gpu=gpu)))
        print('Preview only. No job submitted. Add --submit to submit these two allocations.')
        return
    with (path / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = path / 'submission.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'gpu': gpu, 'jobs': {}}
        if receipt['gpu'] != gpu: raise ValueError('GPU differs from existing submission')
        for stage in ('validation', 'full'):
            old = receipt['jobs'].get(stage)
            if old:
                if not old.get('job_id'):
                    raise RuntimeError('Previous submission outcome is uncertain. Check squeue; do not submit duplicates.')
                continue
            command = submission_command(path, stage, gpu)
            item = dict(command=command, attempted=time.time(), status='submitting')
            receipt['jobs'][stage] = item; save(receipt_path, receipt)
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            item.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            match = re.fullmatch(r'(\d+)(?:;[^\s]+)?', result.stdout.strip()) if result.returncode == 0 else None
            if match: item.update(job_id=match.group(1), status='submitted')
            else: item['status'] = 'uncertain'
            save(receipt_path, receipt)
            if not match: raise RuntimeError('Submission not confirmed; inspect submission.json and squeue before retrying')
            print(stage, item['job_id'], flush=True)
        print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'submit', 'run'))
    parser.add_argument('path', type=Path)
    parser.add_argument('--stage', choices=('validation', 'full'))
    parser.add_argument('--gpu', choices=('a100', 'h200'), default='a100')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args(); path = args.path.resolve()
    if args.command == 'prepare': prepare_sequence(path)
    elif args.command == 'submit': submit(path, args.gpu, args.submit)
    elif not args.stage: parser.error('--stage is required for GPU execution')
    else: execute(path, args.stage)
    return 0
