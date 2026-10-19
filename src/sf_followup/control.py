"""Idempotent Slurm handoff: pin 5k, evaluate, and resume paired runs to 20k."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from arena_startup.core import ROOT, claim, file_hash, save
from sf_followup.checkpoints import pin


def code_files():
    paths = list((ROOT / 'src/sf_followup').glob('*.py'))
    paths += list((ROOT / 'slurm').glob('sf_*.sbatch'))
    paths += [ROOT / 'scripts/sf_multinode_worker.sh']
    paths += [ROOT / 'requirements/sf_followup.lock', ROOT / 'config/sf_followup_dependencies.json']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def check(campaign):
    c = json.loads(Path(campaign).read_text())
    if c['files'] != code_files(): raise ValueError('Follow-up files changed since preparation')
    for path, digest in c['plans'].items():
        if file_hash(path) != digest: raise ValueError(f'Plan changed: {path}')
    source = json.loads(Path(c['source_plan']).read_text())
    for name, digest in source['files'].items():
        if file_hash(ROOT / name) != digest: raise ValueError(f'Original frozen training file changed: {name}')
    return c


def command(name, script, args, gpus=0, nodes=1, hours=24, dependencies=(), constraint=None, time_limit=None):
    cmd = ['sbatch', '--parsable', '--account=gts-jli741-paid', '--qos=inferno',
           f'--partition={"gpu-a100" if gpus else "cpu-small"}', f'--nodes={nodes}',
           f'--ntasks={nodes}', '--ntasks-per-node=1', f'--cpus-per-task={8*gpus if gpus else 1}',
           f'--mem={96 if gpus else 2}G', f'--time={time_limit or format(hours, "02d") + ":00:00"}',
           f'--job-name={name}', f'--chdir={ROOT}', f'--output={ROOT}/logs/{name}-%j.out']
    if gpus: cmd.append(f'--gres=gpu:a100:{gpus}')
    if constraint: cmd.append(f'--constraint={constraint}')
    if dependencies:
        cmd.append('--dependency=afterany:' + ':'.join(str(x) for x in dependencies))
        cmd.append('--kill-on-invalid-dep=yes')
    return cmd + [str(ROOT / script)] + list(map(str, args))


def submit_once(directory, label, cmd):
    directory = Path(directory)
    receipt = directory / 'submission' / f'{label}.json'
    with claim(directory / 'submission') as owned:
        if not owned: raise RuntimeError('Submission lock is held')
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if value['status'] != 'submitted':
                raise RuntimeError('Uncertain submission; inspect Slurm before retrying: ' + str(receipt))
            return value['job_id']
        value = dict(command=cmd, attempted_unix=time.time(), status='unknown')
        save(receipt, value)
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=90)
        value.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        if result.returncode == 0:
            job = result.stdout.strip().split(';')[0]
            if job.isdigit(): value.update(status='submitted', job_id=job)
        save(receipt, value)
        if value['status'] != 'submitted': raise RuntimeError('Submission failed/uncertain: ' + str(receipt))
        print(f"Submitted {label}: {value['job_id']}", flush=True)
        return value['job_id']


def prepare(directory, source, calibration, pinned, continuation, evaluations):
    directory = Path(directory).resolve()
    paths = [Path(source).resolve(), Path(continuation).resolve()]
    paths += [Path(x).resolve() / 'plan.json' for x in evaluations.values()]
    paths += [Path(calibration).resolve() / f'mb{x}/plan.json' for x in (1, 2)]
    original_jobs = {}
    for arm in ('baseline', 'sf'):
        receipt = json.loads((paths[0].parent / 'submission' / f'{arm}.json').read_text())
        if receipt.get('status') != 'submitted' or not str(receipt.get('job_id', '')).isdigit():
            raise ValueError('Original job has no confirmed submission receipt')
        original_jobs[arm] = str(receipt['job_id'])
    c = dict(schema=1, source_plan=str(paths[0]), original_jobs=original_jobs,
             calibration=str(Path(calibration).resolve()), pinned=str(Path(pinned).resolve()),
             continuation_plan=str(Path(continuation).resolve()),
             evaluations={k: str(Path(v).resolve()) for k, v in evaluations.items()},
             initial_step=5000, target_step=20000, global_batch=32, max_concurrent_gpus=8,
             files=code_files(), plans={str(p): file_hash(p) for p in paths},
             note='Initial training untouched. Catch-up only if 5k is missing after a normal time-budget exit. '
                  'Continuation uses matched 5k checkpoints; SF four-GPU start waits for evaluation allocations to exit.')
    if (directory / 'campaign.json').exists(): raise FileExistsError(directory)
    save(directory / 'campaign.json', c)


def active_jobs():
    text = subprocess.check_output(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i'], text=True, timeout=60)
    return set(text.split())


def receipts(directory):
    return {p.stem: json.loads(p.read_text()) for p in (directory / 'submission').glob('*.json')}


def initial_finished(c, arm, live, submitted):
    jobs = [c['original_jobs'][arm]] + [v['job_id'] for k, v in submitted.items()
                                       if k.startswith(f'catchup-{arm}-') and v['status'] == 'submitted']
    return not any(j in live for j in jobs)


def tick(campaign):
    campaign = Path(campaign)
    directory = campaign.parent
    c = check(campaign)
    source = Path(c['source_plan']).parent
    pinned = pin(source, c['pinned'])
    live = active_jobs()
    submitted = receipts(directory)
    if any(v['status'] != 'submitted' for v in submitted.values()):
        raise RuntimeError('Resolve an uncertain submission before automatic handoff')
    pinned_5k = {x['arm'] for x in pinned if x['step'] == c['initial_step']}
    blockers = []
    for arm in ('baseline', 'sf'):
        if arm in pinned_5k or not initial_finished(c, arm, live, submitted): continue
        status_path = source / arm / 'status.json'
        if not status_path.exists():
            blockers.append(f'{arm}: initial job ended without normal saved status'); continue
        status = json.loads(status_path.read_text())
        if status['step'] >= 5000:
            blockers.append(f'{arm}: 5000-step checkpoint missing; no later step substituted'); continue
        if status['stop_reason'] != 'wall_time_budget':
            blockers.append(f'{arm}: stopped for {status["stop_reason"]}; inspect before retry'); continue
        label = f'catchup-{arm}-{status["step"]}'
        if label in submitted and submitted[label]['job_id'] not in live:
            blockers.append(f'{arm}: catch-up did not advance; inspect logs'); continue
        cmd = command('sf-5k-' + arm, 'slurm/sf_catchup.sbatch', [c['source_plan'], arm], gpus=2)
        submit_once(directory, label, cmd)

    eval_jobs = []
    if pinned_5k == {'baseline', 'sf'}:
        for arm, run in c['evaluations'].items():
            evaluation = json.loads((Path(run) / 'plan.json').read_text())
            cmd = command('sf-eval5k-' + arm, 'slurm/sf_evaluation.sbatch', [run],
                          gpus=1, time_limit=evaluation['time_limit'])
            eval_jobs.append(submit_once(directory, 'eval-' + arm, cmd))

    evidence_path = Path(c['calibration']) / 'calibration.json'
    calibrated = evidence_path.exists() and json.loads(evidence_path.read_text()).get('passed', False)
    # Re-read after submissions so newly queued evaluation allocations count toward the cap.
    live = active_jobs()
    submitted = receipts(directory)
    calibration_job = submitted.get('calibration', {}).get('job_id')
    ready = (pinned_5k == {'baseline', 'sf'} and calibrated
             and calibration_job not in live
             and all(initial_finished(c, arm, live, submitted) for arm in ('baseline', 'sf')))
    continuation = Path(c['continuation_plan']).parent
    complete = []
    if ready:
        for arm in ('baseline', 'sf'):
            status_path = continuation / arm / 'status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else None
            done = status and status['step'] == c['target_step'] and status['completed']
            complete.append(bool(done))
            if done: continue
            phase_jobs = {k: v for k, v in submitted.items() if k.startswith('continue-' + arm + '-')}
            if any(v['job_id'] in live for v in phase_jobs.values()): continue
            step = status['step'] if status else 5000
            label = f'continue-{arm}-{step}'
            if label in phase_jobs:
                blockers.append(f'{arm}: continuation did not advance; inspect logs'); continue
            if status and status['stop_reason'] != 'wall_time_budget':
                blockers.append(f'{arm}: continuation stopped for {status["stop_reason"]}'); continue
            # Baseline (4) + evaluations (2) <= 6. SF's four GPUs start only after both eval jobs exit.
            dependencies = [job for job in eval_jobs if job in live] if arm == 'sf' else []
            cmd = command('sf-20k-' + arm, 'slurm/sf_continue.sbatch',
                          [c['continuation_plan'], arm, campaign], gpus=2, nodes=2, dependencies=dependencies)
            submit_once(directory, label, cmd)
    elif not calibrated:
        blockers.append('Awaiting successful batch-2 calibration; no four-GPU continuation submitted')
    if continuation.exists():
        pin(continuation, directory / 'pinned-continuation', steps=(10000, 15000, 20000))
    # Evaluation success is checked separately from Slurm exit status.
    evaluations_complete = all((ROOT / 'reports' / Path(r).name / 'summary.json').exists()
                               for r in c['evaluations'].values())
    comparison = ROOT / 'reports' / (directory.name + '-5k-comparison')
    if evaluations_complete and not (comparison / 'comparison.json').exists():
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        subprocess.run([str(ROOT / '.runtime/sf_train/bin/python'), '-m', 'sf_followup.evaluation', 'compare',
                        c['evaluations']['baseline'], c['evaluations']['sf'], str(comparison)],
                       check=True, env=env, timeout=300)
    state = dict(checked_unix=time.time(), pinned_5k=sorted(pinned_5k), calibration_passed=calibrated,
                 continuation_ready=ready, complete=complete == [True, True] and evaluations_complete,
                 evaluations_complete=evaluations_complete, blockers=blockers)
    save(directory / 'state.json', state)
    return state


def monitor(campaign):
    campaign = Path(campaign).resolve()
    with claim(campaign.parent / 'monitor') as owned:
        if not owned: raise RuntimeError('Another monitor is active')
        started = time.monotonic()
        while time.monotonic() - started < 71 * 3600:
            state = tick(campaign)
            if state['complete']:
                print('Paired evaluation and 20k training complete', flush=True); return
            if any('did not advance' in x or 'stopped for' in x for x in state['blockers']):
                raise RuntimeError('Follow-up requires inspection: ' + '; '.join(state['blockers']))
            time.sleep(60)
        # Continue only the CPU orchestration; training/evaluation remain bounded separate jobs.
        job = os.environ['SLURM_JOB_ID']
        cmd = command('sf-followup-watch', 'slurm/sf_watch.sbatch', [campaign], hours=72, dependencies=[job])
        submit_once(campaign.parent, 'monitor-after-' + job, cmd)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); s = p.add_subparsers(dest='action', required=True)
    for action in ('check', 'tick', 'monitor'):
        q = s.add_parser(action); q.add_argument('campaign')
    q = s.add_parser('start'); q.add_argument('campaign'); q.add_argument('--submit', action='store_true')
    a = p.parse_args()
    if a.action == 'start':
        c = check(a.campaign); directory = Path(a.campaign).parent
        jobs = [
            ('calibration', command('sf-mb2-calibration', 'slurm/sf_calibration.sbatch', [c['calibration']],
                                    gpus=2, hours=1, constraint='A100-40GB')),
            ('monitor', command('sf-followup-watch', 'slurm/sf_watch.sbatch', [a.campaign], hours=72)),
        ]
        for label, cmd in jobs:
            print(shlex.join(cmd), flush=True)
            if a.submit: submit_once(directory, label, cmd)
    elif a.action == 'check': check(a.campaign); print('Follow-up manifests verified')
    elif a.action == 'tick': print(tick(a.campaign))
    else: monitor(a.campaign)
