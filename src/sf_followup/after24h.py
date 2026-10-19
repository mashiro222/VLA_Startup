"""Manual Slurm jobs from the final checkpoint of each initial training allocation."""
import argparse
import contextlib
import fcntl
import json
import os
import subprocess
from pathlib import Path

from arena_startup.core import ROOT, collect, file_hash, frozen_files, save
from sf_followup import continuation, evaluation
from sf_followup.control import active_jobs, code_files, command, submit_once

ARMS = ('baseline', 'sf')


@contextlib.contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prepare(directory, source_plan, calibration, calibration_job):
    directory, source_plan, calibration = map(lambda x: Path(x).resolve(), (directory, source_plan, calibration))
    if directory.exists():
        raise FileExistsError(directory)
    source = json.loads(source_plan.read_text())
    if source['gpus'] != 2 or source['recipe']['global_batch'] != 32:
        raise ValueError('Expected the existing two-GPU, global-batch-32 training plan')
    jobs = {}
    for arm in ARMS:
        receipt = json.loads((source_plan.parent / 'submission' / f'{arm}.json').read_text())
        if receipt.get('status') != 'submitted' or not str(receipt.get('job_id', '')).isdigit():
            raise ValueError('Missing confirmed initial training job')
        jobs[arm] = str(receipt['job_id'])
    if not str(calibration_job).isdigit():
        raise ValueError('Missing existing calibration job')
    template = ROOT / 'runs/arena-balanced1500-001/plan.json'
    calibration_plans = [calibration / f'mb{micro}/plan.json' for micro in (1, 2)]
    c = dict(schema=1, protocol='final-checkpoint-after-initial-24h-manual',
             submission_mode='manual', automatic_resubmission=False,
             source_plan=str(source_plan), source_plan_sha256=file_hash(source_plan), original_jobs=jobs,
             calibration=str(calibration), calibration_job=str(calibration_job),
             additional_updates=15000, global_batch=32, gpus_per_training=4, micro_batch=2, accumulation=4,
             trials_per_task=30, episodes_per_arm=900, max_concurrent_gpus=None,
             pinned=str(directory / 'pinned-final'), comparison=str(ROOT / 'reports' / (directory.name + '-comparison')),
             evaluations={arm: str(ROOT / 'runs' / (directory.name + f'-{arm}-eval')) for arm in ARMS},
             continuations={arm: str(ROOT / 'runs' / (directory.name + f'-{arm}-plus15k')) for arm in ARMS},
             files=code_files(), arena_files=frozen_files(),
             inputs={str(p): file_hash(p) for p in [source_plan, template] + calibration_plans})
    save(directory / 'campaign.json', c)
    return c


def check(campaign):
    c = json.loads(Path(campaign).read_text())
    if c['protocol'] != 'final-checkpoint-after-initial-24h-manual' or c['max_concurrent_gpus'] is not None:
        raise ValueError('Unexpected follow-up protocol or GPU cap')
    if c['submission_mode'] != 'manual' or c['automatic_resubmission']:
        raise ValueError('This campaign permits manual submission only')
    if c['files'] != code_files() or c['arena_files'] != frozen_files():
        raise ValueError('Frozen follow-up/evaluation implementation changed')
    for name, expected in c['inputs'].items():
        if file_hash(name) != expected:
            raise ValueError('Frozen input changed: ' + name)
    source = json.loads(Path(c['source_plan']).read_text())
    for name, expected in source['files'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('Original training implementation changed: ' + name)
    if (c['additional_updates'], c['global_batch'], c['gpus_per_training'], c['micro_batch'], c['accumulation']) != (15000, 32, 4, 2, 4):
        raise ValueError('Unexpected continuation budget')
    return c


def final_checkpoint(source, arm):
    """Read final status, never round to 5k or substitute an older periodic checkpoint."""
    source = Path(source)
    status = json.loads((source / arm / 'status.json').read_text())
    step = status['step']
    if type(step) is not int or step < 1 or status['stop_reason'] not in ('wall_time_budget', 'signal', 'target_reached'):
        raise ValueError('Training did not leave a normal final checkpoint status')
    checkpoint = source / arm / f'step-{step:08d}.pt'
    complete = sorted((source / arm).glob('step-*.pt'))
    if not complete or complete[-1] != checkpoint or checkpoint.stat().st_size == 0:
        raise ValueError('Final status and latest complete checkpoint differ')
    return step, checkpoint


def verify_selection(c, arm, selection):
    if (selection['arm'] != arm or selection['source_plan_sha256'] != c['source_plan_sha256']
            or selection['target_step'] != selection['source_step'] + c['additional_updates']):
        raise ValueError('Saved final-checkpoint selection differs from the campaign')
    if not Path(selection['checkpoint']).is_file():
        raise FileNotFoundError('Pinned final checkpoint missing')
    for name, expected in selection['plans'].items():
        if file_hash(name) != expected:
            raise ValueError('Materialized plan changed: ' + name)
    return selection


def materialize(campaign, arm):
    """Select the final checkpoint during an explicit manual submission."""
    campaign = Path(campaign).resolve()
    c = check(campaign)
    if arm not in ARMS:
        raise ValueError(arm)
    with locked(campaign.parent / f'{arm}.lock'):
        record = campaign.parent / f'{arm}-selection.json'
        if record.exists():
            return verify_selection(c, arm, json.loads(record.read_text()))
        step, source_file = final_checkpoint(Path(c['source_plan']).parent, arm)
        pinned = Path(c['pinned']) / arm / source_file.name
        pinned.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_file, pinned)
        except FileExistsError:
            if not os.path.samefile(source_file, pinned):
                raise ValueError('Pinned checkpoint was replaced')
        evaluation_dir, continuation_dir = Path(c['evaluations'][arm]), Path(c['continuations'][arm])
        if not (evaluation_dir / 'plan.json').exists():
            evaluation.prepare(c['source_plan'], c['pinned'], evaluation_dir, arm,
                               step=step, trials_per_task=c['trials_per_task'])
        if not (continuation_dir / 'plan.json').exists():
            continuation.prepare(c['source_plan'], continuation_dir, c['calibration'], c['pinned'],
                                 target_step=step + c['additional_updates'], source_step=step)
        ep = json.loads((evaluation_dir / 'plan.json').read_text())
        cp = json.loads((continuation_dir / 'plan.json').read_text())
        evaluation.validate(ep)
        continuation.compatible(json.loads(Path(c['source_plan']).read_text()), cp)
        if (ep['trained_policy']['arm'] != arm or ep['trained_policy']['step'] != step
                or ep['trained_policy']['checkpoint'] != str(pinned)
                or ep['expected_episodes'] != c['episodes_per_arm']
                or cp['migration']['source_step'] != step or cp['recipe']['steps'] != step + c['additional_updates']):
            raise ValueError('Evaluation and continuation must use the exact same final checkpoint')
        selection = dict(arm=arm, original_job=c['original_jobs'][arm], source_step=step,
                         target_step=step + c['additional_updates'], checkpoint=str(pinned),
                         source_plan_sha256=c['source_plan_sha256'],
                         plans={str(p): file_hash(p) for p in (evaluation_dir / 'plan.json', continuation_dir / 'plan.json')})
        save(record, selection)
        return selection


def stage_command(campaign, c, arm, stage):
    if arm not in ARMS or stage not in ('eval', 'continue'):
        raise ValueError('Unknown arm/stage')
    return command(f'sf-24h-{stage}-{arm}', f'slurm/sf_after24h_{stage}.sbatch',
                   [str(Path(campaign).resolve()), arm],
                   gpus=1 if stage == 'eval' else 2, nodes=1 if stage == 'eval' else 2, hours=24)


def require_completed_job(job):
    output = subprocess.check_output(
        ['sacct', '-n', '-P', '-j', str(job), '--format=JobIDRaw,State%30,ExitCode'], text=True, timeout=60)
    rows = [line.split('|') for line in output.splitlines() if line.split('|')[0].strip() == str(job)]
    if len(rows) != 1 or rows[0][1].strip() != 'COMPLETED' or rows[0][2].strip() != '0:0':
        raise ValueError(f'Initial training job {job} has not completed successfully; no job submitted')


def submit(campaign, arm, stage, execute=False):
    import shlex
    campaign = Path(campaign).resolve()
    c = check(campaign)
    cmd = stage_command(campaign, c, arm, stage)
    print(shlex.join(cmd), flush=True)
    if not execute:
        print('Preview only. Add --submit when this arm has finished its initial training.')
        return
    require_completed_job(c['original_jobs'][arm])
    selection = materialize(campaign, arm)
    if stage == 'continue':
        evidence = json.loads((Path(c['calibration']) / 'calibration.json').read_text())
        if not evidence.get('passed') or len(evidence.get('rows', [])) != 4:
            raise ValueError('The existing batch-2 calibration has not passed; no job submitted')
    previous = {p.stem: json.loads(p.read_text())
                for p in (campaign.parent / 'submission').glob(f'{stage}-{arm}-*.json')}
    if any(value.get('status') != 'submitted' for value in previous.values()):
        raise RuntimeError('Resolve the uncertain submission receipt before retrying')
    live = active_jobs()
    running = [value['job_id'] for value in previous.values() if value['job_id'] in live]
    if running:
        print(f'Existing {arm} {stage} job: {running[0]}; no duplicate submitted')
        return running[0]
    if stage == 'eval':
        run = Path(c['evaluations'][arm])
        plan = json.loads((run / 'plan.json').read_text())
        progress = len(collect(run, plan))
        if progress == plan['expected_episodes']:
            print(f'{arm}: all {progress} evaluation episodes are already complete')
            return
    else:
        run = Path(c['continuations'][arm])
        status_path = run / arm / 'status.json'
        progress = selection['source_step']
        if status_path.exists():
            status = json.loads(status_path.read_text())
            progress, _ = final_checkpoint(run, arm)
            if status['target'] != selection['target_step'] or progress > selection['target_step']:
                raise ValueError('Continuation target changed')
            if progress == selection['target_step'] and status['completed']:
                print(f'{arm}: all 15000 additional updates are already complete at step {progress}')
                return
            if status['stop_reason'] not in ('wall_time_budget', 'signal'):
                raise ValueError('Inspect the interrupted continuation before resubmitting')
    label = f'{stage}-{arm}-after-{progress}'
    if label in previous:
        raise RuntimeError('Previous job at the same progress has ended; inspect its logs before retrying')
    return submit_once(campaign.parent, label, cmd)


def maybe_compare(campaign):
    campaign = Path(campaign).resolve()
    c = check(campaign)
    with locked(campaign.parent / 'comparison.lock'):
        if not all((ROOT / 'reports' / Path(run).name / 'summary.json').exists() for run in c['evaluations'].values()):
            print('This arm is reported; combined table will be generated by the other evaluation when it finishes.')
            return
        for arm in ARMS:
            verify_selection(c, arm, json.loads((campaign.parent / f'{arm}-selection.json').read_text()))
        evaluation.compare(c['evaluations']['baseline'], c['evaluations']['sf'],
                           c['comparison'], require_equal_steps=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='action', required=True)
    q = s.add_parser('prepare')
    for name in ('directory', 'source', 'calibration', 'calibration_job'): q.add_argument(name)
    q = s.add_parser('submit'); q.add_argument('campaign'); q.add_argument('arm', choices=ARMS)
    q.add_argument('stage', choices=('eval', 'continue')); q.add_argument('--submit', action='store_true')
    for name in ('check', 'compare'):
        q = s.add_parser(name); q.add_argument('campaign')
    for name in ('materialize', 'path'):
        q = s.add_parser(name); q.add_argument('campaign'); q.add_argument('arm', choices=ARMS)
        if name == 'path': q.add_argument('stage', choices=('eval', 'continue'))
    a = p.parse_args()
    if a.action == 'prepare': prepare(a.directory, a.source, a.calibration, a.calibration_job)
    elif a.action == 'submit': submit(a.campaign, a.arm, a.stage, a.submit)
    elif a.action == 'check': check(a.campaign); print('Independent follow-up manifest verified')
    elif a.action == 'materialize': print(json.dumps(materialize(a.campaign, a.arm), indent=2))
    elif a.action == 'compare': maybe_compare(a.campaign)
    else:
        c = check(a.campaign)
        print(c['evaluations'][a.arm] if a.stage == 'eval' else str(Path(c['continuations'][a.arm]) / 'plan.json'))
