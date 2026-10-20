"""Prepare and manually submit an audited SF recovery and a paired 900-episode evaluation."""
import argparse
import json
import os
from pathlib import Path

from arena_startup.core import ROOT, file_hash, save
from sf_followup import continuation, evaluation
from sf_followup.control import command, submit_once


def code_files():
    paths = list((ROOT / 'src/sf_recovery').glob('*.py'))
    paths += [ROOT / 'scripts/recovery_worker.sh', ROOT / 'slurm/recovery_continue.sbatch', ROOT / 'slurm/recovery_audit.sbatch',
              ROOT / 'slurm/recovery_eval.sbatch']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def prepare(directory, audit_file):
    directory, audit_file = Path(directory).resolve(), Path(audit_file).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    audit = json.loads(audit_file.read_text())
    if not audit.get('passed') or not audit.get('production_restore_passed') or audit['step'] != 3000:
        raise ValueError('SF 3000-step CPU restore audit must pass before preparation')
    source = ROOT / 'runs/arena-sf-train-full-002/plan.json'
    if audit['source_plan_sha256'] != file_hash(source):
        raise ValueError('Audited training plan changed')
    pinned = directory / 'pinned'
    destination = pinned / 'sf/step-00003000.pt'
    destination.parent.mkdir(parents=True)
    os.link(audit['source'], destination)
    if file_hash(destination) != audit['checkpoint_sha256']:
        raise ValueError('Audited checkpoint changed')
    calibration = ROOT / 'runs/arena-microbatch-20261018-001'
    training = ROOT / 'runs' / (directory.name + '-sf-plus15k')
    continuation.prepare(source, training, calibration, pinned, target_step=18000, source_step=3000)
    tp = json.loads((training / 'plan.json').read_text())
    tp['recipe']['keep_checkpoints'] = 15002
    save(training / 'plan.json', tp)
    run = ROOT / 'runs' / (directory.name + '-sf-eval900')
    evaluation.prepare(source, pinned, run, 'sf', step=3000, trials_per_task=30)
    baseline = ROOT / 'runs/arena-after24h-20261019-001-baseline-eval'
    bp = json.loads((baseline / 'plan.json').read_text())
    ep = json.loads((run / 'plan.json').read_text())
    evaluation.validate_comparison([bp, ep], require_equal_steps=False)
    spec = dict(schema=1, root=str(ROOT), source_plan=str(source), source_step=3000,
                target_step=18000, additional_updates=15000, arm='sf',
                continuation=str(training), evaluation=str(run), baseline_evaluation=str(baseline),
                comparison=str(ROOT / 'reports' / (directory.name + '-comparison')),
                checkpoint=str(destination), checkpoint_sha256=audit['checkpoint_sha256'],
                teacher_sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0',
                audit=str(audit_file), automatic_resubmission=False,
                files=code_files(), followup_files=evaluation.followup_files(),
                inputs={str(p): file_hash(p) for p in [source, audit_file, training / 'plan.json', run / 'plan.json', baseline / 'plan.json', calibration / 'calibration.json']})
    save(directory / 'recovery.json', spec)
    return spec


def check(path):
    spec = json.loads(Path(path).read_text())
    if spec['files'] != code_files() or spec['followup_files'] != evaluation.followup_files():
        raise ValueError('Recovery implementation changed after preparation')
    for path, expected in spec['inputs'].items():
        if file_hash(path) != expected:
            raise ValueError('Recovery input changed: '+path)
    from sf_training.workflow import files
    source = json.loads(Path(spec['source_plan']).read_text())
    if source['files'] != files():
        raise ValueError('Frozen original training implementation changed')
    target = json.loads((Path(spec['continuation']) / 'plan.json').read_text())
    continuation.compatible(source, target)
    if (target['recipe']['steps'], target['migration']['source_step'], target['gpus'],
        target['recipe']['micro_batch'], target['recipe']['global_batch']) != (18000, 3000, 4, 2, 32):
        raise ValueError('Unexpected SF continuation budget')
    evaluation.validate(json.loads((Path(spec['evaluation']) / 'plan.json').read_text()))
    return spec


def submit(path, stage, execute=False):
    path = Path(path).resolve()
    spec = check(path)
    if stage not in ('continue', 'eval'):
        raise ValueError(stage)
    cmd = command('sf-recover-3000-' + stage, 'slurm/recovery_' + stage + '.sbatch', [path],
                  gpus=2 if stage=='continue' else 1, nodes=2 if stage=='continue' else 1, hours=24)
    import shlex
    print(shlex.join(cmd), flush=True)
    if execute:
        return submit_once(path.parent, stage, cmd)
    print('Preview only; no job submitted')


def compare(path):
    spec = check(path)
    if not all((ROOT / 'reports' / Path(spec[k]).name / 'summary.json').exists()
               for k in ('baseline_evaluation', 'evaluation')):
        print('Waiting for both existing evaluations to finish; no jobs submitted')
        return
    evaluation.compare(spec['baseline_evaluation'], spec['evaluation'], spec['comparison'], require_equal_steps=False)
    # The generic legacy title assumes two final 24-hour checkpoints; this recovery
    # compares explicitly selected unequal steps, so state the actual protocol.
    (Path(spec['comparison']) / 'README.md').write_text(
        '# Arena Safety checkpoint comparison\n\nBaseline: 5,952 updates; SF: 3,000 updates.\n\n'
        'Each arm evaluates the same 900 task/initial-state pairs. Training update counts differ; '
        'this is an interim comparison, not an equal-update SF ablation.\n\n'
        'See comparison.csv, comparison.json and comparison.png for SR and mean CC. '
        'Per-arm reports include termination reasons and cost attribution.\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); s=p.add_subparsers(dest='action',required=True)
    q=s.add_parser('prepare');q.add_argument('directory');q.add_argument('audit')
    for name in ('check','compare'):
        q=s.add_parser(name);q.add_argument('spec')
    q=s.add_parser('submit');q.add_argument('spec');q.add_argument('stage',choices=('continue','eval'));q.add_argument('--submit',action='store_true')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.directory,a.audit)
    elif a.action=='submit':print(submit(a.spec,a.stage,a.submit))
    elif a.action=='compare':compare(a.spec)
    else:check(a.spec);print('Recovery manifest verified')
