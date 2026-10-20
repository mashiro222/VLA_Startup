"""Checkpoint migration to four GPUs with fixed optimizer/data/LR semantics."""
import argparse
import copy
import json
import os
from pathlib import Path

from arena_startup.core import ROOT, file_hash, save


def compatible(source, target):
    allowed = {'steps', 'micro_batch', 'log_every', 'checkpoint_every', 'keep_checkpoints'}
    if {k: v for k, v in source['recipe'].items() if k not in allowed} != {
            k: v for k, v in target['recipe'].items() if k not in allowed}:
        raise ValueError('Migration changes model, data protocol, optimizer or learning-rate schedule')
    for key in ('assets', 'environment', 'files'):
        if source[key] != target[key]: raise ValueError(f'Migration {key} differs')
    # Increasing retention changes storage only, never the optimizer trajectory.
    before = source['recipe'].get('keep_checkpoints', 2)
    after = target['recipe'].get('keep_checkpoints', before)
    if type(after) is not int or after < before:
        raise ValueError('Continuation must not reduce checkpoint retention')
    if target['recipe']['global_batch'] != 32 or target['recipe']['micro_batch'] != 2 or target['gpus'] != 4:
        raise ValueError('Expected 4 GPUs x micro-batch 2 x accumulation 4')


def prepare(source_path, output, calibration, pinned, target_step=20000, source_step=5000):
    source_path, output = Path(source_path).resolve(), Path(output).resolve()
    source = json.loads(source_path.read_text())
    target = copy.deepcopy(source)
    target.update(gpus=4, profile='full', time_limit='24:00:00')
    target['recipe'].update(micro_batch=2, steps=target_step)
    if type(source_step) is not int or source_step < 1 or target_step <= source_step:
        raise ValueError('Continuation must advance beyond a complete source checkpoint')
    target['recipe']['keep_checkpoints'] = source['recipe']['keep_checkpoints']
    target['migration'] = dict(source_plan=str(source_path), source_plan_sha256=file_hash(source_path),
                               source_step=source_step, pinned=str(Path(pinned).resolve()),
                               calibration=str(Path(calibration).resolve()), nodes=2, gpus_per_node=2,
                               rng=f'Reset all rank streams deterministically at update {source_step}; not bitwise replay')
    compatible(source, target)
    if output.exists(): raise FileExistsError(output)
    save(output / 'plan.json', target)


def migrate(plan_path, arm):
    import torch
    from sf_training.workflow import verify_plan
    from sf_training.utils import seed_all
    from sf_training import checkpoint
    plan_path = Path(plan_path).resolve()
    target = verify_plan(plan_path)
    spec = target['migration']
    source = verify_plan(Path(spec['source_plan']))
    compatible(source, target)
    if file_hash(spec['source_plan']) != spec['source_plan_sha256']:
        raise ValueError('Source plan changed')
    evidence = json.loads((Path(spec['calibration']) / 'calibration.json').read_text())
    if not evidence.get('passed') or len(evidence['rows']) != 4:
        raise ValueError('Both arms must pass the batch-2 calibration first')
    d = plan_path.parent / arm
    d.mkdir(exist_ok=True)
    if list(d.glob('step-*.pt')):
        # Native continuation is handled by the original strict resume implementation.
        return
    source_file = Path(spec['pinned']) / arm / f"step-{spec['source_step']:08d}.pt"
    ck = torch.load(source_file, map_location='cpu', weights_only=False, mmap=True)
    base = file_hash(ROOT / '.runtime/training/pi05_base_torch/conversion.json')
    if (ck['identity'] != dict(plan_sha256=spec['source_plan_sha256'], arm=arm, world=source['gpus'], base=base)
            or ck['step'] != spec['source_step']):
        raise ValueError('Wrong source checkpoint identity or update count')
    rngs = []
    for rank in range(target['gpus']):
        seed_all(target['recipe']['seed'] + 1_000_000 + ck['step'] * target['gpus'] + rank)
        rngs.append(checkpoint.rng_state())
    # Preserve weights, Adam moments and step. Only identity/rank RNG mapping changes.
    migrated = dict(ck)
    migrated['identity'] = dict(plan_sha256=file_hash(plan_path), arm=arm, world=4, base=base)
    migrated['rng'] = rngs
    destination = d / f"step-{ck['step']:08d}.pt"
    temporary = destination.with_suffix('.tmp')
    torch.save(migrated, temporary)
    with temporary.open('rb') as f: os.fsync(f.fileno())
    os.replace(temporary, destination)
    save(d / 'migration.json', dict(source=str(source_file), source_sha256=file_hash(source_file),
                                   source_step=ck['step'], target_plan_sha256=file_hash(plan_path),
                                   optimizer_preserved=True, lr_schedule_preserved=True, rng=spec['rng']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); s = p.add_subparsers(dest='command', required=True)
    q = s.add_parser('prepare'); q.add_argument('source'); q.add_argument('output'); q.add_argument('calibration'); q.add_argument('pinned')
    q.add_argument('--target-step', type=int, default=20000)
    q.add_argument('--source-step', type=int, default=5000)
    q = s.add_parser('migrate'); q.add_argument('plan'); q.add_argument('arm', choices=['baseline', 'sf'])
    a = p.parse_args()
    if a.command == 'prepare': prepare(a.source, a.output, a.calibration, a.pinned, a.target_step, a.source_step)
    else: migrate(a.plan, a.arm)
