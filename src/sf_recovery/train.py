"""Use the frozen trainer with local input paths and per-rank stall diagnostics.

No loss, architecture, optimizer, sample order, or gradient-reduction changes.
"""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import time


def install(trainer, local, diagnostic):
    local, diagnostic = Path(local), Path(diagnostic)
    diagnostic.parent.mkdir(parents=True, exist_ok=True)
    current = {'step': None, 'rank': int(os.environ.get('RANK', 0))}

    def phase(name, **details):
        current.update(phase=name, unix=time.time(), **details)
        temporary = diagnostic.with_suffix('.tmp')
        temporary.write_text(json.dumps(current) + '\n')
        os.replace(temporary, diagnostic)

    original_data = trainer.ArenaData
    class LocalData(original_data):
        def __init__(self, root, *args, **kwargs):
            super().__init__(local / 'arena_l0', *args, **kwargs)

        def __getitem__(self, index):
            phase('data_read', sample_index=int(index))
            value = super().__getitem__(index)
            phase('data_ready')
            return value
    trainer.ArenaData = LocalData
    original_policy, original_teacher = trainer.build_policy, trainer.build_teacher
    trainer.build_policy = lambda cfg, _: original_policy(cfg, local / 'base.safetensors')
    trainer.build_teacher = lambda _: original_teacher(local / 'teacher.pt')
    original_indices = trainer.batch_indices
    def indices(size, seed, step, *args, **kwargs):
        phase('update_begin', step=step + 1)
        faulthandler.dump_traceback_later(180, repeat=True)
        return original_indices(size, seed, step, *args, **kwargs)
    trainer.batch_indices = indices
    original_experiment = trainer.Experiment
    class TracedExperiment(original_experiment):
        def forward(self, batch):
            phase('forward_begin')
            result = super().forward(batch)
            phase('forward_end')
            if result[0].requires_grad:
                def entering_backward(gradient):
                    phase('backward_entered')
                    return gradient
                result[0].register_hook(entering_backward)
            return result
    trainer.Experiment = TracedExperiment
    original_restore = trainer.checkpoint.restore
    expected = json.loads((local / 'resume-source.json').read_text())['source']
    def restore(path, *args, **kwargs):
        if str(Path(path).resolve()) != expected:
            raise ValueError('Staged resume checkpoint is not the selected latest checkpoint')
        phase('checkpoint_restore')
        result = original_restore(local / 'resume.pt', *args, **kwargs)
        phase('checkpoint_restored', restored_step=result)
        return result
    trainer.checkpoint.restore = restore
    phase('initialized')
    return phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    parser.add_argument('local', type=Path)
    args = parser.parse_args()
    from sf_recovery.workflow import check
    spec = check(args.spec)
    rank = int(os.environ['RANK'])
    output = Path(spec['continuation']) / 'diagnostics'
    output.mkdir(exist_ok=True)
    trace = (output / f'rank-{rank}-python-{os.environ["SLURM_JOB_ID"]}.log').open('a', buffering=1)
    faulthandler.enable(file=trace, all_threads=True)
    faulthandler.dump_traceback_later(180, repeat=True, file=trace)
    from sf_training import train
    phase = install(train, args.local, args.local / f'rank-{rank}-phase.json')
    try:
        train.run(Path(spec['continuation']) / 'plan.json', 'sf', resume=True)
        phase('completed')
    except BaseException as exc:
        last = json.loads((args.local / f'rank-{rank}-phase.json').read_text())
        (output / f'rank-{rank}-exception-{os.environ["SLURM_JOB_ID"]}.json').write_text(
            json.dumps(dict(last_phase=last, exception=repr(exc)), indent=2))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.disable()
        trace.close()


if __name__ == '__main__':
    main()
