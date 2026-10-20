"""CPU verification of an interrupted training checkpoint, including optimizer restore."""
import argparse
import json
from pathlib import Path
import time


def audit(source_plan, checkpoint, output, local_file=None):
    import torch
    from torch import nn
    from sf_training import checkpoint as checkpoint_io
    from sf_training.utils import sha

    torch.set_num_threads(1)
    source_plan, checkpoint, output = map(Path, (source_plan, checkpoint, output))
    plan = json.loads(source_plan.read_text())
    started = time.time()
    reading = Path(local_file) if local_file else checkpoint
    if local_file:
        copied = json.loads(reading.with_suffix('.copy.json').read_text())
        status = checkpoint.stat()
        if (copied['source'], copied['size'], copied['mtime_ns']) != (
                str(checkpoint.resolve()), status.st_size, status.st_mtime_ns):
            raise ValueError('Local CPU audit copy no longer matches the source checkpoint')
        if sha(reading) != copied['sha256']:
            raise ValueError('Local CPU audit copy checksum mismatch')
    ck = torch.load(reading, map_location='cpu', weights_only=False, mmap=True)
    print('Checkpoint metadata loaded; checking tensors and Adam moments', flush=True)
    identity = dict(plan_sha256=sha(source_plan), arm='sf', world=2,
                    base=sha(Path(plan['root']) / '.runtime/training/pi05_base_torch/conversion.json'))
    if ck['schema'] != 1 or ck['step'] != 3000 or ck['identity'] != identity:
        raise ValueError('Wrong SF checkpoint step/schema/identity')
    manifest = json.loads((checkpoint.parent / 'parameters.json').read_text())
    expected = {p['name']: p for p in manifest['parameters'] if p['trainable']}
    if set(ck['trainable']) != set(expected):
        raise ValueError('Saved parameter names differ from the actual training model')
    for name, value in ck['trainable'].items():
        if list(value.shape) != expected[name]['shape'] or str(value.dtype) != expected[name]['dtype']:
            raise ValueError('Parameter shape/dtype mismatch: ' + name)
    names = list(ck['trainable'])
    ids = [i for group in ck['optimizer']['param_groups'] for i in group['params']]
    if len(ids) != len(names) or len(set(ids)) != len(ids):
        raise ValueError('Optimizer parameter ordering/count mismatch')
    gradients = checkpoint.parent / 'gradients-step-1.json'
    expected_unused = set(json.loads(gradients.read_text())['without_grad']) if gradients.exists() else set()
    observed_unused = {name for name, ident in zip(names, ids) if ident not in ck['optimizer']['state']}
    if observed_unused != expected_unused:
        raise ValueError('Missing Adam states differ from the recorded unused parameters')
    # Build only the trainable parameter tree, with checkpoint-backed CPU tensors.
    # Frozen base weights and GPU forwards are deliberately outside this CPU audit.
    model = nn.Module()
    for name, value in ck['trainable'].items():
        parent = model
        pieces = name.split('.')
        for part in pieces[:-1]:
            if part not in parent._modules:
                parent.add_module(part, nn.Module())
            parent = parent._modules[part]
        parent.register_parameter(pieces[-1], nn.Parameter(value, requires_grad=True))
    if [name for name, _ in model.named_parameters()] != names:
        raise ValueError('Reconstructed optimizer parameter ordering differs')
    tensors = 0
    for index, (name, ident) in enumerate(zip(names, ids)):
        state = ck['optimizer']['state'].get(ident)
        values = [ck['trainable'][name]]
        if state is not None:
            if int(state['step']) != 3000:
                raise ValueError('Inconsistent Adam step: ' + name)
            for key in ('exp_avg', 'exp_avg_sq'):
                if state[key].shape != ck['trainable'][name].shape:
                    raise ValueError('Adam moment shape mismatch: ' + name)
                values.append(state[key])
        for value in values:
            flat = value.reshape(-1)
            for chunk in flat.split(1024 * 1024):
                if not torch.isfinite(chunk).all():
                    raise ValueError('Nonfinite checkpoint tensor: ' + name)
            tensors += 1
        if (index+1) % 200 == 0:
            print('Verified parameter groups', index+1, '/', len(names), flush=True)
    print('All model tensors and Adam moments are finite', flush=True)
    optimizer = torch.optim.AdamW(model.parameters())
    # Exercise the production restore path, not just torch.load / file existence.
    # Model parameters already share these mmap-backed storages; no GPU model allocation.
    step = checkpoint_io.restore(reading, model, optimizer, identity, 0)
    if step != 3000 or len(optimizer.state) != len(ck['optimizer']['state']):
        raise ValueError('Optimizer restoration incomplete')
    if len(ck['rng']) != 2:
        raise ValueError('Missing rank RNG state')
    for rng in ck['rng']:
        checkpoint_io.restore_rng(rng)
    report = dict(passed=True, source=str(checkpoint.resolve()), checkpoint_sha256=copied['sha256'] if local_file else sha(checkpoint),
                  source_plan_sha256=sha(source_plan), step=step, arm='sf', world=2,
                  trainable_tensors=len(names), finite_tensors_checked=tensors,
                  optimizer_states_restored=len(optimizer.state), rank_rng_states_restored=2,
                  recorded_unused_parameters=sorted(observed_unused),
                  production_restore_passed=True, gpu_forward_tested=False,
                  cpu_validation_seconds=time.time()-started)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_plan'); parser.add_argument('checkpoint'); parser.add_argument('output')
    parser.add_argument('--local-file')
    a = parser.parse_args()
    audit(a.source_plan, a.checkpoint, a.output, a.local_file)
