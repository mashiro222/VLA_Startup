"""Stage immutable training inputs on node-local disk before distributed startup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def copy_verified(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    marker = destination.with_suffix(destination.suffix + '.verified.json')
    identity = dict(source=str(source.resolve()), size=source.stat().st_size,
                    mtime_ns=source.stat().st_mtime_ns, expected_sha256=expected)
    if destination.exists() and marker.exists():
        if json.loads(marker.read_text()) == identity and destination.stat().st_size == identity['size']:
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    digest = hashlib.sha256()
    with source.open('rb') as reader, temporary.open('wb') as writer:
        while block := reader.read(8 * 1024 * 1024):
            writer.write(block)
            digest.update(block)
        writer.flush()
        os.fsync(writer.fileno())
    if temporary.stat().st_size != identity['size'] or (expected and digest.hexdigest() != expected):
        temporary.unlink()
        raise ValueError('Local staging checksum/size mismatch: ' + str(source))
    os.chmod(temporary, source.stat().st_mode & 0o777)
    os.replace(temporary, destination)
    marker.write_text(json.dumps(identity))


def stage_runtime(root, local):
    site = root / '.runtime/sf_train/lib/python3.11/site-packages'
    for package in ('torch', 'nvidia'):
        for source in sorted((site / package).rglob('*')):
            relative = source.relative_to(site)
            # Headers are unused by this eager training path. Keep their original
            # paths reachable without copying thousands of build-only files.
            if 'include' in relative.parts:
                continue
            if source.is_file() and '__pycache__' not in source.parts:
                copy_verified(source, local / 'site' / source.relative_to(site))
        for headers in (site / package).glob('*/include'):
            target = local / 'site' / headers.relative_to(site)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): target.symlink_to(headers, target_is_directory=True)
    headers=site / 'torch/include'; target=local / 'site/torch/include'
    if not target.exists(): target.symlink_to(headers, target_is_directory=True)
    print('Node-local PyTorch/CUDA runtime staged', flush=True)


def stage_training(spec, local):
    root = Path(spec['root'])
    home = root / '.runtime/training'
    manifest = json.loads((home / 'arena_l0/packed/complete.json').read_text())
    sources = [(home / 'arena_l0/packed' / name,
                local / 'arena_l0/packed' / name, info['sha256'])
               for name, info in manifest['files'].items()]
    for name in ('meta/episodes.jsonl', 'meta/tasks.jsonl', 'packed/complete.json'):
        sources.append((home / 'arena_l0' / name, local / 'arena_l0' / name, None))
    conversion = json.loads((home / 'pi05_base_torch/conversion.json').read_text())
    sources.append((home / 'pi05_base_torch/model.safetensors', local / 'base.safetensors', conversion['sha256']))
    sources.append((home / 'vggt/model.pt', local / 'teacher.pt', spec['teacher_sha256']))
    run = Path(spec['continuation'])
    checkpoints = sorted((run / 'sf').glob('step-*.pt'))
    if not checkpoints:
        raise FileNotFoundError('Checkpoint migration must finish before local staging')
    sources.append((checkpoints[-1], local / 'resume.pt', None))
    required = sum(p.stat().st_size for p, _, _ in sources)
    if shutil.disk_usage(local).free < required + 2 * 1024**3:
        raise OSError('Insufficient node-local space for immutable training inputs')
    for source, destination, expected in sources:
        print('Staging', source.name, flush=True)
        copy_verified(source, destination, expected)
    (local / 'resume-source.json').write_text(json.dumps(dict(source=str(checkpoints[-1].resolve()))))
    print('Node-local training inputs ready', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    parser.add_argument('local', type=Path)
    parser.add_argument('--runtime-only', action='store_true')
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    args.local.mkdir(parents=True, exist_ok=True)
    stage_runtime(Path(spec['root']), args.local)
    if not args.runtime_only:
        stage_training(spec, args.local)
