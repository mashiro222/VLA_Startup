import hashlib
import json
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def episode_seed(level, task, init, repeat):
    # Condition, gamma, worker and run identifier are deliberately excluded.
    s = f'cbf-static-paired-v1/L{level}/t{task}/i{init}/r{repeat}'
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], 'little')


def environment(server=False):
    from arena_startup.runtime import environment as base
    env = base()
    env['PYTHONPATH'] = f"{ROOT / '.runtime/cbf_deps'}:{env['PYTHONPATH']}"
    if server:
        env['PYTHONPATH'] = f"{ROOT / '.runtime/cbf_server_deps'}:{env['PYTHONPATH']}"
        env.update(JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    return env


def record_path(run, episode):
    return Path(run) / 'results' / f"L{episode['level']}" / episode['name'] / episode['condition'] / f"g{episode['gamma']:g}" / f"seed{episode['repeat']}" / f"init{episode['init_index']:03d}"


def source_files():
    files = list((ROOT / 'src/cbf_static').glob('*.py')) + list((ROOT / 'config/cbf_static').glob('*'))
    files += [ROOT / x for x in ['scripts/cbf_static.py', 'slurm/cbf-static.sbatch',
              'scripts/cbf_sequence.py', 'slurm/cbf-sequence.sbatch',
              'scripts/setup_cbf_static.sh', 'scripts/download_cbf_checkpoint.py',
              'requirements/cbf-static.txt', 'requirements/cbf-static-server.txt',
              'src/arena_startup/adapter.py', 'src/arena_startup/transport.py',
              'src/arena_startup/diagnostics.py', 'src/arena_startup/runtime.py',
              'src/vla_startup/evaluate.py']]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}
