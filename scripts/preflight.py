#!/usr/bin/env python3
"""Read-only readiness checks. Does not create environments or use a GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vla_startup.manifest import build, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--imports', action='store_true')
    args = parser.parse_args()
    sources = json.loads((ROOT / 'config/sources.json').read_text())
    runtime = ROOT / '.runtime'
    failures = []
    for name in ('openpi', 'libero_plus', 'libero_clean'):
        repo = runtime / 'repos' / name
        result = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        if result.returncode or result.stdout.strip() != sources[name]['commit']:
            failures.append(f'Wrong or missing source revision: {name}')
        elif subprocess.run(['git', '-C', str(repo), 'diff', '--quiet', 'HEAD'], capture_output=True).returncode:
            failures.append(f'Modified tracked upstream files: {name}')
    for relative in ('server/bin/python', 'sim/bin/python',
                     'cache/openpi/big_vision/paligemma_tokenizer.model',
                     'checkpoints/pi05_libero/.verified.json',
                     'native-7.1.1/squashfs-root/usr/lib/libMagickWand-7.Q16HDRI.so',
                     'repos/libero_plus/libero/libero/assets/.vla-assets-sha256'):
        if not (runtime / relative).is_file():
            failures.append(f'Missing runtime file: {relative}')
    marker = runtime / 'checkpoints/pi05_libero/.verified.json'
    if marker.exists() and json.loads(marker.read_text()) != json.loads((ROOT / 'config/checkpoint_files.json').read_text()):
        failures.append('Checkpoint verification marker is stale')
    marker = runtime / 'repos/libero_plus/libero/libero/assets/.vla-assets-sha256'
    if marker.exists() and marker.read_text().strip() != sources['assets']['sha256']:
        failures.append('Scene asset verification marker is stale')
    if not (ROOT / 'requirements/sim.lock').is_file():
        failures.append('Missing requirements/sim.lock')
    for row in json.loads((ROOT / 'config/checkpoint_files.json').read_text()):
        path = runtime / 'checkpoints/pi05_libero' / row['name'].split('checkpoints/pi05_libero/', 1)[1]
        if not path.exists() or path.stat().st_size != int(row['size']):
            failures.append(f'Missing or wrong-size checkpoint file: {path.name}')
    if args.run_dir:
        manifest = json.loads((args.run_dir / 'manifest.json').read_text())
        validate(manifest)
        rebuilt = build(ROOT, manifest['profile'], manifest['seed'], manifest['prompt_mode'])
        if rebuilt != manifest:
            failures.append('Run manifest differs from current source data')
    if args.imports and not failures:
        import_env = dict(os.environ, MPLCONFIGDIR=str(runtime / 'cache/matplotlib'),
                          XDG_CACHE_HOME=str(runtime / 'cache/xdg'),
                          NUMBA_CACHE_DIR=str(runtime / 'cache/numba'),
                          HF_HOME=str(runtime / 'cache/huggingface'),
                          WANDB_MODE='disabled', HF_HUB_DISABLE_TELEMETRY='1',
                          OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        for mode in ('clean', 'plus'):
            print(f'Checking simulation imports: {mode}', flush=True)
            env = dict(import_env)
            native = runtime / 'native-7.1.1/squashfs-root/usr'
            env.update(PYTHONPATH=f"{runtime / 'repos' / ('libero_' + mode)}:{runtime / 'repos/openpi/packages/openpi-client/src'}:{ROOT / 'src'}",
                       LIBERO_CONFIG_PATH=str(runtime / 'config' / mode),
                       MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MAGICK_HOME=str(native),
                       LD_LIBRARY_PATH=f"{native / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}")
            subprocess.run([str(runtime / 'sim/bin/python'), '-c',
                            'from libero.libero import benchmark; from libero.libero.envs import OffScreenRenderEnv; '
                            'from openpi_client import image_tools; '
                            'from wand.image import Image; print("Simulation imports OK")'], check=True, env=env)
        print('Checking OpenPI imports in CPU mode; no weights will be loaded', flush=True)
        env = dict(import_env, JAX_PLATFORMS='cpu', OPENPI_DATA_HOME=str(runtime / 'cache/openpi'))
        subprocess.run([str(runtime / 'server/bin/python'), '-c',
                        'from openpi.training.config import get_config; c=get_config("pi05_libero"); '
                        'assert c.model.action_horizon == 10; print("OpenPI config imports OK; weights not loaded")'],
                       check=True, env=env)
    if failures:
        print('\n'.join('FAIL: ' + error for error in failures))
        return 1
    print('PASS: local readiness checks. GPU inference and EGL rendering still require a submitted smoke job.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
