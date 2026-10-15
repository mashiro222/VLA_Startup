"""Environment isolation and GPU-job process supervision (stdlib only)."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from .core import ROOT, collect, save, validate_plan


def environment():
    rt = ROOT / '.runtime'
    native = rt / 'native-7.1.1/squashfs-root/usr'
    env = dict(os.environ)
    env.update(PYTHONPATH=f"{ROOT / 'src'}:{rt / 'repos/vla_arena'}:{rt / 'repos/openpi/packages/openpi-client/src'}:{rt / 'repos/openpi/src'}",
               MAGICK_HOME=str(native), LD_LIBRARY_PATH=f"{native / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}",
               OPENPI_DATA_HOME=str(rt / 'cache/openpi'), HF_HOME=str(rt / 'cache/huggingface'),
               LIBERO_CONFIG_PATH=str(rt / 'config/arena'),
               XDG_CACHE_HOME=str(rt / 'cache/xdg'), MPLCONFIGDIR=str(rt / 'cache/arena-matplotlib'),
               NUMBA_CACHE_DIR=str(rt / 'cache/arena-numba'), PYTHONUNBUFFERED='1',
               WANDB_MODE='disabled', HF_HUB_DISABLE_TELEMETRY='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
    return env


def supervise(run):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU evaluation requires a Slurm allocation; login-node inference is disabled')
    plan = json.loads((run / 'plan.json').read_text())
    validate_plan(plan)
    if len(collect(run, plan)) == plan['expected_episodes']:
        print('Already complete; skipping policy startup')
        return 0
    rt = ROOT / '.runtime'
    worker = f"{os.environ['SLURM_JOB_ID']}-{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    provenance = {'worker': worker, 'job_id': os.environ['SLURM_JOB_ID'],
                  'array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'), 'hostname': socket.gethostname(),
                  'started_unix': time.time(), 'plan_sha256': plan['sha256'],
                  'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv'], text=True),
                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    save(run / 'workers' / f'{worker}.json', provenance)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = environment()
    env.update(JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    env['MPLCONFIGDIR'] += '-' + worker
    env['NUMBA_CACHE_DIR'] += '-' + worker
    processes = []
    interrupted = False

    def stop(*unused):
        nonlocal interrupted
        interrupted = True
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    code = 1
    try:
        with (run / 'workers' / f'{worker}-server.log').open('w') as log:
            server_env = dict(env, PYTHONPATH=str(ROOT / 'src'))
            server = subprocess.Popen([str(rt / 'server/bin/python'), '-m', 'arena_startup.server',
                                       '--checkpoint', str(ROOT / plan['config']['checkpoint']),
                                       '--port', str(port), '--num-steps', str(plan['config']['evaluation']['num_steps'])],
                                      stdout=log, stderr=subprocess.STDOUT, env=server_env)
            processes.append(server)
            timeout = plan['config']['evaluation']['policy_startup_timeout_seconds']
            start = time.monotonic()
            next_log = start
            while True:
                if interrupted or server.poll() is not None:
                    raise RuntimeError('Policy stopped before readiness; see worker server log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f'Policy startup exceeded {timeout}s')
                if time.monotonic() >= next_log:
                    print(f'Waiting for policy: {time.monotonic() - start:.0f}s', flush=True)
                    next_log = time.monotonic() + 60
                time.sleep(2)
            client = subprocess.Popen([str(rt / 'arena/bin/python'), '-m', 'arena_startup.evaluate',
                                       str(run), '--port', str(port), '--worker', worker], env=env)
            processes.append(client)
            code = client.wait()
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        provenance.update(ended_unix=time.time(), exit_code=code, interrupted=interrupted)
        save(run / 'workers' / f'{worker}.json', provenance)
    return code if code >= 0 else 128 - code
