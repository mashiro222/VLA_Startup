#!/usr/bin/env python3
"""Supervise the server and simulator in one already-allocated GPU job."""
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vla_startup.journal import read_records, validate_prefix


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Refusing to run on a login node; submit through Slurm')
    run = ROOT / 'runs' / sys.argv[1]
    lock = (run / '.job.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((run / 'manifest.json').read_text())
    records = read_records(run / 'episodes.jsonl')
    validate_prefix(records, manifest)
    if len(records) == manifest['expected_episodes']:
        print('Already complete; no model will be loaded.')
        return 0
    runtime = ROOT / '.runtime'
    provenance = {'job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(),
                  'started_unix': time.time(), 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv'], text=True)}
    (run / f"provenance-{os.environ['SLURM_JOB_ID']}.json").write_text(json.dumps(provenance, indent=2))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server_env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    sim_env = dict(os.environ)
    native = runtime / 'native-7.1.1/squashfs-root/usr'
    sim_env.update(PYTHONPATH=f"{runtime / 'repos' / ('libero_' + manifest['benchmark'])}:{runtime / 'repos/openpi/packages/openpi-client/src'}:{ROOT / 'src'}",
                   LIBERO_CONFIG_PATH=str(runtime / 'config' / manifest['benchmark']),
                   MAGICK_HOME=str(native), LD_LIBRARY_PATH=f"{native / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    processes = []
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        # Terminate the client first so it can record interruption and close logs.
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    code = 1
    try:
        with (run / f"server-{os.environ['SLURM_JOB_ID']}.log").open('w') as log:
            server = subprocess.Popen([str(runtime / 'server/bin/python'), '-m', 'vla_startup.server',
                                       '--checkpoint', str(runtime / 'checkpoints/pi05_libero'), '--port', str(port),
                                       '--num-steps', str(manifest['evaluation']['num_steps'])],
                                      env=server_env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(server)
            deadline = time.monotonic() + 900
            while True:
                if interrupted or server.poll() is not None:
                    raise RuntimeError('Server stopped before becoming ready; inspect server log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Policy startup timed out')
                time.sleep(2)
            client = subprocess.Popen([str(runtime / 'sim/bin/python'), '-m', 'vla_startup.evaluate',
                                       '--run-dir', str(run), '--port', str(port),
                                       '--videos', '2' if manifest['profile'].startswith('smoke') else '0'], env=sim_env)
            processes.append(client)
            code = client.wait()
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        subprocess.run([sys.executable, '-m', 'vla_startup.summarize', str(run)],
                       env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')), check=True)
        if read_records(run / 'episodes.jsonl'):
            subprocess.run([str(runtime / 'sim/bin/python'), str(ROOT / 'scripts/report.py'), run.name],
                           env=sim_env, check=True)
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(main())
