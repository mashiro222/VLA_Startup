"""Run one explicit array unit. Never submits, cancels, or chains Slurm jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
import urllib.request
from .common import ROOT, digest, environment, record_path, save
from .plan import validate


def evaluate(run, index, port):
    from arena_startup.transport import ArenaClient
    from .rollout import rollout
    plan = json.loads((run / 'plan.json').read_text()); cfg = plan['config']
    expected = {'model': cfg['policy_config'], 'plan_sha256': digest(run / 'plan.json'),
                'norm_sha256': digest(ROOT / cfg['checkpoint'] / cfg['norm_stats']),
                'checkpoint_revision': json.loads((ROOT / 'config/cbf_static/checkpoint.json').read_text())['revision']}
    with ArenaClient(port, expected,
                     request_timeout=cfg['evaluation']['policy_timeout_seconds']) as client:
        for episode in plan['units'][index]['episodes']:
            path = record_path(run, episode)
            if (path / 'episode.json').exists(): continue
            def expired(*unused): raise TimeoutError('Episode exceeded wall-time deadline; no score recorded')
            previous = signal.signal(signal.SIGALRM, expired)
            signal.alarm(cfg['evaluation']['episode_timeout_seconds'])
            try:
                rollout(run, episode, cfg, client)
            except BaseException:
                save(path / 'infrastructure-error.json', {'traceback': traceback.format_exc(), 'job': os.environ.get('SLURM_JOB_ID'), 'time': time.time()})
                raise  # restart the worker explicitly; no silent failed-episode scoring
            finally:
                signal.alarm(0); signal.signal(signal.SIGALRM, previous)


def supervise(run, index):
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Model/EGL execution requires a submitted GPU allocation')
    plan = validate(run); unit = plan['units'][index]; cfg = plan['config']
    if all((record_path(run, e) / 'episode.json').exists() for e in unit['episodes']):
        print('Unit already complete; no model startup'); return 0
    workers = run / 'workers'; workers.mkdir(exist_ok=True)
    lock = (workers / f'{index}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for ep in unit['episodes']:
        for key, check in [('bddl', 'bddl_sha256'), ('init_file', 'init_sha256')]:
            if digest(ROOT / ep[key]) != ep[check]: raise ValueError('Task data changed after preparation')
    name = f"{os.environ['SLURM_JOB_ID']}-{index}"
    provenance = {'unit': index, 'unit_id': unit['id'], 'started': time.time(), 'host': socket.gethostname(),
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,uuid,driver_version', '--format=csv'], text=True),
        'plan_sha256': digest(run / 'plan.json'), 'job': os.environ['SLURM_JOB_ID']}
    save(workers / f'{name}.json', provenance)
    (workers / f'{name}-config.yaml').write_bytes((run / 'config.yaml').read_bytes())
    with socket.socket() as s: s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    processes = []
    def stop(*unused):
        for proc in processes:
            if proc.poll() is None: proc.terminate()
        raise InterruptedError('Slurm signal: finished episode records remain resumable')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    status = 1
    try:
        with (workers / f'{name}-server.log').open('w') as log:
            server = subprocess.Popen([str(ROOT / '.runtime/server/bin/python'), '-m', 'cbf_static.server', str(run), '--port', str(port)],
                env=environment(server=True), stdout=log, stderr=subprocess.STDOUT)
            processes.append(server); started = time.monotonic()
            while True:
                if server.poll() is not None: raise RuntimeError('Policy server exited; inspect server.log')
                if time.monotonic() - started > cfg['evaluation']['startup_timeout_seconds']: raise TimeoutError('Policy startup deadline exceeded')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2) as response:
                        if response.status == 200: break
                except OSError: pass
                time.sleep(2)
            client = subprocess.Popen([str(ROOT / '.runtime/arena/bin/python'), '-m', 'cbf_static.runtime', str(run),
                '--index', str(index), '--port', str(port)], env=environment())
            processes.append(client); status = client.wait()
            if status == 0 and plan['profile'] == 'calibration':
                status = subprocess.run([str(ROOT / '.runtime/arena/bin/python'), '-m', 'cbf_static.report', str(run), '--calibrate'], env=environment()).returncode
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for proc in reversed(processes):
            if proc.poll() is None: proc.terminate()
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        provenance.update(ended=time.time(), exit_code=status); save(workers / f'{name}.json', provenance)
        lock.close()
    return status


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run', type=Path); p.add_argument('--index', type=int, required=True)
    p.add_argument('--port', type=int); a = p.parse_args()
    if a.port: evaluate(a.run, a.index, a.port)
    else: raise SystemExit(supervise(a.run, a.index))
