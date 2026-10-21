"""Restart a timed-out policy and simulator together, preserving scored episodes.

Only transport failures are retried, at most twice in the same allocation. An
unfinished episode is reset to its original initial state and RNG. A task failure
is a scored result and is never retried. No Slurm jobs are created here.
"""
import argparse
import faulthandler
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import traceback
import urllib.request

from arena_startup.core import ROOT, append, claim, key, records, save
from eval_resume.workflow import check, verify_prefixes

RETRY_EXIT = 75


class RetryableTransportError(Exception):
    """Only a failed policy transport request permits an episode replay."""


def recovering_client_class(original_client, phase_path, retry_path):
    from websockets.exceptions import ConnectionClosed
    class ObservedClient(original_client):
        def infer(self, obs):
            save(phase_path, dict(phase='request', rng=obs.get('__vla_rng'), unix=time.time()))
            try:
                result = super().infer(obs)
            except (TimeoutError, ConnectionClosed, OSError) as error:
                # The episode SIGALRM may interrupt recv too; it must not be
                # confused with its narrower, explicitly labelled request timeout.
                if isinstance(error, TimeoutError) and not str(error).startswith('Policy response exceeded '):
                    raise
                if isinstance(error.__cause__, TimeoutError) and str(error.__cause__).startswith('Episode wall-clock limit exceeded'):
                    raise
                save(retry_path, dict(error=repr(error), unix=time.time(), rng=obs.get('__vla_rng')))
                raise RetryableTransportError(str(error)) from error
            save(phase_path, dict(phase='response', rng=obs.get('__vla_rng'), unix=time.time()))
            return result
    return ObservedClient


def archive_incomplete(run, plan, attempt):
    """Keep diagnostics of every interrupted episode before its deterministic replay."""
    for unit in plan['units']:
        directory = run / 'units' / unit['id']
        first = len(records(directory / 'episodes.jsonl'))
        for episode in range(first, unit['trials']):
            for relative in (Path('diagnostics') / str(episode), Path('videos') / f'{episode}.mp4'):
                source = directory / relative
                if source.exists():
                    target = directory / 'interrupted' / attempt / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists(): raise FileExistsError(target)
                    source.rename(target)


def local_inputs(spec, local):
    from sf_recovery.stage import copy_verified, stage_runtime
    plan = json.loads((Path(spec['run']) / 'plan.json').read_text())
    files = [(ROOT / '.runtime/training/pi05_base_torch/model.safetensors', local / 'base.safetensors', spec['base_sha256']),
             (Path(plan['trained_policy']['checkpoint']), local / 'checkpoint.pt', spec['checkpoint_sha256'])]
    if shutil.disk_usage(local).free < sum(p.stat().st_size for p, _, _ in files) + 15 * 1024**3:
        raise OSError('Insufficient local disk for policy/runtime staging')
    print('Staging policy and simulator runtimes', flush=True)
    stage_runtime(ROOT, local)
    stage_simulator_runtime(ROOT, local)
    for source, target, expected in files:
        print('Staging verified policy input:', source.name, flush=True)
        copy_verified(source, target, expected)


def stage_simulator_runtime(root, local):
    """Keep the simulator's CPU PyTorch separate from policy CUDA PyTorch."""
    from sf_recovery.stage import copy_verified
    site = root / '.runtime/arena/lib/python3.11/site-packages'
    for source in sorted((site / 'torch').rglob('*')):
        relative = source.relative_to(site)
        if source.is_file() and 'include' not in relative.parts and '__pycache__' not in relative.parts:
            copy_verified(source, local / 'arena-site' / relative)
    print('Node-local simulator CPU PyTorch staged', flush=True)


class TracedPolicy:
    def __init__(self, policy, phase):
        self.policy, self.phase = policy, phase

    def infer(self, obs):
        started = time.monotonic()
        context = dict(rng=obs.get('__vla_rng'), prompt=obs.get('prompt'), unix=time.time())
        save(self.phase, dict(phase='infer_enter', **context))
        faulthandler.dump_traceback_later(120, repeat=True)
        try:
            result = self.policy.infer(obs)
            save(self.phase, dict(phase='infer_return', seconds=time.monotonic()-started, **context))
            return result
        finally:
            faulthandler.cancel_dump_traceback_later()


def server(spec_path, local, port, attempt):
    from sf_followup import evaluation
    spec = check(spec_path)
    plan = json.loads((Path(spec['run']) / 'plan.json').read_text())
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    # Same policy builder, weights, normalization and inference function. Redirect
    # only large immutable input reads to checksum-verified node-local copies.
    import torch
    import sf_training.model as model
    original_build, original_load, original_hash = model.build_policy, torch.load, evaluation.file_hash
    checkpoint = Path(plan['trained_policy']['checkpoint']).resolve()
    model.build_policy = lambda cfg, path: original_build(cfg, local / 'base.safetensors')
    def load(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).resolve() == checkpoint:
            path = local / 'checkpoint.pt'
        return original_load(path, *args, **kwargs)
    torch.load = load
    evaluation.file_hash = lambda path: original_hash(local / 'checkpoint.pt' if Path(path).resolve() == checkpoint else path)
    # Preserve the original order: validate the training package inventory before
    # exposing the separate WebSocket dependency directory.
    import sf_training.workflow as training
    original_verify = training.verify_plan
    def verify_and_instrument(path):
        train = original_verify(path)
        import sys
        deps = ROOT / '.runtime/sf_followup_deps'
        for name, expected in json.loads((ROOT / 'config/sf_followup_dependencies.json').read_text()).items():
            if original_hash(deps / name) != expected: raise ValueError('Transport sidecar dependency changed')
        sys.path.insert(0, str(deps))
        import arena_startup.transport as transport
        original_server = transport.ArenaPolicyServer
        class ObservedServer(original_server):
            def __init__(self, policy, *args, **kwargs):
                faulthandler.cancel_dump_traceback_later()
                super().__init__(TracedPolicy(policy, local / f'{attempt}-server-phase.json'), *args, **kwargs)
        transport.ArenaPolicyServer = ObservedServer
        return train
    training.verify_plan = verify_and_instrument
    evaluation.server(Path(spec['run']), port)


def client(spec_path, local, port, attempt):
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    from sf_followup import evaluation
    from arena_startup import evaluate, transport
    spec = check(spec_path)
    transport.ArenaClient = recovering_client_class(transport.ArenaClient,
        local / f'{attempt}-client-phase.json', local / f'{attempt}-retry.json')
    original_rollout = evaluate.rollout
    def traced_rollout(unit, episode, *args, **kwargs):
        save(local / f'{attempt}-episode.json', dict(episode_key=key(unit, episode), unix=time.time()))
        return original_rollout(unit, episode, *args, **kwargs)
    evaluate.rollout = traced_rollout
    try:
        evaluation.client(Path(spec['run']), port, attempt)
    except RetryableTransportError:
        traceback.print_exc()
        return RETRY_EXIT
    finally:
        faulthandler.cancel_dump_traceback_later()
    return 0


def stop_process(proc):
    if proc.poll() is None: proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)


def run_attempt(spec_path, local, spec, attempt):
    from arena_startup.runtime import environment
    run = Path(spec['run'])
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    server_env = dict(os.environ)
    server_env['PYTHONPATH'] = ':'.join(map(str, [local / 'site', ROOT / 'src',
        ROOT / '.runtime/repos/spatial_forcing/openpi-SF/src', ROOT / '.runtime/repos/openpi/packages/openpi-client/src']))
    client_env = environment()
    client_env['PYTHONPATH'] = str(local / 'arena-site') + ':' + client_env['PYTHONPATH']
    for env in (server_env, client_env):
        env.update(PYTHONUNBUFFERED='1', TORCHINDUCTOR_CACHE_DIR=str(local / 'inductor'), TRITON_CACHE_DIR=str(local / 'triton'))
    common = ['-m', 'eval_resume.runtime']
    args = [str(spec_path), str(local), '--port', str(port), '--attempt', attempt]
    processes = []
    try:
        with (run / 'workers' / f'{attempt}-server.log').open('x') as log:
            p = subprocess.Popen([str(ROOT / '.runtime/sf_train/bin/python'), *common, 'server', *args],
                                 env=server_env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(p)
            start = time.monotonic()
            while True:
                if p.poll() is not None: raise RuntimeError('Policy startup failed; inspect server log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2) as response:
                        if response.status == 200: break
                except OSError: pass
                if time.monotonic()-start > 1800: raise TimeoutError('Policy startup exceeded 1800s')
                time.sleep(2)
            c = subprocess.Popen([str(ROOT / '.runtime/arena/bin/python'), *common, 'client', *args], env=client_env)
            processes.append(c)
            client_started = time.monotonic()
            while c.poll() is None:
                # Bound simulator cleanup after a known transport failure. Normal
                # episode timeout has a separate, non-retryable watchdog.
                retry = local / f'{attempt}-retry.json'
                episode = local / f'{attempt}-episode.json'
                if not episode.exists() and time.monotonic()-client_started > 1800:
                    raise TimeoutError('Simulator startup exceeded 1800s; inspect client stack')
                if retry.exists() and time.time()-json.loads(retry.read_text())['unix'] > 60:
                    stop_process(c)
                    return RETRY_EXIT
                if not retry.exists() and episode.exists() and time.time()-json.loads(episode.read_text())['unix'] > 1860:
                    raise TimeoutError('Episode watchdog exceeded 1860s; manual diagnosis required')
                time.sleep(1)
            return c.returncode
    finally:
        for proc in reversed(processes): stop_process(proc)
        for path in local.glob(f'{attempt}-*.json'):
            shutil.copy2(path, run / 'workers' / path.name)


def bounded_resume(spec_path, local, spec, attempt_fn=run_attempt):
    run = Path(spec['run'])
    plan = json.loads((run / 'plan.json').read_text())
    for index in range(spec['max_attempts']):
        completed = verify_prefixes(run, plan, spec)
        if completed == plan['expected_episodes']: return
        attempt = f"{os.environ.get('SLURM_JOB_ID', 'test')}-attempt{index+1}"
        archive_incomplete(run, plan, attempt)
        print(f'{attempt}: {completed} complete, {plan["expected_episodes"]-completed} remaining', flush=True)
        code = attempt_fn(spec_path, local, spec, attempt)
        after = verify_prefixes(run, plan, spec)
        append(run / 'workers/resume-attempts.jsonl', dict(attempt=attempt, exit_code=code, before=completed, after=after, unix=time.time()))
        if code == 0:
            if after != plan['expected_episodes']: raise RuntimeError('Worker exited before all episodes completed')
            return
        if code != RETRY_EXIT: raise RuntimeError(f'Non-transport failure (exit {code}); no automatic retry')
    raise RuntimeError('Transport retry budget exhausted; completed journals retained')


def worker(spec_path, local):
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('GPU allocation required')
    spec = check(spec_path)
    run = Path(spec['run'])
    plan = json.loads((run / 'plan.json').read_text())
    def interrupted(*_): raise InterruptedError('Allocation interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    with claim(run / 'resume-worker') as owned:
        if not owned: raise RuntimeError('Another recovery worker is active')
        if verify_prefixes(run, plan, spec) < plan['expected_episodes']:
            local_inputs(spec, local)
            bounded_resume(spec_path, local, spec)
        from arena_startup.runtime import environment
        # The stdlib supervisor has no plotting packages; use the pinned evaluator.
        subprocess.run([str(ROOT / '.runtime/arena/bin/python'), '-m', 'sf_followup.evaluation',
                        'report', str(run)], env=environment(), check=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('worker', 'server', 'client'))
    p.add_argument('spec', type=Path); p.add_argument('local', type=Path)
    p.add_argument('--port', type=int); p.add_argument('--attempt')
    a = p.parse_args()
    if a.mode == 'worker': worker(a.spec, a.local)
    elif a.mode == 'server': server(a.spec, a.local, a.port, a.attempt)
    else: raise SystemExit(client(a.spec, a.local, a.port, a.attempt))
