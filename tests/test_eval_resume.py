"""Resume protocol regressions: real sockets plus the production journal loop."""
import copy
import json
import multiprocessing
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.runtime/repos/openpi/src'),
               str(ROOT / '.runtime/repos/openpi/packages/openpi-client/src')]

from arena_startup.core import append, collect, episode_seed, key, save
from arena_startup import evaluate
from arena_startup.transport import ArenaClient
from eval_resume import runtime, workflow
from test_arena_transport import serve_fixture, METADATA


def row(unit, episode, plan, success=False):
    return dict(episode_key=key(unit, episode), plan_sha256=plan['sha256'],
                success=success, cost=0, steps=300, episode_seed=episode_seed(unit, episode))


def fixture(tmp_path):
    plan = dict(sha256='frozen-plan', expected_episodes=900, videos_per_worker=0,
                config={'evaluation': {'episode_timeout_seconds': 1800}},
                units=[dict(id=f'task{i}', trials=30, bddl='scene', init_file='init',
                            bddl_sha256='hash', init_sha256='hash') for i in range(30)])
    save(tmp_path / 'plan.json', plan)
    (tmp_path / 'workers').mkdir()
    for i, unit in enumerate(plan['units']):
        count = 30 if i < 11 else (11 if i == 11 else 0)
        journal = tmp_path / 'units' / unit['id'] / 'episodes.jsonl'
        if count: journal.parent.mkdir(parents=True)
        for episode in range(count): append(journal, row(unit, episode, plan))
    prefixes, pending = workflow.snapshot(tmp_path, plan)
    spec = dict(run=str(tmp_path), prefixes=prefixes, pending_keys=pending, max_attempts=3)
    return plan, spec


def mock_rollouts(monkeypatch, plan, calls, fail_at=None):
    monkeypatch.setattr(evaluate, 'file_hash', lambda _: 'hash')
    monkeypatch.setattr(evaluate.signal, 'alarm', lambda _: None)
    def rollout(unit, episode, protocol, client, *args):
        calls.append((key(unit, episode), episode_seed(unit, episode)))
        if fail_at is not None and len(calls) == fail_at:
            raise runtime.RetryableTransportError('simulated transport stall')
        return row(unit, episode, plan)
    monkeypatch.setattr(evaluate, 'rollout', rollout)


def test_production_loop_runs_only_559_and_keeps_all_original_bytes(tmp_path, monkeypatch):
    plan, spec = fixture(tmp_path)
    assert len(spec['pending_keys']) == 559
    assert workflow.verify_prefixes(tmp_path, plan, spec) == 341
    calls = []; mock_rollouts(monkeypatch, plan, calls)
    evaluate.evaluate_units(SimpleNamespace(run=tmp_path, worker='retry'), plan, None)
    assert [k for k, _ in calls] == spec['pending_keys']
    assert workflow.verify_prefixes(tmp_path, plan, spec) == 900
    # A completed run starts no server and reruns no task failures (all rows False).
    runtime.bounded_resume(None, tmp_path, spec, lambda *a: pytest.fail('already complete'))
    assert len({r['episode_key'] for r in collect(tmp_path, plan)}) == 900


def test_transport_replay_starts_only_unfinished_episode_with_identical_seed(tmp_path, monkeypatch):
    plan, spec = fixture(tmp_path)
    calls = []; mock_rollouts(monkeypatch, plan, calls, fail_at=4)
    launches = []
    def attempt(_, local, s, name):
        launches.append(name)
        try:
            evaluate.evaluate_units(SimpleNamespace(run=tmp_path, worker=name), plan, None)
        except runtime.RetryableTransportError:
            return runtime.RETRY_EXIT
        return 0
    runtime.bounded_resume(None, tmp_path, spec, attempt)
    assert len(launches) == 2 and len(calls) == 560
    assert calls[3] == calls[4]  # same episode and seed; no in-place action resend
    assert workflow.verify_prefixes(tmp_path, plan, spec) == 900
    assert len(collect(tmp_path, plan)) == 900


def test_retry_budget_is_bounded_and_fatal_errors_are_not_retried(tmp_path):
    plan, spec = fixture(tmp_path)
    calls = []
    def stalled(*args): calls.append(1); return runtime.RETRY_EXIT
    with pytest.raises(RuntimeError, match='budget exhausted'):
        runtime.bounded_resume(None, tmp_path, spec, stalled)
    assert len(calls) == 3 and workflow.verify_prefixes(tmp_path, plan, spec) == 341
    calls.clear()
    def fatal(*args): calls.append(1); return 1
    with pytest.raises(RuntimeError, match='Non-transport'):
        runtime.bounded_resume(None, tmp_path, spec, fatal)
    assert len(calls) == 1


def test_zero_exit_with_incomplete_coverage_is_rejected(tmp_path):
    _, spec = fixture(tmp_path)
    with pytest.raises(RuntimeError, match='before all episodes'):
        runtime.bounded_resume(None, tmp_path, spec, lambda *a: 0)


def test_changed_completed_prefix_is_rejected(tmp_path):
    plan, spec = fixture(tmp_path)
    path = tmp_path / 'units/task0/episodes.jsonl'
    path.write_bytes(path.read_bytes().replace(b'false', b'true ', 1))
    with pytest.raises(ValueError, match='prefix changed'):
        workflow.verify_prefixes(tmp_path, plan, spec)


def test_interrupted_traces_archived_without_touching_completed_traces(tmp_path):
    plan, _ = fixture(tmp_path)
    unit = tmp_path / 'units/task11'
    for episode in (10, 11):
        save(unit / f'diagnostics/{episode}/marker.json', {'original': episode})
        p = unit / f'videos/{episode}.mp4'; p.parent.mkdir(exist_ok=True); p.write_bytes(b'video')
    runtime.archive_incomplete(tmp_path, plan, 'attempt1')
    assert (unit / 'diagnostics/10/marker.json').exists()
    assert (unit / 'videos/10.mp4').exists()
    assert not (unit / 'diagnostics/11').exists()
    assert (unit / 'interrupted/attempt1/diagnostics/11/marker.json').exists()
    assert (unit / 'interrupted/attempt1/videos/11.mp4').read_bytes() == b'video'


@pytest.fixture
def real_server():
    procs = []
    def start(delay=0, error=False):
        ctx = multiprocessing.get_context('spawn'); queue = ctx.Queue()
        proc = ctx.Process(target=serve_fixture, args=(queue, False, delay, error))
        proc.start(); procs.append((proc, queue))
        return queue.get(timeout=60)
    yield start
    for proc, queue in procs:
        proc.terminate(); proc.join(10)
        if proc.is_alive(): proc.kill(); proc.join(5)
        queue.close()


def test_real_socket_timeout_then_new_server_keeps_input_and_rng(tmp_path, real_server):
    client = runtime.recovering_client_class(ArenaClient, tmp_path/'phase.json', tmp_path/'retry.json')
    observation = {'input': 3, '__vla_rng': [0, 1234]}
    with client(real_server(delay=.4), METADATA, request_timeout=.05) as c:
        with pytest.raises(runtime.RetryableTransportError, match='Policy response exceeded'):
            c.infer(observation)
    assert json.loads((tmp_path/'retry.json').read_text())['rng'] == [0, 1234]
    with client(real_server(), METADATA, request_timeout=5) as c:
        result = c.infer(observation)
    assert result['echo'] == observation


@pytest.mark.parametrize('kind', ['metadata', 'model'])
def test_real_protocol_errors_are_fatal(tmp_path, real_server, kind):
    client = runtime.recovering_client_class(ArenaClient, tmp_path/'phase.json', tmp_path/'retry.json')
    metadata = dict(METADATA, model='wrong') if kind == 'metadata' else METADATA
    with client(real_server(error=kind == 'model'), metadata) as c:
        with pytest.raises((ValueError, RuntimeError)):
            c.infer({'__vla_rng': [0, 3]})
    assert not (tmp_path/'retry.json').exists()


def test_episode_deadline_and_invalid_actions_do_not_become_transport_retries(tmp_path):
    for error in (TimeoutError('Episode wall-clock limit exceeded'), ValueError('Invalid action chunk')):
        class Fake:
            def infer(self, obs): raise error
        client = runtime.recovering_client_class(Fake, tmp_path/'phase.json', tmp_path/'retry.json')
        with pytest.raises(type(error), match=str(error)):
            client().infer({'__vla_rng': [0, 1]})
        assert not (tmp_path/'retry.json').exists()


def test_episode_alarm_wrapped_by_original_transport_is_still_fatal(tmp_path):
    class Socket:
        def send(self, data): pass
        def recv(self, **kwargs): raise TimeoutError('Episode wall-clock limit exceeded; infrastructure error')
    client_class = runtime.recovering_client_class(ArenaClient, tmp_path/'phase.json', tmp_path/'retry.json')
    c = client_class(1, METADATA)
    c.ws = Socket()
    with pytest.raises(TimeoutError): c.infer({'__vla_rng': [0, 1]})
    assert not (tmp_path/'retry.json').exists()


def test_failed_policy_does_not_leave_watchdog_armed(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.faulthandler, 'dump_traceback_later', lambda *a, **k: calls.append('armed'))
    monkeypatch.setattr(runtime.faulthandler, 'cancel_dump_traceback_later', lambda: calls.append('cancelled'))
    class Policy:
        def infer(self, obs): raise ValueError('Invalid action chunk')
    with pytest.raises(ValueError, match='Invalid action'):
        runtime.TracedPolicy(Policy(), tmp_path/'phase.json').infer({'__vla_rng': [0, 2]})
    assert calls == ['armed', 'cancelled']
    assert json.loads((tmp_path/'phase.json').read_text())['phase'] == 'infer_enter'


def test_policy_observability_preserves_output_and_rng(tmp_path):
    class Policy:
        def infer(self, obs): return {'actions': [[1]*7]*10, '__vla_rng': [obs['__vla_rng'][0]+1, 123]}
    obs = {'__vla_rng': [7, 123], 'prompt': 'task'}
    expected = Policy().infer(copy.deepcopy(obs))
    result = runtime.TracedPolicy(Policy(), tmp_path/'phase.json').infer(obs)
    assert result == expected and obs['__vla_rng'] == [7, 123]
    assert json.loads((tmp_path/'phase.json').read_text())['phase'] == 'infer_return'


def test_submit_requests_only_one_baseline_gpu_and_no_dependencies(tmp_path, monkeypatch):
    plan, spec = fixture(tmp_path)
    monkeypatch.setattr(workflow, 'check', lambda _: spec)
    submitted = []
    monkeypatch.setattr(workflow, 'submit_once', lambda d, label, cmd: submitted.append(cmd) or '123')
    assert workflow.submit(tmp_path/'resume.json', True) == '123'
    cmd = submitted[0]
    assert '--account=gts-jli741-paid' in cmd and '--gres=gpu:a100:1' in cmd
    assert '--time=24:00:00' in cmd and '--nodes=1' in cmd
    assert not any(s.startswith(('--dependency=', '--array=')) for s in cmd)


def test_supervisor_stops_old_server_before_retry_and_keeps_environments_separate(tmp_path, monkeypatch):
    _, spec = fixture(tmp_path)
    local = tmp_path/'local'; local.mkdir()
    processes = []
    class Process:
        def __init__(self, cmd, **kwargs):
            self.cmd, self.env = cmd, kwargs['env']
            self.returncode = runtime.RETRY_EXIT if 'client' in cmd else None
            self.stopped = False
            processes.append(self)
        def poll(self): return self.returncode
        def terminate(self): self.stopped = True; self.returncode = -15
        def wait(self, **kwargs): return self.returncode
    class Health:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(runtime.subprocess, 'Popen', Process)
    monkeypatch.setattr(runtime.urllib.request, 'urlopen', lambda *a, **k: Health())
    save(local/'attempt-client-phase.json', {'phase': 'request'})
    assert runtime.run_attempt(tmp_path/'spec.json', local, spec, 'attempt') == runtime.RETRY_EXIT
    assert len(processes) == 2 and processes[0].stopped
    assert processes[0].env['PYTHONPATH'].startswith(str(local/'site')+':')
    assert str(local/'site') not in processes[1].env['PYTHONPATH']
    assert processes[1].env['PYTHONPATH'].startswith(str(local/'arena-site')+':')
    assert (tmp_path/'workers/attempt-client-phase.json').exists()


def test_simulator_staging_keeps_cpu_version_and_excludes_build_only_files(tmp_path):
    site = tmp_path/'.runtime/arena/lib/python3.11/site-packages'
    for name in ('torch/__init__.py', 'torch/lib/libtorch_cpu.so', 'torch/include/header.h'):
        p = site/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'cpu-version')
    local = tmp_path/'local'
    runtime.stage_simulator_runtime(tmp_path, local)
    assert (local/'arena-site/torch/lib/libtorch_cpu.so').read_bytes() == b'cpu-version'
    assert not (local/'arena-site/torch/include').exists()
    assert not (local/'site').exists()
