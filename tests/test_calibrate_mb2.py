import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from arena_startup.core import save

spec = importlib.util.spec_from_file_location(
    'calibrate_mb2', Path(__file__).resolve().parents[1] / 'scripts/calibrate_mb2.py')
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)


def completed(root, micro, arm):
    case = root / f'mb{micro}' / arm
    save(case / 'status.json', dict(completed=True, step=12))
    save(case / 'runtime.json', dict(global_batch=32, world=2, micro_batch=micro))
    save(case / 'inference_smoke.json', dict(finite=True))
    (case / 'metrics.jsonl').write_text(''.join(json.dumps(dict(
        step=step, loss=0.1, flow_mse=0.1, grad_norm=1.0, seconds=10.0)) + '\n'
        for step in range(1, 13)))


@pytest.fixture
def calibration(tmp_path):
    for micro in (1, 2):
        save(tmp_path / f'mb{micro}/plan.json', dict(profile='smoke', gpus=2,
             recipe=dict(global_batch=32, micro_batch=micro, steps=12)))
    for arm in ('baseline', 'sf'):
        completed(tmp_path, 1, arm)
    return tmp_path


def test_retry_runs_only_batch2_and_keeps_batch1_evidence(calibration, monkeypatch):
    before = {p: p.read_bytes() for p in (calibration / 'mb1').rglob('*') if p.is_file()}
    calls, reports = [], []
    def train(command, check):
        assert check
        assert command[command.index('--plan') + 1] == str(calibration / 'mb2/plan.json')
        arm = command[command.index('--arm') + 1]
        calls.append(arm)
        completed(calibration, 2, arm)
    monkeypatch.setattr(retry.subprocess, 'run', train)
    monkeypatch.setattr(retry, 'summarize', reports.append)
    retry.run(calibration)
    assert calls == ['baseline', 'sf'] and reports == [calibration]
    assert all(p.read_bytes() == content for p, content in before.items())


def test_completed_batch2_arm_is_not_repeated(calibration):
    completed(calibration, 2, 'baseline')
    assert retry.pending_arms(calibration) == ['sf']


def test_partial_output_is_not_overwritten(calibration):
    case = calibration / 'mb2/baseline'
    case.mkdir()
    (case / 'metrics.jsonl').write_text('{"step": 1}\n')
    with pytest.raises(ValueError, match='partial'):
        retry.pending_arms(calibration)


def test_incomplete_batch1_evidence_is_not_retrained(calibration):
    save(calibration / 'mb1/sf/status.json', dict(completed=False, step=4))
    with pytest.raises(ValueError, match='Incomplete'):
        retry.pending_arms(calibration)


def test_training_failure_never_creates_pass_report(calibration, monkeypatch):
    def fail(command, check):
        raise subprocess.CalledProcessError(1, command)
    def forbidden(*args):
        pytest.fail('Failed training must not produce a calibration report')
    monkeypatch.setattr(retry.subprocess, 'run', fail)
    monkeypatch.setattr(retry, 'summarize', forbidden)
    with pytest.raises(subprocess.CalledProcessError):
        retry.run(calibration)
