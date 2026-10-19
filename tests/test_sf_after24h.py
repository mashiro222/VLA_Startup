import copy
import json
from pathlib import Path

import pytest

from arena_startup.core import file_hash, save
from sf_followup import after24h, evaluation


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    save(source / 'plan.json', dict(gpus=2, recipe=dict(global_batch=32, micro_batch=1, steps=60000,
                                                      keep_checkpoints=2, seed=7),
                                    assets={'norm_stats': 'normalization-hash'}, files={}, environment={}))
    units = [dict(suite=suite, level=level, task_id=task, id=f'{suite}-{level}-{task}',
                  init_count=50, seed=7, trials=20)
             for suite in ('safety_static_obstacles', 'safety_dynamic_obstacles')
             for level in range(3) for task in range(5)]
    save(tmp_path / 'runs/arena-balanced1500-001/plan.json',
         dict(units=units, config=dict(profiles={}, evaluation=dict(init_state_offset=0,
                                                                  init_state_selection_mode='episode_idx'))))
    root = tmp_path / 'campaign'
    root.mkdir()
    c = dict(source_plan=str(source / 'plan.json'), source_plan_sha256=file_hash(source / 'plan.json'),
             original_jobs={'baseline': '111', 'sf': '222'}, calibration_job='333',
             calibration=str(tmp_path / 'calibration'), additional_updates=15000,
             trials_per_task=30, episodes_per_arm=900, pinned=str(root / 'pinned'),
             evaluations={arm: str(tmp_path / f'eval-{arm}') for arm in after24h.ARMS},
             continuations={arm: str(tmp_path / f'continue-{arm}') for arm in after24h.ARMS})
    path = root / 'campaign.json'
    save(path, c)
    monkeypatch.setattr(after24h, 'check', lambda _: copy.deepcopy(c))
    monkeypatch.setattr(evaluation, 'ROOT', tmp_path)
    monkeypatch.setattr(evaluation, 'frozen_files', lambda: {})
    monkeypatch.setattr(evaluation, 'followup_files', lambda: {})
    monkeypatch.setattr(evaluation, 'validate', lambda _: None)
    return path, c


def completed_allocation(source, arm, step):
    save(source / arm / 'status.json', dict(step=step, stop_reason='wall_time_budget'))
    (source / arm / f'step-{step:08d}.pt').write_bytes(b'complete atomic checkpoint')


@pytest.mark.parametrize('final_step', [4875, 6200])
def test_one_arm_can_prepare_without_the_other_and_uses_final_not_5k(campaign, final_step):
    path, c = campaign
    source = Path(c['source_plan']).parent
    completed_allocation(source, 'baseline', final_step)
    (source / 'baseline/step-00004000.pt').write_bytes(b'older periodic checkpoint')
    if final_step > 5000:
        (source / 'baseline/step-00005000.pt').write_bytes(b'older 5k checkpoint')
    selection = after24h.materialize(path, 'baseline')
    assert selection['source_step'] == final_step and selection['target_step'] == final_step + 15000
    assert not (source / 'sf/status.json').exists()
    ep = json.loads((Path(c['evaluations']['baseline']) / 'plan.json').read_text())
    cp = json.loads((Path(c['continuations']['baseline']) / 'plan.json').read_text())
    assert ep['expected_episodes'] == 900 and ep['trained_policy']['step'] == final_step
    assert ep['trained_policy']['checkpoint'] == selection['checkpoint']
    assert cp['migration']['source_step'] == final_step and cp['recipe']['steps'] == final_step + 15000
    assert cp['gpus'] * cp['recipe']['micro_batch'] * 4 == cp['recipe']['global_batch'] == 32
    assert after24h.materialize(path, 'baseline') == selection
    (source / 'baseline' / f'step-{final_step:08d}.pt').unlink()
    assert Path(selection['checkpoint']).read_bytes() == b'complete atomic checkpoint'
    assert after24h.materialize(path, 'baseline') == selection


def test_different_final_steps_are_explicitly_allowed_only_for_24h_comparison(campaign):
    path, c = campaign
    source = Path(c['source_plan']).parent
    for arm, step in [('baseline', 4875), ('sf', 3650)]:
        completed_allocation(source, arm, step)
        selection = after24h.materialize(path, arm)
        assert selection['target_step'] == step + 15000
    plans = [json.loads((Path(c['evaluations'][arm]) / 'plan.json').read_text()) for arm in after24h.ARMS]
    assert plans[0]['units'] == plans[1]['units']
    with pytest.raises(ValueError, match='budget'):
        evaluation.validate_comparison(plans)
    evaluation.validate_comparison(plans, require_equal_steps=False)


def test_missing_final_checkpoint_never_falls_back_to_periodic_file(tmp_path):
    completed_allocation(tmp_path, 'baseline', 4000)
    save(tmp_path / 'baseline/status.json', dict(step=4875, stop_reason='wall_time_budget'))
    (tmp_path / 'baseline/step-00004875.tmp').write_bytes(b'incomplete')
    with pytest.raises(ValueError, match='latest complete'):
        after24h.final_checkpoint(tmp_path, 'baseline')


@pytest.mark.parametrize('step,reason', [(0, 'wall_time_budget'), ('5000', 'wall_time_budget'), (5000, 'error')])
def test_abnormal_final_status_is_not_accepted(tmp_path, step, reason):
    save(tmp_path / 'baseline/status.json', dict(step=step, stop_reason=reason))
    with pytest.raises(ValueError, match='normal final'):
        after24h.final_checkpoint(tmp_path, 'baseline')


def test_job_templates_have_no_total_cap_or_cross_job_dependencies(campaign):
    path, c = campaign
    total = 0
    for arm in after24h.ARMS:
        for stage in ('eval', 'continue'):
            cmd = after24h.stage_command(path, c, arm, stage)
            assert not any(x.startswith(('--dependency=', '--array=')) for x in cmd)
            assert '--account=gts-jli741-paid' in cmd
            if stage == 'continue':
                assert '--nodes=2' in cmd and '--gres=gpu:a100:2' in cmd
                total += 4
            else:
                assert '--nodes=1' in cmd and '--gres=gpu:a100:1' in cmd
                total += 1
    assert total == 10


def test_preview_never_checks_jobs_or_submits(campaign, monkeypatch, capsys):
    path, _ = campaign
    def forbidden(*args, **kwargs):
        raise AssertionError('Preview must not query or submit jobs')
    monkeypatch.setattr(after24h, 'require_completed_job', forbidden)
    monkeypatch.setattr(after24h, 'submit_once', forbidden)
    after24h.submit(path, 'baseline', 'continue')
    assert 'Preview only' in capsys.readouterr().out


def enable_manual_submission(monkeypatch, c):
    monkeypatch.setattr(after24h, 'require_completed_job', lambda job: None)
    monkeypatch.setattr(after24h, 'active_jobs', lambda: set())
    save(Path(c['calibration']) / 'calibration.json', {'passed': True, 'rows': [{}, {}, {}, {}]})


def test_manual_submission_creates_only_selected_task_and_prevents_duplicates(campaign, monkeypatch):
    path, c = campaign
    completed_allocation(Path(c['source_plan']).parent, 'baseline', 4875)
    enable_manual_submission(monkeypatch, c)
    calls = []
    def submitted(directory, label, cmd):
        calls.append((label, cmd))
        save(directory / 'submission' / f'{label}.json', {'status': 'submitted', 'job_id': '999'})
        return '999'
    monkeypatch.setattr(after24h, 'submit_once', submitted)
    assert after24h.submit(path, 'baseline', 'continue', execute=True) == '999'
    assert len(calls) == 1 and calls[0][0] == 'continue-baseline-after-4875'
    assert not (path.parent / 'sf-selection.json').exists()
    monkeypatch.setattr(after24h, 'active_jobs', lambda: {'999'})
    assert after24h.submit(path, 'baseline', 'continue', execute=True) == '999'
    assert len(calls) == 1


def test_manual_resume_keeps_original_plus_15k_target(campaign, monkeypatch):
    path, c = campaign
    completed_allocation(Path(c['source_plan']).parent, 'baseline', 4875)
    selection = after24h.materialize(path, 'baseline')
    run = Path(c['continuations']['baseline'])
    completed_allocation(run, 'baseline', 9000)
    save(run / 'baseline/status.json', dict(step=9000, target=19875, completed=False,
                                           stop_reason='wall_time_budget', resumed_from=4875))
    enable_manual_submission(monkeypatch, c)
    calls = []
    monkeypatch.setattr(after24h, 'submit_once', lambda directory, label, cmd: calls.append((label, cmd)))
    after24h.submit(path, 'baseline', 'continue', execute=True)
    assert len(calls) == 1 and calls[0][0] == 'continue-baseline-after-9000'
    assert not any(x.startswith('--dependency=') for x in calls[0][1])
    plan = json.loads((run / 'plan.json').read_text())
    assert plan['recipe']['steps'] == selection['source_step'] + 15000 == 19875


def test_completed_continuation_never_requests_another_allocation(campaign, monkeypatch):
    path, c = campaign
    completed_allocation(Path(c['source_plan']).parent, 'baseline', 4875)
    after24h.materialize(path, 'baseline')
    run = Path(c['continuations']['baseline'])
    completed_allocation(run, 'baseline', 19875)
    save(run / 'baseline/status.json', dict(step=19875, target=19875, completed=True,
                                           stop_reason='target_reached', resumed_from=9000))
    enable_manual_submission(monkeypatch, c)
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed training must not submit another allocation')
    monkeypatch.setattr(after24h, 'submit_once', forbidden)
    after24h.submit(path, 'baseline', 'continue', execute=True)


@pytest.mark.parametrize('state,exit_code,accepted', [
    ('RUNNING', '0:0', False), ('FAILED', '1:0', False),
    ('COMPLETED', '1:0', False), ('COMPLETED', '0:0', True),
])
def test_manual_submit_requires_successful_initial_job(monkeypatch, state, exit_code, accepted):
    monkeypatch.setattr(after24h.subprocess, 'check_output',
                        lambda *a, **k: f'111|{state}|{exit_code}' + chr(10) + '111.batch|COMPLETED|0:0' + chr(10))
    if accepted:
        after24h.require_completed_job('111')
    else:
        with pytest.raises(ValueError, match='not completed successfully'):
            after24h.require_completed_job('111')
