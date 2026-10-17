import json
from types import SimpleNamespace
import pytest
from sf_training import workflow as w


def setup_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ROOT", tmp_path)
    p = tmp_path / "runs/full"
    p.mkdir(parents=True)
    (p / "plan.json").write_text("{}")
    plan = {
        "profile": "full",
        "gpus": 2,
        "time_limit": "24:00:00",
        "files": {"code": "a"},
        "assets": {"data": "b"},
        "environment": {"version": "c"},
        "recipe": {"global_batch": 32, "steps": 60000},
    }
    monkeypatch.setattr(
        w,
        "verify_plan",
        lambda path: dict(
            plan, profile="smoke" if path.parent.name == "smoke" else "full"
        ),
    )
    return p, plan


def test_formal_submit_requires_successful_smoke(tmp_path, monkeypatch):
    setup_plan(tmp_path, monkeypatch)
    args = SimpleNamespace(
        run_id="full",
        arm="baseline",
        submit=True,
        smoke_run=None,
        after_smoke=False,
        resume=False,
    )
    with pytest.raises(ValueError, match="smoke-run"):
        w.submit(args)


def test_dependency_submission_account_and_duplicate_guard(tmp_path, monkeypatch):
    p, plan = setup_plan(tmp_path, monkeypatch)
    smoke = tmp_path / "runs/smoke/submission"
    smoke.mkdir(parents=True)
    (smoke / "smoke.json").write_text(
        json.dumps({"job_id": "123", "status": "submitted"})
    )
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        return SimpleNamespace(returncode=0, stdout="456\n", stderr="")

    monkeypatch.setattr(w.subprocess, "run", fake_run)
    args = SimpleNamespace(
        run_id="full",
        arm="sf",
        submit=True,
        smoke_run="smoke",
        after_smoke=True,
        resume=False,
    )
    w.submit(args)
    cmd = captured[0]
    assert "--account=gts-jli741-paid" in cmd and "--gres=gpu:a100:2" in cmd
    assert "--dependency=afterok:123" in cmd and "--kill-on-invalid-dep=yes" in cmd
    assert cmd[-3:] == ["full", "sf", "smoke"]
    assert json.loads((p / "submission/sf.json").read_text())["job_id"] == "456"
    with pytest.raises(FileExistsError):
        w.submit(args)
    assert len(captured) == 1


def test_stale_smoke_rejected(tmp_path, monkeypatch):
    _, plan = setup_plan(tmp_path, monkeypatch)
    smoke = tmp_path / "runs/smoke"
    smoke.mkdir(parents=True)
    (smoke / "smoke_passed.json").write_text(
        json.dumps(dict(plan, files={"code": "old"}))
    )
    with pytest.raises(ValueError, match="does not match"):
        w.smoke_gate("smoke", plan)


def test_auto_launch_does_not_submit_before_smoke_passes(tmp_path, monkeypatch):
    setup_plan(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(w, "submit", lambda a: calls.append(a))
    with pytest.raises(FileNotFoundError):
        w.launch_full(SimpleNamespace(smoke_run="smoke", full_run="full"))
    assert calls == []


def test_auto_launch_submits_both_arms_only_after_matching_pass(tmp_path, monkeypatch):
    _, plan = setup_plan(tmp_path, monkeypatch)
    smoke=tmp_path/'runs/smoke'; smoke.mkdir(parents=True)
    (smoke/'smoke_passed.json').write_text(json.dumps(plan))
    calls=[]
    monkeypatch.setattr(w, 'submit', lambda a: calls.append(a))
    w.launch_full(SimpleNamespace(smoke_run='smoke', full_run='full'))
    assert [a.arm for a in calls] == ['baseline','sf']
    assert all(a.submit and a.after_smoke and not a.resume for a in calls)


def test_smoke_full_recipe_or_gpu_mismatch_rejected():
    plan=dict(profile='full',gpus=2,files={},assets={},environment={},recipe={'global_batch':32})
    for changed in (dict(gpus=1),dict(recipe={'global_batch':4})):
        smoke=dict(plan,profile='smoke',**changed)
        with pytest.raises(ValueError): w.matching_full(smoke,plan)


def test_smoke_submission_records_conditional_launch_and_requires_fresh_run(tmp_path, monkeypatch):
    full, plan = setup_plan(tmp_path, monkeypatch)
    smoke = tmp_path / 'runs/smoke'
    smoke.mkdir(parents=True)
    (smoke / 'plan.json').write_text('{}')
    calls=[]
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0,stdout='123\n',stderr='')
    monkeypatch.setattr(w.subprocess,'run',fake_run)
    args=SimpleNamespace(run_id='smoke',arm='smoke',submit=True,smoke_run=None,
                         after_smoke=False,resume=False,then_full_run='full')
    w.submit(args)
    assert len(calls)==1 and calls[0][-4:]==['smoke','smoke','-','full']
    assert '--gres=gpu:a100:2' in calls[0]
    (full/'baseline').mkdir()
    (full/'baseline/step-00001000.pt').touch()
    with pytest.raises(ValueError,match='fresh'):
        w.submit(args)
    assert len(calls)==1


def test_explicit_skip_smoke_submits_full_without_dependency(tmp_path, monkeypatch):
    p, _ = setup_plan(tmp_path, monkeypatch)
    calls=[]
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0,stdout='456\n',stderr='')
    monkeypatch.setattr(w.subprocess,'run',fake_run)
    args=SimpleNamespace(run_id='full',arm='baseline',submit=True,smoke_run=None,
                         after_smoke=False,resume=False,skip_smoke=True)
    w.submit(args)
    assert calls[0][-3:]==['full','baseline','skip-smoke']
    assert not any(x.startswith('--dependency') for x in calls[0])
    assert '--gres=gpu:a100:2' in calls[0] and '--account=gts-jli741-paid' in calls[0]
    assert json.loads((p/'submission/baseline.json').read_text())['gpu_smoke']=='explicitly_skipped'
    args.smoke_run='smoke'
    with pytest.raises(ValueError,match='without smoke'):
        w.submit(args)
