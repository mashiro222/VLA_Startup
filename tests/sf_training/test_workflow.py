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
