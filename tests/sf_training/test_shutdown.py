"""CPU regression checks for the real trainer's graceful checkpoint control flow."""

import contextlib
import importlib.util
import sys
from unittest.mock import patch
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sf_training import checkpoint, workflow


@pytest.fixture
def trainer():
    # Load the actual control flow without importing full policy/teacher frameworks.
    spec = importlib.util.spec_from_file_location("checkpoint_control_flow_test", "src/sf_training/train.py")
    module = importlib.util.module_from_spec(spec)
    stub = SimpleNamespace(Experiment=None, build_policy=None, build_teacher=None)
    with patch.dict(sys.modules, {"sf_training.model": stub}):
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("reason", ["deadline", "signal"])
def test_graceful_checkpoint_before_regular_interval(tmp_path, monkeypatch, reason, trainer):
    train = trainer
    cfg = json.loads(Path("config/sf_training/recipe.json").read_text())
    cfg.update(steps=3, global_batch=1, micro_batch=1, checkpoint_every=1000)
    home = tmp_path / ".runtime/training"
    (home / "pi05_base_torch").mkdir(parents=True)
    (home / "pi05_base_torch/conversion.json").write_text("{}")
    (home / "norm_stats.json").write_text('{"norm_stats": {}}')
    plan = {"root": str(tmp_path), "recipe": cfg, "gpus": 1, "profile": "full"}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(workflow, "verify_plan", lambda _: plan)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = nn.Linear(1, 1, bias=False)
            self.lora_adapter = nn.Linear(1, 1, bias=False)
            self.action_out_proj = nn.Linear(1, 1, bias=False)

        def forward(self, batch):
            loss = (self.action_out_proj(self.lora_adapter(self.vision_tower(batch["x"]))) - 1).square().mean()
            return loss, loss.detach(), loss.detach() * 0

    model = Model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.25)
    initial = {n: p.clone() for n, p in model.state_dict().items()}
    monkeypatch.setattr(train, "build_policy", lambda *_: nn.Identity())
    monkeypatch.setattr(train, "Experiment", lambda *_: model)
    monkeypatch.setattr(train, "parameter_manifest", lambda _: {})
    monkeypatch.setattr(train, "ArenaData", lambda *_: [0])
    monkeypatch.setattr(train, "collate", lambda *_: dict(
        x=torch.ones(1, 1), actions=torch.ones(1, 10, 32), rare_rotation=torch.zeros(1)))

    class CpuTorch:
        def __getattr__(self, name):
            return getattr(torch, name)

        def device(self, *_):
            return torch.device("cpu")

        def autocast(self, *_a, **_k):
            return contextlib.nullcontext()

    monkeypatch.setattr(train, "torch", CpuTorch())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("set_device", "synchronize", "manual_seed_all", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *_: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU test double")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_: SimpleNamespace(total_memory=0))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(train.signal, "signal", lambda *_: None)
    monkeypatch.setattr(train, "STOP", False)
    for key, value in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")):
        monkeypatch.setenv(key, value)
    if reason == "deadline":
        monkeypatch.setenv("SF_DEADLINE", "1")
    else:
        monkeypatch.delenv("SF_DEADLINE", raising=False)
        train.stop_requested()

    train.run(path, "baseline")
    output = tmp_path / "baseline"
    ck_path = output / "step-00000001.pt"
    ck = torch.load(ck_path, weights_only=False)
    status = json.loads((output / "status.json").read_text())
    assert status["step"] == ck["step"] == 1
    assert not status["completed"]
    assert status["stop_reason"] == ("signal" if reason == "signal" else "wall_time_budget")
    assert not list(output.glob("*.tmp"))
    assert len(ck["rng"]) == 1 and ck["optimizer"]["state"]
    assert any(not torch.equal(initial[n], p) for n, p in model.state_dict().items())
    restored = Model()
    optimizer = torch.optim.AdamW(restored.parameters())
    assert checkpoint.restore(ck_path, restored, optimizer, ck["identity"], 0) == 1
    for n, p in restored.state_dict().items():
        torch.testing.assert_close(p, model.state_dict()[n], rtol=0, atol=0)
    assert all(int(s["step"]) == 1 for s in optimizer.state.values())


def test_failed_atomic_write_preserves_complete_checkpoint(tmp_path, monkeypatch):
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "step-00000001.pt"
    checkpoint.save(path, model, optimizer, 1, {"run": "test"}, [checkpoint.rng_state()])
    original = path.read_bytes()

    def fail_write(_value, temp):
        Path(temp).write_bytes(b"incomplete")
        raise OSError("simulated storage failure")

    monkeypatch.setattr(torch, "save", fail_write)
    with pytest.raises(OSError, match="storage failure"):
        checkpoint.save(path, model, optimizer, 2, {"run": "test"}, [])
    assert path.read_bytes() == original
    assert torch.load(path, weights_only=False)["step"] == 1
