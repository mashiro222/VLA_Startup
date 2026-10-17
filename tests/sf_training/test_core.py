import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from sf_training.lora import ArenaLoRALinear, trainable_state, load_trainable
from sf_training.data import chunk, normalize, batch_indices
from sf_training.checkpoint import rng_state, save, restore
from sf_training.utils import accumulation, learning_rate, seed_all


@pytest.mark.parametrize(
    "kind,heads,i,o", [("dense", 1, 5, 7), ("qkv", 2, 5, 8), ("out", 2, 8, 5)]
)
def test_lora_merge_and_grads(kind, heads, i, o):
    torch.manual_seed(4)
    m = ArenaLoRALinear(nn.Linear(i, o), 2, 4, kind, heads)
    x = torch.randn(2, 3, i)
    torch.testing.assert_close(
        m(x), torch.nn.functional.linear(x, m.merged_weight(), m.base.bias)
    )
    m(x).square().mean().backward()
    assert m.base.weight.grad is None
    assert m.lora_a.grad.abs().sum() > 0 and m.lora_b.grad.abs().sum() > 0


def test_attention_independent_heads():
    m = ArenaLoRALinear(nn.Linear(4, 6, bias=False), 2, 2, "qkv", 2)
    with torch.no_grad():
        m.lora_a.zero_()
        m.lora_b.fill_(1)
        m.lora_a[0].fill_(1)
    d = m.delta(torch.ones(1, 4))
    assert torch.count_nonzero(d[:, :3]) == 3 and torch.count_nonzero(d[:, 3:]) == 0


def test_checkpoint_resumes_same_update_and_rng(tmp_path):
    seed_all(42)
    m = nn.Sequential(ArenaLoRALinear(nn.Linear(4, 3), 2, 2), nn.Linear(3, 2))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad])

    def step():
        opt.zero_grad()
        loss = m(torch.randn(2, 4)).square().mean()
        loss.backward()
        opt.step()
        return loss.detach().clone()

    step()
    path = tmp_path / "step.pt"
    identity = {"same": "experiment"}
    save(path, m, opt, 1, identity, [rng_state()])
    loss = step()
    expected = {n: p.clone() for n, p in m.named_parameters()}
    assert restore(path, m, opt, identity, 0) == 1
    torch.testing.assert_close(step(), loss, rtol=0, atol=0)
    for n, p in m.named_parameters():
        torch.testing.assert_close(p, expected[n], rtol=0, atol=0)
    with pytest.raises(ValueError):
        restore(path, m, opt, {"different": True}, 0)


def test_trainable_restore_strict():
    m = nn.Linear(3, 4)
    state = trainable_state(m)
    state.pop("bias")
    with pytest.raises(ValueError):
        load_trainable(m, state)


def test_chunk_never_crosses_episode():
    a = np.arange(21).reshape(3, 7)
    out = chunk(a, 1, 10)
    assert np.array_equal(out[0], a[1])
    assert np.array_equal(out[1:], np.repeat(a[-1:], 9, 0))


def test_quantile_normalization_no_clip():
    out = normalize(np.array([4, 2], np.float32), {"q01": [0, 1], "q99": [2, 3]})
    np.testing.assert_allclose(out, [3, 0], atol=3e-6)


@pytest.mark.parametrize("world", [1, 2, 4])
def test_shard_batch_and_resume(world):
    full = batch_indices(23, 42, 7, 32)
    shards = [batch_indices(23, 42, 7, 32, r, world) for r in range(world)]
    assert [shards[r][i] for i in range(32 // world) for r in range(world)] == full
    assert batch_indices(23, 42, 7, 32) == full and batch_indices(23, 42, 8, 32) != full
    assert accumulation({"global_batch": 32, "micro_batch": 1}, world) == 32 // world


def test_batch_constraints():
    with pytest.raises(ValueError):
        accumulation({"global_batch": 31, "micro_batch": 1}, 2)


def test_lr_matches_arena_schedule():
    cfg = {"warmup_steps": 1000, "peak": 2.5e-5, "decay_steps": 30000, "end": 2.5e-6}
    assert learning_rate(0, cfg) == pytest.approx(2.5e-5 / 1001)
    assert learning_rate(1000, cfg) == pytest.approx(2.5e-5)
    assert learning_rate(30000, cfg) == pytest.approx(2.5e-6)
    assert learning_rate(60000, cfg) == pytest.approx(2.5e-6)


def test_gradient_accumulation_equivalence():
    seed_all(5)
    m = nn.Linear(3, 2)
    x = torch.randn(4, 3)
    y = torch.randn(4, 2)
    ((m(x) - y) ** 2).mean().backward()
    grad = m.weight.grad.clone()
    m.zero_grad()
    for i in range(4):
        (((m(x[i : i + 1]) - y[i : i + 1]) ** 2).mean() / 4).backward()
    torch.testing.assert_close(m.weight.grad, grad)


def test_aligned_augmentation_and_missing_camera():
    from sf_training.model import observation

    b = dict(
        images=torch.rand(2, 2, 3, 224, 224),
        state=torch.zeros(2, 32),
        tokens=torch.ones(2, 200, dtype=torch.long),
        token_mask=torch.ones(2, 200, dtype=torch.bool),
    )
    seed_all(42)
    obs, t = observation(b, True)
    seed_all(42)
    again, _ = observation(b, True)
    assert torch.equal(t, b["images"]) and torch.equal(
        obs.images["base_0_rgb"], again.images["base_0_rgb"]
    )
    assert (
        not obs.image_masks["right_wrist_0_rgb"].any()
        and (obs.images["base_0_rgb"].abs() <= 1).all()
    )


def test_sf_loss_gradient_and_frozen_target():
    from openpi.models_pytorch.projectors import AlignProjector

    proj = AlignProjector(8, 4)
    student = torch.randn(2, 5, 8, requires_grad=True)
    teacher = nn.Linear(8, 8).requires_grad_(False)
    with torch.no_grad():
        target = teacher(torch.randn(2, 5, 8))
    loss = proj(student, target, torch.ones(2, 5, dtype=torch.bool))
    loss.backward()
    assert (
        torch.isfinite(loss)
        and student.grad.abs().sum() > 0
        and proj.fc1.weight.grad.abs().sum() > 0
    )
    assert all(p.grad is None for p in teacher.parameters())


def test_real_pi05_trainable_layout_on_meta():
    from sf_training.model import Policy
    from sf_training.lora import inject_lora, parameter_manifest

    cfg = json.loads(Path("config/sf_training/recipe.json").read_text())
    with torch.device("meta"):
        model = Policy(cfg)
        count = len(inject_lora(model, cfg["lora"]))
    assert (
        model.pi05
        and model.config.action_horizon == 10
        and not model.config.discrete_state_input
        and count == 252
    )
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("vision_tower" in n for n in trainable)
    assert all(
        "lora_" in n
        for n in trainable
        if ".language_model." in n or ".gemma_expert." in n
    )
    assert not any("lm_head" in n for n in trainable)
    Path("reports/preparation").mkdir(parents=True, exist_ok=True)
    Path("reports/preparation/sf-policy-parameters.json").write_text(
        json.dumps(parameter_manifest(model), indent=2) + "\n"
    )


def test_packed_and_parquet_sample_equivalence(tmp_path):
    from io import BytesIO
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from sf_training.data import ArenaData

    root = tmp_path / "data"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 2}) + "\n"
    )
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick up cube"}) + "\n"
    )
    b = BytesIO()
    Image.fromarray(np.full((256, 256, 3), 100, np.uint8)).save(b, format="PNG")
    blob = b.getvalue()
    states = np.arange(16, dtype=np.float32).reshape(2, 8)
    actions = np.arange(14, dtype=np.float32).reshape(2, 7)
    values = {
        "state": states.tolist(),
        "actions": actions.tolist(),
        "task_index": [0, 0],
        "image": [{"bytes": blob}] * 2,
        "wrist_image": [{"bytes": blob}] * 2,
    }
    pq.write_table(
        pa.Table.from_pydict(values), root / "data/chunk-000/episode_000000.parquet"
    )
    tokenizer = Path(".runtime/cache/openpi/big_vision/paligemma_tokenizer.model")
    stats = {
        k: {"q01": [0] * n, "q99": [20] * n} for k, n in [("state", 8), ("actions", 7)]
    }
    raw = ArenaData(root, tokenizer, stats)
    expected = raw[1]
    packed = root / "packed"
    packed.mkdir()
    offsets = []
    with (packed / "images.bin").open("wb") as out:
        for _ in range(2):
            pair = []
            for _ in range(2):
                pair.append([out.tell(), len(blob)])
                out.write(blob)
            offsets.append(pair)
    for n, x in [
        ("state", states),
        ("actions", actions),
        ("task", [0, 0]),
        ("ends", [2]),
        ("offsets", offsets),
    ]:
        np.save(packed / (n + ".npy"), x)
    (packed / "complete.json").write_text("{}")
    actual = ArenaData(root, tokenizer, stats)[1]
    for k in expected:
        if k != "sample_index":
            torch.testing.assert_close(actual[k], expected[k], rtol=0, atol=0)
    assert actual["actions"].shape == (10, 32) and actual["images"].shape == (
        2,
        3,
        224,
        224,
    )


def test_sf_zero_weight_shares_policy_flow_and_keeps_teacher_frozen():
    from sf_training.model import Experiment

    class ToyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(2.0))

        def flow(self, obs, actions, need_features):
            features = self.w * torch.ones((1, 512, 2048)) if need_features else None
            return self.w.square(), features

    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(1), requires_grad=False)

        def forward(self, x):
            return {
                "features": [torch.ones(1, 2, 37 * 37 + 1, 2048)],
                "patch_start_idx": 1,
            }

    cfg = {"sf": {"weight": 0.0, "teacher_layer_index": -1, "position_embedding": True}}
    baseline = Experiment(ToyPolicy(), cfg, "baseline")
    sf = Experiment(ToyPolicy(), cfg, "sf", Teacher().eval())
    batch = {
        "images": torch.rand(1, 2, 3, 224, 224),
        "state": torch.zeros(1, 32),
        "tokens": torch.zeros(1, 200, dtype=torch.long),
        "token_mask": torch.ones(1, 200, dtype=torch.bool),
        "actions": torch.zeros(1, 10, 32),
    }
    seed_all(42)
    a = baseline(batch)[0]
    a.backward()
    seed_all(42)
    b = sf(batch)[0]
    b.backward()
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(baseline.policy.w.grad, sf.policy.w.grad)
    assert not any(n.startswith("teacher") for n, _ in sf.named_parameters())
