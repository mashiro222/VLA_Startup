"""Export a training checkpoint as an ordinary pi0.5 policy; teacher/projector omitted."""

import argparse
import json
from pathlib import Path
import safetensors.torch
import torch
from sf_training.model import build_policy
from sf_training.lora import ArenaLoRALinear
from sf_training.utils import sha, write_json


def export(plan_path, arm, step_file, destination):
    plan = json.loads(Path(plan_path).read_text())
    root = Path(plan["root"])
    cfg = plan["recipe"]
    checkpoint = torch.load(step_file, map_location="cpu", weights_only=False)
    if (
        checkpoint["identity"]["plan_sha256"] != sha(plan_path)
        or checkpoint["identity"]["arm"] != arm
    ):
        raise ValueError("Wrong checkpoint identity")
    policy = build_policy(
        cfg, root / ".runtime/training/pi05_base_torch/model.safetensors"
    )
    state = {
        k.removeprefix("policy."): v
        for k, v in checkpoint["trainable"].items()
        if k.startswith("policy.")
    }
    from sf_training.lora import load_trainable

    load_trainable(policy, state)
    for name, module in list(policy.named_modules()):
        if not isinstance(module, ArenaLoRALinear):
            continue
        parent_name, attribute = name.rsplit(".", 1)
        parent = policy.get_submodule(parent_name)
        with torch.no_grad():
            module.base.weight.copy_(
                module.merged_weight().to(module.base.weight.dtype)
            )
        setattr(parent, attribute, module.base)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    safetensors.torch.save_model(policy, str(destination / "model.safetensors"))
    stats = root / ".runtime/training/norm_stats.json"
    (destination / "norm_stats.json").write_bytes(stats.read_bytes())
    write_json(
        destination / "policy.json",
        dict(
            pi05=True,
            arm=arm,
            step=checkpoint["step"],
            action_horizon=10,
            action_dim=32,
            discrete_state_input=False,
            normalization="quantiles",
            extra_delta_transform=False,
            teacher_required=False,
            model_sha256=sha(destination / "model.safetensors"),
            norm_stats_sha256=sha(stats),
            recipe=cfg,
        ),
    )
    print(destination)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--arm", choices=["baseline", "sf"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    export(a.plan, a.arm, a.checkpoint, a.output)
