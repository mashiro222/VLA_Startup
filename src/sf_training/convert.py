"""Convert verified official pi05_base on an allocated node, using pinned OpenPI mappings.

Only the three pure mapping functions are imported via AST from the pinned official
converter. This avoids importing its unrelated LeRobot training configuration.
"""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import safetensors.torch
from flax.nnx import traversals
from openpi.models.model import restore_params
from openpi.models.gemma import get_config
from sf_training.model import Policy
from sf_training.utils import sha, write_json


def mappings(path):
    tree = ast.parse(Path(path).read_text())
    names = {"slice_paligemma_state_dict", "slice_gemma_state_dict"}
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    if {f.name for f in functions} != names:
        raise RuntimeError("Pinned converter mapping functions changed")
    namespace = {"np": np, "torch": torch}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def main():
    root = Path(os.environ["VLA_ROOT"])
    home = root / ".runtime/training"
    dest = home / "pi05_base_torch"
    dest.mkdir(exist_ok=True)
    cfg = json.loads((root / "config/sf_training/recipe.json").read_text())
    mapping_path = (
        root / ".runtime/repos/openpi/examples/convert_jax_model_to_pytorch.py"
    )
    identity = {
        "source_manifest": sha(root / "config/sf_training/assets.json"),
        "converter": sha(mapping_path),
        "driver": sha(__file__),
        "policy_driver": sha(root / "src/sf_training/model.py"),
        "pi05": True,
    }
    marker = dest / "conversion.json"
    out = dest / "model.safetensors"
    if marker.exists() and out.exists():
        previous = json.loads(marker.read_text())
        if previous["identity"] != identity or previous["sha256"] != sha(out):
            raise ValueError("Converted base verification failed")
        print("Verified existing converted pi05_base", flush=True)
        return
    print(
        "Converting pi05_base on allocated node (CPU memory, no training).", flush=True
    )
    weights = restore_params(
        home / "pi05_base/params", restore_type=np.ndarray, dtype="float32"
    )
    flat = traversals.flatten_mapping(weights.pop("PaliGemma"), sep="/")
    m = mappings(mapping_path)
    conf = SimpleNamespace(
        vision_config=SimpleNamespace(
            hidden_size=1152,
            num_hidden_layers=27,
            num_attention_heads=16,
            intermediate_size=4304,
            patch_size=14,
            projection_dim=2048,
        ),
        text_config=SimpleNamespace(
            hidden_size=2048,
            num_hidden_layers=18,
            num_attention_heads=8,
            head_dim=256,
            intermediate_size=16384,
        ),
    )
    pal, expert = m["slice_paligemma_state_dict"](flat, conf)
    expert = m["slice_gemma_state_dict"](
        expert,
        get_config("gemma_300m"),
        num_expert=1,
        checkpoint_dir=str(home / "pi05_base"),
        pi05=True,
    )
    mapped = {**pal, **expert}
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        value = weights.pop(name)
        for key, suffix in [("kernel", "weight"), ("bias", "bias")]:
            array = value[key]
            array = array["value"] if isinstance(array, dict) else array
            tensor = torch.from_numpy(np.array(array))
            mapped[name + "." + suffix] = tensor.T if key == "kernel" else tensor
    if weights:
        raise ValueError(f"Unmapped JAX roots: {list(weights)}")
    # Shape-only construction: no 12GB random model initialization.
    with torch.device("meta"):
        model = Policy(cfg)
    expected = model.state_dict()
    unused = "paligemma_with_expert.gemma_expert.lm_head.weight"
    missing = set(expected) - set(mapped)
    extra = set(mapped) - set(expected)
    tied = "paligemma_with_expert.paligemma.lm_head.weight"
    embedding = (
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    )
    if tied in missing:
        mapped[tied] = mapped[embedding]
        missing.remove(tied)
    if missing != {unused} or extra:
        raise ValueError(f"Conversion mismatch: missing={missing}, extra={extra}")
    # The expert LM vocabulary head is never used in action inference/training.
    mapped[unused] = torch.zeros(expected[unused].shape, dtype=torch.bfloat16)
    for name, value in mapped.items():
        if value.shape != expected[name].shape:
            raise ValueError(
                f"Conversion shape mismatch {name}: {value.shape} != {expected[name].shape}"
            )
        mapped[name] = value.to(expected[name].dtype)
    model.load_state_dict(mapped, strict=True, assign=True)
    model.paligemma_with_expert.paligemma.tie_weights()
    tmp = dest / "model.safetensors.tmp"
    safetensors.torch.save_model(model, str(tmp))
    tmp.replace(out)
    write_json(
        marker,
        {
            "identity": identity,
            "sha256": sha(out),
            "tensors": len(expected),
            "explicit_zero_unused_tensor": unused,
        },
    )
    print("Converted base verified and saved.", flush=True)


if __name__ == "__main__":
    main()
