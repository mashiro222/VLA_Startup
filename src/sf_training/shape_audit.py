"""Validate official checkpoint mappings using zero-stride shape views, no weight load."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sf_training.convert import mappings
from sf_training.model import Policy
from openpi.models.gemma import get_config
from sf_training.utils import write_json


def main():
    root = Path(os.environ["VLA_ROOT"])
    metadata = json.loads(
        (root / ".runtime/training/pi05_base/params/_METADATA").read_text()
    )["tree_metadata"]
    pal = {}
    projections = {}
    for entry in metadata.values():
        path = [p["key"] for p in entry["key_metadata"]][1:]
        shape = entry["value_metadata"]["write_shape"]
        # Every element aliases a single scalar. Transpose/reshape preserve zero strides.
        value = np.lib.stride_tricks.as_strided(
            np.zeros(1, dtype=np.float32), shape=tuple(shape), strides=(0,) * len(shape)
        )
        if path[0] == "PaliGemma":
            pal["/".join(path[1:])] = value
        else:
            projections.setdefault(path[0], {})[path[1]] = value
    module = mappings(
        root / ".runtime/repos/openpi/examples/convert_jax_model_to_pytorch.py"
    )
    config = SimpleNamespace(
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
    converted, expert = module["slice_paligemma_state_dict"](pal, config)
    converted.update(
        module["slice_gemma_state_dict"](
            expert,
            get_config("gemma_300m"),
            num_expert=1,
            checkpoint_dir="pi05_base",
            pi05=True,
        )
    )
    for name, p in projections.items():
        converted[name + ".weight"] = torch.from_numpy(p["kernel"]).T
        converted[name + ".bias"] = torch.from_numpy(p["bias"])
    recipe = json.loads((root / "config/sf_training/recipe.json").read_text())
    with torch.device("meta"):
        expected = Policy(recipe).state_dict()
    missing = sorted(set(expected) - set(converted))
    extra = sorted(set(converted) - set(expected))
    bad = [
        k
        for k, v in converted.items()
        if k in expected and v.shape != expected[k].shape
    ]
    allowed = {
        "paligemma_with_expert.gemma_expert.lm_head.weight",
        "paligemma_with_expert.paligemma.lm_head.weight",
    }
    if set(missing) - allowed or extra or bad:
        raise ValueError(dict(missing=missing, extra=extra, bad_shapes=bad))
    report = dict(
        mapped_tensors=len(converted),
        expected_tensors=len(expected),
        missing_explicitly_handled=missing,
        unexpected=extra,
        shape_mismatches=bad,
        weights_loaded=False,
    )
    write_json(root / "reports/preparation/sf-base-shape-audit.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
