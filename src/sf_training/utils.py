import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def learning_rate(step, cfg):
    warm = cfg["warmup_steps"]
    peak = cfg["peak"]
    end = cfg["end"]
    # Optax warmup_cosine_decay_schedule starts at peak/(warmup_steps+1) and reaches end at decay_steps.
    if step < warm:
        return peak / (warm + 1) + (peak - peak / (warm + 1)) * step / warm
    progress = min(1.0, (step - warm) / (cfg["decay_steps"] - warm))
    return end + (peak - end) * 0.5 * (1 + math.cos(math.pi * progress))


def accumulation(cfg, world):
    divisor = world * cfg["micro_batch"]
    if cfg["global_batch"] % divisor:
        raise ValueError("global_batch must be divisible by world * micro_batch")
    return cfg["global_batch"] // divisor
