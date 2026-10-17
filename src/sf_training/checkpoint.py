"""Atomic training checkpoint including projector, optimizer and per-rank RNG."""

import os
import random
from pathlib import Path
import numpy as np
import torch
from sf_training.lora import trainable_state, load_trainable


def rng_state():
    return dict(
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python=random.getstate(),
        numpy=np.random.get_state(),
    )


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save(path, model, optimizer, step, identity, rng_states):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            schema=1,
            step=step,
            identity=identity,
            trainable=trainable_state(model),
            optimizer=optimizer.state_dict(),
            rng=rng_states,
        ),
        temp,
    )
    with temp.open("rb") as f:
        os.fsync(f.fileno())
    temp.replace(path)


def restore(path, model, optimizer, identity, rank):
    # Only load checkpoints created by this repository in the local run directory.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["schema"] != 1 or ck["identity"] != identity:
        raise ValueError("Checkpoint identity/config/world-size mismatch")
    load_trainable(model, ck["trainable"])
    optimizer.load_state_dict(ck["optimizer"])
    restore_rng(ck["rng"][rank])
    return ck["step"]
