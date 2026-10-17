"""Pack verified embedded images for random access without rereading full Parquet episodes."""

import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np
from sf_training.preparation import compute_stats, read_episodes
from sf_training.utils import write_json, sha


EXPECTED_EPISODES = 3018
EXPECTED_FRAMES = 361736


def main():
    root = Path(os.environ["VLA_ROOT"])
    home = root / ".runtime/training"
    data = home / "arena_l0"
    pack = data / "packed"
    tmp = data / "packed.tmp"
    verified = json.loads((home / "assets_verified.json").read_text())
    identity = verified["manifest_sha256"]
    if identity != sha(root / "config/sf_training/assets.json"):
        raise ValueError("Assets manifest is not verified")
    if pack.exists():
        marker = json.loads((pack / "complete.json").read_text())
        if marker["source_manifest"] != identity:
            raise ValueError("Packed source identity mismatch")
        print("Packed dataset exists.", flush=True)
    else:
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        specs = {
            "state": ((EXPECTED_FRAMES, 8), "float32"),
            "actions": ((EXPECTED_FRAMES, 7), "float32"),
            "task": ((EXPECTED_FRAMES,), "int64"),
            "offsets": ((EXPECTED_FRAMES, 2, 2), "int64"),
            "ends": ((EXPECTED_EPISODES,), "int64"),
        }
        arrays = {
            name: np.lib.format.open_memmap(
                tmp / (name + ".npy"), mode="w+", dtype=dtype, shape=shape
            )
            for name, (shape, dtype) in specs.items()
        }
        episodes = 0
        total = 0
        image_hash = hashlib.sha256()
        with (tmp / "images.bin").open("wb") as out:
            paths = sorted((data / "data").glob("chunk-*/*.parquet"))
            for i, (p, e) in enumerate(read_episodes(paths)):
                length = len(e["state"])
                if not all(int(x) == i for x in e["episode_index"]):
                    raise ValueError("Unexpected episode order")
                begin = total
                total += length
                if total > EXPECTED_FRAMES or i >= EXPECTED_EPISODES:
                    raise ValueError("Dataset exceeds expected dimensions")
                arrays["state"][begin:total] = e["state"]
                arrays["actions"][begin:total] = e["actions"]
                arrays["task"][begin:total] = e["task_index"]
                arrays["ends"][i] = total
                episodes = i + 1
                for t in range(length):
                    pair = []
                    for key in ("image", "wrist_image"):
                        b = e[key][t]["bytes"]
                        pair.append([out.tell(), len(b)])
                        out.write(b)
                        image_hash.update(b)
                    arrays["offsets"][begin + t] = pair
                if (i + 1) % 250 == 0:
                    print(f"Packed {i + 1}/{EXPECTED_EPISODES} episodes", flush=True)
        if (episodes, total) != (EXPECTED_EPISODES, EXPECTED_FRAMES):
            raise ValueError("Packed dataset incomplete")
        for array in arrays.values():
            array.flush()
        del arrays
        files = {
            p.name: dict(
                size=p.stat().st_size,
                sha256=image_hash.hexdigest() if p.name == "images.bin" else sha(p),
            )
            for p in tmp.iterdir()
        }
        write_json(
            tmp / "complete.json",
            dict(
                source_manifest=identity, frames=total, episodes=episodes, files=files
            ),
        )
        tmp.replace(pack)
    if not (home / "norm_stats.json").exists():
        print("Computing training-only normalization statistics.", flush=True)
        result = compute_stats(data)
        result["source_manifest"] = identity
        write_json(home / "norm_stats.json", result)
    print("Packed dataset and normalization ready.", flush=True)


if __name__ == "__main__":
    main()
