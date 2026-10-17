"""Pack verified embedded images for random access without rereading full Parquet episodes."""

import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sf_training.data import compute_stats
from sf_training.utils import write_json, sha


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
        states = []
        actions = []
        tasks = []
        offsets = []
        ends = []
        total = 0
        image_hash = hashlib.sha256()
        with (tmp / "images.bin").open("wb") as out:
            for i, p in enumerate(sorted((data / "data").glob("chunk-*/*.parquet"))):
                e = pq.read_table(p).to_pydict()
                length = len(e["state"])
                if not all(int(x) == i for x in e["episode_index"]):
                    raise ValueError("Unexpected episode order")
                states.extend(e["state"])
                actions.extend(e["actions"])
                tasks.extend(e["task_index"])
                total += length
                ends.append(total)
                for t in range(length):
                    pair = []
                    for key in ("image", "wrist_image"):
                        b = e[key][t]["bytes"]
                        pair.append([out.tell(), len(b)])
                        out.write(b)
                        image_hash.update(b)
                    offsets.append(pair)
                if (i + 1) % 250 == 0:
                    print(f"Packed {i + 1}/3018 episodes", flush=True)
        if (len(ends), total) != (3018, 361736):
            raise ValueError("Packed dataset incomplete")
        for name, values, dtype in [
            ("state", states, "float32"),
            ("actions", actions, "float32"),
            ("task", tasks, "int64"),
            ("offsets", offsets, "int64"),
            ("ends", ends, "int64"),
        ]:
            np.save(tmp / (name + ".npy"), np.asarray(values, dtype=dtype))
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
                source_manifest=identity, frames=total, episodes=len(ends), files=files
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
