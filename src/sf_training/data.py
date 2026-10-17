"""Pinned LeRobot v2 Parquet reader; episode-local 10-action chunks."""

from collections import OrderedDict
from io import BytesIO
from pathlib import Path
import bisect
import json
import os
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import sentencepiece
import torch


class ArenaData:
    def __init__(
        self, root, tokenizer, stats, horizon=10, max_tokens=200, cache_size=4
    ):
        self.root = Path(root)
        self.horizon = horizon
        self.max_tokens = max_tokens
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.episodes = [
            json.loads(line)
            for line in (self.root / "meta/episodes.jsonl").read_text().splitlines()
        ]
        self.tasks = {
            t["task_index"]: t["task"]
            for t in map(
                json.loads, (self.root / "meta/tasks.jsonl").read_text().splitlines()
            )
        }
        self.ends = np.cumsum([e["length"] for e in self.episodes]).tolist()
        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer))
        self.stats = stats
        packed = self.root / "packed"
        self.packed = None
        if (packed / "complete.json").exists():
            self.packed = {
                n: np.load(packed / (n + ".npy"), mmap_mode="r")
                for n in ("state", "actions", "task", "offsets", "ends")
            }
            if self.packed["ends"].tolist() != self.ends:
                raise ValueError("Packed episode boundaries differ")
            self.image_fd = os.open(packed / "images.bin", os.O_RDONLY)

    def __len__(self):
        return self.ends[-1]

    def episode(self, i):
        if i not in self.cache:
            eid = self.episodes[i]["episode_index"]
            p = self.root / f"data/chunk-{eid // 1000:03d}/episode_{eid:06d}.parquet"
            self.cache[i] = pq.read_table(p).to_pydict()
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(i)
        return self.cache[i]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        i = bisect.bisect_right(self.ends, index)
        t = index - (self.ends[i - 1] if i else 0)
        e = self.episode(i) if self.packed is None else None
        if self.packed is None:
            state = np.asarray(e["state"][t], dtype=np.float32)
            actions = chunk(np.asarray(e["actions"], dtype=np.float32), t, self.horizon)
            task_index = int(e["task_index"][t])
        else:
            state = self.packed["state"][index]
            actions = self.packed["actions"][
                np.minimum(np.arange(index, index + self.horizon), self.ends[i] - 1)
            ]
            task_index = int(self.packed["task"][index])
        state = normalize(state, self.stats["state"])
        actions = normalize(actions, self.stats["actions"])
        imgs = []
        for view, key in enumerate(("image", "wrist_image")):
            if self.packed is None:
                value = e[key][t]
            else:
                offset, length = map(int, self.packed["offsets"][index, view])
                value = {"bytes": os.pread(self.image_fd, length, offset)}
            if not isinstance(value, dict) or value.get("bytes") is None:
                raise ValueError("Expected embedded RGB image bytes")
            im = Image.open(BytesIO(value["bytes"])).convert("RGB")
            if im.size != (256, 256):
                raise ValueError(f"Unexpected image shape {im.size}")
            # Dataset images are already upright; do not apply simulator's 180-degree correction again.
            im = im.resize((224, 224), Image.Resampling.BILINEAR)
            imgs.append(
                torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1).float() / 255
            )
        prompt = self.tasks[task_index].strip().replace("_", " ").replace("\n", " ")
        ids = self.sp.encode(prompt, add_bos=True) + self.sp.encode("\n")
        ids = ids[: self.max_tokens]
        mask = [True] * len(ids) + [False] * (self.max_tokens - len(ids))
        ids += [0] * (self.max_tokens - len(ids))
        return dict(
            images=torch.stack(imgs),
            state=torch.from_numpy(np.pad(state, (0, 32 - len(state)))),
            actions=torch.from_numpy(
                np.pad(actions, ((0, 0), (0, 32 - actions.shape[-1])))
            ),
            tokens=torch.tensor(ids, dtype=torch.long),
            token_mask=torch.tensor(mask),
            sample_index=index,
        )


def chunk(actions, t, horizon):
    return actions[np.minimum(np.arange(t, t + horizon), len(actions) - 1)]


def normalize(x, stats):
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    return (x - low) / (high - low + 1e-6) * 2 - 1


def batch_indices(size, seed, step, global_batch, rank=0, world=1):
    """Counter-based permutation: resume does not depend on DataLoader prefetch state."""
    if global_batch % world:
        raise ValueError("global_batch must be divisible by world size")
    positions = range(step * global_batch, (step + 1) * global_batch)
    permutations = {}
    values = []
    for pos in positions:
        epoch, offset = divmod(pos, size)
        if epoch not in permutations:
            permutations[epoch] = np.random.default_rng(
                np.random.SeedSequence([seed, epoch])
            ).permutation(size)
        values.append(int(permutations[epoch][offset]))
    return values[rank::world]


def collate(items, device):
    return {
        k: torch.stack([x[k] for x in items]).to(device)
        for k in items[0]
        if k != "sample_index"
    }


def compute_stats(root, horizon=10):
    root = Path(root)
    state_parts = []
    action_parts = []
    frames = 0
    episodes = 0
    for p in sorted((root / "data").glob("chunk-*/*.parquet")):
        e = pq.read_table(p, columns=["state", "actions"]).to_pydict()
        s = np.asarray(e["state"], dtype=np.float32)
        a = np.asarray(e["actions"], dtype=np.float32)
        if (
            s.shape[1] != 8
            or a.shape[1] != 7
            or not np.isfinite(s).all()
            or not np.isfinite(a).all()
        ):
            raise ValueError(f"Invalid training data: {p}")
        state_parts.append(s)
        # Statistics cover the same clamped action chunks sampled in training.
        ix = np.minimum(np.arange(len(a))[:, None] + np.arange(horizon), len(a) - 1)
        action_parts.append(a[ix].reshape(-1, 7))
        frames += len(a)
        episodes += 1
    if (episodes, frames) != (3018, 361736):
        raise ValueError(f"Incomplete dataset: {episodes} episodes {frames} frames")
    result = {}
    for key, parts in [("state", state_parts), ("actions", action_parts)]:
        x = np.concatenate(parts)
        result[key] = {
            k: v.tolist()
            for k, v in dict(
                mean=x.mean(0),
                std=x.std(0),
                q01=np.quantile(x, 0.01, axis=0),
                q99=np.quantile(x, 0.99, axis=0),
            ).items()
        }
    return dict(
        norm_stats=result,
        episodes=episodes,
        frames=frames,
        action_horizon=horizon,
        method="exact-frame-weighted-quantiles-clamped-chunks-v1",
    )
