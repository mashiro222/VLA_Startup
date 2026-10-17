"""Bounded-memory CPU preparation; no policy or PyTorch imports."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def read_episodes(paths, columns=None, prefetch=4):
    """Bound I/O concurrency and memory while preserving episode order exactly."""
    if prefetch < 1:
        raise ValueError("prefetch must be positive")
    paths = iter(paths)

    def read(path):
        return pq.read_table(path, columns=columns, use_threads=False).to_pydict()

    with ThreadPoolExecutor(max_workers=prefetch) as pool:
        pending = deque()
        for _ in range(prefetch):
            path = next(paths, None)
            if path is not None:
                pending.append((path, pool.submit(read, path)))
        while pending:
            path, future = pending.popleft()
            value = future.result()
            following = next(paths, None)
            if following is not None:
                pending.append((following, pool.submit(read, following)))
            yield path, value


def compute_stats(root, horizon=10):
    root = Path(root)
    state_parts = []
    action_parts = []
    frames = 0
    episodes = 0
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    for p, e in read_episodes(paths, columns=["state", "actions"]):
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
