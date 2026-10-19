"""Pin completed atomic checkpoint files without copying GPU-sized data."""
import argparse
import json
import os
from pathlib import Path
import time

from arena_startup.core import save, file_hash


def pin(source, output, steps=(3000, 5000)):
    source, output = Path(source), Path(output)
    plan = source / 'plan.json'
    original = json.loads(plan.read_text())
    output.mkdir(parents=True, exist_ok=True)
    saved_plan = output / 'plan.json'
    if saved_plan.exists() and file_hash(saved_plan) != file_hash(plan):
        raise ValueError('Pinned source plan changed')
    if not saved_plan.exists():
        saved_plan.write_bytes(plan.read_bytes())
    pinned = []
    for arm in ('baseline', 'sf'):
        d = output / arm
        d.mkdir(exist_ok=True)
        for step in steps:
            src, dst = source / arm / f'step-{step:08d}.pt', d / f'step-{step:08d}.pt'
            if not dst.exists():
                try:
                    os.link(src, dst)
                except FileExistsError:
                    pass
                except FileNotFoundError:
                    continue
            pinned.append(dict(arm=arm, step=step, path=str(dst), bytes=dst.stat().st_size))
    save(output / 'pins.json', dict(source=str(source), source_plan_sha256=file_hash(plan),
                                  checked_unix=time.time(), checkpoints=pinned,
                                  global_batch=original['recipe']['global_batch']))
    return pinned


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source'); p.add_argument('output')
    a = p.parse_args(); print(pin(a.source, a.output))
