#!/usr/bin/env python3
"""Render one Arena initial scene in an existing Slurm allocation, without a model or submission."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from arena_startup.runtime import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='safety_static_obstacles')
    parser.add_argument('--level', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--task-id', type=int, choices=range(5), default=0)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/scene-preview')
    parser.add_argument('--list', action='store_true', help='List task names/prompts on CPU; no Slurm needed')
    args = parser.parse_args()
    with (ROOT / 'reports/preparation/arena-task-catalog.csv').open() as source:
        catalog = list(csv.DictReader(source))
    if args.list:
        for row in catalog:
            print(row['suite'], 'L' + row['level'], row['task_id'], row['prompt'], sep='\t')
        return 0
    selected = [r for r in catalog if r['suite'] == args.suite and int(r['level']) == args.level and int(r['task_id']) == args.task_id]
    if len(selected) != 1:
        parser.error('Task not found; inspect --list')
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Rendering requires an existing Slurm GPU allocation; this tool never submits a job')
    python = ROOT / '.runtime/arena/bin/python'
    # Compare venv paths, not resolved symlinks (uv interpreters may share a base binary).
    if Path(sys.prefix) != ROOT / '.runtime/arena':
        return subprocess.run([str(python), str(Path(__file__).resolve()), *sys.argv[1:]], env=environment()).returncode
    from arena_startup.core import episode_seed, file_hash
    import imageio.v2 as imageio
    import numpy as np
    import random
    import torch
    from openpi_client import image_tools
    from vla_arena.vla_arena.envs import OffScreenRenderEnv
    row = selected[0]
    for field, checksum in [('bddl', 'bddl_sha256'), ('init_file', 'init_sha256')]:
        if file_hash(ROOT / row[field]) != row[checksum]:
            raise ValueError(f'Task catalog hash mismatch: {field}')
    if args.output.exists():
        parser.error('Output directory already exists; choose a new directory')
    cfg = json.loads((ROOT / 'config/arena.json').read_text())['evaluation']
    unit_id = f'{args.suite}-L{args.level}-t{args.task_id}-s{args.seed}'
    seed = episode_seed({'id': unit_id}, 0)
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    env = OffScreenRenderEnv(bddl_file_name=str(ROOT / row['bddl']),
                            camera_heights=cfg['render_resolution'], camera_widths=cfg['render_resolution'],
                            camera_offset=False, color_randomize=False, add_noise=False, light_adjustment=False)
    try:
        env.seed(seed)
        states = torch.load(ROOT / row['init_file'], weights_only=False)
        env.reset()
        obs = env.set_init_state(states[0])
        for _ in range(cfg['wait_steps']):
            obs, _, _, _ = env.step([0.] * 6 + [-1.])
        args.output.mkdir(parents=True)
        for name, key in [('front', 'agentview_image'), ('wrist', 'robot0_eye_in_hand_image')]:
            frame = np.ascontiguousarray(obs[key][::-1, ::-1])
            imageio.imwrite(args.output / f'{name}-render.png', frame)
            size = cfg['policy_resolution']
            imageio.imwrite(args.output / f'{name}-policy.png', image_tools.convert_to_uint8(image_tools.resize_with_pad(frame, size, size)))
        (args.output / 'scene.json').write_text(json.dumps(
            {'task': row, 'unit_id': unit_id, 'episode_seed': seed, 'initial_state_index': 0,
             'wait_steps': cfg['wait_steps'], 'slurm_job_id': os.environ['SLURM_JOB_ID'],
             'note': 'Initial scene only; no model actions or success/safety score.'}, indent=2) + '\n')
    finally:
        env.close()
    print(args.output.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
