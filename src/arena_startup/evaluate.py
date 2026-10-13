"""Arena rollout protocol with explicit per-episode seeds and error journals.

Environment, task definitions and safety predicates come from pinned VLA-Arena.
Preprocessing/rollout semantics follow its OpenPI evaluator (Apache-2.0).
"""
import argparse
from collections import deque
import json
import logging
import os
import random
import signal
import time
import traceback

from .core import ROOT, append, claim, episode_seed, file_hash, finish_metrics, key, records, validate_plan, validate_rows


def interrupted(*unused):
    raise InterruptedError('Worker interrupted; the incomplete episode will be replayed')


def timed_out(*unused):
    raise TimeoutError('Episode wall-clock limit exceeded; this is an infrastructure error, not task failure')


def rollout(unit, episode, plan, client, video_path=None):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from openpi_client import image_tools
    from vla_arena.vla_arena.envs import OffScreenRenderEnv
    from vla_arena.vla_arena.utils.eval_cost import get_timeout_final_cost
    from vla_arena.vla_arena.utils.eval_init_state import select_init_state_index
    from vla_startup.evaluate import quat_to_axisangle

    proto = plan['config']['evaluation']
    seed = episode_seed(unit, episode)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    started = time.monotonic()
    env, writer = None, None
    try:
        # Fresh environment per episode prevents simulator state leaking across dynamic workers/retries.
        env = OffScreenRenderEnv(bddl_file_name=str(ROOT / unit['bddl']),
                                camera_heights=proto['render_resolution'], camera_widths=proto['render_resolution'],
                                camera_offset=False, color_randomize=False, add_noise=False, light_adjustment=False)
        env.seed(seed)
        initial_states = torch.load(ROOT / unit['init_file'], weights_only=False)
        init_index = select_init_state_index(num_initial_states=len(initial_states), episode_idx=episode,
                                            selection_mode=proto['init_state_selection_mode'],
                                            offset=proto['init_state_offset'], offset_random=False,
                                            rng=np.random.default_rng(seed))
        env.reset()
        observation = env.set_init_state(initial_states[init_index])
        # Follow the upstream warmup protocol; costs in these settling steps are not scored.
        for _ in range(proto['wait_steps']):
            observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
        if video_path:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video_path, fps=20)
        actions = deque()
        rng = [0, seed]
        raw_cost = 0.0
        calls = 0
        inference_seconds = 0.0
        done, info = False, {}
        for steps in range(1, proto['max_steps'] + 1):
            if writer or not actions:
                front = np.ascontiguousarray(observation['agentview_image'][::-1, ::-1])
                wrist = np.ascontiguousarray(observation['robot0_eye_in_hand_image'][::-1, ::-1])
                size = proto['policy_resolution']
                front = image_tools.convert_to_uint8(image_tools.resize_with_pad(front, size, size))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, size, size))
                if writer:
                    writer.append_data(front)
            if not actions:
                state = np.concatenate((observation['robot0_eef_pos'],
                                        quat_to_axisangle(observation['robot0_eef_quat'], np),
                                        observation['robot0_gripper_qpos']))
                before = time.monotonic()
                response = client.infer({'observation/image': front, 'observation/wrist_image': wrist,
                                         'observation/state': state, 'prompt': unit['prompt'], '__vla_rng': rng})
                inference_seconds += time.monotonic() - before
                chunk = np.asarray(response['actions'])
                if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < proto['replan_steps'] or not np.isfinite(chunk).all():
                    raise ValueError(f'Invalid policy action chunk: {chunk.shape}')
                rng = response['__vla_rng']
                actions.extend(chunk[:proto['replan_steps']])
                calls += 1
            observation, _, done, info = env.step(actions.popleft().tolist())
            # Safety tasks must return a cost; never silently replace missing costs with zero.
            value = float(info['cost'])
            if not np.isfinite(value) or value < 0:
                raise ValueError('Non-finite or negative safety cost')
            raw_cost += value
            if done:
                break
        terminal = float(get_timeout_final_cost(env)) if not done else 0.0
        success, cost = finish_metrics(unit['suite'], raw_cost, terminal, done, info,
                                       proto['hazard_cost_scale'], proto['safety_success_filter'])
        return {'episode_key': key(unit, episode), 'episode': episode, 'unit_id': unit['id'],
                'suite': unit['suite'], 'level': unit['level'], 'task_id': unit['task_id'], 'name': unit['name'],
                'seed': unit['seed'], 'episode_seed': seed, 'initial_state_index': init_index,
                'plan_sha256': plan['sha256'], 'prompt': unit['prompt'], 'success': success, 'cost': cost,
                'raw_step_cost': raw_cost, 'timeout_final_cost': terminal, 'steps': steps,
                'termination': 'success' if success else ('environment_done' if done else 'horizon'),
                'inference_calls': calls, 'inference_seconds': inference_seconds,
                'elapsed_seconds': time.monotonic() - started, 'rng_after': rng,
                'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    finally:
        try:
            if writer:
                writer.close()
        finally:
            if env is not None:
                env.close()


def main():
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path); p.add_argument('--port', type=int, required=True); p.add_argument('--worker', required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Refusing to evaluate on the login node')
    plan = json.loads((args.run / 'plan.json').read_text())
    validate_plan(plan)
    import torch
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    torch.set_num_threads(1)
    client = WebsocketClientPolicy('127.0.0.1', args.port)
    meta = client.get_server_metadata()
    sources = json.loads((ROOT / 'config/sources.json').read_text())
    if (meta.get('model') != 'pi05_libero' or meta.get('backend') != 'jax' or meta.get('action_horizon') != 10
            or meta.get('num_steps') != plan['config']['evaluation']['num_steps']
            or meta.get('openpi_commit') != sources['openpi']['commit']
            or meta.get('checkpoint_manifest_sha256') != file_hash(ROOT / 'config/checkpoint_files.json')):
        raise ValueError('Unexpected model/normalization/checkpoint identity')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, timed_out)
    completed, videos = 0, 0
    for unit in plan['units']:
        directory = args.run / 'units' / unit['id']
        with claim(directory) as owned:
            if not owned:
                continue
            rows = records(directory / 'episodes.jsonl', repair=True)
            validate_rows(rows, unit, plan)
            if len(rows) == unit['trials']:
                continue
            for name, expected in [(unit['bddl'], unit['bddl_sha256']), (unit['init_file'], unit['init_sha256'])]:
                if file_hash(ROOT / name) != expected:
                    raise ValueError(f'Task input changed: {name}')
            logging.info('Worker %s claimed %s (%d/%d already done)', args.worker, unit['id'], len(rows), unit['trials'])
            for episode in range(len(rows), unit['trials']):
                try:
                    signal.alarm(plan['config']['evaluation']['episode_timeout_seconds'])
                    video = directory / 'videos' / f'{episode}.mp4' if videos < plan['videos_per_worker'] else None
                    row = rollout(unit, episode, plan, client, video)
                    row['worker'] = args.worker
                    append(directory / 'episodes.jsonl', row)
                    completed += 1
                    videos += video is not None
                    logging.info('%s success=%s CC=%.3f steps=%s worker_completed=%s',
                                 row['episode_key'], row['success'], row['cost'], row['steps'], completed)
                except BaseException as error:
                    append(directory / 'errors.jsonl', {'episode_key': key(unit, episode), 'error': repr(error),
                                                        'traceback': traceback.format_exc(), 'worker': args.worker,
                                                        'time': time.time()})
                    raise
                finally:
                    signal.alarm(0)
    logging.info('Worker finished available units; new episodes=%s. Other workers may still be running.', completed)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
    main()
