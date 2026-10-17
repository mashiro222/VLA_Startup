"""Arena rollout protocol with explicit per-episode seeds and error journals.

Environment, task definitions and safety predicates come from pinned VLA-Arena.
Preprocessing/rollout semantics follow its OpenPI evaluator (Apache-2.0).
"""
import argparse
from collections import deque
import gzip
import json
import logging
import os
import random
import signal
import time
import traceback

from .core import ROOT, append, claim, episode_seed, file_hash, finish_metrics, key, records, validate_plan, validate_rows
from .diagnostics import PredicateObserver, failure_reason


def interrupted(*unused):
    raise InterruptedError('Worker interrupted; the incomplete episode will be replayed')


def timed_out(*unused):
    raise TimeoutError('Episode wall-clock limit exceeded; this is an infrastructure error, not task failure')


def rollout(unit, episode, plan, client, video_path=None, diagnostic_path=None):
    logging.info('%s: loading simulator and initializing scene', key(unit, episode))
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from vla_arena.vla_arena.envs import OffScreenRenderEnv
    from vla_arena.vla_arena.envs.predicates import check_temporal_predicate
    from vla_arena.vla_arena.utils.eval_cost import get_timeout_final_cost
    from vla_arena.vla_arena.utils.eval_init_state import select_init_state_index
    from .adapter import policy_images, policy_state, action_chunk

    proto = plan['config']['evaluation']
    seed = episode_seed(unit, episode)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    started = time.monotonic()
    env, writer, observer, trace = None, None, None, None
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
        logging.info('%s: initial scene ready', key(unit, episode))
        if video_path:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video_path, fps=20)
        if proto.get('predicate_diagnostics', False):
            scale = proto['hazard_cost_scale'] if unit['suite'] == 'safety_hazard_avoidance' else 1.0
            observer = PredicateObserver(env, check_temporal_predicate, scale)
            if diagnostic_path:
                diagnostic_path.mkdir(parents=True, exist_ok=True)
                trace = gzip.open(diagnostic_path / 'steps.partial.jsonl.gz', 'wt')
        actions = deque()
        rng = [0, seed]
        raw_cost = 0.0
        calls = 0
        inference_seconds = 0.0
        done, info = False, {}
        for steps in range(1, proto['max_steps'] + 1):
            if writer or not actions:
                front, wrist = policy_images(observation, proto['policy_resolution'])
                if writer:
                    writer.append_data(front)
            if not actions:
                state = policy_state(observation)
                before = time.monotonic()
                response = client.infer({'observation/image': front, 'observation/wrist_image': wrist,
                                         'observation/state': state, 'prompt': unit['prompt'], '__vla_rng': rng})
                inference_seconds += time.monotonic() - before
                chunk = action_chunk(response, proto['replan_steps'])
                if video_path and diagnostic_path and observer and calls < 3:
                    np.savez_compressed(diagnostic_path / f'policy-call-{calls:03d}.npz',
                                        raw_front=observation['agentview_image'],
                                        raw_wrist=observation['robot0_eye_in_hand_image'],
                                        policy_front=front, policy_wrist=wrist, state=state,
                                        action_chunk=chunk, rng_before=np.asarray(rng, dtype=np.uint32))
                rng = response['__vla_rng']
                actions.extend(chunk[:proto['replan_steps']])
                calls += 1
            action = actions.popleft().tolist()
            state_before = policy_state(observation).tolist() if observer else None
            if observer:
                observer.begin_step(steps)
            observation, _, done, info = env.step(action)
            # Safety tasks must return a cost; never silently replace missing costs with zero.
            value = float(info['cost'])
            if not np.isfinite(value) or value < 0:
                raise ValueError('Non-finite or negative safety cost')
            raw_cost += value
            if observer:
                events = observer.consume(value)
                if trace:
                    trace.write(json.dumps({'step': steps, 'action': action, 'state_before': state_before,
                                           'state_after': policy_state(observation).tolist(),
                                           'done': bool(done), 'task_success': bool(info.get('success', done)),
                                           'environment_timeout': bool(info.get('timeout', False)),
                                           'raw_cost': value, 'raw_cumulative_cost': raw_cost,
                                           'cumulative_cc_before_terminal': raw_cost * observer.scale,
                                           'cost_predicates': events, 'goal_predicates': observer.goals},
                                          allow_nan=False) + '\n')
            if done:
                break
        terminal = float(get_timeout_final_cost(env)) if not done else 0.0
        success, cost = finish_metrics(unit['suite'], raw_cost, terminal, done, info,
                                       proto['hazard_cost_scale'], proto['safety_success_filter'])
        diagnostic = None
        if observer:
            final_events = observer.consume(terminal) if not done else []
            diagnostic = observer.summary(cost)
            if trace:
                trace.write(json.dumps({'phase': 'episode_end', 'step': steps, 'cost_predicates': final_events,
                                       'cc': cost, 'failure_reason': failure_reason(success, done, info)},
                                      allow_nan=False) + '\n')
                trace.close()
                trace = None
                os.replace(diagnostic_path / 'steps.partial.jsonl.gz', diagnostic_path / 'steps.jsonl.gz')
        return {'episode_key': key(unit, episode), 'episode': episode, 'unit_id': unit['id'],
                'suite': unit['suite'], 'level': unit['level'], 'task_id': unit['task_id'], 'name': unit['name'],
                'seed': unit['seed'], 'episode_seed': seed, 'initial_state_index': init_index,
                'plan_sha256': plan['sha256'], 'prompt': unit['prompt'], 'success': success, 'cost': cost,
                'raw_step_cost': raw_cost, 'timeout_final_cost': terminal, 'steps': steps,
                'termination': 'success' if success else ('environment_done' if done else 'horizon'),
                'failure_reason': failure_reason(success, done, info),
                'task_success': bool(done and info.get('success', done)),
                'diagnostics': diagnostic,
                'inference_calls': calls, 'inference_seconds': inference_seconds,
                'elapsed_seconds': time.monotonic() - started, 'rng_after': rng,
                'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    finally:
        try:
            if trace:
                trace.close()
            if observer:
                observer.close()
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
    from .transport import ArenaClient, TRANSPORT_VERSION
    torch.set_num_threads(1)
    sources = json.loads((ROOT / 'config/sources.json').read_text())
    expected = {'model': 'pi05_libero', 'backend': 'jax', 'action_horizon': 10,
                'num_steps': plan['config']['evaluation']['num_steps'],
                'openpi_commit': sources['openpi']['commit'],
                'checkpoint_manifest_sha256': file_hash(ROOT / 'config/checkpoint_files.json'),
                'transport': TRANSPORT_VERSION}
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, timed_out)
    with ArenaClient(args.port, expected, plan['config']['evaluation']['policy_request_timeout_seconds']) as client:
        evaluate_units(args, plan, client)


def evaluate_units(args, plan, client):
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
                    row = rollout(unit, episode, plan, client, video, directory / 'diagnostics' / str(episode))
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
