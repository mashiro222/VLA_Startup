"""Closed-loop evaluator; the upstream environment applies all perturbations."""
import argparse
from collections import deque
import contextlib
import fcntl
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import traceback

from .journal import append, read_records, validate_prefix
from .manifest import ROOT, digest, validate


def timeout_handler(signum, frame):
    raise TimeoutError('Episode exceeded the configured wall-clock limit')


def quat_to_axisangle(quat, np):
    quat = np.asarray(quat, dtype=np.float64)
    w = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1 - w * w)
    if denominator == 0:
        return np.zeros(3)
    return quat[:3] * 2 * np.arccos(w) / denominator


def numpy_state(np):
    algorithm, values, position, has_gaussian, cached = np.random.get_state()
    return [algorithm, values.tolist(), position, has_gaussian, cached]


def restore_numpy(np, state):
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--episode-timeout', type=int, default=1800)
    parser.add_argument('--videos', type=int, default=2, help='Maximum videos per run; 0 disables')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((run / 'manifest.json').read_text())
    validate(manifest)
    protocol = manifest['evaluation']
    import numpy as np
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    import imageio.v2 as imageio
    np.random.seed(manifest['seed'])
    random.seed(manifest['seed'])
    torch.set_num_threads(1)
    client = WebsocketClientPolicy('127.0.0.1', args.port)
    metadata = client.get_server_metadata()
    expected_checkpoint = hashlib.sha256((ROOT / 'config/checkpoint_files.json').read_bytes()).hexdigest()
    if (metadata.get('model') != 'pi05_libero' or metadata.get('num_steps') != protocol['num_steps']
            or metadata.get('action_horizon') != 10
            or metadata.get('openpi_commit') != manifest['sources']['openpi']['commit']
            or metadata.get('checkpoint_manifest_sha256') != expected_checkpoint):
        raise ValueError('Policy-server identity differs from the locked experiment')
    settings = {'server': metadata, **protocol,
                'seed': manifest['seed'], 'prompt_mode': manifest['prompt_mode'],
                'manifest_sha256': manifest['manifest_sha256']}
    # Resuming across code changes would invalidate RNG and metric guarantees.
    tracked_inputs = list((ROOT / 'src/vla_startup').glob('*.py')) + [
        ROOT / 'scripts/patch_sim.py', ROOT / 'config/imagemagick.json',
        ROOT / '.runtime/repos/openpi/uv.lock']
    settings['code_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(tracked_inputs)}
    settings['simulation_lock_sha256'] = hashlib.sha256((ROOT / 'requirements/sim.lock').read_bytes()).hexdigest()
    fingerprint = digest(settings)
    records = read_records(run / 'episodes.jsonl', repair_tail=True)
    expected_keys = validate_prefix(records, manifest, fingerprint)
    config_path = run / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise ValueError('Run configuration changed; use a new run directory')
    config_path.write_text(json.dumps(settings, indent=2) + '\n')
    rng_state = records[-1]['rng_after'] if records else [0, 0]
    resume_np = records[-1]['numpy_after'] if records else None
    resume_task = (records[-1]['suite'], records[-1]['task_id']) if records else None
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.signal(signal.SIGTERM, lambda *unused: (_ for _ in ()).throw(InterruptedError('Slurm termination')))
    cursor = 0
    suites = {}
    for row in manifest['tasks']:
        suite_name, task_id = row['suite'], row['task_id']
        trials = manifest['trials_per_task']
        if cursor + trials <= len(records):
            cursor += trials
            continue
        if suite_name not in suites:
            with contextlib.redirect_stdout(io.StringIO()):
                suites[suite_name] = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        suite = suites[suite_name]
        task = suite.get_task(task_id)
        if task.name != row['name']:
            raise ValueError('Task indices no longer match manifest names')
        env = None
        current_key = expected_keys[max(cursor, len(records))]
        try:
            signal.alarm(args.episode_timeout)
            # Keep the logical variant filename. The Plus wrapper decodes it.
            env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
                                     camera_heights=protocol['render_resolution'], camera_widths=protocol['render_resolution'])
            env.seed(manifest['seed'])
            initial_states = suite.get_task_init_states(task_id)
            if len(initial_states) < trials:
                raise ValueError('Not enough initial states for requested trials')
            prompt = task.language if manifest['prompt_mode'] == 'upstream' else env.language_instruction
            if resume_task == (suite_name, task_id) and resume_np is not None:
                restore_numpy(np, resume_np)
            for episode in range(trials):
                index = cursor + episode
                if index < len(records):
                    continue
                current_key = expected_keys[index]
                signal.alarm(args.episode_timeout)
                start = time.monotonic()
                env.reset()
                observation = env.set_init_state(initial_states[episode])
                # Match official OpenPI preprocessing and action semantics.
                for _ in range(protocol['wait_steps']):
                    observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
                actions = deque()
                succeeded = False
                calls = 0
                inference_seconds = 0.0
                steps = 0
                writer = None
                if index < args.videos:
                    (run / 'videos').mkdir(exist_ok=True)
                    writer = imageio.get_writer(run / 'videos' / f'{suite_name}_{task_id}_{episode}.mp4', fps=10)
                try:
                    for steps in range(1, row['max_steps'] + 1):
                        front = np.ascontiguousarray(observation['agentview_image'][::-1, ::-1])
                        wrist = np.ascontiguousarray(observation['robot0_eye_in_hand_image'][::-1, ::-1])
                        resolution = protocol['policy_resolution']
                        front = image_tools.convert_to_uint8(image_tools.resize_with_pad(front, resolution, resolution))
                        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resolution, resolution))
                        if writer:
                            writer.append_data(front)
                        if not actions:
                            state = np.concatenate((observation['robot0_eef_pos'],
                                                    quat_to_axisangle(observation['robot0_eef_quat'], np),
                                                    observation['robot0_gripper_qpos']))
                            request = {'observation/image': front, 'observation/wrist_image': wrist,
                                       'observation/state': state, 'prompt': str(prompt), '__vla_rng': rng_state}
                            call_start = time.monotonic()
                            response = client.infer(request)
                            inference_seconds += time.monotonic() - call_start
                            chunk = np.asarray(response['actions'])
                            if chunk.ndim != 2 or chunk.shape[0] < protocol['replan_steps'] or chunk.shape[1] != 7 or not np.isfinite(chunk).all():
                                raise ValueError(f'Invalid action chunk: {chunk.shape}')
                            rng_state = response['__vla_rng']
                            actions.extend(chunk[:protocol['replan_steps']])
                            calls += 1
                        observation, _, done, _ = env.step(actions.popleft().tolist())
                        if done:
                            succeeded = True
                            break
                finally:
                    if writer:
                        writer.close()
                record = {**row, 'episode': episode, 'episode_key': current_key,
                          'manifest_sha256': manifest['manifest_sha256'], 'run_fingerprint': fingerprint,
                          'success': succeeded, 'steps': steps, 'prompt': str(prompt),
                          'inference_calls': calls, 'inference_seconds': inference_seconds,
                          'elapsed_seconds': time.monotonic() - start,
                          'rng_after': rng_state, 'numpy_after': numpy_state(np),
                          'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
                append(run / 'episodes.jsonl', record)
                records.append(record)
                logging.info('%s success=%s completed=%s/%s seconds=%.2f', current_key, succeeded,
                             len(records), len(expected_keys), record['elapsed_seconds'])
            cursor += trials
        except BaseException as error:
            append(run / 'errors.jsonl', {'episode_key': current_key, 'error': repr(error),
                                         'traceback': traceback.format_exc(), 'time': time.time()})
            raise
        finally:
            signal.alarm(0)
            if env is not None:
                env.close()
    logging.info('Finished %s episodes', len(records))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    main()
