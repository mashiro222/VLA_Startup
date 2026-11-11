"""Fresh Arena environment per paired episode; filter each executed action."""
from collections import deque
import gzip
import json
from pathlib import Path
import random
import time
import numpy as np
from .common import ROOT, save, record_path
from .filter import filter_action
from .geometry import nearest_segment, barrier
from .metrics import classify
from .scene import Scene


def rollout(run, episode, cfg, client):
    import torch
    import imageio.v2 as imageio
    from vla_arena.vla_arena.envs import OffScreenRenderEnv
    from vla_arena.vla_arena.envs.predicates import check_temporal_predicate
    from vla_arena.vla_arena.utils.eval_cost import get_timeout_final_cost, is_success_done
    from arena_startup.adapter import policy_images, policy_state, action_chunk
    from arena_startup.diagnostics import PredicateObserver
    ev = cfg['evaluation']; fcfg = dict(cfg['filter'], gamma=episode['gamma'])
    path = record_path(run, episode); path.mkdir(parents=True, exist_ok=True)
    random.seed(episode['seed']); np.random.seed(episode['seed']); torch.manual_seed(episode['seed']); torch.set_num_threads(1)
    env = scene = observer = writer = contact_file = None
    started = time.monotonic()
    trace = {k: [] for k in ['eef_before', 'eef_after', 'eef_quat', 'gripper_qpos', 'gripper_command',
        'action_nominal', 'action_executed', 'h', 'h_after', 'min_h', 'n_active', 'intervention_action',
        'intervention_m', 'filter_seconds', 'inference_seconds', 'dist_subgoal_m', 'held', 'collision',
        'gripper_collision', 'robot_link_collision', 'held_object_collision', 'target_object_collision',
        'raw_cost', 'solver_failed', 'prediction_error_m', 'actual_delta_controller', 'obstacle_centers',
        'constraint_slack', 'constraint_dual', 'action_effective_intervention_m', 'free_motion']}
    try:
        env = OffScreenRenderEnv(bddl_file_name=str(ROOT / episode['bddl']),
            camera_heights=ev['render_size'], camera_widths=ev['render_size'],
            camera_offset=False, color_randomize=False, add_noise=False, light_adjustment=False)
        env.seed(episode['seed']); env.reset()
        states = torch.load(ROOT / episode['init_file'], weights_only=False)
        obs = env.set_init_state(states[episode['init_index']])
        for _ in range(ev['settle_steps']): obs, _, _, _ = env.step([0.] * 6 + [-1.])
        scene = Scene(env, fcfg)
        if not np.allclose(scene.eef(), obs['robot0_eef_pos'], atol=1e-6):
            raise ValueError('Observation EE position and gripper proxy site disagree')
        if int(env.env.control_freq) != ev['hz']: raise ValueError('Unexpected control frequency')
        centers, matrices = scene.ellipsoids()
        selected = ([] if episode['condition'] == 'C0' or not len(centers) else
                    ([nearest_segment(centers, scene.eef(), scene.position(scene.target))]
                     if episode['condition'] == 'C1' else list(range(len(centers)))))
        scene.begin_step(); initial_contacts = scene.contacts()
        initial_collision = any(x['main_collision'] for x in initial_contacts)
        initial_h, _ = barrier(scene.eef(), centers, matrices)
        save(path / 'geometry.json', dict(scene.audit, initial_contacts=initial_contacts,
             initial_h=initial_h.tolist(), selected_obstacles=[scene.names[i] for i in selected],
             fitted_scale_m=fcfg['action_scale_m']))
        for name, frame in zip(('front', 'wrist'), policy_images(obs, ev['policy_size'])):
            imageio.imwrite(path / f'{name}-initial.png', frame)
        observer = PredicateObserver(env, check_temporal_predicate, scale=1.0)
        writer = imageio.get_writer(path / 'video.partial.mp4', fps=ev['hz'])
        contact_file = gzip.open(path / 'contacts.jsonl.gz', 'wt')
        queue = deque(); rng = [0, episode['seed']]; cc = 0.; done = False; info = {}
        statuses = []; collision_bodies = set(); components = set()
        for t in range(ev['horizon']):
            inference_seconds = 0.
            if not queue:
                front, wrist = policy_images(obs, ev['policy_size']); begin = time.monotonic()
                response = client.infer({'observation/image': front, 'observation/wrist_image': wrist,
                    'observation/state': policy_state(obs), 'prompt': episode['prompt'], '__vla_rng': rng})
                inference_seconds = time.monotonic() - begin
                rng = response['__vla_rng']; queue.extend(action_chunk(response, ev['replan_steps'])[:ev['replan_steps']])
            nominal = np.asarray(queue.popleft()); p = scene.eef(); centers, matrices = scene.ellipsoids()
            action, diagnostic = filter_action(nominal, p, centers, matrices, selected, fcfg,
                    enabled=episode['condition'] != 'C0', rotation=scene.rotation)
            if not np.array_equal(action[3:], nominal[3:]): raise AssertionError('Rotation/gripper altered')
            observer.begin_step(t + 1); scene.begin_step()
            obs, _, done, info = env.step(action.tolist())
            contacts = scene.contacts(); held = scene.grasp(); p_after = scene.eef()
            cost = float(info['cost'])
            if not np.isfinite(cost) or cost < 0: raise ValueError('Invalid official Arena cost')
            events = observer.consume(cost); cc += cost
            main = [x for x in contacts if x['main_collision']]
            for x in main:
                collision_bodies.add(tuple(x['bodies']))
                for label in ('gripper', 'robot_link', 'held_object'):
                    if x[label]: components.add(label)
            contact_file.write(json.dumps({'step': t, 'sim_time': float(env.sim.data.time), 'contacts': contacts,
                'cost_predicates': events, 'goal_predicates': observer.goals, 'solver_status': diagnostic['status'],
                'fallback_satisfies_linear_cbf': diagnostic['fallback_satisfies_linear_cbf']}) + '\n')
            predicted = fcfg['action_scale_m'] * scene.rotation @ np.clip(action[:3], -1, 1)
            h_after, _ = barrier(p_after, *scene.ellipsoids())
            values = dict(eef_before=p, eef_after=p_after, eef_quat=obs['robot0_eef_quat'],
                gripper_qpos=obs['robot0_gripper_qpos'], gripper_command=action[6], action_nominal=nominal,
                action_executed=action, h=diagnostic['h'], h_after=h_after,
                min_h=float(np.min(diagnostic['h'])) if len(centers) else np.nan,
                n_active=diagnostic['n_active'], intervention_action=diagnostic['intervention_action'],
                intervention_m=diagnostic['intervention_m'], filter_seconds=diagnostic['filter_seconds'],
                inference_seconds=inference_seconds, held=held, collision=bool(main),
                gripper_collision=any(x['gripper'] for x in main), robot_link_collision=any(x['robot_link'] for x in main),
                held_object_collision=any(x['held_object'] for x in main),
                target_object_collision=any(x['target_object'] and x['obstacle'] and not x['robot'] for x in contacts),
                raw_cost=cost, solver_failed=diagnostic['solver_failed'],
                free_motion=not any(x['robot'] and not all(g in scene.robot_ids for g in x['geom_ids']) for x in contacts),
                prediction_error_m=np.linalg.norm(p_after - p - predicted),
                actual_delta_controller=scene.rotation.T @ (p_after - p), obstacle_centers=centers,
                constraint_slack=diagnostic['slack'], constraint_dual=diagnostic['dual'],
                action_effective_intervention_m=fcfg['action_scale_m'] * np.linalg.norm(np.clip(nominal[:3], -1, 1) - np.clip(action[:3], -1, 1)),
                dist_subgoal_m=np.linalg.norm(p_after - scene.position(scene.receptacle if held else scene.target)))
            for key, value in values.items(): trace[key].append(value)
            statuses.append(diagnostic['status'])
            writer.append_data(np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]))
            if done: break
        success = is_success_done(done, info)
        timeout = bool(not success and (not done or info.get('timeout', False)))
        terminal = float(get_timeout_final_cost(env)) if not done else 0.
        cc += terminal
        terminal_events = observer.consume(terminal) if not done else []
        labels = classify(trace, success, timeout, cfg['diagnostics'], ev['hz'], initial_collision)
        row = dict(episode, **labels, n_steps=len(trace['eef_after']), CC=cc, terminal_cost=terminal,
            initial_collision=initial_collision, initial_h=initial_h.tolist(),
            selected_obstacles=[scene.names[i] for i in selected], collision_bodies=sorted(collision_bodies),
            collision_components=sorted(components), solver_failures=int(np.sum(trace['solver_failed'])),
            solver_status_counts={s: statuses.count(s) for s in sorted(set(statuses))},
            mean_intervention_m=float(np.mean(trace['intervention_m'])),
            intervention_frequency=float(np.mean(np.asarray(trace['intervention_m']) > cfg['diagnostics']['intervention_m'])),
            mean_prediction_error_m=float(np.mean(trace['prediction_error_m'])),
            filter_seconds=float(np.sum(trace['filter_seconds'])),
            inference_seconds=float(np.sum(trace['inference_seconds'])), wall_seconds=time.monotonic() - started,
            official_diagnostics=observer.summary(cc), terminal_events=terminal_events,
            termination='success' if success else ('horizon_timeout' if timeout else 'environment_done_without_success'))
        # Close artifacts before publishing the atomic completion record. Partial episodes are retried, never scored.
        writer.close(); writer = None; contact_file.close(); contact_file = None
        np.savez_compressed(path / 'trace.npz', **{k: np.asarray(v) for k, v in trace.items()})
        keep = episode['ordinal'] < ev['video_first'] or (episode['condition'] != 'C0' and labels['stuck'])
        if keep: (path / 'video.partial.mp4').replace(path / 'video.mp4')
        else: (path / 'video.partial.mp4').unlink()
        row['video'] = 'video.mp4' if keep else None
        save(path / 'episode.json', row)
        print(f"L{episode['level']} {episode['name']} {episode['condition']} init={episode['init_index']} seed={episode['repeat']} SR={int(success)} CC={cc:g} stuck={labels['stuck']}", flush=True)
        return row
    finally:
        if contact_file: contact_file.close()
        if writer: writer.close()
        if observer: observer.close()
        if scene: scene.close()
        if env: env.close()
