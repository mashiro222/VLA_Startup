"""Outcomes, contact labels, stalling diagnostics and action-scale calibration."""
import numpy as np


def trapped_window(positions, widths, cfg, hz):
    n = int(round(cfg['window_seconds'] * hz))
    if len(positions) < n:
        return False, None
    pos = np.asarray(positions)[-n:]; width = np.asarray(widths)[-n:]
    stationary = np.max(np.linalg.norm(pos - pos.mean(0), axis=1)) <= cfg['radius_m']
    unchanged = np.max(np.ptp(width, axis=0)) <= cfg['gripper_width_tolerance_m']
    return bool(stationary and unchanged), len(positions) - n


def classify(trace, success, timeout, cfg, hz, initial_collision=False):
    pos = np.asarray(trace['eef_after']); widths = np.asarray(trace['gripper_qpos'])
    stationary, start = trapped_window(pos, widths, cfg, hz)
    stuck = bool(timeout and stationary)
    collisions = np.asarray(trace['collision'], dtype=bool)
    any_collision = bool(initial_collision or collisions.any())
    first = -1 if initial_collision else (int(np.flatnonzero(collisions)[0]) if collisions.any() else None)
    fraction = float(np.mean(np.asarray(trace['intervention_m'])[start:] > cfg['intervention_m'])) if stuck else 0.
    path_points = np.vstack([np.asarray(trace['eef_before'])[0], pos]) if 'eef_before' in trace else pos
    path = float(np.linalg.norm(np.diff(path_points, axis=0), axis=1).sum())
    net = float(np.linalg.norm(path_points[-1] - path_points[0]))
    near = bool(np.median(np.asarray(trace['dist_subgoal_m'])[start:]) <= cfg['near_distance_m']) if stuck else False
    return {'success': bool(success), 'timeout': bool(timeout), 'collision': any_collision,
            'collision_free': not any_collision, 'safe_success': bool(success and not any_collision),
            'stuck': stuck, 'safe_stuck': stuck and not any_collision, 'collision_stuck': stuck and any_collision,
            'first_collision_step': first, 'stuck_window_start_step': start if stuck else None,
            'collision_before_stuck': bool(stuck and first is not None and first < start),
            'collision_during_stuck': bool(stuck and first is not None and first >= start),
            'near_stuck': stuck and near, 'far_stuck': stuck and not near,
            'filter_associated_stuck': bool(stuck and fraction > cfg['intervention_fraction']),
            'stuck_intervention_fraction': fraction, 'wandering_timeout': bool(timeout and not stuck),
            'safe_wandering_timeout': bool(timeout and not stuck and not any_collision),
            'collision_wandering_timeout': bool(timeout and not stuck and any_collision),
            'tail_stationary_on_success': bool(success and stationary),
            'path_length_m': path, 'net_displacement_m': net,
            'path_to_net_ratio': path / net if net > 1e-8 else None,
            'mean_active_when_stuck': float(np.mean(trace['n_active'][start:])) if stuck else None}


def fit_scale(command, displacement, cfg):
    u = np.asarray(command, dtype=float); d = np.asarray(displacement, dtype=float)
    if len(u) == 0:
        return {'passed': False, 'reason': 'insufficient_excited_samples', 'samples': 0}
    if u.ndim != 2 or u.shape[1] != 3 or d.shape != u.shape:
        raise ValueError('Calibration requires matched N x 3 arrays')
    valid = np.isfinite(u).all(1) & np.isfinite(d).all(1) & (np.linalg.norm(u, axis=1) > .05)
    u, d = u[valid], d[valid]
    if len(u) < cfg['min_samples']:
        return {'passed': False, 'reason': 'insufficient_excited_samples', 'samples': len(u)}
    # Alternating held-out samples provide a simple fixed, reproducible diagnostic.
    train = np.arange(len(u)) % 2 == 0
    k = float(np.sum(u[train] * d[train]) / np.sum(u[train] ** 2))
    residual = d[~train] - k * u[~train]
    rel = float(np.sqrt(np.sum(residual**2) / max(np.sum(d[~train]**2), 1e-20)))
    axis_n = np.sum(np.abs(u) > .05, axis=0)
    axis_k = np.sum(u * d, axis=0) / np.maximum(np.sum(u**2, axis=0), 1e-20)
    passed = bool(cfg['min_scale_m'] <= k <= cfg['max_scale_m'] and rel <= cfg['max_relative_rmse']
                  and np.min(axis_n) >= cfg['min_axis_samples']
                  and np.max(np.abs(axis_k - k)) / max(k, 1e-20) <= cfg['max_axis_scale_relative_deviation'])
    return {'passed': passed, 'action_scale_m': k, 'samples': len(u), 'relative_rmse_held_out': rel,
            'axis_samples': axis_n.tolist(), 'axis_scales_m': axis_k.tolist(),
            'reason': 'accepted_approximation' if passed else 'scalar_dynamics_fit_failed',
            'note': 'Empirical approximation, not a robust safety certificate; rotation and contact cause tracking errors.'}
