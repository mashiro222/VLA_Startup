"""A hard-constraint, translation-only discrete CBF QP; no safety slack."""
import time
import numpy as np
from scipy import sparse
import osqp
from .geometry import barrier


def filter_action(action, p, centers, matrices, selected, cfg, enabled=True, rotation=None):
    started = time.perf_counter()
    nominal = np.asarray(action, dtype=float)
    if nominal.shape != (7,) or not np.isfinite(nominal).all():
        raise ValueError('Expected finite seven-dimensional OSC_POSE action')
    h, grad = barrier(p, centers, matrices)
    selected = np.asarray(selected, dtype=int)
    out = nominal.copy(); status = 'bypass'; active = 0; failure = False
    dual = np.zeros(len(selected)); slack = np.zeros(len(selected))
    if enabled and len(selected):
        rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        g = grad[selected] @ rotation * cfg['action_scale_m']; lower = -cfg['gamma'] * h[selected]
        if np.max(np.abs(nominal[:3])) <= 1 and np.all(g @ nominal[:3] >= lower):
            status = 'nominal_feasible'  # Exact minimizer: objective value is zero.
        else:
            solver = osqp.OSQP()
            solver.setup(P=sparse.eye(3, format='csc') * 2, q=-2 * nominal[:3],
                         A=sparse.csc_matrix(np.vstack([g, np.eye(3)])),
                         l=np.r_[lower, [-1.] * 3], u=np.r_[[np.inf] * len(selected), [1.] * 3],
                         eps_abs=cfg['qp_tolerance'], eps_rel=cfg['qp_tolerance'],
                         max_iter=10000, polishing=False, verbose=False)
            result = solver.solve(raise_error=False); status = result.info.status
            candidate = np.clip(result.x, -1, 1) if result.x is not None else None
            feasible = (result.x is not None and np.isfinite(result.x).all()
                        and np.max(np.abs(result.x)) <= 1 + cfg['active_tolerance']
                        and np.all(g @ candidate >= lower - cfg['active_tolerance']))
            if result.info.status_val in (1, 2) and feasible:
                out[:3] = candidate
                dual = result.y[:len(selected)]
            else:
                out[:3] = 0.; failure = True
        slack = g @ out[:3] - lower
        active = int(np.count_nonzero((np.abs(slack) <= cfg['active_tolerance']) | (np.abs(dual) > cfg['active_tolerance'])))
    # C0 and empty-obstacle C2 are bit-identical to the nominal policy command.
    intervention = float(np.linalg.norm(out[:3] - nominal[:3]))
    return out, {'h': h, 'n_active': active, 'dual': dual, 'slack': slack,
                 'status': status, 'solver_failed': failure,
                 'unsafe_at_start': bool(np.any(h[selected] < 0)),
                 'fallback_satisfies_linear_cbf': bool(not failure or np.all(slack >= -cfg['active_tolerance'])),
                 'intervention_action': intervention,
                 'intervention_m': cfg['action_scale_m'] * intervention,
                 'filter_seconds': time.perf_counter() - started}
