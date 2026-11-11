"""Local CSV/PNG/Markdown reports. Never interprets infrastructure errors as SR=0."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .common import ROOT, record_path, save, digest
from .metrics import classify, fit_scale

RATES = ['success', 'collision_free', 'safe_success', 'stuck', 'safe_stuck', 'collision_stuck',
         'collision_before_stuck', 'collision_during_stuck', 'near_stuck', 'far_stuck',
         'wandering_timeout', 'safe_wandering_timeout', 'collision_wandering_timeout', 'filter_associated_stuck']


def wilson(k, n):
    if n == 0: return None, None
    z = 1.95996398454; p = k/n; denominator = 1 + z*z/n
    mid = (p + z*z/(2*n))/denominator
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return min(p, max(0., mid-half)), max(p, min(1., mid+half))


def csv_write(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def collect(run):
    plan = json.loads((run / 'plan.json').read_text()); rows = []
    for unit in plan['units']:
        for ep in unit['episodes']:
            p = record_path(run, ep) / 'episode.json'
            if p.exists(): rows.append(dict(json.loads(p.read_text()), artifact_dir=str(p.parent.resolve())))
    return plan, rows


def calibrate(run, plan, rows):
    if plan['profile'] != 'calibration' or len(rows) != plan['expected_episodes']:
        raise ValueError('Calibration requires every planned L0 C0 episode to finish')
    commands, deltas = [], []
    for row in rows:
        with np.load(Path(row['artifact_dir']) / 'trace.npz') as data:
            a = data['action_executed']; mask = data['free_motion'] & ~data['held'] & (np.linalg.norm(a[:,3:6], axis=1) < .3)
            commands.extend(np.clip(a[mask, :3], -1, 1)); deltas.extend(data['actual_delta_controller'][mask])
    result = fit_scale(commands, deltas, plan['config']['calibration'])
    result.update(checkpoint_revision=json.loads((ROOT / 'config/cbf_static/checkpoint.json').read_text())['revision'],
        source_run=str(run.resolve()), source_plan_sha256=digest(run / 'plan.json'),
        selection='No robot/environment contact; not grasping; rotation norm < 0.3; translation norm > 0.05',
        controller_nominal_scale_m=.05)
    save(run / 'calibration.json', result)
    print(json.dumps(result, indent=2)); return result


def report(run, do_calibrate=False, extra=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = Path(run); plan, rows = collect(run); expected = plan['expected_episodes']
    for other in extra:
        extra_plan, more = collect(Path(other))
        if extra_plan['config'] != plan['config']: raise ValueError('Cannot merge differing policy/filter/diagnostic configurations')
        rows.extend(more); expected += extra_plan['expected_episodes']
    keys = [(r['level'], r['task_id'], r['condition'], r['gamma'], r['init_index'], r['repeat']) for r in rows]
    if len(keys) != len(set(keys)): raise ValueError('Duplicate paired episodes across input runs')
    out = run / 'summary'; out.mkdir(exist_ok=True)
    if do_calibrate: calibrate(run, plan, rows)
    if not rows:
        (out / 'README.md').write_text(f'# CBF experiment\n\n0/{expected} completed episodes. No performance scores available.\n')
        return
    csv_write(out / 'episodes.csv', rows)
    groups = defaultdict(list)
    for r in rows: groups[(r['level'], r['task_id'], r['condition'], r['gamma'])].append(r)
    task_rows = []
    for (level, task, condition, gamma), items in sorted(groups.items()):
        row = dict(level=level, task_id=task, condition=condition, gamma=gamma, observed=len(items))
        for metric in RATES:
            k = sum(x[metric] for x in items); row[metric] = k / len(items)
            row[metric + '_ci_low'], row[metric + '_ci_high'] = wilson(k, len(items))
        for metric in ['CC', 'mean_intervention_m', 'intervention_frequency', 'wall_seconds', 'filter_seconds', 'inference_seconds']:
            row[metric] = float(np.mean([x[metric] for x in items]))
        row['solver_failures'] = sum(x['solver_failures'] for x in items)
        row['mean_active_when_stuck'] = float(np.mean([x['mean_active_when_stuck'] for x in items if x['stuck']])) if any(x['stuck'] for x in items) else None
        task_rows.append(row)
    csv_write(out / 'by-task.csv', task_rows)
    macro = []
    for key in sorted({(x['level'], x['condition'], x['gamma']) for x in task_rows}):
        items = [r for r in task_rows if (r['level'], r['condition'], r['gamma']) == key]
        macro.append(dict(level=key[0], condition=key[1], gamma=key[2], tasks_observed=len(items),
            episodes_observed=sum(x['observed'] for x in items),
            **{m: float(np.mean([x[m] for x in items])) for m in RATES + ['CC', 'mean_intervention_m', 'intervention_frequency']}))
    csv_write(out / 'macro-average.csv', macro)
    reference = {1:[90.,83.,77.,72.,85.], 2:[8.,0.,89.3,81.3,74.]}
    comparisons = []
    for row in task_rows:
        if row['condition'] != 'C0' or row['level'] not in reference: continue
        published = reference[row['level']][row['task_id']]
        gap = 100*row['success'] - published
        comparisons.append(dict(level=row['level'], task_id=row['task_id'], observed=row['observed'],
            SR_percent=100*row['success'], FailBank_reference_SR_percent=published, gap_percentage_points=gap,
            investigate=abs(gap)>15, protocol='different initialization/sampling/version protocol; advisory comparison only'))
    csv_write(out / 'reference-comparison.csv', comparisons)
    # Pairing is by task/init/sampling seed, with C0 reused across gamma values.
    controls = {(r['level'], r['task_id'], r['init_index'], r['repeat']): r for r in rows if r['condition'] == 'C0'}
    paired = []
    for row in rows:
        if row['condition'] == 'C0': continue
        c0 = controls.get((row['level'], row['task_id'], row['init_index'], row['repeat']))
        if c0 is None: continue
        paired.append({k: row[k] for k in ('level', 'task_id', 'condition', 'gamma', 'init_index', 'repeat')} | {
            'C0_success': c0['success'], 'C0_collision': c0['collision'],
            'paired_success_to_stuck': c0['success'] and row['stuck'],
            'collision_to_safe_success': c0['collision'] and row['safe_success'],
            'both_fail_unresolved': not c0['success'] and not row['success'],
            'filtered_success': row['success'], 'filtered_collision': row['collision']})
    csv_write(out / 'paired-episodes.csv', paired)
    paired_summary = []
    for key in sorted({(x['level'], x['condition'], x['gamma']) for x in paired}):
        items = [x for x in paired if (x['level'], x['condition'], x['gamma']) == key]
        n_success = sum(x['C0_success'] for x in items); n_collision = sum(x['C0_collision'] for x in items)
        k = sum(x['paired_success_to_stuck'] for x in items); rescued = sum(x['collision_to_safe_success'] for x in items)
        paired_summary.append(dict(level=key[0], condition=key[1], gamma=key[2], complete_pairs=len(items),
            C0_success_pairs=n_success, success_to_stuck_count=k, success_to_stuck_rate=k/len(items),
            success_to_stuck_conditional=k/n_success if n_success else None,
            C0_collision_pairs=n_collision, rescued_count=rescued,
            rescued_conditional=rescued/n_collision if n_collision else None))
    csv_write(out / 'paired-summary.csv', paired_summary)
    fig, ax = plt.subplots(figsize=(7, 4))
    for condition in ('C1', 'C2'):
        items = [x for x in paired_summary if x['condition'] == condition and x['gamma'] == plan['config']['filter']['gamma']]
        if not items: continue
        x = [r['level'] for r in items]; y = [r['success_to_stuck_rate'] for r in items]
        bounds = [wilson(r['success_to_stuck_count'], r['complete_pairs']) for r in items]
        err = [[rate-lo, hi-rate] for rate, (lo, hi) in zip(y, bounds)]
        ax.errorbar(x, y, yerr=np.array(err).T, marker='o', capsize=4, label=condition)
    ax.set(xlabel='Obstacle count / level', ylabel='C0 success AND filtered stuck / complete pairs',
           title='Paired filter-associated stalling (descriptive Wilson 95% CI)', xticks=[0,1,2], ylim=(-.02,1.02))
    if paired_summary: ax.legend()
    fig.tight_layout(); fig.savefig(out/'paired-induced-stuck.png',dpi=180);plt.close(fig)
    sensitivity = []; body_rows = []; checks = []
    for row in rows:
        trace_path = Path(row['artifact_dir']) / 'trace.npz'
        with np.load(trace_path) as data:
            for w in (2, 3, 5):
                for radius in (.02, .03, .05):
                    cfg = dict(plan['config']['diagnostics'], window_seconds=w, radius_m=radius)
                    labels = classify(data, row['success'], row['timeout'], cfg, plan['config']['evaluation']['hz'], row['initial_collision'])
                    sensitivity.append(dict(level=row['level'], task_id=row['task_id'], condition=row['condition'], gamma=row['gamma'], init_index=row['init_index'], repeat=row['repeat'], window_seconds=w, radius_m=radius,
                        **{k: labels[k] for k in ['stuck', 'safe_stuck', 'collision_stuck', 'tail_stationary_on_success']}))
            for label in ('gripper_collision', 'robot_link_collision', 'held_object_collision', 'target_object_collision'):
                body_rows.append(dict(level=row['level'], condition=row['condition'], gamma=row['gamma'], component=label, present=bool(data[label].any()), control_steps=int(data[label].sum())))
        if row['level'] in (0,1) and row['condition'] == 'C2':
            c0 = (controls.get((0, row['task_id'], row['init_index'], row['repeat'])) if row['level']==0 else
                  next((x for x in rows if x['level']==1 and x['task_id']==row['task_id'] and x['condition']=='C1' and x['gamma']==row['gamma'] and x['init_index']==row['init_index'] and x['repeat']==row['repeat']),None))
            if c0:
                with np.load(Path(c0['artifact_dir']) / 'trace.npz') as a, np.load(trace_path) as b:
                    checks.append({'level':row['level'], 'comparison':'C0/C2' if row['level']==0 else 'C1/C2',
                        'task': row['task_id'], 'init': row['init_index'], 'seed': row['repeat'],
                        'identical_actions': a['action_executed'].shape == b['action_executed'].shape and bool(np.allclose(a['action_executed'], b['action_executed'], rtol=0, atol=1e-6)),
                        'identical_positions': a['eef_after'].shape == b['eef_after'].shape and bool(np.allclose(a['eef_after'], b['eef_after'], rtol=0, atol=1e-6))})
    csv_write(out / 'sensitivity.csv', sensitivity); csv_write(out / 'contact-components.csv', body_rows); save(out / 'sanity-checks.json', checks)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, ['success', 'collision_free', 'safe_stuck'], ['Success rate', 'Collision avoidance rate', 'Collision-free stuck rate']):
        for condition in ('C0', 'C1', 'C2'):
            x, y, errors = [], [], []
            for level in (0, 1, 2):
                selected = [r for r in rows if r['level'] == level and r['condition'] == condition and r['gamma'] == .3]
                if not selected: continue
                n = len(selected); k = sum(r[metric] for r in selected); lo, hi = wilson(k, n)
                x.append(level); y.append(k/n); errors.append([k/n-lo, hi-k/n])
            if x: ax.errorbar(x, y, yerr=np.array(errors).T, label=condition, marker='o', capsize=3)
        ax.set(title=title, xlabel='Obstacle count / level', ylim=(-.02, 1.02), xticks=[0, 1, 2]); ax.legend()
    fig.suptitle(f'Observed episodes: {len(rows)}/{expected}; descriptive Wilson 95% intervals')
    fig.tight_layout(); fig.savefig(out / 'rates.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4)); component_names = ['gripper_collision', 'robot_link_collision', 'held_object_collision']
    labels = sorted({(x['level'], x['condition'], x['gamma']) for x in body_rows}); positions = np.arange(len(labels))
    for j, comp in enumerate(component_names):
        means = [np.mean([x['present'] for x in body_rows if (x['level'], x['condition'], x['gamma']) == key and x['component'] == comp]) for key in labels]
        ax.bar(positions + (j-1)*.25, means, width=.25, label=comp)
    ax.set_xticks(positions, [f'L{k[0]} {k[1]} g{k[2]}' for k in labels], rotation=45, ha='right'); ax.set_ylabel('Episodes with contact / observed episodes')
    ax.set_title('Contact categories overlap; not a partition of CC'); ax.legend(); fig.tight_layout(); fig.savefig(out/'contact-components.png',dpi=180); plt.close(fig)
    costs=[]
    for row in rows:
        for c in row['official_diagnostics']['cost_components']:
            costs.append(dict(level=row['level'],task_id=row['task_id'],condition=row['condition'],gamma=row['gamma'],
                init_index=row['init_index'],repeat=row['repeat'],predicate=c['predicate'],cc=c['cc']))
    csv_write(out/'cost-predicates.csv',costs)
    fig,ax=plt.subplots(figsize=(9,4));bottom=np.zeros(len(labels))
    for predicate in sorted({c['predicate'][0] for c in costs}):
        values=[]
        for key in labels:
            n=sum((r['level'],r['condition'],r['gamma'])==key for r in rows)
            values.append(sum(c['cc'] for c in costs if (c['level'],c['condition'],c['gamma'])==key and c['predicate'][0]==predicate)/n)
        ax.bar(positions,values,bottom=bottom,label=predicate);bottom+=values
    ax.set_xticks(positions,[f'L{k[0]} {k[1]} g{k[2]}' for k in labels],rotation=45,ha='right');ax.set_ylabel('Mean official CC contribution')
    if costs:ax.legend()
    fig.tight_layout();fig.savefig(out/'cost-predicates.png',dpi=180);plt.close(fig)
    example_rows = sorted(rows, key=lambda x: (not x['stuck'], x['level'], x['task_id']))[:6]
    for i, row in enumerate(example_rows):
        with np.load(Path(row['artifact_dir']) / 'trace.npz') as d:
            t = np.arange(row['n_steps']) / plan['config']['evaluation']['hz']; fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
            axes[0].plot(t, d['dist_subgoal_m']); axes[0].set_ylabel('Subgoal distance (m)')
            axes[1].plot(t, d['intervention_m']); axes[1].set_ylabel('Correction (m)')
            axes[2].plot(t, d['n_active']); axes[2].set_ylabel('Active constraints')
            axes[3].plot(t, np.linalg.norm(d['action_nominal'][:,:3],axis=1),label='nominal'); axes[3].plot(t, np.linalg.norm(d['action_executed'][:,:3],axis=1),label='executed'); axes[3].legend(); axes[3].set(xlabel='Simulated time (s)',ylabel='Translation command norm')
            if row['stuck']:
                for ax in axes: ax.axvline(row['stuck_window_start_step']/plan['config']['evaluation']['hz'], ls='--',color='red')
            fig.suptitle(f"L{row['level']} {row['name']} {row['condition']} init {row['init_index']}; success={row['success']} stuck={row['stuck']}")
            fig.tight_layout(); fig.savefig(out/f'process-{i:02d}.png',dpi=160); plt.close(fig)
    csv_write(out / 'process-examples.csv', [{'plot':f'process-{i:02d}.png','artifact_dir':r['artifact_dir']} for i,r in enumerate(example_rows)])
    md = [f'# Static Obstacles CBF results', '', f'Completed: **{len(rows)}/{expected}**. ' + ('COMPLETE.' if len(rows)==expected else '**PARTIAL; missing episodes are not failures.**'), '',
        '[Episode table](episodes.csv) · [Per-task table](by-task.csv) · [Task macro averages](macro-average.csv) · [Paired analysis](paired-summary.csv) · [Threshold sensitivity](sensitivity.csv)', '',
        '![Paired stalling](paired-induced-stuck.png)', '', '![Rates](rates.png)', '', '![Contact components](contact-components.png)', '', '![Official CC](cost-predicates.png)', '',
        'Wilson intervals are descriptive episode-level intervals. Two seeds reuse each initial state, so these are not independent-scene confidence bounds. Macro averages and pooled observed rates can differ for incomplete runs.', '',
        'Paired success→stuck is an association with inserting the filter; it is not proof of a particular dynamical cause. Both policies failing does not identify policy failure as the cause. The gripper sphere does not protect arm links or carried objects.', '',
        '| Level | Condition | Gamma | Observed | SR | CAR | Safe stuck | CC |', '|---|---|---|---|---|---|---|---|']
    for x in macro: md.append(f"| L{x['level']} | {x['condition']} | {x['gamma']} | {x['episodes_observed']} | {x['success']:.1%} | {x['collision_free']:.1%} | {x['safe_stuck']:.1%} | {x['CC']:.2f} |")
    failures = sum(r['solver_failures'] for r in rows); md += ['',f'QP fallback steps: {failures}. Inspect contacts.jsonl.gz for statuses and infeasibility. Initial collisions: {sum(r["initial_collision"] for r in rows)} episodes.', '',
        'Successful C0 trajectories are also checked for stationary tail windows in sensitivity.csv; the timeout prerequisite is removed for this false-positive diagnostic. Thresholds are fixed before inspecting filtered results.', '',
        '[Process figure index and video directories](process-examples.csv). Full per-step arrays are trace.npz; contact bodies, solver status, and official predicate costs are in contacts.jsonl.gz.']
    (out / 'README.md').write_text('\n'.join(md)+'\n'); print(out.resolve())


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('run',type=Path); p.add_argument('--calibrate',action='store_true'); a=p.parse_args(); report(a.run,a.calibrate)
