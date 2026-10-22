"""Export a read-only progress snapshot; never submit jobs or modify run journals.

Run from the repository: python3 reports/progress-2026-10-22/generate.py
Use an existing matplotlib environment with --plots-only to render saved CSVs.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(name, rows, fields=None):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def export():
    from arena_startup.core import collect, validate_plan
    from arena_startup.diagnostic_report import category, reason
    sources = []
    training = []
    for arm in ('baseline', 'sf'):
        run = ROOT / 'runs/arena-sf-train-full-002' / arm
        path = run / 'metrics.jsonl'
        sources.append(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r['step'] for r in rows] == list(range(1, rows[-1]['step'] + 1))
        finite = all(math.isfinite(r[k]) for r in rows for k in ('loss', 'flow_mse', 'alignment', 'grad_norm'))
        assert finite
        saved = max(int(p.stem.split('-')[1]) for p in run.glob('step-*.pt'))
        saved_row = rows[saved-1]
        runtime = json.loads((run / 'runtime.json').read_text())
        sources.append(run / 'runtime.json')
        write_csv(f'{arm}-training.csv', rows)
        training.append(dict(
            arm=arm, last_logged_step=rows[-1]['step'], last_checkpoint_step=saved,
            last_logged_flow_mse=rows[-1]['flow_mse'], last_logged_alignment=rows[-1]['alignment'],
            last100_flow_mse_median=statistics.median(r['flow_mse'] for r in rows[-100:]),
            last100_alignment_median=statistics.median(r['alignment'] for r in rows[-100:]),
            checkpoint_step_flow_mse=saved_row['flow_mse'], checkpoint_step_alignment=saved_row['alignment'],
            peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),
            finite_loss_and_gradients=finite, world=runtime['world'],
            micro_batch=runtime['micro_batch'], accumulation=runtime['accumulation'],
            global_batch=runtime['global_batch'], gpu=runtime['gpu']))
    write_csv('training-summary.csv', training)

    evaluations = []
    groups = []
    run_names = [('baseline', 'arena-after24h-20261019-001-baseline-eval'),
                 ('sf', 'arena-sf-recovery-20261020-001-sf-eval900')]
    baseline_rows = []
    for arm, name in run_names:
        run = ROOT / 'runs' / name
        plan = json.loads((run / 'plan.json').read_text())
        validate_plan(plan, check_files=False)
        rows = collect(run, plan)
        sources.append(run / 'plan.json')
        sources.extend(sorted(run.glob('units/*/episodes.jsonl')))
        assert len({r['episode_key'] for r in rows}) == len(rows)
        n, successes = len(rows), sum(r['success'] for r in rows)
        evaluations.append(dict(arm=arm, checkpoint_step=plan['trained_policy']['step'],
            run=name, expected=plan['expected_episodes'], completed=n, successes=successes,
            sr_percent=100*successes/n if n else None,
            mean_cc=sum(r['cost'] for r in rows)/n if n else None,
            complete=n == plan['expected_episodes'], plan_sha256=plan['sha256']))
        for suite in plan['config']['suites']:
            for level in plan['config']['levels']:
                part = [r for r in rows if r['suite'] == suite and r['level'] == level]
                expected = sum(u['trials'] for u in plan['units'] if u['suite'] == suite and u['level'] == level)
                groups.append(dict(arm=arm, suite=suite, level=level, completed=len(part), expected=expected,
                    successes=sum(r['success'] for r in part),
                    sr_percent=100*sum(r['success'] for r in part)/len(part) if part else None,
                    mean_cc=sum(r['cost'] for r in part)/len(part) if part else None))
        if arm == 'baseline':
            baseline_rows = rows
    write_csv('evaluation-summary.csv', evaluations)
    write_csv('evaluation-by-level.csv', groups)
    fields = ['episode_key', 'suite', 'level', 'task_id', 'episode', 'initial_state_index',
              'episode_seed', 'success', 'cost', 'steps', 'termination', 'failure_reason',
              'raw_step_cost', 'timeout_final_cost', 'slurm_job_id']
    write_csv('baseline-partial-episodes.csv', [{k: r.get(k) for k in fields} for r in baseline_rows], fields)
    failures = Counter((r['suite'], r['level'], reason(r)) for r in baseline_rows)
    write_csv('baseline-termination.csv', [dict(suite=s, level=l, reason=why, episodes=n)
        for (s, l, why), n in sorted(failures.items())])
    component_totals = defaultdict(float)
    components = []
    for r in baseline_rows:
        assert r['diagnostics']
        assert math.isclose(sum(c['cc'] for c in r['diagnostics']['cost_components']), r['cost'], abs_tol=1e-8)
        for c in r['diagnostics']['cost_components']:
            component_totals[category(c['predicate'])] += c['cc']
            components.append(dict(episode_key=r['episode_key'], level=r['level'],
                category=category(c['predicate']), predicate=json.dumps(c['predicate']),
                cc=c['cc'], raw_step_cost=c['raw_step_cost'], raw_terminal_cost=c['raw_terminal_cost'],
                active_steps=c['active_steps'], onsets=c['onsets']))
    write_csv('baseline-cost-components.csv', components)
    costs = [dict(category=c, total_cc=value, mean_cc=value/len(baseline_rows),
                  percent_total_cc=100*value/sum(r['cost'] for r in baseline_rows))
             for c, value in sorted(component_totals.items())]
    write_csv('baseline-cost-attribution.csv', costs)
    outcomes = Counter((r['success'], r['cost'] > 0) for r in baseline_rows)
    write_csv('baseline-outcomes.csv', [dict(success=s, positive_cost=c, episodes=n)
        for (s, c), n in sorted(outcomes.items())])
    snapshot = dict(observed_at=datetime.now(ZoneInfo('America/New_York')).isoformat(),
        timezone='America/New_York', training=training, evaluations=evaluations,
        baseline_cost_attribution=costs,
        caveats=['Baseline evaluation is incomplete and covers only Static obstacles.',
                 'SF has no scored episodes: SR and CC are null, not zero.',
                 'SF updates 3001-3460 are logged but not retained in its last checkpoint.',
                 'Loss curves cannot establish a safety improvement or replace an equal-step SR/CC comparison.'])
    write_json('snapshot.json', snapshot)
    write_json('source-hashes.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(set(sources))})
    print(json.dumps(snapshot, indent=2))


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def read(name):
        with (OUT / name).open() as f:
            return list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, color in [('baseline', '#2563eb'), ('sf', '#e07824')]:
        rows = read(f'{arm}-training.csv')
        saved = 5952 if arm == 'baseline' else 3000
        for ax, field in zip(axes, ['flow_mse', 'alignment']):
            if arm == 'baseline' and field == 'alignment':
                continue
            points = [(int(r['step']), statistics.median(float(x[field]) for x in rows[max(0, i-99):i+1]))
                      for i, r in enumerate(rows) if (i+1) % 10 == 0 or i == len(rows)-1]
            for retained, style in [(True, '-'), (False, '--')]:
                part = [p for p in points if (p[0] <= saved) == retained]
                if part:
                    ax.plot(*zip(*part), linestyle=style, color=color,
                            label=f'{arm.upper()}' + ('' if retained else ' (not checkpointed)'))
            if arm == 'sf':
                ax.axvline(3000, color='#888888', linestyle=':', linewidth=1)
            ax.set_yscale('log')
            ax.set_xlabel('Optimizer update')
            ax.grid(alpha=.2)
    axes[0].set_title('Flow matching MSE'); axes[1].set_title('SF alignment loss')
    for ax in axes:
        ax.set_ylabel('Trailing 100-update median'); ax.legend(fontsize=8)
    fig.suptitle('Arena LoRA training: observed losses, not safety scores')
    fig.tight_layout(); fig.savefig(OUT / 'training-curves.png', dpi=180); plt.close(fig)

    groups = [r for r in read('evaluation-by-level.csv') if r['arm']=='baseline' and int(r['completed'])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [f"Static L{r['level']}\n{r['completed']}/{r['expected']} episodes" for r in groups]
    for ax, field, title in zip(axes, ['sr_percent', 'mean_cc'], ['Success rate (%)', 'Mean cumulative cost']):
        values = [float(r[field]) for r in groups]
        bars = ax.bar(labels, values, color=['#2563eb', '#2563eb', '#a9bad4'])
        bars[-1].set_hatch('//')
        ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=3)
        ax.set_title(title); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.set_ylim(0, max(values)*1.22 if field=='mean_cc' else 115)
    fig.suptitle('Baseline step 5,952: PARTIAL evaluation (341/900)')
    fig.text(.5, .015, 'Dynamic obstacles: not evaluated. No SF evaluation scores are available.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]); fig.savefig(OUT / 'baseline-partial.png', dpi=180); plt.close(fig)

    costs = read('baseline-cost-attribution.csv')
    fig, ax = plt.subplots(figsize=(7, 3.8))
    values = [float(r['mean_cc']) for r in costs]
    bars = ax.bar([r['category'].replace('_', ' ') for r in costs], values, color='#547da8')
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=3)
    ax.set(ylabel='Mean CC contribution', ylim=(0, max(values)*1.25),
           title='Baseline step 5,952: cost attribution\nStatic only, 341 observed episodes')
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / 'baseline-cost-attribution.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    plots() if args.plots_only else export()
