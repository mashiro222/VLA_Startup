"""Coverage-checked local CSV/Markdown/PNG reports. Never uploads or submits jobs."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
from statistics import mean

from .core import ROOT, collect, file_hash, save, validate_plan


def write_csv(path, rows, fields=None):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(run, allow_partial=False, output=None):
    plan = json.loads((run / 'plan.json').read_text())
    validate_plan(plan)
    rows = collect(run, plan)
    complete = len(rows) == plan['expected_episodes']
    if not complete and not allow_partial:
        raise ValueError(f"Incomplete: {len(rows)}/{plan['expected_episodes']}; use --partial for a labeled snapshot")
    if not rows:
        raise ValueError('No completed episodes to report')
    output = output or ROOT / 'reports' / plan['run_id']
    output.mkdir(parents=True, exist_ok=True)
    cfg = plan['config']
    groups, seed_groups = [], []
    for i, suite in enumerate(cfg['suites']):
        for level in cfg['levels']:
            part = [r for r in rows if r['suite'] == suite and r['level'] == level]
            expected = sum(u['trials'] for u in plan['units'] if u['suite'] == suite and u['level'] == level)
            groups.append({'suite': suite, 'level': level, 'completed': len(part), 'expected': expected,
                           'successes': sum(r['success'] for r in part),
                           'sr_percent': 100 * mean(r['success'] for r in part) if part else None,
                           'cc': mean(r['cost'] for r in part) if part else None,
                           'episodes_with_cost': sum(r['cost'] > 0 for r in part),
                           'successes_with_cost': sum(r['success'] and r['cost'] > 0 for r in part),
                           'paper_finetuned_sr_percent': cfg['reference']['sr_percent'][i][level],
                           'paper_finetuned_cc': cfg['reference']['cc'][i][level]})
            for seed in cfg['profiles'][plan['profile']]['seeds']:
                part_seed = [r for r in part if r['seed'] == seed]
                seed_groups.append({'suite': suite, 'level': level, 'seed': seed, 'completed': len(part_seed),
                                    'sr_percent': 100 * mean(r['success'] for r in part_seed) if part_seed else None,
                                    'cc': mean(r['cost'] for r in part_seed) if part_seed else None})
    save(output / 'plan.json', plan)
    write_csv(output / 'episodes.csv', [{k: v for k, v in r.items() if k != 'rng_after'} for r in rows])
    write_csv(output / 'summary.csv', groups)
    write_csv(output / 'by_seed.csv', seed_groups)
    with (output / 'episodes.jsonl.gz').open('wb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as gz:
            for row in rows:
                gz.write((json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n').encode())
    errors = []
    for path in (run / 'units').glob('*/errors.jsonl'):
        if path.stat().st_size:
            errors.append(str(path.relative_to(run)))
            destination = output / 'errors' / path.parent.name / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    for path in (run / 'workers').glob('*.json'):
        destination = output / 'provenance' / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    result = {'completed': len(rows), 'expected': plan['expected_episodes'], 'complete': complete,
              'successes': sum(r['success'] for r in rows), 'sr_percent': 100 * mean(r['success'] for r in rows),
              'cc': mean(r['cost'] for r in rows), 'groups': groups, 'error_logs': errors,
              'training_condition': plan['training_condition']}
    save(output / 'summary.json', result)
    def fmt(v): return 'N/A' if v is None else f'{v:.2f}'
    lines = [f"# {plan['run_id']}", '',
             '**Zero-shot transfer from LIBERO-finetuned pi05_libero. No Arena fine-tuning.**', '',
             f"Coverage: {len(rows)} / {plan['expected_episodes']} ({'complete selected plan' if complete else 'PARTIAL snapshot'}).",
             f"SR: {result['sr_percent']:.2f}%; mean CC: {result['cc']:.2f}.", '',
             f"Profile: {plan['profile']}. Smoke is diagnostic only. Infrastructure error logs: {len(errors)}.", '',
             '| Suite | Level | Completed / expected | Our zero-shot SR % | Our CC | Paper Arena-finetuned SR % | Paper CC |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['suite']} | {g['level']} | {g['completed']} / {g['expected']} | {fmt(g['sr_percent'])} | {fmt(g['cc'])} | {fmt(g['paper_finetuned_sr_percent'])} | {fmt(g['paper_finetuned_cc'])} |")
    lines += ['', 'Paper values are contextual references under different training conditions, not a reproduction target.',
              'CC is cumulative safety cost, not collision probability. A successful episode can have nonzero CC.',
              'Per-episode environment/policy seeds make worker ordering and retries independent; this differs from a single continuous upstream RNG stream.',
              'Initial state selection and all inference settings are in plan.json. Seeds here are evaluation seeds, not training seeds.',
              'The cumulative plot follows canonical manifest order, not wall-clock completion order.', '',
              f"Reference: {cfg['reference']['source']} (Table 7).", '']
    (output / 'summary.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for suite in cfg['suites']:
        selected = [g for g in groups if g['suite'] == suite]
        for ax, metric in zip(axes, ['sr_percent', 'cc']):
            ax.plot([g['level'] for g in selected], [np.nan if g[metric] is None else g[metric] for g in selected],
                    marker='o', label=suite.removeprefix('safety_').replace('_', ' '))
            ax.set_xticks([0, 1, 2], ['L0', 'L1', 'L2']); ax.grid(alpha=.2)
    axes[0].set_ylabel('Task success (%)'); axes[0].set_ylim(0, 100)
    axes[1].set_ylabel('Mean cumulative safety cost (lower is better)'); axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"pi05_libero zero-shot / Arena Safety / {plan['profile']} / {len(rows)}/{plan['expected_episodes']}")
    fig.tight_layout(); fig.savefig(output / 'safety_by_level.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, len(rows) + 1), 100 * np.cumsum([r['success'] for r in rows]) / np.arange(1, len(rows) + 1))
    ax.set(xlabel='Completed episodes in manifest order', ylabel='Cumulative task success (%)', ylim=(0, 100),
           title='pi05_libero zero-shot / Arena Safety' + ('' if complete else ' / PARTIAL'))
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output / 'cumulative_success.png', dpi=160); plt.close(fig)
    save(output / 'sha256.json', {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'sha256.json'})
    print(f'Local report: {output}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path); parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    build(args.run, args.partial)
