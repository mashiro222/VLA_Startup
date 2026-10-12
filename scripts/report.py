#!/usr/bin/env python3
"""Create small, auditable tables and PNG plots from real evaluation records."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vla_startup.journal import read_records, validate_prefix
from vla_startup.summarize import markdown, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        raise ValueError('Invalid run ID')
    run = ROOT / 'runs' / args.run_id
    manifest = json.loads((run / 'manifest.json').read_text())
    records = read_records(run / 'episodes.jsonl')
    validate_prefix(records, manifest)
    result = summarize(manifest, records, json.loads((ROOT / 'config/references.json').read_text()))
    if not records:
        raise ValueError('No completed episodes; refusing to generate empty or synthetic result plots')
    output = ROOT / 'reports' / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json', 'run_config.json', 'errors.jsonl', 'parallel_execution.json'):
        if (run / name).exists():
            shutil.copy2(run / name, output / name)
    for path in run.glob('provenance-*.json'):
        shutil.copy2(path, output / path.name)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    text = markdown(result)
    if 'parallel_aggregate' in manifest:
        text += '\nParallel execution: ' + manifest['parallel_aggregate']['rng_note'] + '\n'
    (output / 'summary.md').write_text(text)
    fields = ['episode_key', 'suite', 'task_id', 'name', 'base_task', 'category', 'difficulty',
              'episode', 'success', 'steps', 'elapsed_seconds', 'inference_calls', 'inference_seconds', 'prompt']
    with (output / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    with (output / 'summary.csv').open('w', newline='') as stream:
        fields = ['group_type', 'group', 'expected', 'completed', 'missing', 'successes',
                  'failures', 'success_percent_completed', 'success_percent_full']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({'group_type': 'total', 'group': 'all', **result['totals']})
        for group_type, groups in result['groups'].items():
            for name, values in groups.items():
                writer.writerow({'group_type': group_type, 'group': name, **values})
    # Keep RNG states in a compressed journal so resumption and auditing remain possible.
    with (output / 'episodes.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            # Use exactly the same completed-record snapshot as tables/plots.
            # A running job may append more episodes while this report is made.
            for record in records:
                zipped.write((json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n').encode())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    completed, cumulative = [], []
    successes = 0
    for index, row in enumerate(records, 1):
        successes += row['success']
        completed.append(index)
        cumulative.append(100 * successes / index)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    ax.plot(completed, cumulative, color='#215bb0', linewidth=1.5)
    ax.set(xlabel='Completed evaluation episodes', ylabel='Cumulative success (%)', ylim=(0, 100),
           title=f"{manifest['profile']} — {len(records)}/{manifest['expected_episodes']} episodes")
    ax.grid(alpha=0.2)
    if result['reference_comparison_enabled']:
        ax.axhline(result['reference']['overall_percent'], color='#c36b23', linestyle='--', label='Published reference (protocol caveats apply)')
        ax.legend(fontsize=8)
    fig.savefig(output / 'cumulative_success.png', dpi=160)
    plt.close(fig)
    dimension = 'category' if manifest['benchmark'] == 'plus' else 'suite'
    groups = result['groups'][dimension]
    names = list(groups)
    rates = [groups[name]['success_percent_completed'] or 0 for name in names]
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    ax.barh(names, rates, color='#215bb0')
    for i, name in enumerate(names):
        data = groups[name]
        label = f"{data['successes']}/{data['completed']}" if data['completed'] else 'N/A (not evaluated)'
        ax.text(min(rates[i] + 1, 88), i, label, va='center', fontsize=9)
    ax.set(xlim=(0, 105), xlabel='Success among completed episodes (%)',
           title=f"{manifest['profile']} — {'complete selected plan' if result['complete'] else 'PARTIAL coverage'}")
    fig.savefig(output / 'success_by_group.png', dpi=160)
    plt.close(fig)
    hashes = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != 'sha256.json':
            if path.stat().st_size > 50 * 1024 ** 2:
                raise ValueError(f'Report file exceeds the GitHub backup size budget: {path.name}')
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print('Tables, plots and compressed raw journal:', output)


if __name__ == '__main__':
    main()
