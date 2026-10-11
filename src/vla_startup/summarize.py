"""Summarize completed episodes and explicitly show missing coverage."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from .journal import read_records, validate_prefix
from .manifest import ROOT, REFERENCE_EVALUATION, validate


def statistics(records, expected):
    successes = sum(r['success'] for r in records)
    return {'expected': expected, 'completed': len(records), 'missing': expected - len(records),
            'successes': successes, 'failures': len(records) - successes,
            'success_percent_completed': 100 * successes / len(records) if records else None,
            'success_percent_full': 100 * successes / expected if len(records) == expected and expected else None}


def summarize(manifest, records, references):
    validate(manifest)
    validate_prefix(records, manifest)
    totals = statistics(records, manifest['expected_episodes'])
    groups = {}
    for field in ('suite', 'category', 'difficulty'):
        expected = defaultdict(int)
        completed = defaultdict(list)
        for task in manifest['tasks']:
            expected[str(task[field])] += manifest['trials_per_task']
        for row in records:
            completed[str(row[field])].append(row)
        groups[field] = {key: statistics(completed[key], count) for key, count in expected.items()}
    mode = manifest['benchmark']
    reference = references[mode]
    complete = totals['missing'] == 0
    comparable = (complete and manifest['profile'] == 'full-' + mode
                  and manifest['prompt_mode'] == 'upstream' and manifest.get('seed') == 7
                  and manifest.get('evaluation') == REFERENCE_EVALUATION)
    comparisons = []
    dimension = 'suite' if mode == 'clean' else 'category'
    if comparable:
        for key, value in reference[dimension + '_percent'].items():
            measured = groups[dimension][key]['success_percent_full']
            comparisons.append({'group': key, 'ours_percent': measured, 'reference_percent': value,
                                'difference_percentage_points': measured - value})
        comparisons.append({'group': 'Overall (micro)', 'ours_percent': totals['success_percent_full'],
                            'reference_percent': reference['overall_percent'],
                            'difference_percentage_points': totals['success_percent_full'] - reference['overall_percent']})
    macro = None
    if complete and mode == 'plus' and len(groups['category']) == 7:
        macro = sum(x['success_percent_full'] for x in groups['category'].values()) / 7
    return {'schema_version': 1, 'profile': manifest['profile'], 'benchmark': mode,
            'manifest_sha256': manifest['manifest_sha256'], 'complete': complete,
            'totals': totals, 'groups': groups, 'category_macro_percent': macro,
            'reference': reference, 'reference_comparison_enabled': comparable, 'comparisons': comparisons,
            'interpretation': ('Full coverage; differences are descriptive, not a pass/fail replication test. '
                               'LIBERO-Plus community protocol is incomplete.' if comparable else
                               'Partial coverage, diagnostic subset, or changed protocol: do not claim a benchmark reproduction.')}


def markdown(result):
    lines = [f"# {result['profile']}", '', result['interpretation'], '',
             f"Completed: {result['totals']['completed']} / {result['totals']['expected']}",
             f"Successes: {result['totals']['successes']}", '',
             '| Group | Completed / expected | Successes | Success rate (completed) |',
             '|---|---:|---:|---:|']
    for group in ('suite', 'category'):
        for name, data in result['groups'][group].items():
            rate = data['success_percent_completed']
            formatted = f'{rate:.2f}%' if rate is not None else 'N/A'
            lines.append(f"| {name} | {data['completed']} / {data['expected']} | {data['successes']} | {formatted} |")
    if result['comparisons']:
        lines += ['', '| Group | Ours | Reference | Difference (pp) |', '|---|---:|---:|---:|']
        for row in result['comparisons']:
            lines.append(f"| {row['group']} | {row['ours_percent']:.2f}% | {row['reference_percent']:.2f}% | {row['difference_percentage_points']:+.2f} |")
    lines += ['', f"Reference: {result['reference'].get('source', 'Unspecified')}", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    records = read_records(args.run_dir / 'episodes.jsonl')
    result = summarize(manifest, records, json.loads((ROOT / 'config/references.json').read_text()))
    (args.run_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.run_dir / 'summary.md').write_text(markdown(result))
    print(json.dumps(result['totals'], indent=2))


if __name__ == '__main__':
    main()
