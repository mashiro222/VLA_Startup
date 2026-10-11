"""Pure-Python task selection: inspect metadata without importing a simulator."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
HORIZONS = dict(zip(SUITES, (220, 280, 300, 520)))
REFERENCE_EVALUATION = {'num_steps': 10, 'replan_steps': 5, 'wait_steps': 10,
                        'render_resolution': 256, 'policy_resolution': 224,
                        'horizons': HORIZONS}
CATEGORIES = ('Camera Viewpoints', 'Robot Initial States', 'Language Instructions',
              'Light Conditions', 'Background Textures', 'Sensor Noise', 'Objects Layout')
ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def task_map(repo):
    tree = ast.parse((repo / 'libero/libero/benchmark/libero_suite_task_map.py').read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in n.targets))
    return ast.literal_eval(assignment.value)


def physical_bddl(name):
    return name.split('_view_')[0] + '.bddl'


def init_file(mode, suite, name):
    if mode == 'clean':
        return f'{suite}/{name}.pruned_init'
    # Match the pinned upstream Benchmark.get_task_init_states rules exactly.
    if '_language_' in name:
        stem = name.split('_language_')[0]
    elif '_view_' in name:
        stem = name.split('_view_')[0]
    elif '_add_' in name or '_level' in name:
        return f'libero_newobj/{suite}/{name}.pruned_init'
    elif '_light_' in name:
        stem = name.split('_light_')[0]
    elif '_table_' in name or '_tb_' in name:
        # Base task text may itself contain "from_table_center"; only strip
        # numeric texture suffixes. Upstream uses two independent if statements.
        stem = re.sub(r'_(?:table|tb)_\d+', '', name)
    else:
        raise ValueError(f'Unrecognized upstream initial-state rule: {name}')
    return f'{suite}/{stem}.pruned_init'


def build(root, profile, seed=7, prompt_mode='upstream'):
    evaluation = json.loads((root / 'config/evaluation.json').read_text())
    for key in ('num_steps', 'replan_steps', 'render_resolution', 'policy_resolution'):
        if type(evaluation[key]) is not int or evaluation[key] <= 0:
            raise ValueError(f'Invalid positive integer setting: {key}')
    if not 1 <= evaluation['replan_steps'] <= 10:
        raise ValueError('replan_steps must fit the checkpoint action horizon (10)')
    if type(evaluation['wait_steps']) is not int or evaluation['wait_steps'] < 0:
        raise ValueError('wait_steps must be a nonnegative integer')
    if any(type(evaluation['horizons'][s]) is not int or evaluation['horizons'][s] <= 0 for s in SUITES):
        raise ValueError('Each horizon must be a positive integer')
    mode = 'clean' if profile.endswith('clean') else 'plus'
    repo = root / '.runtime/repos' / ('libero_' + mode)
    mapping = task_map(repo)
    base_mapping = task_map(root / '.runtime/repos/libero_clean')
    classifications = {}
    if mode == 'plus':
        data = json.loads((repo / 'libero/libero/benchmark/task_classification.json').read_text())
        for suite in SUITES:
            classifications[suite] = {r['name']: r for r in data[suite]}
            if set(mapping[suite]) != set(classifications[suite]):
                raise ValueError(f'Metadata mismatch: {suite}')
    rows = []
    for suite in SUITES:
        for task_id, name in enumerate(mapping[suite]):
            metadata = classifications.get(suite, {}).get(name, {})
            bases = [b for b in base_mapping[suite] if name == b or name.startswith(b + '_')]
            if not bases:
                raise ValueError(f'Cannot identify base task: {name}')
            row = {'suite': suite, 'task_id': task_id, 'name': name,
                   'base_task': max(bases, key=len), 'category': metadata.get('category', 'Clean'),
                   'difficulty': metadata.get('difficulty_level'),
                   'classification_id': metadata.get('id'),
                   'bddl': f'{suite}/{physical_bddl(name)}',
                   'init_file': init_file(mode, suite, name), 'max_steps': evaluation['horizons'][suite]}
            rows.append(row)
    if profile.startswith('smoke'):
        if mode == 'clean':
            rows = [next(r for r in rows if r['suite'] == suite) for suite in SUITES]
        else:
            selected, used = [], set()
            rng = random.Random(seed)
            for category in CATEGORIES:
                candidates = [r for r in rows if r['suite'] == 'libero_spatial' and r['category'] == category]
                rng.shuffle(candidates)
                choice = next((r for r in candidates if r['base_task'] not in used), candidates[0])
                used.add(choice['base_task'])
                selected.append(choice)
            rows = selected
    elif profile == 'pilot-plus':
        rng = random.Random(seed)
        selected = []
        for suite in SUITES:
            for category in CATEGORIES:
                candidates = [r for r in rows if r['suite'] == suite and r['category'] == category]
                rng.shuffle(candidates)
                seen = set()
                for row in candidates:
                    if row['base_task'] not in seen:
                        selected.append(row)
                        seen.add(row['base_task'])
                    if len(seen) == 2:
                        break
        rows = selected
    rows.sort(key=lambda r: (SUITES.index(r['suite']), r['task_id']))
    for row in rows:
        for key, subdir in (('bddl', 'bddl_files'), ('init_file', 'init_files')):
            if not (repo / 'libero/libero' / subdir / row[key]).is_file():
                raise FileNotFoundError(f'{key}: {row[key]}')
    trials = 50 if profile == 'full-clean' else 1
    sources = json.loads((root / 'config/sources.json').read_text())
    result = {'schema_version': 1, 'profile': profile, 'benchmark': mode, 'seed': seed,
              'prompt_mode': prompt_mode, 'evaluation': evaluation, 'trials_per_task': trials,
              'expected_episodes': len(rows) * trials, 'tasks': rows,
              'sources': sources, 'selection': 'all' if profile.startswith('full') else 'diagnostic subset'}
    result['manifest_sha256'] = digest(result)
    return result


def validate(manifest):
    unsigned = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if digest(unsigned) != manifest['manifest_sha256']:
        raise ValueError('Manifest checksum mismatch')
    keys = [(r['suite'], r['task_id']) for r in manifest['tasks']]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate tasks')
    if len(keys) * manifest['trials_per_task'] != manifest['expected_episodes']:
        raise ValueError('Invalid expected episode count')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, choices=['smoke-clean', 'smoke-plus', 'pilot-plus', 'full-clean', 'full-plus'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--prompt-mode', choices=['upstream', 'bddl'], default='upstream')
    args = parser.parse_args()
    manifest = build(ROOT, args.profile, args.seed, args.prompt_mode)
    validate(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'profile': args.profile, 'tasks': len(manifest['tasks']),
                      'episodes': manifest['expected_episodes'],
                      'categories': dict(Counter(r['category'] for r in manifest['tasks'])),
                      'manifest_sha256': manifest['manifest_sha256']}, indent=2))


if __name__ == '__main__':
    main()
