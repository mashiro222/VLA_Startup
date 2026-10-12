import copy
import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

from vla_startup.journal import episode_keys
from vla_startup.manifest import digest

spec = importlib.util.spec_from_file_location('parallel', Path(__file__).resolve().parents[1] / 'scripts/parallel.py')
parallel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel)


def fixture():
    return parallel.signed({
        'profile': 'full-clean', 'benchmark': 'clean', 'trials_per_task': 3,
        'expected_episodes': 24, 'prompt_mode': 'upstream', 'seed': 7,
        'tasks': [{'suite': 'suite', 'task_id': i, 'name': f'task-{i}',
                   'max_steps': horizon, 'category': 'Clean', 'difficulty': None}
                  for i, horizon in enumerate([220, 220, 280, 280, 300, 300, 520, 520])]})


def completed_shards(parent, shards):
    output = []
    for i, shard in enumerate(shards):
        config = {'manifest_sha256': shard['manifest_sha256'], 'code_sha256': {'eval': 'abc'},
                  'server': {'initial_rng': [0, 0]}, 'seed': 7}
        records = [{**task, 'episode': e, 'episode_key': f"suite:{task['task_id']}:{e}",
                    'success': (task['task_id'] + e) % 2 == 0,
                    'manifest_sha256': shard['manifest_sha256'], 'run_fingerprint': digest(config),
                    'rng_after': [i, e]}
                   for task in shard['tasks'] for e in range(shard['trials_per_task'])]
        output.append((f'child-{i}', shard, records, config))
    return output


class ParallelTests(unittest.TestCase):
    def test_plus_only_plan_excludes_clean_and_round_trips(self):
        base = fixture()
        base = parallel.signed({**base, 'profile': 'full-plus', 'benchmark': 'plus'})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(parallel, 'ROOT', Path(directory)), \
                    patch.object(parallel, 'frozen_inputs', return_value={'code': 'abc'}), \
                    patch.object(parallel, 'build', return_value=base) as build_mock:
                parallel.prepare('plus-only', 4, 'plus')
                plan = parallel.load_plan('plus-only', check_inputs=True)
                self.assertEqual(set(plan['benchmarks']), {'plus'})
                build_mock.assert_called_once_with(Path(directory), 'full-plus')
                self.assertFalse((Path(directory) / 'runs/plus-only-full-clean').exists())

    def test_balanced_partition_is_disjoint_complete_and_keeps_whole_tasks(self):
        base = fixture()
        parent, shards = parallel.split_manifest(base, 4)
        self.assertEqual((parent, shards), parallel.split_manifest(base, 4))
        flattened = [k for shard in shards for k in episode_keys(shard)]
        self.assertEqual(set(flattened), set(episode_keys(base)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(all(s['trials_per_task'] == 3 for s in shards))
        self.assertTrue(all(s['profile'] == 'shard-full-clean' for s in shards))
        loads = [sum(t['max_steps'] * 3 for t in s['tasks']) for s in shards]
        self.assertLessEqual(max(loads) - min(loads), 3 * 520)

    def test_merge_uses_canonical_order_and_retains_original_identity(self):
        parent, shards = parallel.split_manifest(fixture(), 4)
        inputs = completed_shards(parent, shards)
        rows, config = parallel.merge_records(parent, list(reversed(inputs)))
        self.assertEqual([r['episode_key'] for r in rows], episode_keys(parent))
        self.assertEqual(sum(r['success'] for r in rows), 12)
        self.assertTrue(all(r['manifest_sha256'] == parent['manifest_sha256'] for r in rows))
        originals = {name: shard['manifest_sha256'] for name, shard, _, _ in inputs}
        self.assertTrue(all(r['source_manifest_sha256'] == originals[r['source_run_id']] for r in rows))
        self.assertTrue(all(r['source_run_fingerprint'] != r['run_fingerprint'] for r in rows))
        self.assertEqual(config['execution'], 'parallel_aggregate_report_only')

    def test_merge_rejects_missing_duplicates_changed_code_and_wrong_parent(self):
        parent, shards = parallel.split_manifest(fixture(), 4)
        inputs = completed_shards(parent, shards)
        bad_inputs = []
        missing = copy.deepcopy(inputs)
        missing[0][2].pop()
        bad_inputs.append(missing)
        bad_inputs.append(inputs + [inputs[0]])
        changed = copy.deepcopy(inputs)
        changed[0][3]['code_sha256']['eval'] = 'changed'
        for row in changed[0][2]:
            row['run_fingerprint'] = digest(changed[0][3])
        bad_inputs.append(changed)
        wrong = copy.deepcopy(inputs)
        wrong[0][1]['parallel_shard']['parent_manifest_sha256'] = 'other'
        bad_inputs.append(wrong)
        for value in bad_inputs:
            with self.assertRaises(ValueError):
                parallel.merge_records(parent, value)

    def test_merge_rejects_falsified_task_metadata(self):
        parent, shards = parallel.split_manifest(fixture(), 4)
        inputs = completed_shards(parent, shards)
        inputs[0][2][0]['max_steps'] = 1
        with self.assertRaises(ValueError):
            parallel.merge_records(parent, inputs)


if __name__ == '__main__':
    unittest.main()
