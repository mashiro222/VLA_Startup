import json
from pathlib import Path
import tempfile
import unittest

from vla_startup.journal import append, read_records, validate_prefix
from vla_startup.manifest import REFERENCE_EVALUATION, digest, init_file, physical_bddl, validate
from vla_startup.summarize import summarize


def plan():
    tasks = [dict(suite='s', task_id=i, category=category, difficulty=None)
             for i, category in enumerate(('A', 'A', 'B'))]
    value = dict(profile='pilot-plus', benchmark='plus', tasks=tasks, prompt_mode='upstream',
                 trials_per_task=1, expected_episodes=3)
    value['manifest_sha256'] = digest(value)
    return value


def records(manifest):
    return [{**task, 'episode_key': f"s:{task['task_id']}:0", 'success': success,
             'run_fingerprint': 'version-a', 'manifest_sha256': manifest['manifest_sha256']}
            for task, success in zip(manifest['tasks'], (True, True, False))]


class ResultsTests(unittest.TestCase):
    def test_changed_protocol_cannot_claim_default_reference_comparison(self):
        for replan, expected in ((5, True), (8, False)):
            manifest = plan()
            manifest.pop('manifest_sha256')
            manifest.update(profile='full-plus', seed=7,
                            evaluation={**REFERENCE_EVALUATION, 'replan_steps': replan})
            manifest['manifest_sha256'] = digest(manifest)
            reference = {'plus': {'category_percent': {'A': 100, 'B': 0}, 'overall_percent': 66.7}}
            result = summarize(manifest, records(manifest), reference)
            self.assertEqual(result['reference_comparison_enabled'], expected)

    def test_virtual_bddl_is_resolved_without_losing_language_perturbation(self):
        self.assertEqual(physical_bddl('base_language_10_view_0_0_100_0_0_initstate_0'), 'base_language_10.bddl')

    def test_texture_suffix_does_not_confuse_table_in_base_task_name(self):
        self.assertEqual(init_file('plus', 's', 'pick_from_table_center_tb_12'), 's/pick_from_table_center.pruned_init')
        self.assertEqual(init_file('plus', 's', 'pick_from_table_center_table_12'), 's/pick_from_table_center.pruned_init')
        self.assertEqual(init_file('plus', 's', 'base_add_2'), 'libero_newobj/s/base_add_2.pruned_init')

    def test_missing_results_do_not_become_a_full_benchmark_score(self):
        manifest = plan()
        reference = {'plus': {'category_percent': {}, 'overall_percent': 84.2}}
        result = summarize(manifest, records(manifest)[:1], reference)
        self.assertEqual(result['totals']['missing'], 2)
        self.assertEqual(result['totals']['success_percent_completed'], 100)
        self.assertIsNone(result['totals']['success_percent_full'])
        self.assertFalse(result['reference_comparison_enabled'])

    def test_overall_rate_weights_episodes_not_categories(self):
        manifest = plan()
        result = summarize(manifest, records(manifest), {'plus': {}})
        self.assertAlmostEqual(result['totals']['success_percent_full'], 200 / 3)
        self.assertEqual(result['groups']['category']['A']['success_percent_full'], 100)
        self.assertEqual(result['groups']['category']['B']['success_percent_full'], 0)

    def test_duplicate_or_out_of_order_records_are_rejected(self):
        manifest = plan()
        rows = records(manifest)
        for bad in ([rows[0], rows[0]], [rows[1], rows[0]]):
            with self.assertRaises(ValueError):
                validate_prefix(bad, manifest)

    def test_resume_rejects_changed_experiment(self):
        manifest = plan()
        with self.assertRaises(ValueError):
            validate_prefix(records(manifest), manifest, 'version-b')

    def test_truncated_tail_is_replayed_but_corrupted_committed_record_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'episodes.jsonl'
            append(path, {'completed': True})
            with path.open('ab') as stream:
                stream.write(b'{"incomplete":')
            self.assertEqual(read_records(path, repair_tail=True), [{'completed': True}])
            append(path, {'completed': False})
            self.assertEqual(len(read_records(path)), 2)
            with path.open('ab') as stream:
                stream.write(b'not json\n')
            with self.assertRaises(json.JSONDecodeError):
                read_records(path)

    def test_manifest_tampering_is_detected(self):
        manifest = plan()
        manifest['tasks'][0]['task_id'] = 99
        with self.assertRaises(ValueError):
            validate(manifest)

    def test_runtime_error_is_not_accepted_as_a_task_failure(self):
        manifest = plan()
        rows = records(manifest)
        rows[0]['success'] = None
        with self.assertRaises(ValueError):
            validate_prefix(rows, manifest)


if __name__ == '__main__':
    unittest.main()
