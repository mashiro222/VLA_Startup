"""Submission boundaries, calibration gating and preservation of the full protocol."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import yaml
from cbf_static.common import ROOT, save
from cbf_static.plan import episodes
from cbf_static.sequence import execute, smoke_gate, submission_command, submit


class SequenceTests(unittest.TestCase):
    def test_full_is_exact_original_4000_with_main_first(self):
        cfg = yaml.safe_load((ROOT / 'config/cbf_static/experiment.yaml').read_text())
        with (ROOT / 'reports/preparation/arena-task-catalog.csv').open() as f:
            catalog = list(csv.DictReader(f))
        full = episodes('full', catalog, cfg)
        self.assertEqual(full, episodes('main', catalog, cfg) + episodes('sanity', catalog, cfg))
        self.assertEqual(len(full), 4000)
        self.assertEqual(len({(x['level'], x['task_id'], x['condition'], x['repeat'], x['init_index']) for x in full}), 4000)

    def test_paid_single_gpu_limits_and_no_dependency(self):
        validation = submission_command(Path('/tmp/run'), 'validation')
        full = submission_command(Path('/tmp/run'), 'full')
        for command in (validation, full):
            self.assertIn('--gres=gpu:a100:1', command)
            self.assertIn('--constraint=A100-80GB', command)
            self.assertIn('--account=gts-jli741-paid', command)
            self.assertFalse(any('--array' in x for x in command))
            self.assertFalse(any('--dependency' in x for x in command))
        self.assertIn('--time=02:00:00', validation)
        self.assertIn('--time=24:00:00', full)

    def test_unknown_submission_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as d, patch('cbf_static.sequence.check_sequence'):
            path = Path(d)
            with patch('cbf_static.sequence.subprocess.run', side_effect=subprocess.TimeoutExpired('sbatch', 90)) as run:
                with self.assertRaises(subprocess.TimeoutExpired): submit(path, 'a100', True)
                self.assertEqual(run.call_count, 1)
            with patch('cbf_static.sequence.subprocess.run') as run:
                with self.assertRaisesRegex(RuntimeError, 'uncertain'): submit(path, 'a100', True)
                run.assert_not_called()

    def test_two_submissions_are_idempotent_and_independent(self):
        with tempfile.TemporaryDirectory() as d, patch('cbf_static.sequence.check_sequence'):
            path = Path(d)
            responses = [subprocess.CompletedProcess([], 0, '101\n', ''), subprocess.CompletedProcess([], 0, '102\n', '')]
            with patch('cbf_static.sequence.subprocess.run', side_effect=responses) as run:
                submit(path, 'a100', True)
                self.assertFalse(any('--dependency' in x for x in run.call_args_list[1].args[0]))
                submit(path, 'a100', True)
                self.assertEqual(run.call_count, 2)

    def test_failed_calibration_stops_before_smoke(self):
        with tempfile.TemporaryDirectory() as d, patch.dict('os.environ', SLURM_JOB_ID='unit-test'), \
                patch('cbf_static.sequence.check_sequence'), patch('cbf_static.sequence.child') as child, \
                patch('cbf_static.sequence.materialize', side_effect=ValueError('Calibration rejected')):
            with self.assertRaisesRegex(ValueError, 'Calibration rejected'): execute(Path(d), 'validation')
            self.assertEqual(child.call_count, 1)
            self.assertIn(str(Path(d) / 'calibration'), child.call_args.args[0])
            self.assertFalse((Path(d) / 'validation-passed.json').exists())

    def test_full_can_start_first_without_waiting_for_smoke(self):
        with tempfile.TemporaryDirectory() as d, patch.dict('os.environ', SLURM_JOB_ID='unit-test'), \
                patch('cbf_static.sequence.check_sequence'), patch('cbf_static.sequence.child') as child, \
                patch('cbf_static.sequence.materialize', return_value=Path(d) / 'full') as materialize:
            execute(Path(d), 'full')
            self.assertEqual(child.call_count, 3)
            self.assertIn(str(Path(d) / 'full-calibration'), child.call_args_list[0].args[0])
            self.assertIn(str(Path(d) / 'full'), child.call_args_list[1].args[0])
            materialize.assert_called_once_with(Path(d), 'full', Path(d) / 'full-calibration/calibration.json')

    def test_full_reuses_already_available_calibration_without_smoke_gate(self):
        with tempfile.TemporaryDirectory() as d, patch.dict('os.environ', SLURM_JOB_ID='unit-test'), \
                patch('cbf_static.sequence.check_sequence'), patch('cbf_static.sequence.child') as child, \
                patch('cbf_static.sequence.materialize', return_value=Path(d) / 'full'):
            path = Path(d); save(path / 'calibration/calibration.json', {'passed': True})
            execute(path, 'full')
            self.assertEqual(child.call_count, 2)
            self.assertIn(str(path / 'full'), child.call_args_list[0].args[0])

    def test_smoke_gate_checks_execution_not_success_rate(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d); folder = path / 'fixture'; folder.mkdir()
            (path / 'plan.json').write_text('{}')
            save(path / 'summary/sanity-checks.json', [dict(identical_actions=True, identical_positions=True)] * 10)
            nominal = np.zeros((2, 7)); action = nominal.copy(); action[:, 0] = .1
            np.savez(folder / 'trace.npz', action_nominal=nominal, action_executed=action,
                     eef_after=np.zeros((2, 3)), h=np.ones((2, 2)), h_after=np.ones((2, 2)), intervention_m=[.01, .01])
            row = dict(artifact_dir=str(folder), selected_obstacles=['a', 'b'], level=2, condition='C2',
                       official_diagnostics={'cost_components': [{'cc': 2.}]}, CC=2.,
                       solver_status_counts={'solved': 2}, solver_failures=0, wall_seconds=10., success=False)
            rows = [copy.deepcopy(row) for _ in range(40)]
            frame = np.zeros((224, 224, 3), np.uint8); frame[0, 0] = 255
            with patch('cbf_static.report.collect', return_value=({'profile': 'smoke', 'expected_episodes': 40}, rows)), \
                    patch('imageio.v2.imread', return_value=frame):
                self.assertTrue(smoke_gate(path)['passed'])
                rows[0]['selected_obstacles'] = []
                with self.assertRaisesRegex(ValueError, 'selected obstacles'): smoke_gate(path)
                rows[0]['selected_obstacles'] = ['a', 'b']
                rows[0]['CC'] = 4.
                with self.assertRaisesRegex(ValueError, 'CC'): smoke_gate(path)


if __name__ == '__main__': unittest.main()
