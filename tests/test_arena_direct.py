"""Submission boundary tests; every scheduler/process call is mocked."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
spec = importlib.util.spec_from_file_location('arena_direct_test', REPO / 'scripts/arena_direct.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class DirectSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'runs/arena-full-test'
        self.plan = {'run_id': 'arena-full-test', 'profile': 'full', 'workers': 4,
                     'time_limit': '12:00:00', 'expected_episodes': 2250, 'sha256': 'plan-hash'}
        self.args = argparse.Namespace(run_id='arena-full-test', gpu='a100', time_limit=None,
                                       submit=False, test_only=False, reason='User explicitly waived separate smoke')
        for name in cli.LAUNCH_FILES:
            p = self.root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('frozen launcher')
        for p in [patch.object(cli, 'ROOT', self.root), patch.object(cli, 'run_path', return_value=self.run),
                  patch.object(cli.arena, 'preflight'), patch.object(cli.arena, 'load', return_value=self.plan),
                  patch.object(cli, 'collect', return_value=[])]:
            p.start(); self.addCleanup(p.stop)
        cli.prepare(self.args)

    def test_dry_run_does_not_contact_scheduler(self):
        with patch.object(cli.subprocess, 'run') as launch, patch.object(cli.subprocess, 'check_output') as query:
            cli.submit(self.args)
            launch.assert_not_called(); query.assert_not_called()
        self.assertFalse((self.run / 'submission/unknown.json').exists())

    def test_probe_uses_test_only_and_does_not_create_submission_marker(self):
        self.args.test_only = True
        with patch.object(cli.subprocess, 'run') as launch:
            cli.submit(self.args)
        self.assertIn('--test-only', launch.call_args.args[0])
        self.assertFalse((self.run / 'submission/unknown.json').exists())

    def test_submit_uses_paid_account_one_gpu_per_worker_and_no_smoke_dependency(self):
        self.args.submit = True; self.args.gpu = 'h100'
        response = subprocess.CompletedProcess([], 0, stdout='12345;cluster\n', stderr='')
        with patch.object(cli.subprocess, 'check_output', return_value=''), \
             patch.object(cli.subprocess, 'run', return_value=response) as launch:
            cli.submit(self.args)
        cmd = launch.call_args.args[0]
        for flag in ['--account=gts-jli741-paid', '--partition=gpu-h100', '--gres=gpu:h100:1', '--array=0-3%4']:
            self.assertIn(flag, cmd)
        self.assertFalse(any(x.startswith('--dependency') for x in cmd))
        self.assertTrue((self.run / 'submission/12345.json').exists())
        self.assertFalse((self.run / 'submission/unknown.json').exists())

    def test_active_job_blocks_duplicate_submission(self):
        self.args.submit = True
        with patch.object(cli.subprocess, 'check_output', return_value='12345_0\n'), \
             patch.object(cli.subprocess, 'run') as launch:
            with self.assertRaisesRegex(RuntimeError, 'Already queued'):
                cli.submit(self.args)
            launch.assert_not_called()

    def test_uncertain_submission_blocks_retry(self):
        self.args.submit = True
        with patch.object(cli.subprocess, 'check_output', return_value=''), \
             patch.object(cli.subprocess, 'run', side_effect=subprocess.TimeoutExpired('sbatch', 60)):
            with self.assertRaises(subprocess.TimeoutExpired): cli.submit(self.args)
        self.assertTrue((self.run / 'submission/unknown.json').exists())
        with patch.object(cli.subprocess, 'run') as launch, patch.object(cli.subprocess, 'check_output') as query:
            with self.assertRaisesRegex(RuntimeError, 'result unknown'): cli.submit(self.args)
            launch.assert_not_called(); query.assert_not_called()

    def test_changed_launcher_or_plan_prevents_launch(self):
        (self.root / cli.LAUNCH_FILES[0]).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'does not match'): cli.load(self.run)
        (self.root / cli.LAUNCH_FILES[0]).write_text('frozen launcher')
        self.plan['sha256'] = 'different-plan'
        with self.assertRaisesRegex(ValueError, 'does not match'): cli.load(self.run)

    def test_record_checksum_and_full_profile_are_checked(self):
        p = self.run / 'direct-launch.json'
        record = json.loads(p.read_text()); record['reason'] = 'altered'
        p.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'checksum'): cli.load(self.run)
        self.plan['profile'] = 'smoke'
        p.unlink()
        with self.assertRaisesRegex(ValueError, 'only for the full'): cli.prepare(self.args)

    def test_worker_requires_allocation_and_records_waiver_without_smoke_gate(self):
        with patch.dict(cli.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'Slurm allocation'): cli.worker(self.args)
        with patch.dict(cli.os.environ, {'SLURM_JOB_ID': '12345', 'SLURM_ARRAY_TASK_ID': '2'}), \
             patch.object(cli, 'supervise', return_value=0) as launch, \
             patch.object(cli.arena, 'require_smoke', side_effect=AssertionError('must not wait')):
            self.assertEqual(cli.worker(self.args), 0)
            launch.assert_called_once_with(self.run)
        self.assertTrue((self.run / 'workers/12345-2-direct-launch.json').exists())


if __name__ == '__main__': unittest.main()
