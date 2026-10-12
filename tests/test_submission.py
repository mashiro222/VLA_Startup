"""Exercise the submission shell against fake Slurm commands; never contact Slurm."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='vla-submit-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for directory in ('scripts', 'config', 'runs/batch', 'fake-bin'):
            (self.root / directory).mkdir(parents=True)
        for name in ('common.sh', 'submit_parallel.sh'):
            shutil.copy2(ROOT / 'scripts' / name, self.root / 'scripts' / name)
        (self.root / 'config/pace.env').write_text(
            'ACCOUNT=test\nPARTITION=gpu-a100\nQOS=test\nGRES=gpu:a100:1\nCPUS=8\nMEMORY=64G\n')
        (self.root / 'runs/batch/batch.json').write_text(json.dumps({'workers': 4}))
        (self.root / 'scripts/preflight.py').write_text('pass\n')
        (self.root / 'scripts/parallel.py').write_text(
            'import os,sys\n'
            'if "--require-smoke" in sys.argv and os.environ.get("FAKE_SMOKE_READY") != "1": sys.exit(1)\n'
            'if sys.argv[1] == "remaining": print("0,1,2,3")\n')
        commands = {
            'squeue': '#!/bin/sh\nprintf "%s" "$FAKE_QUEUE"\nexit "${FAKE_QUEUE_CODE:-0}"\n',
            'sbatch': '#!/bin/sh\nprintf "call\\n" >> "$FAKE_CALLS"\nprintf "%s\\n" "$@" > "$FAKE_CAPTURE"\nexit "${FAKE_SUBMIT_CODE:-0}"\n'}
        for name, content in commands.items():
            path = self.root / 'fake-bin' / name
            path.write_text(content)
            path.chmod(0o755)
        self.capture = self.root / 'args'
        self.calls = self.root / 'calls'
        self.env = dict(os.environ, PATH=str(self.root / 'fake-bin') + ':' + os.environ['PATH'],
                        FAKE_CAPTURE=str(self.capture), FAKE_CALLS=str(self.calls),
                        FAKE_QUEUE='', FAKE_SMOKE_READY='1')

    def run_script(self, *arguments, **env):
        return subprocess.run(['bash', str(self.root / 'scripts/submit_parallel.sh'), 'batch', *arguments],
                              env=dict(self.env, **env), capture_output=True, text=True)

    def test_dry_run_never_calls_sbatch(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.capture.exists())

    def test_incomplete_smoke_or_active_job_blocks_submission(self):
        for env in ({'FAKE_SMOKE_READY': '0'}, {'FAKE_QUEUE': '12345'}, {'FAKE_QUEUE_CODE': '1'}):
            result = self.run_script('--submit', **env)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.capture.exists())

    def test_explicit_submission_requests_four_one_gpu_workers_without_dependency(self):
        result = self.run_script('--submit', TIME_LIMIT='08:00:00')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        self.assertIn('--array=0,1,2,3%4', args)
        self.assertIn('--gres=gpu:a100:1', args)
        self.assertIn('--time=08:00:00', args)
        self.assertFalse(any(a.startswith('--dependency') for a in args))
        self.assertEqual(self.calls.read_text(), 'call\n')

    def test_h100_override_requests_one_h100_per_worker(self):
        result = self.run_script(VLA_GPU_TYPE='h100')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--partition=gpu-h100', result.stdout)
        self.assertIn('--gres=gpu:h100:1', result.stdout)
        self.assertFalse(self.capture.exists())

    def test_dependency_waits_for_whole_existing_array(self):
        result = self.run_script('--submit', VLA_GPU_TYPE='h100',
                                 VLA_AFTERANY_JOB_ID='13967217')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        self.assertIn('--dependency=afterany:13967217', args)
        self.assertIn('--partition=gpu-h100', args)
        self.assertIn('--gres=gpu:h100:1', args)

    def test_invalid_dependency_is_rejected_before_submission(self):
        result = self.run_script('--submit', VLA_AFTERANY_JOB_ID='13967217_0')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.capture.exists())

    def test_sbatch_timeout_is_not_automatically_retried(self):
        result = self.run_script('--submit', FAKE_SUBMIT_CODE='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls.read_text(), 'call\n')
        self.assertIn('do not retry before checking squeue', result.stderr)


if __name__ == '__main__':
    unittest.main()
