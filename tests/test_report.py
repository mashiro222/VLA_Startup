"""Report integration checks use synthetic unit-test fixtures in a temporary dir.

These are never evaluation results and are never published to reports/.
Run with the simulation Python to enable the matplotlib-dependent test.
"""
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_results import plan, records
from vla_startup.journal import append


@unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Run with simulation Python for plot checks')
class ReportTests(unittest.TestCase):
    def test_partial_report_preserves_denominators_and_raw_records(self):
        repo = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('report_under_test', repo / 'scripts/report.py')
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        with tempfile.TemporaryDirectory(prefix='vla-report-test-') as directory:
            root = Path(directory)
            run = root / 'runs' / 'fixture'
            run.mkdir(parents=True)
            (root / 'config').mkdir()
            (root / 'config/references.json').write_text(json.dumps({'plus': {}}))
            manifest = plan()
            (run / 'manifest.json').write_text(json.dumps(manifest))
            rows = records(manifest)[:1]
            for row in rows:
                append(run / 'episodes.jsonl', row)
            with patch.object(report, 'ROOT', root), patch.object(sys, 'argv', ['report.py', 'fixture']):
                report.main()
            output = root / 'reports/fixture'
            with (output / 'summary.csv').open() as stream:
                totals = next(csv.DictReader(stream))
            self.assertEqual(totals['missing'], '2')
            self.assertEqual(totals['success_percent_full'], '')
            self.assertEqual(json.loads((output / 'summary.json').read_text())['complete'], False)
            with gzip.open(output / 'episodes.jsonl.gz', 'rt') as stream:
                self.assertEqual([json.loads(line) for line in stream], rows)
            for name in ('cumulative_success.png', 'success_by_group.png'):
                self.assertEqual((output / name).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            hashes = json.loads((output / 'sha256.json').read_text())
            self.assertIn('summary.csv', hashes)
            self.assertIn('episodes.jsonl.gz', hashes)


if __name__ == '__main__':
    unittest.main()
