#!/usr/bin/env python3
"""Patch robosuite 1.4.0's shared /tmp logfile for multi-user PACE nodes."""
from importlib.metadata import version
from pathlib import Path
import sysconfig

assert version('robosuite') == '1.4.0', 'Recheck this patch after upgrading robosuite'
path = Path(sysconfig.get_paths()['purelib']) / 'robosuite/utils/log_utils.py'
text = path.read_text()
old = 'logging.FileHandler("/tmp/robosuite.log")'
new = 'logging.FileHandler(os.path.join(tempfile.gettempdir(), f"robosuite-{os.getuid()}.log"))'
if old in text:
    assert text.count(old) == 1
    text = text.replace('import logging\n', 'import logging\nimport os\nimport tempfile\n', 1)
    path.write_text(text.replace(old, new))
else:
    assert new in text, 'Unexpected robosuite source: review patch before proceeding'
print('robosuite: per-user temporary logfile enabled (no simulation changes)')
