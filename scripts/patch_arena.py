"""Only make the robosuite log path safe on a shared multi-user host."""
from importlib.metadata import version
from pathlib import Path
import sysconfig

assert version('robosuite') == '1.5.1'
path = Path(sysconfig.get_paths()['purelib']) / 'robosuite/utils/log_utils.py'
text = path.read_text()
old = 'logging.FileHandler("/tmp/robosuite.log")'
new = 'logging.FileHandler(os.path.join(tempfile.gettempdir(), f"robosuite-{os.getuid()}.log"))'
if old in text:
    text = text.replace('import logging\n', 'import logging\nimport os\nimport tempfile\n', 1)
    path.write_text(text.replace(old, new))
elif '/tmp/robosuite.log' in text:
    raise RuntimeError('Unexpected log path syntax; review before use')
print('Arena robosuite logfile checked; simulation code unchanged')
