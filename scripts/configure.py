#!/usr/bin/env python3
"""Write noninteractive, project-local LIBERO configs; never touch ~/.libero."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for mode in ('clean', 'plus'):
    base = ROOT / '.runtime/repos' / ('libero_' + mode) / 'libero/libero'
    config = ROOT / '.runtime/config' / mode
    config.mkdir(parents=True, exist_ok=True)
    values = {'benchmark_root': base, 'bddl_files': base / 'bddl_files',
              'init_states': base / 'init_files', 'assets': base / 'assets',
              'datasets': ROOT / '.runtime/unused_datasets'}
    # JSON is valid YAML, so this does not depend on PyYAML at setup time.
    (config / 'config.yaml').write_text(json.dumps({k: str(v.resolve()) for k, v in values.items()}, indent=2) + '\n')
    print('Configured:', config)
