#!/usr/bin/env python3
"""Combined 50-episode validation and independent single-GPU 4,000-episode run."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cbf_static.common import environment

if __name__ == '__main__':
    if Path(sys.prefix) != ROOT / '.runtime/arena':
        raise SystemExit(subprocess.run(
            [str(ROOT / '.runtime/arena/bin/python'), str(Path(__file__).resolve()), *sys.argv[1:]],
            env=environment()).returncode)
    from cbf_static.sequence import main
    raise SystemExit(main())
