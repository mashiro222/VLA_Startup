"""Inspect task files/initial states without constructing a simulator or policy."""
import json
from pathlib import Path
import sys

from .core import ROOT, file_hash


def build():
    from vla_arena.vla_arena import benchmark, get_vla_arena_path
    cfg = json.loads((ROOT / 'config/arena.json').read_text())
    rows = []
    for name in cfg['suites']:
        suite = benchmark.get_benchmark(name)()
        for level in cfg['levels']:
            for task_id in range(5):
                task = suite.get_task_by_level_id(level, task_id)
                bddl = Path(get_vla_arena_path('bddl_files')) / task.problem_folder / f'level_{level}' / task.bddl_file
                init = Path(get_vla_arena_path('init_states')) / task.problem_folder / f'level_{level}' / task.init_states_file
                states = suite.get_task_init_states(level, task_id)
                if len(states) == 0:
                    raise ValueError(f'Empty initial states: {init}')
                prompt = task.language[0] if isinstance(task.language, list) else task.language
                rows.append({'suite': name, 'level': level, 'task_id': task_id, 'name': task.name,
                             'prompt': prompt, 'bddl': str(bddl.relative_to(ROOT)),
                             'bddl_sha256': file_hash(bddl), 'init_file': str(init.relative_to(ROOT)),
                             'init_sha256': file_hash(init), 'init_count': len(states)})
    return rows


if __name__ == '__main__':
    from .core import save
    save(Path(sys.argv[1]), build())
