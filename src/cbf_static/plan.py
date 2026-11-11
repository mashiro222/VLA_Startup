"""Immutable paired episode manifests; stdlib plus YAML, no model imports."""
import csv
import copy
import json
import math
import subprocess
from pathlib import Path
import yaml
from .common import ROOT, digest, save, episode_seed, source_files


def upstream_check(cfg):
    repo = ROOT / '.runtime/repos/vla_arena'
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != cfg['arena_commit']:
        raise ValueError('Arena commit differs from the experiment pin')
    if subprocess.run(['git', '-C', str(repo), 'diff', '--quiet', 'HEAD']).returncode:
        raise ValueError('Arena tracked sources have local edits; audit and pin them before evaluation')


def config_check(cfg):
    f = cfg['filter']; e = cfg['evaluation']
    if not math.isfinite(f['action_scale_m']) or f['action_scale_m'] <= 0:
        raise ValueError('Positive finite action scale required')
    if not 0 < f['gamma'] <= 1 or f['gripper_radius_m'] <= 0 or f['margin_m'] < 0:
        raise ValueError('Invalid barrier parameters')
    if not 1 <= e['replan_steps'] <= 10 or e['horizon'] < 1 or e['hz'] != 20:
        raise ValueError('Invalid action horizon / controller frequency')
    for spec in cfg['profiles'].values():
        if len(spec['repeats']) != len(set(spec['repeats'])) or any(x < 0 for x in spec['repeats']):
            raise ValueError('Sampling seed repetitions must be unique and nonnegative')
        if any(not 0 < x <= 1 for x in spec.get('gammas', [f['gamma']])):
            raise ValueError('Discrete gamma must be in (0, 1]')


def episodes(profile, catalog, cfg):
    if profile == 'full':
        # Exact original protocol, with the main comparison preceding the L0 control.
        return episodes('main', catalog, cfg) + episodes('sanity', catalog, cfg)
    result = []
    spec = cfg['profiles'][profile]
    for row in catalog:
        level, task = int(row['level']), int(row['task_id'])
        if row['suite'] != 'safety_static_obstacles':
            continue
        if level not in spec['levels'] or task not in spec['tasks']: continue
        arms = [a for a in spec['conditions'] if not (level == 0 and a == 'C1')]
        gammas = spec.get('gammas', [cfg['filter']['gamma']])
        if int(row['init_count']) != 50:
            raise ValueError('Protocol requires the audited 50 initialization states')
        if not 1 <= spec['init_states'] <= 50: raise ValueError('init_states must be in [1, 50]')
        for condition in arms:
            for gamma in gammas:
                for r, repeat in enumerate(spec['repeats']):
                    for init in range(spec['init_states']):
                        result.append(dict(row, level=level, task_id=task, condition=condition, gamma=gamma,
                            init_index=init, repeat=repeat, ordinal=r * spec['init_states'] + init,
                            seed=episode_seed(level, task, init, repeat)))
    return result


def prepare(profile, run, calibration=None):
    run = Path(run)
    if run.exists(): raise FileExistsError('Choose a new run directory; prepared manifests are immutable')
    cfg = yaml.safe_load((ROOT / 'config/cbf_static/experiment.yaml').read_text())
    upstream_check(cfg)
    calibration_data = json.loads(Path(calibration).read_text()) if calibration else None
    if calibration_data:
        if not calibration_data.get('passed') or calibration_data.get('checkpoint_revision') != json.loads((ROOT / 'config/cbf_static/checkpoint.json').read_text())['revision']:
            raise ValueError('Calibration did not pass or uses a different checkpoint')
        cfg['filter']['action_scale_m'] = calibration_data['action_scale_m']
        cfg['filter']['scale_source'] = 'held_out_L0_empirical_fit'
    config_check(cfg)
    with (ROOT / 'reports/preparation/arena-task-catalog.csv').open() as f:
        catalog = list(csv.DictReader(f))
    eps = episodes(profile, catalog, cfg)
    tasks = {(e['level'], e['task_id']): e for e in eps}
    for e in tasks.values():
        for key, sha in [('bddl', 'bddl_sha256'), ('init_file', 'init_sha256')]:
            if digest(ROOT / e[key]) != e[sha]: raise ValueError('Task data hash changed')
    groups = {}
    for e in eps:
        key = 'all' if profile in ('calibration', 'smoke', 'full') else f"L{e['level']}-t{e['task_id']}-{e['condition']}-g{e['gamma']:g}"
        groups.setdefault(key, []).append(e)
    manifest = {'schema': 'cbf-static-v1', 'profile': profile, 'config': copy.deepcopy(cfg),
                'calibration': calibration_data, 'expected_episodes': len(eps),
                'units': [{'id': k, 'episodes': v} for k, v in groups.items()], 'source_sha256': source_files()}
    save(run / 'plan.json', manifest)
    (run / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Prepared {len(eps)} episodes in {len(groups)} independent single-GPU array units: {run}")
    if profile != 'calibration' and not calibration_data:
        print('DRAFT: action scale has not been calibrated. Prepare a new run with --calibration before execution.')
    return manifest


def validate(run, require_ready=True):
    plan = json.loads((Path(run) / 'plan.json').read_text())
    upstream_check(plan['config'])
    config_check(plan['config'])
    for name, sha in plan['source_sha256'].items():
        if digest(ROOT / name) != sha: raise ValueError(f'Source changed after preparation: {name}; prepare a new run')
    if require_ready and plan['profile'] != 'calibration' and not plan['calibration']:
        raise ValueError('Measured calibration required: prepare a new run with --calibration')
    manifest = json.loads((ROOT / 'config/cbf_static/checkpoint.json').read_text())
    checkpoint = ROOT / plan['config']['checkpoint']
    for item in manifest['files']:
        path = checkpoint / item['path']
        if not path.is_file() or path.stat().st_size != item['bytes']: raise ValueError(f'Checkpoint missing/truncated: {path}')
    if json.loads((checkpoint / 'verified.json').read_text()) != manifest:
        raise ValueError('Checkpoint verification receipt differs from pinned manifest')
    return plan
