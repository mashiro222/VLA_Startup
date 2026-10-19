"""Evaluate matched trained pi0.5 policies with the existing Arena cost diagnostics."""
import argparse
import copy
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

from arena_startup.core import ROOT, collect, digest, file_hash, frozen_files, save, validate_plan


def followup_files():
    paths = list((ROOT / 'src/sf_followup').glob('*.py'))
    paths += [ROOT / 'slurm/sf_evaluation.sbatch']
    paths += [ROOT / 'requirements/sf_followup.lock', ROOT / 'config/sf_followup_dependencies.json']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def validate(plan):
    validate_plan(plan)
    if plan['followup_files'] != followup_files():
        raise ValueError('Fine-tuned evaluation code changed after preparation')
    from sf_training.workflow import files
    source = json.loads(Path(plan['trained_policy']['plan']).read_text())
    if source['files'] != files():
        raise ValueError('Training implementation changed')
    if file_hash(plan['trained_policy']['plan']) != plan['trained_policy']['plan_sha256']:
        raise ValueError('Training plan changed')


def matched_units(plan, trials):
    """Use distinct initial-state indices and the same 30 task configurations for both arms."""
    suites = ('safety_static_obstacles', 'safety_dynamic_obstacles')
    units = copy.deepcopy([u for u in plan['units'] if u['suite'] in suites])
    expected = {(suite, level, task) for suite in suites for level in range(3) for task in range(5)}
    actual = {(u['suite'], u['level'], u['task_id']) for u in units}
    if len(units) != 30 or actual != expected or len({u['id'] for u in units}) != 30:
        raise ValueError('Expected exactly 30 distinct matched task configurations')
    protocol = plan['config']['evaluation']
    offset = protocol['init_state_offset']
    if type(trials) is not int or trials < 1 or offset < 0 or protocol['init_state_selection_mode'] != 'episode_idx':
        raise ValueError('Evaluation requires positive trials and distinct initial-state indices')
    for unit in units:
        if unit['seed'] != 7 or unit['init_count'] < offset + trials:
            raise ValueError('Expected seed 7 and sufficient distinct initial states for every task')
        unit['trials'] = trials
    return units


def prepare(source_plan, pinned, output, arm, step=5000, trials_per_task=30):
    source_plan, pinned, output = map(lambda p: Path(p).resolve(), (source_plan, pinned, output))
    if output.exists():
        raise FileExistsError(output)
    source = json.loads(source_plan.read_text())
    plan = copy.deepcopy(json.loads((ROOT / 'runs/arena-balanced1500-001/plan.json').read_text()))
    suites = ['safety_static_obstacles', 'safety_dynamic_obstacles']
    plan['units'] = matched_units(plan, trials_per_task)
    episodes = sum(u['trials'] for u in plan['units'])
    profile = f'matched{episodes}'
    plan.update(run_id=output.name, profile=profile, workers=1, time_limit='24:00:00',
                videos_per_worker=6, expected_episodes=episodes,
                training_condition=f'pi05_base + Arena LoRA {arm}, {step} optimizer updates, global batch 32')
    plan['config']['suites'] = suites
    plan['config'].pop('checkpoint', None)
    plan['config']['policy_config'] = 'pi05_arena_trained_pytorch'
    plan['config']['profiles'][profile] = dict(seeds=[7], trials=trials_per_task)
    if source['recipe']['global_batch'] != 32:
        raise ValueError('Expected training batch 32')
    plan['files'] = frozen_files()
    plan['followup_files'] = followup_files()
    plan['trained_policy'] = dict(plan=str(source_plan), plan_sha256=file_hash(source_plan),
                                  checkpoint=str(pinned / arm / f'step-{step:08d}.pt'),
                                  arm=arm, step=step, norm_stats_sha256=source['assets']['norm_stats'])
    plan['sha256'] = digest({k: v for k, v in plan.items() if k != 'sha256'})
    save(output / 'plan.json', plan)
    (output / 'workers').mkdir()
    print(f'Prepared {output}: {episodes} episodes; checkpoint must reach exactly {step}.')


def rng_seed(state):
    if not isinstance(state, (list, tuple)) or len(state) != 2:
        raise ValueError('Expected [policy_call_index, episode_seed]')
    if any(type(x) is not int or not 0 <= x < 2**32 for x in state):
        raise ValueError('RNG words must be uint32 integers')
    counter, seed = state
    if counter == 2**32 - 1:
        raise ValueError('RNG counter exhausted')
    return ((seed << 32) | counter) % (2**63 - 1), [counter + 1, seed]


def input_batch(obs, sp, stats, max_tokens):
    import numpy as np
    import torch
    from sf_training.normalization import normalize
    images = []
    for key in ('observation/image', 'observation/wrist_image'):
        im = np.asarray(obs[key])
        if im.shape != (224, 224, 3) or im.dtype != np.uint8:
            raise ValueError('Policy expects upright 224x224 RGB uint8 images')
        images.append(torch.from_numpy(im.copy()).permute(2, 0, 1).float() / 255)
    state = np.asarray(obs['observation/state'], np.float32)
    if state.shape != (8,):
        raise ValueError('Expected eight-dimensional Arena state')
    state = normalize(state, stats['state'])
    prompt = str(obs['prompt']).strip().replace('_', ' ').replace('\n', ' ')
    ids = (sp.encode(prompt, add_bos=True) + sp.encode('\n'))[:max_tokens]
    mask = [True] * len(ids) + [False] * (max_tokens - len(ids))
    ids += [0] * (max_tokens - len(ids))
    return dict(images=torch.stack(images)[None], state=torch.from_numpy(np.pad(state, (0, 24)))[None],
                tokens=torch.tensor([ids], dtype=torch.long), token_mask=torch.tensor([mask]))


def expected_metadata(plan):
    return dict(model='pi05_arena_trained', backend='pytorch', action_horizon=10,
                num_steps=plan['config']['evaluation']['num_steps'],
                training_plan_sha256=plan['trained_policy']['plan_sha256'],
                arm=plan['trained_policy']['arm'], step=plan['trained_policy']['step'],
                norm_stats_sha256=plan['trained_policy']['norm_stats_sha256'],
                rng_protocol='episode-seed-call-counter-v1')


def validate_checkpoint(ck, spec, base):
    if (ck['identity']['plan_sha256'] != spec['plan_sha256'] or ck['identity']['arm'] != spec['arm']
            or ck['step'] != spec['step'] or ck['identity']['base'] != base):
        raise ValueError('Checkpoint identity/step mismatch')


def server(run, port):
    import numpy as np
    import sentencepiece
    import torch
    from sf_training.model import build_policy, observation
    from sf_training.lora import load_trainable
    from sf_training.normalization import unnormalize
    from sf_training.workflow import verify_plan
    plan = json.loads((run / 'plan.json').read_text()); validate(plan)
    spec = plan['trained_policy']
    train = verify_plan(Path(spec['plan']))
    # Keep the running training environment unchanged. Transport is an isolated sidecar,
    # added only after validating the original environment's pinned package inventory.
    import sys
    deps = ROOT / '.runtime/sf_followup_deps'
    manifest = json.loads((ROOT / 'config/sf_followup_dependencies.json').read_text())
    for name, expected in manifest.items():
        if file_hash(deps / name) != expected: raise ValueError('Transport sidecar dependency changed')
    sys.path.insert(0, str(deps))
    from arena_startup.transport import ArenaPolicyServer
    home = ROOT / '.runtime/training'
    if file_hash(home / 'norm_stats.json') != spec['norm_stats_sha256']:
        raise ValueError('Training and inference normalization mismatch')
    ck = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    validate_checkpoint(ck, spec, file_hash(home / 'pi05_base_torch/conversion.json'))
    if not torch.cuda.is_available():
        raise RuntimeError('GPU allocation required')
    device = torch.device('cuda', 0)
    policy = build_policy(train['recipe'], home / 'pi05_base_torch/model.safetensors')
    load_trainable(policy, {k.removeprefix('policy.'): v for k, v in ck['trainable'].items() if k.startswith('policy.')})
    del ck
    policy.gradient_checkpointing_disable(); policy.to(device).eval()
    stats = json.loads((home / 'norm_stats.json').read_text())['norm_stats']
    sp = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / '.runtime/cache/openpi/big_vision/paligemma_tokenizer.model'))
    cfg = train['recipe']

    class TrainedPolicy:
        @torch.inference_mode()
        def infer(self, obs):
            seed, next_state = rng_seed(obs['__vla_rng'])
            batch = {k: v.to(device) for k, v in input_batch(obs, sp, stats, cfg['max_token_len']).items()}
            processed, _ = observation(batch, augment=False)
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn((1, cfg['action_horizon'], cfg['action_dim']), device=device, generator=generator)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                normalized = policy.sample_actions(device, processed, noise=noise,
                                                   num_steps=plan['config']['evaluation']['num_steps'])
            actions = unnormalize(normalized[0, :, :7].float().cpu().numpy(), stats['actions'])
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise ValueError('Invalid action chunk')
            return {'actions': actions, '__vla_rng': next_state}

    metadata = expected_metadata(plan)
    metadata['checkpoint_sha256'] = file_hash(spec['checkpoint'])
    save(run / 'workers' / f"policy-{os.environ.get('SLURM_JOB_ID', 'local')}.json", metadata)
    ArenaPolicyServer(TrainedPolicy(), host='127.0.0.1', port=port, metadata=metadata).serve_forever()


def client(run, port, worker):
    from types import SimpleNamespace
    from arena_startup.evaluate import evaluate_units, interrupted, timed_out
    from arena_startup.transport import ArenaClient
    plan = json.loads((run / 'plan.json').read_text()); validate(plan)
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, timed_out)
    with ArenaClient(port, expected_metadata(plan), plan['config']['evaluation']['policy_request_timeout_seconds']) as c:
        evaluate_units(SimpleNamespace(run=run, worker=worker), plan, c)


def worker(run):
    from arena_startup.runtime import environment
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Evaluation requires a Slurm GPU job')
    plan = json.loads((run / 'plan.json').read_text()); validate(plan)
    if not Path(plan['trained_policy']['checkpoint']).is_file():
        raise FileNotFoundError('The requested paired checkpoint is not available; no shorter checkpoint substituted')
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    ident = os.environ['SLURM_JOB_ID']
    env = environment()
    server_env = dict(os.environ)
    server_env['PYTHONPATH'] = ':'.join([str(ROOT / 'src'), str(ROOT / '.runtime/repos/spatial_forcing/openpi-SF/src'),
                                       str(ROOT / '.runtime/repos/openpi/packages/openpi-client/src')])
    server_env['JAX_PLATFORMS'] = 'cpu'
    children = []
    def stop(*_):
        for proc in children:
            if proc.poll() is None: proc.terminate()
        raise InterruptedError('Evaluation allocation interrupted')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        with (run / 'workers' / f'{ident}-server.log').open('w') as out:
            cmd = [str(ROOT / '.runtime/sf_train/bin/python'), '-m', 'sf_followup.evaluation', 'server', str(run), '--port', str(port)]
            p = subprocess.Popen(cmd, env=server_env, stdout=out, stderr=subprocess.STDOUT); children.append(p)
            start = time.monotonic()
            while True:
                if p.poll() is not None: raise RuntimeError('Policy startup failed; see server log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2) as response:
                        if response.status == 200: break
                except OSError: pass
                if time.monotonic() - start > 1800: raise TimeoutError('Policy startup exceeded 1800s')
                time.sleep(2)
            cmd = [str(ROOT / '.runtime/arena/bin/python'), '-m', 'sf_followup.evaluation', 'client', str(run),
                   '--port', str(port), '--worker', ident]
            p = subprocess.Popen(cmd, env=env); children.append(p)
            if p.wait(): raise RuntimeError('Rollout failed; see diagnostic journals')
    finally:
        for p in reversed(children):
            if p.poll() is None: p.terminate()
            try: p.wait(timeout=20)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
    report(run)


def report(run):
    import csv
    from statistics import mean
    from arena_startup.diagnostic_report import build_diagnostic_tables
    plan = json.loads((run / 'plan.json').read_text())
    validate_plan(plan, check_files=False)
    rows = collect(run, plan)
    if len(rows) != plan['expected_episodes']: raise ValueError('Evaluation incomplete')
    output = ROOT / 'reports' / run.name; output.mkdir(exist_ok=True)
    def write_csv(path, data, fields=None):
        with Path(path).open('w') as f:
            w = csv.DictWriter(f, fieldnames=fields or list(data[0])); w.writeheader(); w.writerows(data)
    groups = []
    for suite in plan['config']['suites']:
        for level in plan['config']['levels']:
            part = [r for r in rows if r['suite'] == suite and r['level'] == level]
            groups.append(dict(suite=suite, level=level, episodes=len(part),
                               sr_percent=100*mean(r['success'] for r in part), cc=mean(r['cost'] for r in part)))
    write_csv(output / 'summary.csv', groups)
    write_csv(output / 'episodes.csv', [{k: v for k, v in r.items() if k not in ('diagnostics', 'rng_after')} for r in rows])
    diagnostics = build_diagnostic_tables(output, rows, write_csv)
    save(output / 'summary.json', dict(complete=True, episodes=len(rows), groups=groups,
                                      training_condition=plan['training_condition'], diagnostics=diagnostics))
    save(output / 'plan.json', plan)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for suite in plan['config']['suites']:
        g = [x for x in groups if x['suite'] == suite]
        for ax, key in zip(axes, ('sr_percent', 'cc')):
            ax.plot([x['level'] for x in g], [x[key] for x in g], marker='o', label=suite)
            ax.set(xlabel='Difficulty level', ylabel=key); ax.set_xticks([0, 1, 2]); ax.grid(alpha=.2)
    axes[0].set_ylim(0, 100); axes[1].legend(fontsize=6)
    fig.suptitle(plan['training_condition']); fig.tight_layout(); fig.savefig(output / 'safety_by_level.png', dpi=160); plt.close(fig)
    (output / 'README.md').write_text(f"# {run.name}\n\n{plan['training_condition']}\n\n"
                                    f'Complete: {len(rows)} episodes. See summary.csv, safety_by_level.png, and diagnostic tables.\n'
                                    'L0 is within the training task distribution; L1/L2 evaluate difficulty generalization.\n')


def compare(baseline, sf, output):
    import csv
    from statistics import mean
    plans = [json.loads((Path(r) / 'plan.json').read_text()) for r in (baseline, sf)]
    for plan in plans: validate_plan(plan, check_files=False)
    if (plans[0]['units'] != plans[1]['units'] or plans[0]['config']['evaluation'] != plans[1]['config']['evaluation']
            or plans[0]['trained_policy']['step'] != plans[1]['trained_policy']['step']
            or plans[0]['trained_policy']['plan_sha256'] != plans[1]['trained_policy']['plan_sha256']
            or [p['trained_policy']['arm'] for p in plans] != ['baseline', 'sf']):
        raise ValueError('Comparison requires identical training budget, data and evaluation episodes')
    rows = [collect(Path(r), p) for r, p in zip((baseline, sf), plans)]
    if any(len(r) != p['expected_episodes'] for r, p in zip(rows, plans)):
        raise ValueError('Both evaluations must be complete')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    groups = []
    for suite in plans[0]['config']['suites']:
        for level in plans[0]['config']['levels']:
            b, s = [[r for r in arm if r['suite'] == suite and r['level'] == level] for arm in rows]
            groups.append(dict(suite=suite, level=level, episodes=len(b),
                               baseline_sr=100*mean(r['success'] for r in b), sf_sr=100*mean(r['success'] for r in s),
                               baseline_cc=mean(r['cost'] for r in b), sf_cc=mean(r['cost'] for r in s)))
    with (output / 'comparison.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(groups[0])); w.writeheader(); w.writerows(groups)
    save(output / 'comparison.json', dict(step=plans[0]['trained_policy']['step'], groups=groups,
                                         baseline=str(baseline), sf=str(sf), training_seed_count=1))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, suite in enumerate(plans[0]['config']['suites']):
        part = [r for r in groups if r['suite'] == suite]
        for j, metric in enumerate(('sr', 'cc')):
            for arm in ('baseline', 'sf'):
                axes[i, j].plot([r['level'] for r in part], [r[f'{arm}_{metric}'] for r in part], marker='o', label=arm)
            axes[i, j].set(title=suite, xlabel='Difficulty level', ylabel='SR (%)' if metric == 'sr' else 'Mean CC')
            axes[i, j].set_xticks([0, 1, 2]); axes[i, j].legend(); axes[i, j].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / 'comparison.png', dpi=160); plt.close(fig)
    (output / 'README.md').write_text('# Matched 5,000-step comparison\n\n'
                                    f"Each arm: global batch 32, {plans[0]['expected_episodes']} episodes, identical task/initial-state schedule.\n"
                                    'See comparison.csv and comparison.png; per-arm reports contain failure and cost attribution.\n'
                                    'One paired training seed and an early checkpoint; convergence is not established.\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    p = argparse.ArgumentParser(description=__doc__); s = p.add_subparsers(dest='command', required=True)
    q = s.add_parser('prepare'); q.add_argument('source'); q.add_argument('pinned'); q.add_argument('output')
    q.add_argument('--arm', required=True, choices=['baseline', 'sf']); q.add_argument('--step', type=int, default=5000)
    q.add_argument('--trials-per-task', type=int, default=30)
    q = s.add_parser('compare'); q.add_argument('baseline'); q.add_argument('sf'); q.add_argument('output')
    for command in ('server', 'client', 'worker', 'report'):
        q = s.add_parser(command); q.add_argument('run', type=Path)
        if command in ('server', 'client'): q.add_argument('--port', type=int, required=True)
        if command == 'client': q.add_argument('--worker', required=True)
    a = p.parse_args()
    if a.command == 'prepare': prepare(a.source, a.pinned, a.output, a.arm, a.step, a.trials_per_task)
    elif a.command == 'compare': compare(a.baseline, a.sf, a.output)
    elif a.command == 'server': server(a.run, a.port)
    elif a.command == 'client': client(a.run, a.port, a.worker)
    else: globals()[a.command](a.run)
