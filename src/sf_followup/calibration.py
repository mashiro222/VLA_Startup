"""Measure real two-GPU optimizer updates at fixed global batch 32."""
import argparse
import copy
import json
from pathlib import Path
import statistics

from arena_startup.core import ROOT, save


def prepare(source, output):
    from sf_training.workflow import verify_plan
    source = Path(source).resolve()
    original = verify_plan(source)
    if original['recipe']['global_batch'] != 32:
        raise ValueError('Calibration requires the paired global batch of 32')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    for micro in (1, 2):
        plan = copy.deepcopy(original)
        plan.update(profile='smoke', gpus=2, time_limit='01:00:00')
        plan['recipe'].update(micro_batch=micro, steps=12, checkpoint_every=12, log_every=1)
        plan['calibration'] = dict(source=str(source), warmup_updates=3,
                                   purpose='Memory and throughput, not an experiment checkpoint')
        save(output / f'mb{micro}' / 'plan.json', plan)


def summarize(output):
    output = Path(output)
    rows = []
    for micro in (1, 2):
        for arm in ('baseline', 'sf'):
            d = output / f'mb{micro}' / arm
            status = json.loads((d / 'status.json').read_text())
            runtime = json.loads((d / 'runtime.json').read_text())
            inference = json.loads((d / 'inference_smoke.json').read_text())
            metrics = [json.loads(x) for x in (d / 'metrics.jsonl').read_text().splitlines()]
            if not (status['completed'] and status['step'] == 12 and len(metrics) == 12
                    and inference['finite'] and runtime['global_batch'] == 32
                    and runtime['micro_batch'] == micro and runtime['world'] == 2):
                raise ValueError(f'Incomplete calibration: {d}')
            if any(not 0 <= x['flow_mse'] <= 1000 for x in metrics):
                raise ValueError('Invalid calibration loss')
            measured = metrics[3:]
            rows.append(dict(arm=arm, micro_batch=micro, accumulation=16 // micro,
                             gpu=runtime['gpu'], total_memory=runtime['total_memory'],
                             measured_updates=len(measured),
                             median_seconds=statistics.median(x['seconds'] for x in measured),
                             mean_seconds=statistics.mean(x['seconds'] for x in measured),
                             peak_allocated_gib=max(x['peak_allocated_gib'] for x in metrics),
                             peak_reserved_gib=max(x['peak_reserved_gib'] for x in metrics)))
    for arm in ('baseline', 'sf'):
        a, b = [r for r in rows if r['arm'] == arm]
        b['speedup_vs_mb1'] = a['median_seconds'] / b['median_seconds']
    result = dict(passed=True, global_batch=32, rows=rows,
                  scope='Two-GPU memory/throughput and finite inference; not a convergence test')
    save(output / 'calibration.json', result)
    report = ROOT / 'reports' / output.name
    save(report / 'calibration.json', result)
    lines = ['# Micro-batch calibration', '',
             '| Arm | Micro-batch | Accumulation | GPU | Seconds/update (median) | Allocated GiB | Reserved GiB |',
             '|---|---:|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['arm']} | {r['micro_batch']} | {r['accumulation']} | {r['gpu']} | "
                     f"{r['median_seconds']:.3f} | {r['peak_allocated_gib']:.2f} | {r['peak_reserved_gib']:.2f} |")
    lines += ['', 'Global batch = 32. First 3 of 12 updates excluded from timing.',
              'Four-GPU cross-node communication is not measured by this job.', '']
    (report / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='command', required=True)
    q = s.add_parser('prepare'); q.add_argument('source'); q.add_argument('output')
    q = s.add_parser('summarize'); q.add_argument('output')
    a = p.parse_args()
    prepare(a.source, a.output) if a.command == 'prepare' else summarize(a.output)
