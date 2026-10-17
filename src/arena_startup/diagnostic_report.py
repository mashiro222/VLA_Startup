"""Failure modes and cost attribution; missing historical telemetry stays missing."""
from collections import Counter, defaultdict
import json


def reason(row):
    if row['success']:
        return 'success'
    if row.get('failure_reason'):
        return row['failure_reason']
    return {'horizon': 'horizon_exhausted', 'environment_done': 'legacy_environment_done'}.get(
        row.get('termination'), 'not_recorded')


def category(predicate):
    name = predicate[0].lower()
    if name in ('checkgrippercontact', 'checkgrippercontactpart'):
        return 'gripper_contact'
    if name in ('incontact', 'incontactpart'):
        return 'object_contact'
    if 'distance' in name:
        return 'proximity_threshold'
    return name


def build_diagnostic_tables(output, rows, write_csv):
    failures = Counter((r['suite'], r['level'], reason(r)) for r in rows)
    failure_rows = [{'suite': s, 'level': l, 'termination_reason': why, 'episodes': n}
                    for (s, l, why), n in sorted(failures.items())]
    write_csv(output / 'failure_modes.csv', failure_rows)
    outcomes = []
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['suite'], r['level']].append(r)
    for (suite, level), part in sorted(grouped.items()):
        with_diagnostics = sum(bool(r.get('diagnostics')) for r in part)
        outcomes.append({'suite': suite, 'level': level, 'episodes': len(part),
                         'diagnostic_episodes': with_diagnostics,
                         'success_zero_cost': sum(r['success'] and r['cost'] == 0 for r in part),
                         'success_positive_cost': sum(r['success'] and r['cost'] > 0 for r in part),
                         'failure_zero_cost': sum(not r['success'] and r['cost'] == 0 for r in part),
                         'failure_positive_cost': sum(not r['success'] and r['cost'] > 0 for r in part),
                         'mean_steps': sum(r.get('steps', 0) for r in part) / len(part),
                         'cc_per_100_control_steps': 100 * sum(r['cost'] for r in part) / sum(r['steps'] for r in part)
                         if all(r.get('steps') for r in part) else None})
    write_csv(output / 'outcomes.csv', outcomes)
    components = []
    for r in rows:
        for c in (r.get('diagnostics') or {}).get('cost_components', []):
            components.append({'episode_key': r['episode_key'], 'suite': r['suite'], 'level': r['level'],
                               'category': category(c['predicate']), **c,
                               'predicate': json.dumps(c['predicate'], separators=(',', ':'))})
    fields = ['episode_key', 'suite', 'level', 'category', 'predicate', 'temporal', 'active_steps', 'onsets',
              'first_step', 'last_step', 'predicate_value_sum', 'raw_step_cost', 'raw_terminal_cost',
              'contact_predicate', 'cc']
    write_csv(output / 'cost_components.csv', components, fields)
    grouped_components = defaultdict(list)
    for c in components:
        grouped_components[c['suite'], c['level'], c['category']].append(c)
    attribution = []
    for (suite, level, cat), part in sorted(grouped_components.items()):
        n = sum(bool(r.get('diagnostics')) for r in grouped[suite, level])
        attribution.append({'suite': suite, 'level': level, 'category': cat, 'diagnostic_episodes': n,
                            'cc_total': sum(c['cc'] for c in part), 'cc_mean': sum(c['cc'] for c in part) / n,
                            'predicate_onsets_total': sum(c['onsets'] for c in part),
                            'predicate_active_steps_total': sum(c['active_steps'] for c in part)})
    write_csv(output / 'cost_by_category.csv', attribution,
              ['suite', 'level', 'category', 'diagnostic_episodes', 'cc_total', 'cc_mean',
               'predicate_onsets_total', 'predicate_active_steps_total'])
    lines = ['# Episode termination and safety cost', '',
             f"Predicate telemetry: {sum(bool(r.get('diagnostics')) for r in rows)} / {len(rows)} episodes.", '',
             'Missing historical telemetry is not reconstructed from videos and is not treated as zero.',
             'A horizon termination describes the stopping rule, not the physical cause of failure.',
             'Contact onsets count false-to-true transitions per benchmark predicate at 20 Hz. They are not unique physical impacts.',
             'Active steps accumulate contact duration and can double count overlapping predicates.',
             'Zero cost means no recorded benchmark violation; it does not imply task completion or general safety.', '',
             '| Suite | Level | Reason | Episodes |', '|---|---:|---|---:|']
    lines += [f"| {r['suite']} | {r['level']} | {r['termination_reason']} | {r['episodes']} |" for r in failure_rows]
    (output / 'diagnostics.md').write_text('\n'.join(lines) + '\n')
    if attribution:
        import matplotlib.pyplot as plt
        labels = sorted({(r['suite'], r['level']) for r in attribution})
        categories = sorted({r['category'] for r in attribution})
        fig, ax = plt.subplots(figsize=(12, 5))
        bottom = [0.] * len(labels)
        for cat in categories:
            values = [sum(r['cc_mean'] for r in attribution if (r['suite'], r['level']) == label and r['category'] == cat)
                      for label in labels]
            ax.bar(range(len(labels)), values, bottom=bottom, label=cat)
            bottom = [a+b for a, b in zip(bottom, values)]
        ax.set_xticks(range(len(labels)), [f"{s.removeprefix('safety_')} L{l}" for s, l in labels], rotation=65, ha='right')
        ax.set_ylabel('Mean CC contribution (observed episodes)')
        ax.legend(); fig.tight_layout(); fig.savefig(output / 'cost_attribution.png', dpi=160); plt.close(fig)
    return {'episodes_with_predicate_telemetry': sum(bool(r.get('diagnostics')) for r in rows),
            'failure_modes': failure_rows, 'outcomes': outcomes}
