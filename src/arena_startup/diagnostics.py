"""Observe Arena's existing predicate calls without evaluating predicates twice.

In dynamic tasks _check_cost also stops obstacle motion after contact. Calling it
again to explain a score would change the environment. This observer wraps the
original calls and records their return values; the original method remains the
only source of cost, success and simulator side effects.
"""
import copy
import json
import math

CONTACT_PREDICATES = {'incontact', 'incontactpart', 'checkgrippercontact', 'checkgrippercontactpart'}


def failure_reason(success, done, info):
    if success:
        return None
    if done and info.get('success', False):
        return 'safety_filter_rejected'
    if info.get('timeout', False):
        return 'environment_timeout'
    return 'environment_done_without_success' if done else 'horizon_exhausted'


class PredicateObserver:
    """Observe one episode after settling; all recorded steps are policy steps."""
    version = 'arena-predicate-observer-v1'

    def __init__(self, env, temporal_check, scale=1.0):
        self.env = getattr(env, 'env', env)
        self.temporal_check = temporal_check
        self.scale = scale
        self.context = None
        self.step = 0
        self.events = []
        self.components = {}
        self.goals = []
        self.originals = {}
        for name in ('_eval_predicate', '_check_cost', '_check_final_cost', '_check_success'):
            self.originals[name] = (name in self.env.__dict__, getattr(self.env, name))
        self.env._eval_predicate = self._predicate
        self.env._check_cost = self._cost
        self.env._check_final_cost = self._final_cost
        self.env._check_success = self._success

    def _call(self, method, kind, *args, **kwargs):
        previous = self.context
        captured = []
        self.context = (kind, captured)
        try:
            value = self.originals[method][1](*args, **kwargs)
        finally:
            self.context = previous
        if kind == 'goal':
            self.goals = captured
        else:
            expected = sum(item['raw_cost'] for item in captured)
            # Arena methods return cost before the outer factor of ten.
            if not math.isclose(expected, float(value) * 10, abs_tol=1e-8, rel_tol=1e-8):
                raise ValueError(f'Predicate cost decomposition mismatch: {expected} != {value} * 10')
            self.events.extend(captured)
        return value

    def _predicate(self, state):
        value = self.originals['_eval_predicate'][1](state)
        if self.context is not None:
            kind, items = self.context
            if kind == 'goal':
                items.append({'predicate': copy.deepcopy(state), 'satisfied': bool(value)})
            else:
                temporal = self.temporal_check(state[0])
                weight = 10 * (self.env.temporal_cost_shaping if temporal else 1)
                items.append({'predicate': copy.deepcopy(state), 'phase': kind,
                              'temporal': bool(temporal), 'value': int(value),
                              'raw_cost': int(value) * weight})
        return value

    def _cost(self, *args, **kwargs):
        return self._call('_check_cost', 'step', *args, **kwargs)

    def _final_cost(self, *args, **kwargs):
        return self._call('_check_final_cost', 'timeout_final', *args, **kwargs)

    def _success(self, *args, **kwargs):
        return self._call('_check_success', 'goal', *args, **kwargs)

    def begin_step(self, step):
        self.step = step
        self.events = []

    def consume(self, expected_cost):
        """Check against info['cost'] or timeout final cost and aggregate once."""
        events, self.events = self.events, []
        actual = sum(e['raw_cost'] for e in events)
        if not math.isclose(actual, expected_cost, abs_tol=1e-8, rel_tol=1e-8):
            raise ValueError(f'Observed cost {actual} != authoritative cost {expected_cost}')
        for event in events:
            ident = json.dumps(event['predicate'], separators=(',', ':'))
            if ident not in self.components:
                self.components[ident] = {
                    'predicate': event['predicate'], 'temporal': event['temporal'],
                    'active_steps': 0, 'onsets': 0, 'first_step': None, 'last_step': None,
                    'predicate_value_sum': 0, 'raw_step_cost': 0.0, 'raw_terminal_cost': 0.0,
                    'contact_predicate': event['predicate'][0].lower() in CONTACT_PREDICATES,
                }
            item = self.components[ident]
            active = event['value'] != 0
            if event['temporal'] and active:
                if item['last_step'] != self.step - 1:
                    item['onsets'] += 1
                item['active_steps'] += 1
            if active:
                item['first_step'] = item['first_step'] or self.step
                item['last_step'] = self.step
            item['predicate_value_sum'] += event['value']
            field = 'raw_terminal_cost' if event['phase'] == 'timeout_final' or not event['temporal'] else 'raw_step_cost'
            item[field] += event['raw_cost']
            event['cc'] = event['raw_cost'] * self.scale
        return events

    def summary(self, expected_cc):
        components = list(self.components.values())
        for row in components:
            row['cc'] = (row['raw_step_cost'] + row['raw_terminal_cost']) * self.scale
        if not math.isclose(sum(c['cc'] for c in components), expected_cc, abs_tol=1e-8, rel_tol=1e-8):
            raise ValueError('Episode CC does not equal the sum of predicate contributions')
        return {'version': self.version, 'cost_components': components,
                'final_goal_predicates': self.goals,
                'contact_predicate_onsets': sum(c['onsets'] for c in components if c['contact_predicate']),
                'contact_predicate_active_steps': sum(c['active_steps'] for c in components if c['contact_predicate'])}

    def close(self):
        for name, (was_instance_attribute, original) in self.originals.items():
            if was_instance_attribute:
                setattr(self.env, name, original)
            else:
                delattr(self.env, name)
