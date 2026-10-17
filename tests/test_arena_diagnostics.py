"""Passive instrumentation must preserve upstream scores and dynamic side effects."""
import ast
import copy
import gzip
import json
import math
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from arena_startup.core import ROOT
from arena_startup.diagnostics import PredicateObserver, failure_reason
from arena_startup.adapter import action_chunk, policy_images, policy_state


class PredicateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vla_arena.vla_arena.envs.bddl_base_domain import BDDLBaseDomain
        from vla_arena.vla_arena.envs.predicates import check_temporal_predicate
        cls.domain = BDDLBaseDomain
        cls.temporal = staticmethod(check_temporal_predicate)

    def env(self):
        states = [['incontact', 'apple_1', 'mickey_1'], ['checkgrippercontact', 'mickey_1'], ['spilled', 'bowl_1']]
        env = SimpleNamespace(parsed_problem={'cost_state': states, 'goal_state': [['on', 'apple_1', 'bowl_1']]},
                              temporal_cost_shaping=.1, mocap_joint_names=['mickey_1_main_mocap'],
                              mocap_motion_generators={'mickey_1': object()}, values={}, calls=[])
        def predicate(state):
            env.calls.append(copy.deepcopy(state))
            return env.values.get(state[0], 0)
        env._eval_predicate = predicate
        for method in ('_check_cost', '_check_final_cost', '_check_success'):
            setattr(env, method, MethodType(getattr(self.domain, method), env))
        return env

    def test_exact_upstream_scores_side_effects_and_no_extra_predicate_calls(self):
        reference, observed = self.env(), self.env()
        tracker = PredicateObserver(observed, self.temporal)
        total = 0
        for step, contact in enumerate([0, 1, 1, 0, 1], 1):
            for env in [reference, observed]:
                env.values = {'incontact': contact, 'checkgrippercontact': contact, 'spilled': 3}
            tracker.begin_step(step)
            self.assertEqual(reference._check_success(), observed._check_success())
            a, b = reference._check_cost(False)*10, observed._check_cost(False)*10
            self.assertEqual(a, b)
            tracker.consume(b); total += b
            self.assertEqual(reference.calls, observed.calls)
            self.assertEqual(reference.mocap_joint_names, observed.mocap_joint_names)
            self.assertEqual(list(reference.mocap_motion_generators), list(observed.mocap_motion_generators))
        final = observed._check_final_cost()*10
        self.assertEqual(reference._check_final_cost()*10, final)
        tracker.consume(final)
        result = tracker.summary(total + final)
        self.assertEqual(total + final, 36)
        contact = result['cost_components'][0]
        self.assertEqual((contact['onsets'], contact['active_steps'], contact['cc']), (2, 3, 3))
        self.assertEqual(result['cost_components'][-1]['raw_terminal_cost'], 30)
        self.assertEqual(result['contact_predicate_onsets'], 4)  # Two predicates, not four impacts.
        self.assertEqual(reference.calls, observed.calls)
        tracker.close()
        self.assertFalse(hasattr(observed._check_cost, '__self__') and isinstance(observed._check_cost.__self__, PredicateObserver))

    def test_environment_done_terminal_cost_scaled_once(self):
        env = self.env(); env.values = {'incontact': 1, 'spilled': 2, 'on': True}
        tracker = PredicateObserver(env, self.temporal, scale=.05)
        tracker.begin_step(1)
        env._check_success()
        total = env._check_cost(True)*10
        tracker.consume(total)
        result = tracker.summary(1.05)
        self.assertEqual(total, 21)
        self.assertEqual(result['final_goal_predicates'][0]['satisfied'], True)
        self.assertAlmostEqual(sum(c['cc'] for c in result['cost_components']), 1.05)
        tracker.close()

    def test_mismatched_authoritative_cost_is_not_silently_reported(self):
        tracker = PredicateObserver(self.env(), self.temporal)
        tracker.begin_step(1)
        tracker.env._check_cost(False)
        with self.assertRaisesRegex(ValueError, 'authoritative cost'):
            tracker.consume(9)
        tracker.close()

    def test_termination_labels_distinguish_horizon_timeout_and_filter(self):
        self.assertEqual(failure_reason(False, False, {}), 'horizon_exhausted')
        self.assertEqual(failure_reason(False, True, {'timeout': True}), 'environment_timeout')
        self.assertEqual(failure_reason(False, True, {'success': True}), 'safety_filter_rejected')
        self.assertIsNone(failure_reason(True, True, {'success': True}))


class AdapterTests(unittest.TestCase):
    def test_state_agrees_with_actual_upstream_quaternion_function(self):
        source = (ROOT / '.runtime/repos/vla_arena/vla_arena/models/openpi/evaluator.py').read_text()
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_quat2axisangle')
        namespace = {'np': np, 'math': math}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<pinned-arena-quaternion>', 'exec'), namespace)
        for quat in [np.array([0., 0., 0., 1.]), np.array([0., 0., 1., 0.]),
                     *np.random.default_rng(1).normal(size=(20, 4))]:
            quat = quat / np.linalg.norm(quat)
            obs = {'robot0_eef_pos': np.array([.2, -.3, .4]), 'robot0_eef_quat': quat,
                   'robot0_gripper_qpos': np.array([.01, -.02])}
            expected = np.r_[obs['robot0_eef_pos'], namespace['_quat2axisangle'](quat.copy()), obs['robot0_gripper_qpos']]
            np.testing.assert_allclose(policy_state(obs), expected, atol=1e-8)

    def test_image_orientation_rgb_dtype_and_camera_order(self):
        front = np.arange(8*8*3, dtype=np.uint8).reshape(8,8,3)
        wrist = 255-front
        a,b = policy_images({'agentview_image': front, 'robot0_eye_in_hand_image': wrist}, 8)
        np.testing.assert_array_equal(a, np.rot90(front, 2))
        np.testing.assert_array_equal(b, np.rot90(wrist, 2))
        self.assertEqual(a.dtype, np.uint8); self.assertTrue(a.flags.c_contiguous)

    def test_no_action_scaling_or_gripper_inversion_and_invalid_actions_raise(self):
        actions = np.arange(70, dtype=float).reshape(10,7)/100
        np.testing.assert_array_equal(action_chunk({'actions': actions}, 5), actions)
        for invalid in [np.zeros((10,8)), np.zeros((2,7)), np.full((10,7), np.nan)]:
            with self.assertRaises(ValueError): action_chunk({'actions': invalid}, 5)


class DiagnosticReportTests(unittest.TestCase):
    def test_historical_missing_and_failure_cost_are_preserved(self):
        from arena_startup.diagnostic_report import build_diagnostic_tables
        from arena_startup.report import write_csv
        rows = [{'episode_key': 'u:0', 'suite': 'safety_static_obstacles', 'level': 2,
                 'success': False, 'cost': 99, 'steps': 300, 'termination': 'horizon'}]
        with tempfile.TemporaryDirectory() as d:
            result = build_diagnostic_tables(Path(d), rows, write_csv)
            self.assertEqual(result['episodes_with_predicate_telemetry'], 0)
            self.assertEqual(result['outcomes'][0]['failure_positive_cost'], 1)
            self.assertEqual(result['failure_modes'][0]['termination_reason'], 'horizon_exhausted')
            self.assertEqual(len((Path(d)/'cost_components.csv').read_text().splitlines()), 1)


class RolloutIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vla_arena.vla_arena.envs.bddl_base_domain import BDDLBaseDomain
        cls.domain = BDDLBaseDomain

    env = PredicateTests.env

    def test_rollout_trace_matches_journal_and_records_model_inputs(self):
        from arena_startup.evaluate import rollout
        from vla_arena.vla_arena import envs
        inner = self.env()
        cfg = json.loads((ROOT / 'config/arena.json').read_text())
        cfg['evaluation'].update(max_steps=3, wait_steps=1, replan_steps=2, render_resolution=8, policy_resolution=8)
        actions = np.tile(np.arange(7, dtype=float)/10, (10,1))
        executed = []
        class Env:
            def __init__(self, **kw): self.env = inner; self.n = 0
            def seed(self, seed): pass
            def reset(self): pass
            def set_init_state(self, state): return self.obs()
            def obs(self):
                return {'agentview_image': np.arange(192,dtype=np.uint8).reshape(8,8,3),
                        'robot0_eye_in_hand_image': np.ones((8,8,3),dtype=np.uint8),
                        'robot0_eef_pos': np.zeros(3), 'robot0_eef_quat': np.array([0.,0.,0.,1.]),
                        'robot0_gripper_qpos': np.array([.01,-.01])}
            def step(self, action):
                self.n += 1; executed.append(action)
                inner.values = {'incontact': 1, 'spilled': 2}
                success = inner._check_success()
                return self.obs(), 0., False, {'cost': inner._check_cost(False)*10, 'success': success}
            def get_final_cost(self): return inner._check_final_cost()*10
            def close(self): pass
        class Client:
            def infer(self, obs): return {'actions': actions, '__vla_rng': obs['__vla_rng']}
        class Writer:
            def append_data(self, data): pass
            def close(self): pass
        unit = {'id': 'test-s7', 'suite': 'safety_static_obstacles', 'level': 2, 'task_id': 0,
                'name': 'test', 'prompt': 'test', 'seed': 7, 'bddl': 'unused', 'init_file': 'unused'}
        with tempfile.TemporaryDirectory() as tmp, patch.object(envs, 'OffScreenRenderEnv', Env), \
                patch('torch.load', return_value=[np.zeros(2)]), patch('imageio.v2.get_writer', return_value=Writer()):
            path = Path(tmp)
            row = rollout(unit, 0, {'config': cfg, 'sha256': 'test'}, Client(), path/'0.mp4', path/'diagnostics')
            self.assertEqual(row['cost'], 23)
            self.assertEqual(row['failure_reason'], 'horizon_exhausted')
            self.assertEqual(row['diagnostics']['contact_predicate_onsets'], 1)
            with gzip.open(path/'diagnostics/steps.jsonl.gz', 'rt') as f:
                trace = [json.loads(line) for line in f]
            self.assertEqual(len(trace), 4)
            self.assertEqual(trace[-1]['cc'], row['cost'])
            self.assertEqual(trace[0]['action'], actions[0].tolist())
            self.assertEqual(sum(e['raw_cost'] for r in trace for e in r['cost_predicates']), row['cost'])
            self.assertFalse((path/'diagnostics/steps.partial.jsonl.gz').exists())
            with np.load(path/'diagnostics/policy-call-000.npz') as capture:
                np.testing.assert_array_equal(capture['policy_front'], capture['raw_front'][::-1,::-1])
                np.testing.assert_array_equal(capture['policy_wrist'], capture['raw_wrist'][::-1,::-1])
                np.testing.assert_array_equal(capture['action_chunk'], actions)
        np.testing.assert_array_equal(executed[1:], actions[:3])


if __name__ == '__main__':
    unittest.main()
