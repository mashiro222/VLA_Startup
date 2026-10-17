"""Protocol tests: no GPU, policy weights, or real simulator rollouts."""
import copy
import argparse
import importlib.util
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arena_startup.core import (ROOT, append, claim, collect, digest, episode_seed, finish_metrics,
                               frozen_files, key, records, save, validate_plan, validate_rows)


def try_claim(path, queue):
    with claim(Path(path)) as acquired:
        queue.put(acquired)


class JournalTests(unittest.TestCase):
    def test_timeout_and_environment_timeout_are_not_success(self):
        self.assertEqual(finish_metrics('safety_static_obstacles', 3, 20, False, {}), (False, 23))
        self.assertEqual(finish_metrics('safety_static_obstacles', 23, 20, True, {'success': False}), (False, 23))

    def test_goal_success_can_have_cost_and_hazard_is_scaled_once(self):
        self.assertEqual(finish_metrics('safety_static_obstacles', 23, 20, True, {'success': True}), (True, 23))
        self.assertEqual(finish_metrics('safety_hazard_avoidance', 20, 10, False, {}), (False, 1.5))
        self.assertEqual(finish_metrics('safety_hazard_avoidance', 20, 10, True, {'success': True}), (True, 1))
        with self.assertRaises(ValueError): finish_metrics('s', float('nan'), 0, True, {})

    def test_seeds_do_not_depend_on_worker_or_order(self):
        u = {'id': 'task-s7', 'worker': 0}
        a = [episode_seed(u, i) for i in range(10)]
        u['worker'] = 3
        self.assertEqual(a[::-1], [episode_seed(u, i) for i in reversed(range(10))])
        self.assertEqual(len(set(a)), 10)
        self.assertNotEqual(a[0], episode_seed({'id': 'task-s17'}, 0))

    def test_interrupted_tail_and_wrong_run_are_rejected(self):
        unit = {'id': 'u', 'trials': 2}; plan = {'sha256': 'abc'}
        row = {'episode_key': 'u:0', 'plan_sha256': 'abc', 'success': False, 'cost': 2}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'episodes.jsonl'
            append(p, row)
            with p.open('ab') as f: f.write(b'{"interrupted":')
            self.assertEqual(records(p), [row])
            self.assertEqual(records(p, repair=True), [row])
            validate_rows(records(p), unit, plan)
            with self.assertRaises(ValueError): validate_rows([row, row], unit, plan)
            with self.assertRaises(ValueError): validate_rows([row], unit, {'sha256': 'other'})

    def test_claim_exclusion_and_release_across_processes(self):
        ctx = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as tmp:
            queue = ctx.Queue()
            with claim(Path(tmp)) as acquired:
                self.assertTrue(acquired)
                proc = ctx.Process(target=try_claim, args=(tmp, queue)); proc.start()
                self.assertFalse(queue.get(timeout=10)); proc.join(10)
                self.assertEqual(proc.exitcode, 0)
            proc = ctx.Process(target=try_claim, args=(tmp, queue)); proc.start()
            self.assertTrue(queue.get(timeout=10)); proc.join(10)
            self.assertEqual(proc.exitcode, 0)


class RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from arena_startup.evaluate import rollout
        from vla_arena.vla_arena import envs
        cls.rollout = staticmethod(rollout)
        cls.envs = envs
        cls.np = np

    def fixture(self, final_success=False, error=False):
        np = self.np
        cfg = json.loads((ROOT / 'config/arena.json').read_text())
        cfg['evaluation'].update(max_steps=3, wait_steps=1, replan_steps=2, render_resolution=8, policy_resolution=8,
                                 predicate_diagnostics=False)
        unit = {'id': 'test-s7', 'suite': 'safety_static_obstacles', 'level': 0, 'task_id': 0,
                'name': 'test', 'prompt': 'test', 'seed': 7, 'bddl': 'unused', 'init_file': 'unused'}
        class FakeEnv:
            closed = False
            def __init__(self, **kwargs): self.n = 0
            def seed(self, value): pass
            def reset(self): pass
            def set_init_state(self, state): return self.obs()
            def obs(self):
                return {'agentview_image': np.zeros((8, 8, 3), dtype=np.uint8),
                        'robot0_eye_in_hand_image': np.zeros((8, 8, 3), dtype=np.uint8),
                        'robot0_eef_pos': np.zeros(3), 'robot0_eef_quat': np.array([0, 0, 0, 1]),
                        'robot0_gripper_qpos': np.zeros(2)}
            def step(self, action):
                self.n += 1
                if error and self.n > 1: raise RuntimeError('synthetic simulator fault')
                done = self.n == 4 and final_success
                return self.obs(), 0, done, {'success': done, 'cost': 100 if self.n == 1 else 2}
            def get_final_cost(self): return 10
            def close(self): FakeEnv.closed = True
        class Client:
            def infer(self, request): return {'actions': np.zeros((10, 7)), '__vla_rng': request['__vla_rng']}
        return unit, {'config': cfg, 'sha256': 'test'}, FakeEnv, Client()

    def test_rollout_counts_timeout_cost_and_ignores_warmup_cost(self):
        unit, plan, env, client = self.fixture()
        with patch.object(self.envs, 'OffScreenRenderEnv', env), patch('torch.load', return_value=[self.np.zeros(2)]):
            row = self.rollout(unit, 0, plan, client)
        self.assertFalse(row['success']); self.assertEqual(row['cost'], 16)
        self.assertEqual(row['inference_calls'], 2); self.assertEqual(row['steps'], 3)
        self.assertTrue(env.closed)

    def test_rollout_goal_does_not_double_count_final_cost(self):
        unit, plan, env, client = self.fixture(final_success=True)
        with patch.object(self.envs, 'OffScreenRenderEnv', env), patch('torch.load', return_value=[self.np.zeros(2)]):
            row = self.rollout(unit, 0, plan, client)
        self.assertTrue(row['success']); self.assertEqual(row['cost'], 6)

    def test_infrastructure_exception_propagates_instead_of_scoring_failure(self):
        unit, plan, env, client = self.fixture(error=True)
        with patch.object(self.envs, 'OffScreenRenderEnv', env), patch('torch.load', return_value=[self.np.zeros(2)]):
            with self.assertRaisesRegex(RuntimeError, 'synthetic simulator fault'):
                self.rollout(unit, 0, plan, client)
        self.assertTrue(env.closed)


class ReportTests(unittest.TestCase):
    def test_historical_plan_is_readable_but_cannot_execute_changed_code(self):
        plan = {'units': [], 'expected_episodes': 0, 'files': {'old.py': 'old-hash'}}
        plan['sha256'] = digest(plan)
        validate_plan(plan, check_files=False)
        with self.assertRaisesRegex(ValueError, 'Frozen inputs changed'):
            validate_plan(plan)
        plan['expected_episodes'] = 1
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_plan(plan, check_files=False)

    def test_partial_coverage_refused_then_complete_table_and_plot(self):
        from arena_startup.report import build
        cfg = json.loads((ROOT / 'config/arena.json').read_text())
        unit = {'id': 'u', 'trials': 2, 'suite': cfg['suites'][0], 'level': 0}
        plan = {'run_id': 'arena-synthetic-test', 'profile': 'smoke', 'config': cfg,
                'units': [unit], 'expected_episodes': 2, 'files': frozen_files(), 'training_condition': 'SYNTHETIC TEST'}
        plan['sha256'] = digest(plan)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'; out = Path(tmp) / 'report'
            save(run / 'plan.json', plan)
            dest = run / 'units/u'; dest.mkdir(parents=True)
            row = {'episode_key': 'u:0', 'plan_sha256': plan['sha256'], 'success': True, 'cost': 4,
                   'suite': unit['suite'], 'level': 0, 'seed': 7}
            append(dest / 'episodes.jsonl', row)
            with self.assertRaisesRegex(ValueError, 'Incomplete'): build(run, output=out)
            row.update(episode_key='u:1', success=False, cost=8)
            append(dest / 'episodes.jsonl', row)
            result = build(run, output=out)
            self.assertTrue(result['complete']); self.assertEqual(result['sr_percent'], 50); self.assertEqual(result['cc'], 6)
            for name in ['summary.csv', 'by_seed.csv', 'episodes.csv', 'safety_by_level.png', 'cumulative_success.png', 'sha256.json']:
                self.assertTrue((out / name).is_file())


class SubmissionTests(unittest.TestCase):
    def test_dry_run_never_contacts_slurm_and_full_requires_smoke(self):
        spec = importlib.util.spec_from_file_location('arena_cli_test', ROOT / 'scripts/arena.py')
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        args = argparse.Namespace(run_id='arena-test', gpu='a100', time_limit=None,
                                  smoke_run='arena-smoke-test', submit=False)
        plan = {'expected_episodes': 15, 'workers': 4, 'time_limit': '02:00:00', 'profile': 'full'}
        with patch.object(cli, 'preflight'), patch.object(cli, 'load', return_value=plan), \
             patch.object(cli, 'collect', return_value=[]), \
             patch.object(cli.subprocess, 'run') as launch, patch.object(cli.subprocess, 'check_output') as query:
            cli.submit(args)
            launch.assert_not_called(); query.assert_not_called()
            args.submit = True
            with patch.object(cli, 'require_smoke', side_effect=ValueError('smoke incomplete')):
                with self.assertRaisesRegex(ValueError, 'smoke incomplete'): cli.submit(args)
            launch.assert_not_called(); query.assert_not_called()


if __name__ == '__main__': unittest.main()
