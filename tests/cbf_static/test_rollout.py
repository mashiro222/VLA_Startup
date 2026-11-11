"""Exercise real Arena physics + costs + artifact/report plumbing, with a fake policy.

No real inference or rendering occurs. Black images are explicit test fixtures,
stored only in a TemporaryDirectory; they are never experiment evidence.
"""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import yaml
from cbf_static.common import ROOT, record_path, save


class RolloutIntegration(unittest.TestCase):
    def test_real_physics_cost_journal_resume_and_partial_reporting(self):
        from vla_arena.vla_arena.envs.env_wrapper import ControlEnv
        from cbf_static.rollout import rollout
        from cbf_static.plan import episodes
        from cbf_static.report import report, collect
        cfg=yaml.safe_load((ROOT/'config/cbf_static/experiment.yaml').read_text())
        cfg['evaluation']['horizon']=3
        with (ROOT/'reports/preparation/arena-task-catalog.csv').open() as f:catalog=list(csv.DictReader(f))
        eps=episodes('smoke',catalog,cfg)
        selected=[e for e in eps if e['level']==1 and e['init_index']==0 and e['condition'] in ('C0','C2')]

        class Client:
            def infer(self, obs):
                assert obs['observation/image'].shape==(224,224,3)
                assert obs['observation/state'].shape==(8,)
                return {'actions':np.tile([0.,0,0,0,0,0,-1.],(10,1)), '__vla_rng':obs['__vla_rng']}

        class Headless(ControlEnv):
            @staticmethod
            def images(obs):
                for key in ('agentview_image','robot0_eye_in_hand_image'):obs[key]=np.zeros((256,256,3),np.uint8)
                return obs
            def reset(self):return self.images(super().reset())
            def set_init_state(self,state):return self.images(super().set_init_state(state))
            def step(self,a):
                obs,r,d,i=super().step(a);return self.images(obs),r,d,i

        def factory(**kw):return Headless(**kw,use_camera_obs=False,has_offscreen_renderer=False)
        with tempfile.TemporaryDirectory(prefix='cbf-test-') as temp:
            root=Path(temp)
            with patch('vla_arena.vla_arena.envs.OffScreenRenderEnv',side_effect=factory):
                for ep in selected:
                    row=rollout(root,ep,cfg,Client())
                    self.assertEqual(row['n_steps'],3);self.assertFalse(row['success']);self.assertTrue(row['timeout'])
                    p=record_path(root,ep);self.assertTrue((p/'video.mp4').exists());self.assertTrue((p/'contacts.jsonl.gz').exists())
                    with np.load(p/'trace.npz') as d:
                        self.assertEqual(d['h'].shape,(3,1));self.assertEqual(d['action_executed'].shape,(3,7))
                        np.testing.assert_array_equal(d['action_executed'][:,3:],d['action_nominal'][:,3:])
            # An unfinished third episode with only an error must not enter the denominator.
            missing=copy.deepcopy(selected[0]);missing['init_index']=1
            save(record_path(root,missing)/'infrastructure-error.json',{'error':'test-only'})
            save(root/'plan.json',{'profile':'smoke','config':cfg,'expected_episodes':3,'units':[{'episodes':selected+[missing]}]})
            self.assertEqual(len(collect(root)[1]),2)
            report(root)
            self.assertIn('2/3',(root/'summary/README.md').read_text())
            self.assertTrue((root/'summary/paired-induced-stuck.png').exists())
            with (root/'summary/paired-summary.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),1)


if __name__=='__main__':unittest.main()
