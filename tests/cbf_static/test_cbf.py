import copy
import csv
import tempfile
from pathlib import Path
import unittest
import numpy as np
import yaml
from cbf_static.common import ROOT, episode_seed, save
from cbf_static.geometry import barrier, inflate, mvee, nearest_segment
from cbf_static.filter import filter_action
from cbf_static.metrics import classify, fit_scale
from cbf_static.plan import episodes
from cbf_static.report import wilson

CFG = yaml.safe_load((ROOT/'config/cbf_static/experiment.yaml').read_text())


class GeometryTests(unittest.TestCase):
    def test_mvee_encloses_mesh_and_is_rigid_transform_equivariant(self):
        rng=np.random.default_rng(3); pts=rng.normal(size=(60,3))*[.1,.2,.3]+[1,2,3]
        c,a=mvee(pts); self.assertTrue(np.all(barrier(pts[0], [c], [a])[0]<=1e-8))
        self.assertLessEqual(np.max(np.einsum('ni,ij,nj->n',pts-c,a,pts-c)),1+1e-8)
        r=np.array([[0.,-1,0],[1,0,0],[0,0,1]]); c2,a2=mvee(pts@r.T)
        np.testing.assert_allclose(c2,r@c,atol=1e-6);np.testing.assert_allclose(a2,r@a@r.T,atol=1e-5)

    def test_minkowski_outer_bound(self):
        rng=np.random.default_rng(5); q=np.diag([.01,.04,.25]); a=inflate(np.linalg.inv(q),.06)
        v=rng.normal(size=(20000,3));v/=np.linalg.norm(v,axis=1)[:,None]
        w=rng.normal(size=(20000,3));w/=np.linalg.norm(w,axis=1)[:,None]
        pts=v@np.sqrt(q)+.06*w
        self.assertLessEqual(np.max(np.einsum('ni,ij,nj->n',pts,a,pts)),1+1e-9)

    def test_barrier_gradient(self):
        p=np.array([.4,.3,-.2]);c=np.array([[.1,.2,0.]]);a=np.array([np.diag([2.,3.,5.])]);h,g=barrier(p,c,a)
        numeric=np.array([(barrier(p+np.eye(3)[i]*1e-6,c,a)[0][0]-barrier(p-np.eye(3)[i]*1e-6,c,a)[0][0])/2e-6 for i in range(3)])
        np.testing.assert_allclose(g[0],numeric,rtol=1e-7)

    def test_fixed_obstacle_selection_distance_to_segment(self):
        self.assertEqual(nearest_segment([[.9,.01,0],[0,.1,0]],[0,0,0],[1,0,0]),0)


class FilterTests(unittest.TestCase):
    def test_empty_and_C0_are_identity_including_outside_box(self):
        u=np.array([1.2,-1.3,.5,.2,.3,.4,-1.])
        for enabled,c,a,idx in [(True,[],[],[]),(False,[[0,0,0]],[np.eye(3)],[0])]:
            out,_=filter_action(u,[.3,0,0],c,a,idx,CFG['filter'],enabled)
            np.testing.assert_array_equal(out,u)

    def test_single_sphere_forward_invariance_and_mujoco_no_contact(self):
        import mujoco
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1"/><body mocap="true" name="grip"><geom type="sphere" size="0.05"/></body></worldbody></mujoco>')
        data=mujoco.MjData(model);p=np.array([-.4,0,0]);a=inflate(np.eye(3)/.1**2,.06)
        intervened=False
        for _ in range(150):
            out,info=filter_action([1,0,0,.1,.2,.3,-1],p,[[0,0,0]],[a],[0],CFG['filter'])
            p+=.05*out[:3];intervened|=info['intervention_m']>0
            self.assertGreaterEqual(barrier(p,[[0,0,0]],[a])[0][0],-1e-6)
            data.mocap_pos[0]=p;mujoco.mj_forward(model,data);self.assertEqual(data.ncon,0)
            np.testing.assert_array_equal(out[3:],[.1,.2,.3,-1])
        self.assertTrue(intervened)

    def test_two_obstacles_feasible_gap(self):
        p=np.array([-.4,0,0.]); c=[[0,-.25,0],[0,.25,0]];a=[np.eye(3)/.16**2]*2
        for _ in range(20):
            out,info=filter_action([1,0,0,0,0,0,-1],p,c,a,[0,1],CFG['filter']);p+=.05*out[:3]
            self.assertFalse(info['solver_failed']);self.assertTrue(np.all(barrier(p,c,a)[0]>=-1e-6))
        self.assertGreater(p[0],.3)

    def test_controller_frame_rotation_is_applied(self):
        r=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        p=np.array([-.17,0,0]); c=[[0,0,0]];a=[np.eye(3)/.16**2]
        out,info=filter_action([0,-1,0,0,0,0,-1],p,c,a,[0],CFG['filter'],rotation=r)
        self.assertGreater(info['intervention_m'],0);self.assertGreaterEqual(barrier(p+.05*r@out[:3],c,a)[0][0],-1e-6)

    def test_infeasible_is_explicit_and_zero_is_not_claimed_safe(self):
        out,info=filter_action([1,0,0,0,0,0,1],[0,0,0],[[0,0,0]],[np.eye(3)],[0],CFG['filter'])
        self.assertTrue(info['solver_failed']);self.assertFalse(info['fallback_satisfies_linear_cbf']);np.testing.assert_array_equal(out[:3],0)


class ProtocolTests(unittest.TestCase):
    def test_counts_and_pairing(self):
        with (ROOT/'reports/preparation/arena-task-catalog.csv').open() as f: catalog=list(csv.DictReader(f))
        for profile,n in [('calibration',10),('smoke',40),('main',3000),('sanity',1000),('gamma',2000)]:
            eps=episodes(profile,catalog,CFG);self.assertEqual(len(eps),n)
        paired=episodes('main',catalog,CFG); groups={}
        for e in paired:groups.setdefault((e['level'],e['task_id'],e['init_index'],e['repeat']),[]).append(e)
        for g in groups.values():self.assertEqual(len({e['seed'] for e in g}),1);self.assertEqual(len(g),3)
        self.assertNotEqual(episode_seed(1,0,0,0),episode_seed(1,0,0,1))

    def trace(self):
        return dict(eef_after=np.zeros((100,3)),gripper_qpos=np.zeros((100,2)),collision=np.zeros(100,bool),
                    intervention_m=np.ones(100)*.01,dist_subgoal_m=np.ones(100)*.05,n_active=np.ones(100))

    def test_stuck_success_timeout_collision_are_independent(self):
        d=self.trace();cfg=CFG['diagnostics'];r=classify(d,False,True,cfg,20)
        self.assertTrue(r['safe_stuck']);self.assertTrue(r['filter_associated_stuck']);self.assertEqual(r['stuck_window_start_step'],40)
        self.assertFalse(classify(d,True,False,cfg,20)['stuck'])
        self.assertTrue(classify(d,True,False,cfg,20)['tail_stationary_on_success'])
        d['collision'][3]=True;r=classify(d,False,True,cfg,20)
        self.assertTrue(r['collision_before_stuck']);self.assertFalse(r['safe_stuck'])
        d['collision'][:]=False;d['collision'][60]=True
        self.assertTrue(classify(d,False,True,cfg,20)['collision_during_stuck'])

    def test_gripper_changes_and_wandering_exclude_stuck(self):
        d=self.trace();d['gripper_qpos'][-1]=.03
        self.assertTrue(classify(d,False,True,CFG['diagnostics'],20)['wandering_timeout'])
        d=self.trace();d['eef_after'][:,0]=np.arange(100)*.01
        self.assertFalse(classify(d,False,True,CFG['diagnostics'],20)['stuck'])

    def test_scale_fit_and_rejection(self):
        rng=np.random.default_rng(8);u=rng.uniform(-1,1,(400,3));d=.02*u
        fit=fit_scale(u,d,CFG['calibration']);self.assertTrue(fit['passed']);self.assertAlmostEqual(fit['action_scale_m'],.02)
        self.assertFalse(fit_scale(u,d+rng.normal(0,.1,d.shape),CFG['calibration'])['passed'])
        self.assertFalse(fit_scale(u[:5],d[:5],CFG['calibration'])['passed'])
        self.assertFalse(fit_scale([],[],CFG['calibration'])['passed'])

    def test_wilson_extremes_and_atomic_records(self):
        low,high=wilson(0,100);self.assertAlmostEqual(low,0);self.assertGreater(high,0)
        with tempfile.TemporaryDirectory() as d:
            save(Path(d)/'row.json',{'success':False});self.assertEqual(len(list(Path(d).iterdir())),1)


if __name__=='__main__':unittest.main()
