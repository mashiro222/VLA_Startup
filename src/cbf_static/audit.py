"""Small CPU-only, no-render scene integration audit. No model or submission."""
import csv
import json
import numpy as np
import yaml
from .common import ROOT, save


def main():
    import torch
    from vla_arena.vla_arena.envs.env_wrapper import ControlEnv
    from vla_arena.vla_arena.envs.bddl_utils import robosuite_parse_problem
    from .scene import Scene
    cfg=yaml.safe_load((ROOT/'config/cbf_static/experiment.yaml').read_text())
    with (ROOT/'reports/preparation/arena-task-catalog.csv').open() as f:
        tasks=[r for r in csv.DictReader(f) if r['suite']=='safety_static_obstacles']
    catalog=[]; audits=[]; torch.set_num_threads(1)
    for row in tasks:
        p=robosuite_parse_problem(str(ROOT/row['bddl']));objects={x for v in p['objects'].values() for x in v};obstacles=sorted(objects-set(p['obj_of_interest']))
        assert len(obstacles)==int(row['level'])
        states=torch.load(ROOT/row['init_file'],weights_only=False);assert len(states)==50
        catalog.append(dict(level=int(row['level']),task_id=int(row['task_id']),name=row['name'],obstacles=obstacles,targets=p['obj_of_interest'],cost=p['cost_state'],init_count=len(states)))
        if int(row['task_id'])!=0:continue
        env=ControlEnv(bddl_file_name=str(ROOT/row['bddl']),use_camera_obs=False,has_offscreen_renderer=False)
        scene=None
        try:
            env.reset();obs=env.set_init_state(states[0])
            for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
            scene=Scene(env,cfg['filter']);np.testing.assert_allclose(scene.eef(),obs['robot0_eef_pos'],atol=1e-6)
            scene.begin_step();env.step([0.]*6+[-1.]);contacts=scene.contacts();c,a=scene.ellipsoids()
            assert c.shape==(int(row['level']),3);assert a.shape==(int(row['level']),3,3)
            audits.append(dict(level=int(row['level']),**scene.audit,contacts_observed=len(contacts)))
            print('Scene geometry/contact audit PASS:',row['name'],flush=True)
        finally:
            if scene:scene.close()
            env.close()
    save(ROOT/'reports/cbf-static-preparation/scene-audit.json',{'task_catalog':catalog,'representative_physics_scenes':audits,'rendering_verified':False,'model_inference_verified':False})


if __name__=='__main__':main()
