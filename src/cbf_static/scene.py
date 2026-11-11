"""GT geometry, controller assertions, and non-invasive physics contact tracing."""
import numpy as np
from .geometry import mvee, inflate, geom_vertices


class Scene:
    def __init__(self, env, cfg):
        self.env = env.env; self.sim = env.sim; self.robot = env.robots[0]
        self.controller = self.robot.part_controllers['right']
        c = self.controller
        if (type(c).__name__ != 'OperationalSpaceController' or c.input_type != 'delta'
                or not np.allclose(c.input_min[:3], -1) or not np.allclose(c.input_max[:3], 1)
                or not np.allclose(c.output_min[:3], -.05) or not np.allclose(c.output_max[:3], .05)
                or c._goal_update_mode != 'achieved'):
            raise ValueError('Controller differs from audited OSC_POSE delta configuration')
        self.rotation = np.eye(3) if c.input_ref_frame == 'world' else np.asarray(c.origin_ori).copy()
        if self.rotation.shape != (3, 3) or not np.allclose(self.rotation.T @ self.rotation, np.eye(3)):
            raise ValueError('Invalid controller frame transform')
        parsed = self.env.parsed_problem
        self.targets = list(parsed['obj_of_interest'])
        goal = parsed['goal_state']
        if len(goal) != 1 or goal[0][0] not in ('on', 'in') or len(goal[0]) != 3:
            raise ValueError('Expected one object-to-receptacle placement goal')
        self.target, self.receptacle = goal[0][1:]
        if set(self.targets) != {self.target, self.receptacle}:
            raise ValueError('Goal does not match obj_of_interest')
        objects = self.env.objects_dict
        self.names = sorted(set(objects) - set(self.targets))
        cost_objects = {x for predicate in parsed['cost_state'] for x in predicate[1:] if x in objects} - set(self.targets)
        if cost_objects != set(self.names):
            raise ValueError(f'Geometry/cost obstacle mismatch: {self.names} vs {cost_objects}')
        self.body_ids = {name: self.sim.model.body_name2id(o.root_body) for name, o in objects.items()}
        self.geom_objects = {self.sim.model.geom_name2id(g): name for name, o in objects.items() for g in o.contact_geoms}
        gripper = self.robot.gripper['right']
        self.gripper_ids = {self.sim.model.geom_name2id(g) for g in gripper.contact_geoms}
        self.robot_ids = {self.sim.model.geom_name2id(g) for g in self.robot.robot_model.contact_geoms} | self.gripper_ids
        self.site = self.sim.model.site_name2id(gripper.important_sites['grip_site'])
        self.local = []
        self.audit = {'obstacles': self.names, 'targets': self.targets, 'cost_predicates': parsed['cost_state'],
                      'controller_scale_m': .05, 'controller_frame': c.input_ref_frame,
                      'controller_rotation_to_world': self.rotation.tolist(), 'geometry': {}}
        for name in self.names:
            bid = self.body_ids[name]
            pts = np.concatenate([geom_vertices(self.sim, self.sim.model.geom_name2id(g)) for g in objects[name].contact_geoms])
            r = self.sim.data.body_xmat[bid].reshape(3, 3); p = self.sim.data.body_xpos[bid]
            local_pts = (pts - p) @ r
            center, a = mvee(local_pts, cfg['mvee_tolerance'])
            a = inflate(a, cfg['gripper_radius_m'] + cfg['margin_m'])
            self.local.append((bid, center, a))
            self.audit['geometry'][name] = {'vertices': len(pts), 'body_id': int(bid), 'center_local': center.tolist(), 'inflated_A_local': a.tolist()}
        self.grasp = lambda: bool(self.env._check_grasp(gripper, objects[self.target].contact_geoms))
        self.original_step = self.sim.step
        self.pairs = {}
        self.sim.step = self._physics_step

    def position(self, name):
        return self.sim.data.body_xpos[self.body_ids[name]].copy()

    def eef(self):
        return self.sim.data.site_xpos[self.site].copy()

    def ellipsoids(self):
        centers, matrices = [], []
        for bid, local_c, local_a in self.local:
            r = self.sim.data.body_xmat[bid].reshape(3, 3)
            centers.append(r @ local_c + self.sim.data.body_xpos[bid])
            matrices.append(r @ local_a @ r.T)
        return np.asarray(centers).reshape(-1, 3), np.asarray(matrices).reshape(-1, 3, 3)

    def begin_step(self):
        self.pairs = {}; self.held_before = self.grasp()

    def _capture(self):
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]; a, b = sorted((int(c.geom1), int(c.geom2)))
            oa, ob = self.geom_objects.get(a), self.geom_objects.get(b)
            if not ({a, b} & self.robot_ids or self.target in (oa, ob)):
                continue
            key = (a, b)
            if key not in self.pairs:
                self.pairs[key] = {'geom_ids': [a, b], 'geom_names': [self.sim.model.geom_id2name(x) for x in key],
                    'bodies': [self.sim.model.body_id2name(int(self.sim.model.geom_bodyid[x])) for x in key],
                    'objects': [oa, ob], 'physics_samples': 0, 'first_sim_time': float(self.sim.data.time),
                    'gripper': bool({a, b} & self.gripper_ids),
                    'robot_link': bool(({a, b} & self.robot_ids) - self.gripper_ids),
                    'robot': bool({a, b} & self.robot_ids),
                    'obstacle': oa in self.names or ob in self.names,
                    'target_object': self.target in (oa, ob)}
            row = self.pairs[key]; row['physics_samples'] += 1
            row['last_sim_time'] = float(self.sim.data.time)

    def _physics_step(self, *args, **kwargs):
        value = self.original_step(*args, **kwargs)
        self._capture()
        return value

    def contacts(self):
        self._capture()
        held = self.held_before or self.grasp()
        for row in self.pairs.values():
            row['held_object'] = bool(held and row['target_object'] and not row['robot'])
            row['main_collision'] = bool(row['obstacle'] and (row['robot'] or row['held_object']))
        return list(self.pairs.values())

    def close(self):
        self.sim.step = self.original_step
