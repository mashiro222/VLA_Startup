"""Official Arena OpenPI policy, checkpoint normalization and per-episode JAX RNG."""
import argparse
import json
import logging
from .common import ROOT, digest


def main():
    p = argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('--port', type=int, required=True)
    args = p.parse_args(); plan = json.loads((ROOT / args.run / 'plan.json').read_text()); cfg = plan['config']
    import jax
    import numpy as np
    from vla_arena.models.openpi.src.openpi.training.config import get_config
    from vla_arena.models.openpi.src.openpi.policies.policy_config import create_trained_policy
    from vla_arena.models.openpi.src.openpi.shared import normalize
    from arena_startup.transport import ArenaPolicyServer
    if not any(x.platform == 'gpu' for x in jax.devices()): raise RuntimeError('GPU required')
    checkpoint = ROOT / cfg['checkpoint']; norm_path = checkpoint / cfg['norm_stats']
    # Explicit official checkpoint stats prevent fallback to a different training dataset.
    stats = normalize.load(norm_path.parent)
    config = get_config(cfg['policy_config'])
    policy = create_trained_policy(config, checkpoint, norm_stats=stats,
                                   sample_kwargs={'num_steps': cfg['evaluation']['num_steps']})

    class Seeded:
        def infer(self, obs):
            obs = dict(obs); rng = np.asarray(obs.pop('__vla_rng'), dtype=np.uint32)
            if rng.shape != (2,): raise ValueError('Expected JAX key words')
            policy._rng = jax.random.wrap_key_data(rng)
            result = policy.infer(obs)
            result['__vla_rng'] = np.asarray(jax.random.key_data(policy._rng)).tolist()
            return result

    metadata = {'model': cfg['policy_config'], 'plan_sha256': digest(ROOT / args.run / 'plan.json'),
                'norm_sha256': digest(norm_path), 'checkpoint_revision': json.loads((ROOT / 'config/cbf_static/checkpoint.json').read_text())['revision']}
    logging.info('Official policy metadata: %s', metadata)
    ArenaPolicyServer(Seeded(), host='127.0.0.1', port=args.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True); main()
