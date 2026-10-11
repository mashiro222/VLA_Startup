"""OpenPI JAX policy with explicit RNG state for faithful interrupted-run replay."""
import argparse
import hashlib
import json
import logging
from pathlib import Path

from .manifest import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--num-steps', type=int, default=10)
    args = parser.parse_args()
    import jax
    import numpy as np
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training.config import get_config
    if not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('A CUDA GPU is required; refusing CPU model inference')
    config = get_config('pi05_libero')
    policy = policy_config.create_trained_policy(config, args.checkpoint, sample_kwargs={'num_steps': args.num_steps})
    if not hasattr(policy, '_rng'):
        raise RuntimeError('Pinned JAX policy interface changed')

    class ReplayablePolicy:
        def infer(self, observation):
            observation = dict(observation)
            state = observation.pop('__vla_rng', None)
            if state is None or len(state) != 2 or any(not 0 <= int(x) < 2 ** 32 for x in state):
                raise ValueError('Client must supply the two uint32 words of the JAX RNG state')
            # Restoring the previous key preserves upstream split/key(0) behavior.
            # It does not introduce per-task seeds or a different sampling scheme.
            policy._rng = jax.random.wrap_key_data(np.asarray(state, dtype=np.uint32))
            result = policy.infer(observation)
            result['__vla_rng'] = np.asarray(jax.random.key_data(policy._rng)).tolist()
            return result

    sources = json.loads((ROOT / 'config/sources.json').read_text())
    metadata = {'model': 'pi05_libero', 'backend': 'jax', 'num_steps': args.num_steps,
                'action_horizon': config.model.action_horizon, 'initial_rng': [0, 0],
                'openpi_commit': sources['openpi']['commit'],
                'checkpoint_manifest_sha256': hashlib.sha256((ROOT / 'config/checkpoint_files.json').read_bytes()).hexdigest()}
    logging.info('Serving checkpoint %s with %s', args.checkpoint, metadata)
    WebsocketPolicyServer(ReplayablePolicy(), host='127.0.0.1', port=args.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    main()
