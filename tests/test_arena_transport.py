"""Real localhost WebSocket regressions, with a fake policy; no GPU or simulator."""
import asyncio
import ctypes
import multiprocessing
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '.runtime/repos/openpi/src'),
               str(ROOT / '.runtime/repos/openpi/packages/openpi-client/src')]
from arena_startup.transport import ArenaClient, ArenaPolicyServer, TRANSPORT_VERSION
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.sync.client
from websockets.exceptions import ConnectionClosedError


METADATA = {'model': 'test-policy', 'transport': TRANSPORT_VERSION}


def serve_fixture(queue, old_heartbeat, policy_delay, policy_error):
    class Policy:
        def infer(self, observation):
            # Deliberately block the server event loop, as a cold JIT can do.
            time.sleep(policy_delay)
            if policy_error:
                raise RuntimeError('test policy exception')
            return {'echo': observation, 'actions': [[0.0] * 7] * 10}
    async def run():
        instance = ArenaPolicyServer(Policy(), host='127.0.0.1', port=0, metadata=METADATA)
        listener = (websockets.asyncio.server.serve(
            instance._handler, '127.0.0.1', 0, compression=None, max_size=None,
            ping_interval=.05, ping_timeout=.1) if old_heartbeat else instance.listener())
        async with listener as server:
            queue.put(server.sockets[0].getsockname()[1])
            await server.serve_forever()
    asyncio.run(run())


def native_pause(seconds=.45):
    # PyDLL retains the GIL; unlike time.sleep, this also pauses the client's
    # receive thread. The independent server can then reproduce a missed Pong.
    usleep = ctypes.PyDLL(None).usleep
    usleep.argtypes = [ctypes.c_uint]
    usleep.restype = ctypes.c_int
    usleep(int(seconds * 1_000_000))


class TransportTests(unittest.TestCase):
    def start(self, old=False, delay=0, error=False):
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        proc = ctx.Process(target=serve_fixture, args=(queue, old, delay, error))
        proc.start()
        def cleanup():
            proc.terminate(); proc.join(10)
            if proc.is_alive(): proc.kill(); proc.join(5)
            queue.close()
        self.addCleanup(cleanup)
        return queue.get(timeout=60)

    def test_old_heartbeat_reproduces_1011_after_native_initialization_pause(self):
        port = self.start(old=True)
        with websockets.sync.client.connect(f'ws://127.0.0.1:{port}', compression=None, close_timeout=1) as ws:
            msgpack_numpy.unpackb(ws.recv(timeout=5))
            native_pause()
            with self.assertRaises(ConnectionClosedError) as error:
                ws.send(msgpack_numpy.packb({'input': 1}))
                ws.recv(timeout=2)
            self.assertIn('keepalive ping timeout', str(error.exception))

    def test_fixed_connection_survives_native_pause_and_slow_policy(self):
        port = self.start(delay=.4)
        with ArenaClient(port, METADATA, request_timeout=5) as client:
            first = client.infer({'__vla_rng': [0, 7], 'input': 1})
            native_pause()
            second = client.infer({'__vla_rng': [0, 8], 'input': 2})
        self.assertEqual(first['echo']['__vla_rng'], [0, 7])
        self.assertEqual(second['echo']['input'], 2)
        self.assertEqual(len(second['actions']), 10)

    def test_constructor_does_not_connect_before_scene_is_ready(self):
        with patch('arena_startup.transport.websockets.sync.client.connect') as connect:
            with ArenaClient(1, METADATA):
                connect.assert_not_called()

    def test_request_timeout_still_rejects_a_stalled_policy(self):
        port = self.start(delay=.5)
        with ArenaClient(port, METADATA, request_timeout=.1) as client:
            with self.assertRaisesRegex(TimeoutError, 'Policy response exceeded'):
                client.infer({'input': 1})

    def test_wrong_checkpoint_metadata_is_rejected(self):
        port = self.start()
        with ArenaClient(port, dict(METADATA, model='wrong-checkpoint')) as client:
            with self.assertRaisesRegex(ValueError, 'identity'):
                client.infer({'input': 1})

    def test_server_exception_is_not_scored_or_retried(self):
        port = self.start(error=True)
        with ArenaClient(port, METADATA) as client:
            with self.assertRaisesRegex(RuntimeError, 'test policy exception'):
                client.infer({'input': 1})


if __name__ == '__main__': unittest.main()
