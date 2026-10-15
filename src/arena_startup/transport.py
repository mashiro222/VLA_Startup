"""Local OpenPI wire protocol, tolerant of simulator/JIT pauses and bounded by deadlines."""
import inspect
import logging

from openpi_client import msgpack_numpy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer, _health_check
import websockets.asyncio.server
import websockets.sync.client

TRANSPORT_VERSION = 'arena-local-no-heartbeat-v1'


class ArenaPolicyServer(WebsocketPolicyServer):
    """Retain upstream metadata, inference handler, errors and health endpoint.

    A slow native simulator call may hold the client's GIL. JAX compilation can
    similarly block the server event loop. Neither is a dead network connection.
    The client request deadline and episode/Slurm limits provide liveness bounds.
    """
    def listener(self):
        return websockets.asyncio.server.serve(
            self._handler, self._host, self._port, compression=None, max_size=None,
            process_request=_health_check, ping_interval=None, ping_timeout=None)

    async def run(self):
        async with self.listener() as server:
            await server.serve_forever()


class ArenaClient:
    """Connect lazily on first inference, after scene setup; never retry an action."""
    def __init__(self, port, expected_metadata, request_timeout=600):
        if request_timeout <= 0:
            raise ValueError('Request timeout must be positive')
        self.uri = f'ws://127.0.0.1:{port}'
        self.expected_metadata = expected_metadata
        self.request_timeout = request_timeout
        self.ws = None
        self.packer = msgpack_numpy.Packer()

    def _connect(self):
        logging.info('Scene ready; connecting to policy at %s', self.uri)
        options = {}
        # The pinned simulator uses 14.2 (no sync-client heartbeat). In 15.x a
        # client heartbeat was added; disable that too if the environment changes.
        if 'ping_interval' in inspect.signature(websockets.sync.client.connect).parameters:
            options.update(ping_interval=None, ping_timeout=None)
        ws = websockets.sync.client.connect(
            self.uri, compression=None, max_size=None, open_timeout=30, close_timeout=5, **options)
        try:
            metadata = msgpack_numpy.unpackb(ws.recv(timeout=30))
            if any(metadata.get(k) != v for k, v in self.expected_metadata.items()):
                raise ValueError('Unexpected model/normalization/checkpoint/transport identity')
        except BaseException:
            ws.close()
            raise
        self.ws = ws

    def infer(self, observation):
        if self.ws is None:
            self._connect()
        self.ws.send(self.packer.pack(observation))
        try:
            response = self.ws.recv(timeout=self.request_timeout)
        except TimeoutError as error:
            raise TimeoutError(f'Policy response exceeded {self.request_timeout}s; infrastructure failure') from error
        if isinstance(response, str):
            raise RuntimeError(f'Policy server failed:\n{response}')
        return msgpack_numpy.unpackb(response)

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        if self.ws is not None:
            self.ws.close()
