"""Exercise request/response alignment with the real SkillNet websocket client."""

import asyncio
import threading

import numpy as np
import websockets.asyncio.server

from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from skillnet_client.websocket_client_policy import WebsocketClientPolicy


class EchoPolicy:
    def infer(self, observation):
        return {"actions": np.array([[observation["request_id"]]], dtype=np.float32)}


def test_each_request_receives_its_own_action():
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    state = {}

    async def start():
        policy_server = WebsocketPolicyServer(EchoPolicy())
        state["server"] = await websockets.asyncio.server.serve(
            policy_server._handler, "127.0.0.1", 0, ping_interval=None
        )
        state["port"] = state["server"].sockets[0].getsockname()[1]
        ready.set()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start())
        loop.run_forever()
        loop.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    client = None
    try:
        assert ready.wait(10), "Websocket test server failed to start"
        client = WebsocketClientPolicy("127.0.0.1", state["port"])
        for request_id in range(8):
            response = client.infer({"request_id": request_id})
            assert response["actions"][0, 0] == request_id
    finally:
        if client is not None:
            client._ws.close()

        async def stop():
            state["server"].close()
            await state["server"].wait_closed()

        if ready.is_set():
            asyncio.run_coroutine_threadsafe(stop(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        assert not thread.is_alive()
