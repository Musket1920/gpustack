import pytest
from starlette import status

from gpustack.server.worker_control import WorkerControlSessionRegistry


class FakeWebSocket:
    def __init__(self):
        self.close_calls = []

    async def close(self, code: int, reason: str):
        self.close_calls.append({"code": code, "reason": reason})


@pytest.mark.asyncio
async def test_register_replaces_previous_worker_session_with_new_generation():
    registry = WorkerControlSessionRegistry()
    old_socket = FakeWebSocket()
    new_socket = FakeWebSocket()

    first_session = await registry.register(
        worker_id=7,
        worker_uuid="worker-7",
        websocket=old_socket,
        session_id="session-a",
    )
    second_session = await registry.register(
        worker_id=7,
        worker_uuid="worker-7",
        websocket=new_socket,
        session_id="session-b",
    )

    assert first_session.generation == 1
    assert second_session.generation == 2
    assert first_session.replaced_by_generation == 2
    assert old_socket.close_calls == [
        {
            "code": status.WS_1008_POLICY_VIOLATION,
            "reason": "replaced by newer worker control session",
        }
    ]

    active_session = await registry.get_active_session(7)
    assert active_session is second_session


@pytest.mark.asyncio
async def test_unregister_does_not_remove_newer_active_session():
    registry = WorkerControlSessionRegistry()

    first_session = await registry.register(
        worker_id=11,
        worker_uuid="worker-11",
        websocket=FakeWebSocket(),
    )
    second_session = await registry.register(
        worker_id=11,
        worker_uuid="worker-11",
        websocket=FakeWebSocket(),
    )

    await registry.unregister(worker_id=11, generation=first_session.generation, reason="old")

    active_session = await registry.get_active_session(11)
    assert active_session is second_session


@pytest.mark.asyncio
async def test_register_does_not_replace_newer_session_with_older_generation():
    registry = WorkerControlSessionRegistry()
    newer_socket = FakeWebSocket()
    older_socket = FakeWebSocket()

    newer_session = await registry.register(
        worker_id=19,
        worker_uuid="worker-19",
        websocket=newer_socket,
        generation=2,
        session_id="session-newer",
    )
    older_session = await registry.register(
        worker_id=19,
        worker_uuid="worker-19",
        websocket=older_socket,
        generation=1,
        session_id="session-older",
    )

    active_session = await registry.get_active_session(19)

    assert active_session is not None
    assert active_session is newer_session
    assert active_session.generation == 2
    assert older_session.replaced_by_generation == 2
    assert older_session.disconnect_reason == "replaced"
    assert newer_socket.close_calls == []
    assert older_socket.close_calls == [
        {
            "code": status.WS_1008_POLICY_VIOLATION,
            "reason": "replaced by newer worker control session",
        }
    ]
