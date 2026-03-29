import asyncio
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest

# Inject an fcntl stub before importing any gpustack.worker module so that
# gpustack.worker.__init__ → gpustack.utils.locks → fcntl does not fail on
# Windows where fcntl is unavailable.
if "fcntl" not in sys.modules:
    _fcntl_stub = types.ModuleType("fcntl")
    _fcntl_stub.LOCK_EX = 1  # type: ignore[attr-defined]
    _fcntl_stub.LOCK_UN = 2  # type: ignore[attr-defined]
    _fcntl_stub.lockf = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    _fcntl_stub.flock = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["fcntl"] = _fcntl_stub

from gpustack.config.config import Config
from gpustack.config.registration import write_worker_token
from gpustack.schemas.workers import (
    WorkerCommandAckMessage,
    WorkerCommandMessage,
    WorkerCommandResultMessage,
    WorkerControlCommandStateEnum,
    WorkerHelloMessage,
    WorkerPingMessage,
    WorkerReachabilityModeEnum,
)
from gpustack.server.worker_command_service import COMMAND_RECONCILE_NOW
from gpustack.worker.control_client import (
    WorkerControlClient,
    worker_control_capabilities,
)


def make_config(tmp_path: Path) -> Config:
    return Config(
        token="test",
        jwt_secret_key="test",
        data_dir=str(tmp_path),
        server_url="http://127.0.0.1:30080",
        worker_name="worker-1",
    )


def make_server_hello(session_id: str) -> WorkerHelloMessage:
    return WorkerHelloMessage(
        session_id=session_id,
        worker_uuid="worker-uuid",
        capabilities=worker_control_capabilities(),
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
    )


def text_message(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        type=aiohttp.WSMsgType.TEXT,
        data=json.dumps(payload, default=str),
        extra=None,
    )


def close_message(reason: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None, extra=reason)


class FakeWebSocket:
    def __init__(self, messages: list[SimpleNamespace], *, close_code: int = 1000):
        self._messages = asyncio.Queue()
        for message in messages:
            self._messages.put_nowait(message)
        self.sent_json: list[dict] = []
        self.closed = False
        self.close_code = close_code
        self._wake_event = asyncio.Event()

    async def receive(self):
        while self._messages.empty() and not self.closed:
            await self._wake_event.wait()
            self._wake_event.clear()

        if not self._messages.empty():
            return await self._messages.get()

        return close_message("closed")

    async def send_json(self, payload: dict):
        self.sent_json.append(payload)

    async def close(self, code: int = 1000, reason: str = ""):
        self.closed = True
        self.close_code = code
        self._wake_event.set()


class FakeSession:
    def __init__(self, websockets: list[FakeWebSocket]):
        self._websockets = list(websockets)
        self.connect_calls: list[tuple[str, dict]] = []
        self.closed = False

    async def ws_connect(self, url: str, **kwargs):
        self.connect_calls.append((url, kwargs))
        if not self._websockets:
            raise AssertionError("ws_connect called more times than expected")
        return self._websockets.pop(0)

    async def close(self):
        self.closed = True


class FakeCommandExecutor:
    def __init__(self):
        self.handled_commands: list[WorkerCommandMessage] = []
        self.stopped = False

    async def handle_command(self, command: WorkerCommandMessage, websocket):
        self.handled_commands.append(command)
        await websocket.send_json(
            WorkerCommandAckMessage(
                session_id=command.session_id,
                protocol_version=command.protocol_version,
                command_id=command.command_id,
            ).model_dump(mode="json")
        )
        await websocket.send_json(
            WorkerCommandResultMessage(
                session_id=command.session_id,
                protocol_version=command.protocol_version,
                command_id=command.command_id,
                state=WorkerControlCommandStateEnum.SUCCEEDED,
                result={"recovered": True},
            ).model_dump(mode="json")
        )

    async def stop(self):
        self.stopped = True


async def wait_until(predicate, timeout: float = 1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not satisfied before timeout")


@pytest.mark.asyncio
async def test_connect_and_reconnect(tmp_path: Path):
    cfg = make_config(tmp_path)
    write_worker_token(str(tmp_path), "worker-token")

    first_session_id = "session-1"
    ws1 = FakeWebSocket(
        [
            text_message(make_server_hello(first_session_id).model_dump(mode="json")),
            close_message("network drop"),
        ],
        close_code=1011,
    )
    ws2 = FakeWebSocket(
        [
            text_message(make_server_hello(first_session_id).model_dump(mode="json")),
            text_message(WorkerPingMessage(session_id=first_session_id).model_dump(mode="json")),
        ]
    )
    session = FakeSession([ws1, ws2])

    client = WorkerControlClient(
        cfg=cfg,
        worker_id_getter=lambda: 1,
        worker_uuid_getter=lambda: "worker-uuid",
        session_factory=lambda: session,
        ping_interval_seconds=0.01,
        reconnect_initial_delay_seconds=0.0,
        reconnect_max_delay_seconds=0.0,
        reconnect_jitter_ratio=0.0,
    )

    task = asyncio.create_task(client.start())
    await wait_until(lambda: client.session_state.session_generation >= 2)
    await wait_until(
        lambda: any(msg["message_type"] == "pong" for msg in ws2.sent_json),
    )
    await wait_until(
        lambda: any(msg["message_type"] == "ping" for msg in ws2.sent_json),
    )

    await client.stop()
    await task

    assert len(session.connect_calls) == 2
    assert ws1.sent_json[0]["message_type"] == "hello"
    assert ws1.sent_json[0]["session_id"] is None
    assert ws2.sent_json[0]["message_type"] == "hello"
    assert ws2.sent_json[0]["session_id"] == first_session_id
    assert client.session_state.session_generation == 2
    assert client.session_state.session_id == first_session_id
    assert client.session_state.reachability_mode == WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS


@pytest.mark.asyncio
async def test_auth_failure_backoff_or_give_up(tmp_path: Path):
    cfg = make_config(tmp_path)
    write_worker_token(str(tmp_path), "worker-token")

    ws1 = FakeWebSocket([close_message("policy violation")], close_code=1008)
    ws2 = FakeWebSocket([close_message("policy violation")], close_code=1008)
    session = FakeSession([ws1, ws2])

    client = WorkerControlClient(
        cfg=cfg,
        worker_id_getter=lambda: 1,
        worker_uuid_getter=lambda: "worker-uuid",
        session_factory=lambda: session,
        ping_interval_seconds=60.0,
        reconnect_initial_delay_seconds=0.1,
        reconnect_max_delay_seconds=0.1,
        reconnect_jitter_ratio=0.0,
        max_consecutive_auth_failures=2,
    )

    sleeps: list[float] = []

    async def record_sleep(delay: float):
        sleeps.append(delay)

    client._sleep = record_sleep  # type: ignore[method-assign]

    await client.start()

    assert len(session.connect_calls) == 2
    assert sleeps == [0.1]
    assert client.session_state.session_generation == 0
    assert client.connected_event.is_set() is False


@pytest.mark.asyncio
async def test_session_replacement(tmp_path: Path):
    cfg = make_config(tmp_path)
    write_worker_token(str(tmp_path), "worker-token")

    old_session_id = "session-old"
    new_session_id = "session-new"
    ws1 = FakeWebSocket(
        [
            text_message(make_server_hello(old_session_id).model_dump(mode="json")),
            close_message("replaced"),
        ],
        close_code=1008,
    )
    ws2 = FakeWebSocket(
        [
            text_message(make_server_hello(new_session_id).model_dump(mode="json")),
        ]
    )
    session = FakeSession([ws1, ws2])

    client = WorkerControlClient(
        cfg=cfg,
        worker_id_getter=lambda: 1,
        worker_uuid_getter=lambda: "worker-uuid",
        session_factory=lambda: session,
        ping_interval_seconds=60.0,
        reconnect_initial_delay_seconds=0.0,
        reconnect_max_delay_seconds=0.0,
        reconnect_jitter_ratio=0.0,
    )

    task = asyncio.create_task(client.start())
    await wait_until(lambda: client.session_state.session_generation >= 2)

    await client.stop()
    await task

    assert ws1.sent_json[0]["session_id"] is None
    assert ws2.sent_json[0]["session_id"] == old_session_id
    assert client.session_state.advertised_session_id == old_session_id
    assert client.session_state.session_id == new_session_id
    assert client.session_state.session_generation == 2


@pytest.mark.asyncio
async def test_server_restart_reconcile(tmp_path: Path):
    cfg = make_config(tmp_path)
    write_worker_token(str(tmp_path), "worker-token")

    resumed_session_id = "session-restart"
    recovery_command = WorkerCommandMessage(
        session_id=resumed_session_id,
        command_id="cmd-reconcile-after-restart",
        command_type=COMMAND_RECONCILE_NOW,
        payload={
            "reason": "server restart recovery",
            "full_reconcile": True,
        },
    )
    ws1 = FakeWebSocket(
        [
            text_message(make_server_hello(resumed_session_id).model_dump(mode="json")),
            close_message("server restart"),
        ],
        close_code=1012,
    )
    ws2 = FakeWebSocket(
        [
            text_message(make_server_hello(resumed_session_id).model_dump(mode="json")),
            text_message(recovery_command.model_dump(mode="json")),
        ]
    )
    session = FakeSession([ws1, ws2])
    executor = FakeCommandExecutor()

    client = WorkerControlClient(
        cfg=cfg,
        worker_id_getter=lambda: 1,
        worker_uuid_getter=lambda: "worker-uuid",
        session_factory=lambda: session,
        ping_interval_seconds=60.0,
        reconnect_initial_delay_seconds=0.0,
        reconnect_max_delay_seconds=0.0,
        reconnect_jitter_ratio=0.0,
        command_executor=executor,
    )

    task = asyncio.create_task(client.start())
    await wait_until(lambda: len(executor.handled_commands) == 1)

    await client.stop()
    await task

    assert ws1.sent_json[0]["message_type"] == "hello"
    assert ws1.sent_json[0]["session_id"] is None
    assert ws2.sent_json[0]["message_type"] == "hello"
    assert ws2.sent_json[0]["session_id"] == resumed_session_id
    assert executor.handled_commands[0].command_id == recovery_command.command_id
    assert executor.handled_commands[0].payload["full_reconcile"] is True
    assert ws2.sent_json[1]["message_type"] == "command_ack"
    assert ws2.sent_json[2]["message_type"] == "command_result"
    assert executor.stopped is True
