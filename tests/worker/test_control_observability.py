import asyncio
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import aiohttp
import pytest
from starlette.requests import Request

if "fcntl" not in sys.modules:
    _fcntl_stub = types.ModuleType("fcntl")
    _fcntl_stub.LOCK_EX = 1  # type: ignore[attr-defined]
    _fcntl_stub.LOCK_UN = 2  # type: ignore[attr-defined]
    _fcntl_stub.lockf = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    _fcntl_stub.flock = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["fcntl"] = _fcntl_stub

from gpustack.api.exceptions import ConflictException
from gpustack.config.config import Config
from gpustack.config.registration import write_worker_token
from gpustack.routes import model_instances
from gpustack.schemas.workers import (
    WorkerHelloMessage,
    WorkerReachabilityCapabilities,
    WorkerReachabilityModeEnum,
    WorkerStatus,
    Worker,
)
from gpustack.server.worker_control_observability import (
    reset_control_observability_metrics_for_tests,
    worker_control_capability_route_rejects_total,
    worker_control_session_reconnects_total,
)
from gpustack.worker.control_client import WorkerControlClient, worker_control_capabilities
from gpustack.worker.logs import LogOptions


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
        self.closed = False

    async def ws_connect(self, url: str, **kwargs):
        if not self._websockets:
            raise AssertionError("ws_connect called more times than expected")
        return self._websockets.pop(0)

    async def close(self):
        self.closed = True


async def wait_until(predicate, timeout: float = 1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not satisfied before timeout")


def _build_request(client) -> Request:
    state = SimpleNamespace(http_client=client, http_client_no_proxy=client)
    app = SimpleNamespace(state=state)
    scope = {
        "type": "http",
        "app": app,
        "method": "GET",
        "path": "/model_instances/1/logs",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 8000),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def _build_worker() -> Worker:
    return Worker(
        id=1,
        name="worker-1",
        labels={},
        cluster_id=1,
        hostname="worker-host",
        ip="10.0.0.10",
        advertise_address="10.0.0.10",
        ifname="eth0",
        port=10150,
        worker_uuid="worker-uuid-1",
        status=WorkerStatus.get_default_status(),
        capabilities=WorkerReachabilityCapabilities(
            outbound_control_ws=True,
            reverse_http=False,
        ),
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
    )


def _build_model_instance() -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        worker_id=1,
        name="instance-1",
        state="running",
        model_files=[],
    )


@pytest.mark.asyncio
async def test_control_session_metrics_reconnect_counter(tmp_path: Path):
    reset_control_observability_metrics_for_tests()
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

    assert worker_control_session_reconnects_total.labels(worker_id="1")._value.get() == 1


@pytest.mark.asyncio
async def test_nat_route_reject_metric_records_conflict(monkeypatch: pytest.MonkeyPatch):
    reset_control_observability_metrics_for_tests()
    request = _build_request(SimpleNamespace())
    worker = _build_worker()

    async def fake_fetch_model_instance(session, id):
        return _build_model_instance()

    async def fake_fetch_worker(session, worker_id):
        return worker

    monkeypatch.setattr(model_instances, "fetch_model_instance", fake_fetch_model_instance)
    monkeypatch.setattr(model_instances, "fetch_worker", fake_fetch_worker)
    monkeypatch.setattr(model_instances, "use_proxy_env_for_url", lambda url: False)

    with pytest.raises(ConflictException):
        await model_instances.get_serving_logs(
            request,
            cast(Any, None),
            1,
            LogOptions(follow=False),
        )

    assert (
        worker_control_capability_route_rejects_total.labels(
            worker_id="1",
            operation="streaming serving logs",
            reachability_mode="outbound_control_ws",
        )._value.get()
        == 1
    )
