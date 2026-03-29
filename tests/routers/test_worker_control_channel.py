from contextlib import contextmanager
import asyncio
import sys
import types

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.websockets import WebSocketDisconnect

from gpustack import envs
sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))

from gpustack.server.app import create_app
from gpustack.schemas.users import User, UserRole
from gpustack.schemas.workers import (
    Worker,
    WorkerControlMessageTypeEnum,
    WorkerReachabilityCapabilities,
    WorkerReachabilityModeEnum,
    WorkerStatus,
)


def _build_worker() -> Worker:
    return Worker(
        id=1,
        name="worker-control",
        labels={},
        cluster_id=1,
        hostname="worker-host",
        ip="127.0.0.1",
        ifname="eth0",
        port=10150,
        worker_uuid="worker-uuid-1",
        status=WorkerStatus.get_default_status(),
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
    )


def _build_worker_user() -> User:
    worker = _build_worker()
    return User(
        id=1,
        username="system/worker-test",
        hashed_password="",
        is_system=True,
        role=UserRole.Worker,
        worker=worker,
        worker_id=worker.id,
        cluster_id=worker.cluster_id,
    )


@contextmanager
def _worker_control_client(monkeypatch: pytest.MonkeyPatch, config):
    app = create_app(config)
    engine = create_async_engine(f"sqlite+aiosqlite:///{config.data_dir}/worker_control_channel.db")

    async def _prepare_database():
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    asyncio.run(_prepare_database())

    def sessionmaker():
        return AsyncSession(engine, expire_on_commit=False)

    class TestWorkerCommandDispatchService:
        def __new__(cls, *args, **kwargs):
            from gpustack.server.worker_command_service import WorkerCommandDispatchService

            kwargs.setdefault("session_factory", sessionmaker)
            return WorkerCommandDispatchService(*args, **kwargs)

    async def fake_auth(_websocket):
        return _build_worker_user()

    monkeypatch.setattr(
        "gpustack.routes.worker_control.authenticate_worker_websocket",
        fake_auth,
    )
    monkeypatch.setattr("gpustack.routes.worker_control.async_session", sessionmaker)
    monkeypatch.setattr(
        "gpustack.routes.worker_control.WorkerCommandDispatchService",
        TestWorkerCommandDispatchService,
    )

    with TestClient(app) as client:
        yield client, app

    asyncio.run(engine.dispose())


def _hello_message(session_id: str | None = None) -> dict:
    return {
        "message_type": WorkerControlMessageTypeEnum.HELLO.value,
        "session_id": session_id,
        "protocol_version": 1,
        "worker_uuid": "worker-uuid-1",
        "capabilities": {"outbound_control_ws": True, "reverse_http": False},
        "reachability_mode": WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS.value,
    }


def _pong_message(session_id: str) -> dict:
    return {
        "message_type": WorkerControlMessageTypeEnum.PONG.value,
        "session_id": session_id,
        "protocol_version": 1,
    }


def test_connect_and_replace_duplicate_session(monkeypatch: pytest.MonkeyPatch, config):
    with _worker_control_client(monkeypatch, config) as (client, app):
        with client.websocket_connect("/v2/workers/control/ws") as ws1:
            ws1.send_json(_hello_message(session_id="session-1"))
            hello1 = ws1.receive_json()
            assert hello1["session_id"] == "session-1"

            with client.websocket_connect("/v2/workers/control/ws") as ws2:
                ws2.send_json(_hello_message(session_id="session-2"))
                hello2 = ws2.receive_json()
                assert hello2["session_id"] == "session-2"

                with pytest.raises(WebSocketDisconnect) as excinfo:
                    ws1.receive_json()

                assert excinfo.value.code == 1008
                ws2.send_json(_pong_message(hello2["session_id"]))
                route_paths = [getattr(route, "path", None) for route in app.routes]
                assert route_paths.count("/v2/workers/control/ws") == 1


def test_oversized_message(monkeypatch: pytest.MonkeyPatch, config):
    monkeypatch.setattr(envs, "WORKER_CONTROL_WS_MAX_MESSAGE_BYTES", 256)

    with _worker_control_client(monkeypatch, config) as (client, _app):
        with client.websocket_connect("/v2/workers/control/ws") as websocket:
            websocket.send_json(_hello_message(session_id="session-oversize"))
            websocket.receive_json()

            websocket.send_text("x" * 257)

            with pytest.raises(WebSocketDisconnect) as excinfo:
                websocket.receive_text()

            assert excinfo.value.code == 1009


def test_rate_limit(monkeypatch: pytest.MonkeyPatch, config):
    monkeypatch.setattr(envs, "WORKER_CONTROL_WS_RATE_LIMIT_MESSAGES", 2)
    monkeypatch.setattr(envs, "WORKER_CONTROL_WS_RATE_LIMIT_WINDOW_SECONDS", 60.0)

    with _worker_control_client(monkeypatch, config) as (client, _app):
        with client.websocket_connect("/v2/workers/control/ws") as websocket:
            websocket.send_json(_hello_message(session_id="session-rate"))
            hello = websocket.receive_json()

            websocket.send_json(_pong_message(hello["session_id"]))
            websocket.send_json(_pong_message(hello["session_id"]))

            with pytest.raises(WebSocketDisconnect) as excinfo:
                websocket.receive_text()

            assert excinfo.value.code == 1008


def test_heartbeat_timeout(monkeypatch: pytest.MonkeyPatch, config):
    monkeypatch.setattr(envs, "WORKER_CONTROL_WS_HEARTBEAT_TIMEOUT_SECONDS", 1)

    with _worker_control_client(monkeypatch, config) as (client, _app):
        with client.websocket_connect("/v2/workers/control/ws") as websocket:
            websocket.send_json(_hello_message(session_id="session-timeout"))
            websocket.receive_json()

            with pytest.raises(WebSocketDisconnect) as excinfo:
                websocket.receive_text()

            assert excinfo.value.code == 1008
