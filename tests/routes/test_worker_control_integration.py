from contextlib import contextmanager
import asyncio
import sys
import time
import types

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))

from gpustack.config.config import Config
from gpustack.routes.worker_control import WorkerHelloMessage
from gpustack.schemas.users import User, UserRole
from gpustack.schemas.workers import (
    Worker,
    WorkerCommand,
    WorkerControlCommandStateEnum,
    WorkerReachabilityCapabilities,
    WorkerReachabilityModeEnum,
    WorkerStatus,
)
from gpustack.server.app import create_app
from gpustack.server.worker_command_service import WorkerCommandDispatchService


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
        heartbeat_time=None,
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
def _worker_control_client(config: Config, monkeypatch: pytest.MonkeyPatch):
    app = create_app(config)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{config.data_dir}/worker_control_integration.db"
    )

    async def _prepare_database():
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    asyncio.run(_prepare_database())

    def sessionmaker():
        return AsyncSession(engine, expire_on_commit=False)

    class TestWorkerCommandDispatchService:
        def __new__(cls, *args, **kwargs):
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
        yield client, app, sessionmaker

    asyncio.run(engine.dispose())


def _hello_message(session_id: str | None = None) -> dict:
    return {
        "message_type": "hello",
        "session_id": session_id,
        "protocol_version": 1,
        "worker_uuid": "worker-uuid-1",
        "capabilities": {"outbound_control_ws": True, "reverse_http": False},
        "reachability_mode": WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS.value,
    }


async def _load_command(sessionmaker, command_id: str) -> WorkerCommand | None:
    async with sessionmaker() as session:
        return await WorkerCommand.one_by_field(session, "command_id", command_id)


def _wait_for_command_state(
    sessionmaker,
    command_id: str,
    expected_state: WorkerControlCommandStateEnum,
    timeout: float = 1.0,
) -> WorkerCommand:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        command = asyncio.run(_load_command(sessionmaker, command_id))
        if command is not None and command.state == expected_state:
            return command
        time.sleep(0.01)
    raise AssertionError(
        f"command {command_id} did not reach state {expected_state.value} before timeout"
    )


def test_nat_worker_command_round_trip_without_reverse_probe(
    monkeypatch: pytest.MonkeyPatch, config: Config
):
    with _worker_control_client(config, monkeypatch) as (client, app, sessionmaker):
        with client.websocket_connect("/v2/workers/control/ws") as websocket:
            websocket.send_json(_hello_message())
            server_hello = WorkerHelloMessage.model_validate(websocket.receive_json())

            dispatch_service = WorkerCommandDispatchService(
                session_factory=sessionmaker,
                session_registry=app.state.worker_control_session_registry,
            )
            dispatched_commands = asyncio.run(
                dispatch_service.emit_sync_runtime_state(
                    worker_id=1,
                    idempotency_token="nat-round-trip",
                    reason="integration regression",
                )
            )

            assert len(dispatched_commands) == 1
            command = dispatched_commands[0]
            assert command.state == WorkerControlCommandStateEnum.SENT

            dispatched_message = websocket.receive_json()
            assert dispatched_message["command_id"] == command.command_id
            assert dispatched_message["command_type"] == "sync_runtime_state"
            assert dispatched_message["session_id"] == server_hello.session_id
            assert dispatched_message["payload"] == {
                "worker_id": 1,
                "reason": "integration regression",
            }

            websocket.send_json(
                {
                    "message_type": "command_ack",
                    "session_id": server_hello.session_id,
                    "protocol_version": server_hello.protocol_version,
                    "command_id": command.command_id,
                    "accepted": True,
                    "state": "acknowledged",
                }
            )
            websocket.send_json(
                {
                    "message_type": "command_result",
                    "session_id": server_hello.session_id,
                    "protocol_version": server_hello.protocol_version,
                    "command_id": command.command_id,
                    "state": "succeeded",
                    "result": {"reconciled": True, "source": "worker"},
                }
            )

            completed_command = _wait_for_command_state(
                sessionmaker,
                command.command_id,
                WorkerControlCommandStateEnum.SUCCEEDED,
            )

            assert completed_command.acknowledged_at is not None
            assert completed_command.completed_at is not None
            assert completed_command.result == {
                "reconciled": True,
                "source": "worker",
            }
