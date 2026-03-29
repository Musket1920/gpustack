from datetime import datetime, timezone
import asyncio

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.clusters import Cluster
from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum, SourceEnum
from gpustack.schemas.workers import (
    Worker,
    WorkerCommand,
    WorkerControlChannelEnum,
    WorkerControlCommandStateEnum,
    WorkerReachabilityCapabilities,
    WorkerReachabilityModeEnum,
    WorkerSession,
    WorkerStatus,
)
from gpustack.server.bus import Event, EventType
import gpustack.server.worker_command_service as worker_command_service_module
from gpustack.server.worker_command_controller import WorkerCommandController
from gpustack.server.worker_command_service import COMMAND_RECONCILE_MODEL_INSTANCE, COMMAND_RECONCILE_NOW
from gpustack.server.worker_command_service import (
    StaleWorkerSessionError,
    WorkerCommandDispatchService,
    WorkerCommandService,
)
from gpustack.server.worker_control import WorkerControlSessionRegistry


def _utc(year: int, month: int, day: int, hour: int, minute: int, second: int) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def freeze_worker_command_service_clock(monkeypatch):
    monkeypatch.setattr(
        worker_command_service_module,
        "utcnow",
        lambda: _utc(2026, 3, 28, 0, 0, 0),
    )


@pytest_asyncio.fixture
async def command_service(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'worker_command_service.db'}")

    def sessionmaker():
        return AsyncSession(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with sessionmaker() as session:
        cluster = Cluster(name="cluster-control")
        session.add(cluster)
        await session.commit()
        await session.refresh(cluster)

        worker = Worker(
            name="worker-control",
            labels={},
            cluster_id=cluster.id,
            hostname="worker-host",
            ip="127.0.0.1",
            ifname="eth0",
            port=10150,
            worker_uuid="worker-uuid-1",
            status=WorkerStatus.get_default_status(),
            capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
            reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        )
        other_worker = Worker(
            name="worker-control-2",
            labels={},
            cluster_id=cluster.id,
            hostname="worker-host-2",
            ip="127.0.0.2",
            ifname="eth1",
            port=10151,
            worker_uuid="worker-uuid-2",
            status=WorkerStatus.get_default_status(),
            capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
            reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        )
        session.add(worker)
        session.add(other_worker)
        await session.commit()
        await session.refresh(worker)
        await session.refresh(other_worker)

        yield {
            "engine": engine,
            "sessionmaker": sessionmaker,
            "service": WorkerCommandService(session=session, replay_window_size=1),
            "worker": worker,
            "other_worker": other_worker,
        }

    await engine.dispose()


class FakeWebSocket:
    def __init__(self):
        self.sent_json = []

    async def send_json(self, payload: dict):
        self.sent_json.append(payload)


class AckingWebSocket(FakeWebSocket):
    def __init__(self, on_send):
        super().__init__()
        self._on_send = on_send

    async def send_json(self, payload: dict):
        self.sent_json.append(payload)
        await self._on_send(payload)


async def _open_session(
    service: WorkerCommandService,
    worker_id: int,
    *,
    session_id: str,
    now: datetime,
):
    return await service.open_session(
        worker_id,
        session_id=session_id,
        control_channel=WorkerControlChannelEnum.OUTBOUND_CONTROL_WS,
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        now=now,
    )


@pytest.mark.asyncio
async def test_ack_result_idempotency(command_service):
    service = command_service["service"]
    worker = command_service["worker"]
    opened_session = await _open_session(
        service,
        worker.id,
        session_id="session-1",
        now=_utc(2026, 3, 28, 12, 0, 0),
    )

    created_command = await service.create_command(
        worker.id,
        command_type="reconcile_now",
        payload={"model_instance_id": 1},
        idempotency_key="command:reconcile:1",
        now=_utc(2026, 3, 28, 12, 0, 1),
    )
    duplicate_command = await service.create_command(
        worker.id,
        command_type="reconcile_now",
        payload={"model_instance_id": 1},
        idempotency_key="command:reconcile:1",
        now=_utc(2026, 3, 28, 12, 0, 2),
    )

    assert duplicate_command.id == created_command.id
    assert duplicate_command.sequence == created_command.sequence

    sent_command = await service.mark_command_sent(
        created_command.command_id,
        session_id=opened_session.session_id,
        session_generation=opened_session.generation,
        now=_utc(2026, 3, 28, 12, 0, 3),
    )
    acknowledged_command = await service.acknowledge_command(
        created_command.command_id,
        session_id=opened_session.session_id,
        session_generation=opened_session.generation,
        now=_utc(2026, 3, 28, 12, 0, 4),
    )
    duplicate_ack = await service.acknowledge_command(
        created_command.command_id,
        session_id=opened_session.session_id,
        session_generation=opened_session.generation,
        now=_utc(2026, 3, 28, 12, 0, 5),
    )
    completed_command = await service.record_command_result(
        created_command.command_id,
        session_id=opened_session.session_id,
        session_generation=opened_session.generation,
        state=WorkerControlCommandStateEnum.SUCCEEDED,
        result={"status": "ok"},
        now=_utc(2026, 3, 28, 12, 0, 6),
    )
    duplicate_result = await service.record_command_result(
        created_command.command_id,
        session_id=opened_session.session_id,
        session_generation=opened_session.generation,
        state=WorkerControlCommandStateEnum.SUCCEEDED,
        result={"status": "ignored-duplicate"},
        now=_utc(2026, 3, 28, 12, 0, 7),
    )

    assert sent_command.dispatch_attempts == 1
    assert acknowledged_command.acknowledged_at == _utc(2026, 3, 28, 12, 0, 4)
    assert duplicate_ack.acknowledged_at == acknowledged_command.acknowledged_at
    assert completed_command.state == WorkerControlCommandStateEnum.SUCCEEDED
    assert completed_command.result == {"status": "ok"}
    assert duplicate_result.result == {"status": "ok"}
    assert duplicate_result.completed_at == completed_command.completed_at


@pytest.mark.asyncio
async def test_create_command_serializes_same_worker_idempotency_and_sequence(
    command_service,
):
    sessionmaker = command_service["sessionmaker"]
    worker = command_service["worker"]

    async def create_command(idempotency_key: str):
        async with sessionmaker() as session:
            service = WorkerCommandService(session=session)
            return await service.create_command(
                worker.id,
                command_type="reconcile_now",
                payload={"idempotency_key": idempotency_key},
                idempotency_key=idempotency_key,
                now=_utc(2026, 3, 28, 12, 30, 0),
            )

    duplicate_results = await asyncio.gather(
        *[
            create_command("command:serialize:shared")
            for _ in range(8)
        ]
    )

    assert {command.command_id for command in duplicate_results} == {
        duplicate_results[0].command_id
    }
    assert {command.sequence for command in duplicate_results} == {1}

    unique_results = await asyncio.gather(
        *[
            create_command(f"command:serialize:{index}")
            for index in range(1, 5)
        ]
    )

    assert sorted(command.sequence for command in unique_results) == [2, 3, 4, 5]

    async with sessionmaker() as session:
        commands = (
            await session.exec(
                select(WorkerCommand).where(WorkerCommand.worker_id == worker.id)
            )
        ).all()
    commands = sorted(commands, key=lambda command: command.sequence)

    assert [command.idempotency_key for command in commands] == [
        "command:serialize:shared",
        "command:serialize:1",
        "command:serialize:2",
        "command:serialize:3",
        "command:serialize:4",
    ]
    assert [command.sequence for command in commands] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_stale_session_result_rejected(command_service):
    service = command_service["service"]
    worker = command_service["worker"]
    first_session = await _open_session(
        service,
        worker.id,
        session_id="session-stale-1",
        now=_utc(2026, 3, 28, 13, 0, 0),
    )
    command = await service.create_command(
        worker.id,
        command_type="sync_runtime_state",
        payload={"worker_id": worker.id},
        now=_utc(2026, 3, 28, 13, 0, 1),
    )
    await service.mark_command_sent(
        command.command_id,
        session_id=first_session.session_id,
        session_generation=first_session.generation,
        now=_utc(2026, 3, 28, 13, 0, 2),
    )

    second_session = await _open_session(
        service,
        worker.id,
        session_id="session-stale-2",
        now=_utc(2026, 3, 28, 13, 0, 3),
    )

    with pytest.raises(StaleWorkerSessionError):
        await service.record_command_result(
            command.command_id,
            session_id=first_session.session_id,
            session_generation=first_session.generation,
            state=WorkerControlCommandStateEnum.SUCCEEDED,
            result={"status": "should-not-apply"},
            now=_utc(2026, 3, 28, 13, 0, 4),
        )

    command_after_rejection = await service._require_worker_command(
        command.command_id,
        worker.id,
    )
    first_session_row = await WorkerSession.one_by_field(
        service.session,
        "session_id",
        first_session.session_id,
    )
    second_session_row = await WorkerSession.one_by_field(
        service.session,
        "session_id",
        second_session.session_id,
    )

    assert command_after_rejection.state == WorkerControlCommandStateEnum.SENT
    assert command_after_rejection.completed_at is None
    assert first_session_row is not None
    assert first_session_row.state.value == "stale"
    assert second_session_row is not None
    assert second_session_row.state.value == "active"


@pytest.mark.asyncio
async def test_replay_gap_falls_back_to_full_reconcile(command_service):
    service = command_service["service"]
    worker = command_service["worker"]
    sessionmaker = command_service["sessionmaker"]
    opened_session = await _open_session(
        service,
        worker.id,
        session_id="session-replay",
        now=_utc(2026, 3, 28, 14, 0, 0),
    )

    for sequence in range(3):
        await service.create_command(
            worker.id,
            command_type="reconcile_now",
            payload={"attempt": sequence},
            now=_utc(2026, 3, 28, 14, 0, sequence + 1),
        )

    await service.touch_session(
        opened_session.session_id,
        opened_session.generation,
        replay_cursor=1,
        now=_utc(2026, 3, 28, 14, 0, 9),
    )

    registry = WorkerControlSessionRegistry()
    websocket = FakeWebSocket()
    await registry.register(
        worker_id=worker.id,
        worker_uuid=worker.worker_uuid,
        websocket=websocket,
        generation=opened_session.generation,
        session_id=opened_session.session_id,
    )
    dispatch_service = WorkerCommandDispatchService(
        session_factory=sessionmaker,
        session_registry=registry,
        replay_window_size=1,
    )

    replay_commands = await dispatch_service.dispatch_replay_window(
        session_id=opened_session.session_id,
        session_generation=opened_session.generation,
        after_sequence=1,
    )

    async with sessionmaker() as session:
        stored_session = await WorkerSession.one_by_field(
            session,
            "session_id",
            opened_session.session_id,
        )
        all_commands = list(
            (await session.exec(select(WorkerCommand).where(WorkerCommand.worker_id == worker.id).order_by("sequence"))).all()
        )

    assert len(replay_commands) == 1
    assert replay_commands[0].command_type == COMMAND_RECONCILE_NOW
    assert replay_commands[0].state == WorkerControlCommandStateEnum.SENT
    assert replay_commands[0].worker_session_generation == opened_session.generation
    assert replay_commands[0].payload["full_reconcile"] is True
    assert replay_commands[0].payload["replay_floor_sequence"] == 3
    assert replay_commands[0].payload["replay_through_sequence"] == 3
    assert websocket.sent_json[0]["command_type"] == COMMAND_RECONCILE_NOW
    assert websocket.sent_json[0]["payload"]["full_reconcile"] is True
    assert stored_session is not None
    assert stored_session.requires_full_reconcile is True
    assert stored_session.full_reconcile_reason is not None
    assert len(all_commands) == 4
    assert all_commands[-1].command_id == replay_commands[0].command_id


@pytest.mark.asyncio
async def test_server_restart_reconcile_replays_resume_commands(command_service):
    service = command_service["service"]
    worker = command_service["worker"]
    sessionmaker = command_service["sessionmaker"]
    opened_session = await _open_session(
        service,
        worker.id,
        session_id="session-server-restart",
        now=_utc(2026, 3, 28, 15, 30, 0),
    )
    command = await service.create_command(
        worker.id,
        command_type="sync_runtime_state",
        payload={"worker_id": worker.id, "reason": "server restart replay"},
        now=_utc(2026, 3, 28, 15, 30, 1),
    )
    await service.mark_command_sent(
        command.command_id,
        session_id=opened_session.session_id,
        session_generation=opened_session.generation,
        now=_utc(2026, 3, 28, 15, 30, 2),
    )

    async with sessionmaker() as session:
        resumed_service = WorkerCommandService(session=session)
        resumed_session = await _open_session(
            resumed_service,
            worker.id,
            session_id=opened_session.session_id,
            now=_utc(2026, 3, 28, 15, 31, 0),
        )

    registry = WorkerControlSessionRegistry()
    websocket = FakeWebSocket()
    await registry.register(
        worker_id=worker.id,
        worker_uuid=worker.worker_uuid,
        websocket=websocket,
        generation=resumed_session.generation,
        session_id=resumed_session.session_id,
    )
    dispatch_service = WorkerCommandDispatchService(
        session_factory=sessionmaker,
        session_registry=registry,
    )

    replay_commands = await dispatch_service.dispatch_replay_window(
        session_id=resumed_session.session_id,
        session_generation=resumed_session.generation,
        resume_requested=True,
    )

    async with sessionmaker() as session:
        stored_command = await WorkerCommand.one_by_field(
            session,
            "command_id",
            command.command_id,
        )

    assert len(replay_commands) == 1
    assert replay_commands[0].command_id == command.command_id
    assert websocket.sent_json[0]["command_id"] == command.command_id
    assert stored_command is not None
    assert stored_command.state == WorkerControlCommandStateEnum.SENT
    assert stored_command.dispatch_attempts == 2
    assert stored_command.worker_session_generation == resumed_session.generation


@pytest.mark.asyncio
async def test_worker_restart_supersedes_old_session(command_service):
    service = command_service["service"]
    worker = command_service["worker"]
    sessionmaker = command_service["sessionmaker"]
    first_session = await _open_session(
        service,
        worker.id,
        session_id="session-worker-restart-1",
        now=_utc(2026, 3, 28, 16, 0, 0),
    )
    command = await service.create_command(
        worker.id,
        command_type="sync_runtime_state",
        payload={"worker_id": worker.id, "reason": "worker restart"},
        now=_utc(2026, 3, 28, 16, 0, 1),
    )
    await service.mark_command_sent(
        command.command_id,
        session_id=first_session.session_id,
        session_generation=first_session.generation,
        now=_utc(2026, 3, 28, 16, 0, 2),
    )
    second_session = await _open_session(
        service,
        worker.id,
        session_id="session-worker-restart-2",
        now=_utc(2026, 3, 28, 16, 0, 3),
    )

    registry = WorkerControlSessionRegistry()
    websocket = FakeWebSocket()
    await registry.register(
        worker_id=worker.id,
        worker_uuid=worker.worker_uuid,
        websocket=websocket,
        generation=second_session.generation,
        session_id=second_session.session_id,
    )
    dispatch_service = WorkerCommandDispatchService(
        session_factory=sessionmaker,
        session_registry=registry,
    )

    recovery_commands = await dispatch_service.dispatch_replay_window(
        session_id=second_session.session_id,
        session_generation=second_session.generation,
        resume_requested=False,
    )

    with pytest.raises(StaleWorkerSessionError):
        await service.record_command_result(
            command.command_id,
            session_id=first_session.session_id,
            session_generation=first_session.generation,
            state=WorkerControlCommandStateEnum.SUCCEEDED,
            result={"status": "late-old-session-result"},
            now=_utc(2026, 3, 28, 16, 0, 4),
        )

    async with sessionmaker() as session:
        stored_command = await WorkerCommand.one_by_field(
            session,
            "command_id",
            command.command_id,
        )
        all_commands = list(
            (
                await session.exec(
                    select(WorkerCommand)
                    .where(WorkerCommand.worker_id == worker.id)
                    .order_by("sequence")
                )
            ).all()
        )

    assert len(recovery_commands) == 1
    assert recovery_commands[0].command_type == COMMAND_RECONCILE_NOW
    assert recovery_commands[0].payload["full_reconcile"] is True
    assert websocket.sent_json[0]["command_type"] == COMMAND_RECONCILE_NOW
    assert stored_command is not None
    assert stored_command.state == WorkerControlCommandStateEnum.SUPERSEDED
    assert stored_command.completed_at is not None
    assert all_commands[-1].command_id == recovery_commands[0].command_id


@pytest.mark.asyncio
async def test_open_session_serializes_concurrent_generation_and_stale_marking(
    command_service,
    monkeypatch,
):
    sessionmaker = command_service["sessionmaker"]
    worker = command_service["worker"]
    original_next_generation = WorkerCommandService._next_session_generation

    async def delayed_next_generation(self, worker_id: int) -> int:
        generation = await original_next_generation(self, worker_id)
        await asyncio.sleep(0.05)
        return generation

    monkeypatch.setattr(
        WorkerCommandService,
        "_next_session_generation",
        delayed_next_generation,
    )

    async def open_in_fresh_session(session_id: str, now: datetime) -> WorkerSession:
        async with sessionmaker() as session:
            service = WorkerCommandService(session=session)
            return await _open_session(
                service,
                worker.id,
                session_id=session_id,
                now=now,
            )

    first_open, second_open = await asyncio.gather(
        open_in_fresh_session("session-race-1", _utc(2026, 3, 28, 16, 10, 0)),
        open_in_fresh_session("session-race-2", _utc(2026, 3, 28, 16, 10, 1)),
    )

    async with sessionmaker() as session:
        stored_sessions = list(
            (
                await session.exec(
                    select(WorkerSession)
                    .where(WorkerSession.worker_id == worker.id)
                    .order_by("generation")
                )
            ).all()
        )

    assert sorted([first_open.generation, second_open.generation]) == [1, 2]
    assert [stored_session.generation for stored_session in stored_sessions] == [1, 2]
    assert [stored_session.state.value for stored_session in stored_sessions] == [
        "stale",
        "active",
    ]
    assert stored_sessions[0].disconnected_at == _utc(2026, 3, 28, 16, 10, 1)
    assert stored_sessions[1].session_id == "session-race-2"


@pytest.mark.asyncio
async def test_replay_expiration_requires_full_reconcile(command_service):
    service = command_service["service"]
    worker = command_service["worker"]
    sessionmaker = command_service["sessionmaker"]
    opened_session = await _open_session(
        service,
        worker.id,
        session_id="session-replay-expiration",
        now=_utc(2026, 3, 28, 16, 30, 0),
    )
    command = await service.create_command(
        worker.id,
        command_type="sync_runtime_state",
        payload={"worker_id": worker.id, "reason": "replay expiration"},
        expires_at=_utc(2020, 1, 1, 0, 0, 0),
        now=_utc(2026, 3, 28, 16, 30, 1),
    )
    await service.mark_command_sent(
        command.command_id,
        session_id=opened_session.session_id,
        session_generation=opened_session.generation,
        now=_utc(2026, 3, 28, 16, 30, 1),
    )

    async with sessionmaker() as session:
        resumed_service = WorkerCommandService(session=session)
        resumed_session = await _open_session(
            resumed_service,
            worker.id,
            session_id=opened_session.session_id,
            now=_utc(2026, 3, 28, 16, 31, 0),
        )

    registry = WorkerControlSessionRegistry()
    websocket = FakeWebSocket()
    await registry.register(
        worker_id=worker.id,
        worker_uuid=worker.worker_uuid,
        websocket=websocket,
        generation=resumed_session.generation,
        session_id=resumed_session.session_id,
    )
    dispatch_service = WorkerCommandDispatchService(
        session_factory=sessionmaker,
        session_registry=registry,
    )

    recovery_commands = await dispatch_service.dispatch_replay_window(
        session_id=resumed_session.session_id,
        session_generation=resumed_session.generation,
        resume_requested=True,
    )

    async with sessionmaker() as session:
        stored_command = await WorkerCommand.one_by_field(
            session,
            "command_id",
            command.command_id,
        )
        stored_session = await WorkerSession.one_by_field(
            session,
            "session_id",
            resumed_session.session_id,
        )

    assert len(recovery_commands) == 1
    assert recovery_commands[0].command_type == COMMAND_RECONCILE_NOW
    assert recovery_commands[0].payload["full_reconcile"] is True
    assert websocket.sent_json[0]["command_type"] == COMMAND_RECONCILE_NOW
    assert stored_command is not None
    assert stored_command.state == WorkerControlCommandStateEnum.TIMED_OUT
    assert stored_command.completed_at is not None
    assert stored_session is not None
    assert stored_session.requires_full_reconcile is True
    assert stored_session.full_reconcile_reason is not None


@pytest.mark.asyncio
async def test_emit_reconcile_for_assigned_worker(command_service):
    sessionmaker = command_service["sessionmaker"]
    worker = command_service["worker"]
    other_worker = command_service["other_worker"]

    async with sessionmaker() as session:
        model = Model(
            name="model-control",
            source=SourceEnum.LOCAL_PATH,
            local_path="/models/control",
            cluster_id=worker.cluster_id,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)
        assert model.id is not None

        model_instance = ModelInstance(
            name="instance-control",
            source=SourceEnum.LOCAL_PATH,
            local_path="/models/control",
            worker_id=worker.id,
            worker_name=worker.name,
            worker_ip=worker.ip,
            worker_ifname=worker.ifname,
            state=ModelInstanceStateEnum.RUNNING,
            model_id=model.id,
            model_name=model.name,
            cluster_id=worker.cluster_id,
        )
        session.add(model_instance)
        await session.commit()
        await session.refresh(model_instance)

        command_service_for_session = WorkerCommandService(session)
        opened_session = await _open_session(
            command_service_for_session,
            worker.id,
            session_id="session-dispatch",
            now=_utc(2026, 3, 28, 15, 0, 0),
        )

    registry = WorkerControlSessionRegistry()
    websocket = FakeWebSocket()
    await registry.register(
        worker_id=worker.id,
        worker_uuid=worker.worker_uuid,
        websocket=websocket,
        generation=opened_session.generation,
        session_id=opened_session.session_id,
    )

    controller = WorkerCommandController()
    controller._dispatch_service = WorkerCommandDispatchService(
        session_factory=sessionmaker,
        session_registry=registry,
    )

    await controller._handle_model_instance_event(
        Event(
            type=EventType.UPDATED,
            data=model_instance,
            changed_fields={"state": ([ModelInstanceStateEnum.STARTING], [ModelInstanceStateEnum.RUNNING])},
        )
    )

    async with sessionmaker() as session:
        commands = list(
            (await session.exec(select(WorkerCommand).order_by("sequence"))).all()
        )

    assert len(websocket.sent_json) == 1
    assert websocket.sent_json[0]["command_type"] == COMMAND_RECONCILE_MODEL_INSTANCE
    assert websocket.sent_json[0]["payload"] == {
        "model_id": model_instance.model_id,
        "model_instance_id": model_instance.id,
    }
    assert websocket.sent_json[0]["session_id"] == opened_session.session_id
    assert len(commands) == 1
    assert commands[0].worker_id == worker.id
    assert commands[0].worker_id != other_worker.id
    assert commands[0].state == WorkerControlCommandStateEnum.SENT
    assert commands[0].worker_session_generation == opened_session.generation
    assert commands[0].payload == {
        "model_id": model_instance.model_id,
        "model_instance_id": model_instance.id,
    }


@pytest.mark.asyncio
async def test_dispatch_leases_command_before_websocket_send(command_service):
    sessionmaker = command_service["sessionmaker"]
    worker = command_service["worker"]

    async with sessionmaker() as session:
        service = WorkerCommandService(session=session)
        opened_session = await _open_session(
            service,
            worker.id,
            session_id="session-dispatch-race",
            now=_utc(2026, 3, 28, 17, 0, 0),
        )

    async def acknowledge_during_send(payload: dict):
        async with sessionmaker() as session:
            service = WorkerCommandService(session=session)
            command = await WorkerCommand.one_by_field(
                session,
                "command_id",
                payload["command_id"],
            )
            assert command is not None
            assert command.worker_session_generation == opened_session.generation
            assert command.state == WorkerControlCommandStateEnum.LEASED
            await service.acknowledge_command(
                payload["command_id"],
                session_id=opened_session.session_id,
                session_generation=opened_session.generation,
                now=_utc(2026, 3, 28, 17, 0, 2),
            )

    registry = WorkerControlSessionRegistry()
    websocket = AckingWebSocket(acknowledge_during_send)
    await registry.register(
        worker_id=worker.id,
        worker_uuid=worker.worker_uuid,
        websocket=websocket,
        generation=opened_session.generation,
        session_id=opened_session.session_id,
    )

    dispatch_service = WorkerCommandDispatchService(
        session_factory=sessionmaker,
        session_registry=registry,
    )

    commands = await dispatch_service.emit_reconcile_now(
        worker_ids=[worker.id],
        idempotency_token="dispatch-race",
        reason="dispatch race regression",
    )

    async with sessionmaker() as session:
        stored_command = await WorkerCommand.one_by_field(
            session,
            "command_id",
            commands[0].command_id,
        )

    assert len(websocket.sent_json) == 1
    assert commands[0].state == WorkerControlCommandStateEnum.ACKNOWLEDGED
    assert stored_command is not None
    assert stored_command.state == WorkerControlCommandStateEnum.ACKNOWLEDGED
    assert stored_command.worker_session_generation == opened_session.generation
    assert stored_command.acknowledged_at == _utc(2026, 3, 28, 17, 0, 2)
    assert stored_command.dispatch_attempts == 1
