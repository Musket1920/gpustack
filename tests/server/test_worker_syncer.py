from datetime import datetime, timedelta, timezone
import sys
import types

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack import envs
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.workers import (
    Worker,
    WorkerControlChannelEnum,
    WorkerControlPlaneHealthEnum,
    WorkerReachabilityCapabilities,
    WorkerReachabilityHealthEnum,
    WorkerReachabilityModeEnum,
    WorkerRuntimeHealthEnum,
    WorkerSession,
    WorkerSessionStateEnum,
    WorkerStateEnum,
    WorkerStatus,
    WorkerTransportHealthEnum,
)
from gpustack.server import cache as server_cache
from gpustack.server import worker_status_buffer
from gpustack.server.worker_syncer import WorkerSyncer

sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))

from gpustack.worker.control_client import worker_control_reachability_mode


@pytest_asyncio.fixture
async def worker_syncer_sessionmaker(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'worker_syncer.db'}")

    def sessionmaker():
        return AsyncSession(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    monkeypatch.setattr("gpustack.server.worker_syncer.async_session", sessionmaker)
    monkeypatch.setattr("gpustack.server.worker_status_buffer.async_session", sessionmaker)
    monkeypatch.setattr(server_cache.logger, "trace", server_cache.logger.debug, raising=False)

    yield sessionmaker

    await engine.dispose()


async def _create_worker(
    sessionmaker,
    *,
    name: str,
    now: datetime,
    capabilities: WorkerReachabilityCapabilities | None = None,
    reachability_mode: WorkerReachabilityModeEnum = WorkerReachabilityModeEnum.REVERSE_PROBE,
) -> Worker:
    async with sessionmaker() as session:
        cluster = Cluster(name=f"cluster-{name}")
        session.add(cluster)
        await session.commit()
        await session.refresh(cluster)

        worker = Worker(
            name=name,
            labels={},
            cluster_id=cluster.id,
            hostname=f"{name}-host",
            ip="127.0.0.1",
            ifname="eth0",
            port=10150,
            worker_uuid=f"{name}-uuid",
            status=WorkerStatus.get_default_status(),
            heartbeat_time=now,
            capabilities=capabilities,
            reachability_mode=reachability_mode,
        )
        worker.compute_state()
        session.add(worker)
        await session.commit()
        await session.refresh(worker)
        return worker


async def _create_session(
    sessionmaker,
    *,
    worker_id: int,
    session_id: str,
    last_seen_at: datetime,
    expires_at: datetime,
) -> WorkerSession:
    async with sessionmaker() as session:
        worker_session = WorkerSession(
            session_id=session_id,
            worker_id=worker_id,
            generation=1,
            control_channel=WorkerControlChannelEnum.OUTBOUND_CONTROL_WS,
            reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
            state=WorkerSessionStateEnum.ACTIVE,
            connected_at=last_seen_at,
            last_seen_at=last_seen_at,
            expires_at=expires_at,
        )
        session.add(worker_session)
        await session.commit()
        await session.refresh(worker_session)
        return worker_session


async def _get_worker(sessionmaker, worker_id: int) -> Worker:
    async with sessionmaker() as session:
        worker = await Worker.one_by_id(session, worker_id)
        assert worker is not None
        return worker


def test_legacy_unreachable_still_works():
    worker = Worker(
        id=1,
        name="legacy-worker",
        labels={},
        cluster_id=1,
        hostname="legacy-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="legacy-worker-uuid",
        status=WorkerStatus.get_default_status(),
        heartbeat_time=datetime.now(timezone.utc),
        unreachable=True,
        reachability_mode=WorkerReachabilityModeEnum.REVERSE_PROBE,
    )

    changed_workers = WorkerSyncer.filter_state_change_workers([worker])

    assert changed_workers == [worker]
    assert worker.state == WorkerStateEnum.UNREACHABLE
    assert "healthz" in (worker.state_message or "")
    assert worker.health_status.runtime == WorkerRuntimeHealthEnum.READY
    assert worker.health_status.reachability == WorkerReachabilityHealthEnum.UNREACHABLE
    assert worker.health_status.transport == WorkerTransportHealthEnum.UNSUPPORTED
    assert worker.health_status.control_plane == WorkerControlPlaneHealthEnum.UNSUPPORTED


def test_legacy_mode_flag_keeps_reverse_probe_default(monkeypatch):
    monkeypatch.setattr(envs, "WORKER_CONTROL_ROLLOUT_MODE", "legacy_only")

    assert worker_control_reachability_mode() == WorkerReachabilityModeEnum.REVERSE_PROBE


@pytest.mark.asyncio
async def test_skip_reverse_probe_for_ws_worker(
    worker_syncer_sessionmaker, monkeypatch, caplog
):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    monkeypatch.setattr(envs, "WORKER_UNREACHABLE_CHECK_MODE", "enabled")
    monkeypatch.setattr(envs, "WORKER_CONTROL_SESSION_LOSS_TIMEOUT_SECONDS", 45)

    worker = await _create_worker(
        worker_syncer_sessionmaker,
        name="ws-worker-active",
        now=now,
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
    )
    assert worker.id is not None
    await _create_session(
        worker_syncer_sessionmaker,
        worker_id=worker.id,
        session_id="session-active",
        last_seen_at=now,
        expires_at=now + timedelta(seconds=300),
    )

    call_count = 0

    async def _unexpected_probe(self, current_worker):
        nonlocal call_count
        call_count += 1
        raise AssertionError(f"reverse probe should be skipped for {current_worker.name}")

    monkeypatch.setattr(WorkerSyncer, "is_worker_reachable", _unexpected_probe)

    with caplog.at_level("INFO"):
        await WorkerSyncer(interval=1, worker_unreachable_timeout=1)._sync_workers_states()

    refreshed_worker = await _get_worker(worker_syncer_sessionmaker, worker.id)

    assert call_count == 0
    assert refreshed_worker.state == WorkerStateEnum.READY
    assert refreshed_worker.unreachable is False
    assert refreshed_worker.state_message is None
    assert "bypassed:active_outbound_control_session" in caplog.text


@pytest.mark.asyncio
async def test_transport_timeout_blocks_scheduling(
    worker_syncer_sessionmaker, monkeypatch, caplog
):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    monkeypatch.setattr(envs, "WORKER_UNREACHABLE_CHECK_MODE", "enabled")
    monkeypatch.setattr(envs, "WORKER_CONTROL_SESSION_LOSS_TIMEOUT_SECONDS", 30)

    worker = await _create_worker(
        worker_syncer_sessionmaker,
        name="ws-worker-stale",
        now=now,
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
    )
    assert worker.id is not None
    await _create_session(
        worker_syncer_sessionmaker,
        worker_id=worker.id,
        session_id="session-stale",
        last_seen_at=now - timedelta(seconds=31),
        expires_at=now + timedelta(seconds=300),
    )

    async def _failed_probe(self, current_worker):
        assert current_worker.name == "ws-worker-stale"
        return False

    monkeypatch.setattr(WorkerSyncer, "is_worker_reachable", _failed_probe)

    with caplog.at_level("INFO"):
        await WorkerSyncer(interval=1, worker_unreachable_timeout=1)._sync_workers_states()

    refreshed_worker = await _get_worker(worker_syncer_sessionmaker, worker.id)
    refreshed_worker.set_active_control_session(None)
    assert refreshed_worker.state == WorkerStateEnum.NOT_READY
    assert refreshed_worker.health_status.transport == WorkerTransportHealthEnum.DISCONNECTED
    assert refreshed_worker.health_status.control_plane == WorkerControlPlaneHealthEnum.UNAVAILABLE
    assert refreshed_worker.health_status.reachability == WorkerReachabilityHealthEnum.NOT_APPLICABLE
    assert "blocked from new scheduling" in (refreshed_worker.state_message or "")
    assert "Reverse probe failed" in (refreshed_worker.state_message or "")
    assert "reverse probe failed" in caplog.text

    worker_status_buffer.worker_status_flush_buffer[worker.id] = {
        "status": WorkerStatus.get_default_status(),
        "heartbeat_time": now,
    }

    await worker_status_buffer.flush_worker_status()
    post_flush_worker = await _get_worker(worker_syncer_sessionmaker, worker.id)

    assert post_flush_worker.state == WorkerStateEnum.NOT_READY
    assert "blocked from new scheduling" in (post_flush_worker.state_message or "")
