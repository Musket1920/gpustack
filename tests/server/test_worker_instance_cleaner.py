from datetime import datetime, timedelta, timezone
from typing import cast

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from unittest.mock import AsyncMock

from gpustack.config.config import Config
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum, SourceEnum
from gpustack.schemas.workers import (
    Worker,
    WorkerReachabilityCapabilities,
    WorkerReachabilityModeEnum,
    WorkerStateEnum,
    WorkerStatus,
)
from gpustack.server import cache as server_cache
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import WorkerController
from gpustack.server.worker_instance_cleaner import WorkerInstanceCleaner


@pytest_asyncio.fixture
async def worker_instance_sessionmaker(tmp_path, monkeypatch):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'worker_instance_cleaner.db'}"
    )

    def sessionmaker():
        return AsyncSession(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    monkeypatch.setattr(
        "gpustack.server.worker_instance_cleaner.async_session", sessionmaker
    )
    monkeypatch.setattr("gpustack.server.controllers.async_session", sessionmaker)
    monkeypatch.setattr(
        server_cache.logger, "trace", server_cache.logger.debug, raising=False
    )
    monkeypatch.setattr(
        "gpustack.server.services.delete_cache_by_key",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "gpustack.mixins.active_record.delete_cache_by_key",
        AsyncMock(),
    )

    async def _batch_update(self, model_instances, source=None):
        names = [model_instance.name for model_instance in model_instances]
        for model_instance in model_instances:
            for key, value in (source or {}).items():
                setattr(model_instance, key, value)
            self.session.add(model_instance)
        await self.session.commit()
        return names

    async def _batch_delete(self, model_instances):
        names = [model_instance.name for model_instance in model_instances]
        for model_instance in model_instances:
            await self.session.delete(model_instance)
        await self.session.commit()
        return names

    monkeypatch.setattr(
        "gpustack.server.services.ModelInstanceService.batch_update",
        _batch_update,
    )
    monkeypatch.setattr(
        "gpustack.server.services.ModelInstanceService.batch_delete",
        _batch_delete,
    )

    yield sessionmaker

    await engine.dispose()


async def _create_cluster(sessionmaker, name: str) -> Cluster:
    async with sessionmaker() as session:
        cluster = Cluster(name=name)
        session.add(cluster)
        await session.commit()
        await session.refresh(cluster)
        assert cluster.id is not None
        return cluster


async def _create_worker(
    sessionmaker,
    *,
    cluster_id: int,
    name: str,
    heartbeat_time: datetime,
    state: WorkerStateEnum,
    reachability_mode: WorkerReachabilityModeEnum,
    capabilities: WorkerReachabilityCapabilities | None = None,
    unreachable: bool = False,
    state_message: str | None = None,
) -> Worker:
    async with sessionmaker() as session:
        worker = Worker(
            name=name,
            labels={},
            cluster_id=cluster_id,
            hostname=f"{name}-host",
            ip="127.0.0.1",
            ifname="eth0",
            port=10150,
            worker_uuid=f"{name}-uuid",
            status=WorkerStatus.get_default_status(),
            heartbeat_time=heartbeat_time,
            state=state,
            state_message=state_message,
            capabilities=capabilities,
            reachability_mode=reachability_mode,
            unreachable=unreachable,
        )
        session.add(worker)
        await session.commit()
        await session.refresh(worker)
        return worker


async def _create_model(sessionmaker, *, cluster_id: int, name: str) -> Model:
    async with sessionmaker() as session:
        model = Model(
            name=name,
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            cluster_id=cluster_id,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)
        assert model.id is not None
        return model


async def _create_instance(
    sessionmaker,
    *,
    model: Model,
    worker: Worker,
    name: str,
    state: ModelInstanceStateEnum = ModelInstanceStateEnum.RUNNING,
) -> ModelInstance:
    async with sessionmaker() as session:
        assert model.id is not None
        assert model.cluster_id is not None
        assert worker.id is not None
        instance = ModelInstance(
            name=name,
            model_id=model.id,
            model_name=model.name,
            source=model.source,
            huggingface_repo_id=model.huggingface_repo_id,
            cluster_id=model.cluster_id,
            worker_id=worker.id,
            worker_name=worker.name,
            state=state,
        )
        session.add(instance)
        await session.commit()
        await session.refresh(instance)
        assert instance.id is not None
        return instance


async def _get_instance(sessionmaker, instance_id: int) -> ModelInstance | None:
    async with sessionmaker() as session:
        return await ModelInstance.one_by_id(session, instance_id)


@pytest.mark.asyncio
async def test_legacy_cleanup_path_marks_reverse_probe_instances_unreachable(
    worker_instance_sessionmaker,
):
    cluster = await _create_cluster(worker_instance_sessionmaker, "legacy-controller")
    assert cluster.id is not None
    worker = await _create_worker(
        worker_instance_sessionmaker,
        cluster_id=cluster.id,
        name="legacy-worker",
        heartbeat_time=datetime.now(timezone.utc),
        state=WorkerStateEnum.NOT_READY,
        reachability_mode=WorkerReachabilityModeEnum.REVERSE_PROBE,
        unreachable=False,
        state_message="Heartbeat lost",
    )
    model = await _create_model(
        worker_instance_sessionmaker,
        cluster_id=cluster.id,
        name="legacy-model",
    )
    instance = await _create_instance(
        worker_instance_sessionmaker,
        model=model,
        worker=worker,
        name="legacy-instance",
    )
    assert instance.id is not None

    await WorkerController(cfg=cast(Config, None))._reconcile(
        Event(
            type=EventType.UPDATED,
            data=worker,
            changed_fields={"state": (WorkerStateEnum.READY, WorkerStateEnum.NOT_READY)},
        )
    )

    refreshed_instance = await _get_instance(worker_instance_sessionmaker, instance.id)
    assert refreshed_instance is not None
    assert refreshed_instance.state == ModelInstanceStateEnum.UNREACHABLE


@pytest.mark.asyncio
async def test_legacy_cleanup_path_preserves_ws_runtime_instances_during_transport_loss(
    worker_instance_sessionmaker,
):
    cluster = await _create_cluster(worker_instance_sessionmaker, "ws-controller")
    assert cluster.id is not None
    worker = await _create_worker(
        worker_instance_sessionmaker,
        cluster_id=cluster.id,
        name="ws-worker",
        heartbeat_time=datetime.now(timezone.utc),
        state=WorkerStateEnum.NOT_READY,
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
        state_message="Outbound worker control websocket is unavailable.",
    )
    model = await _create_model(
        worker_instance_sessionmaker,
        cluster_id=cluster.id,
        name="ws-model",
    )
    instance = await _create_instance(
        worker_instance_sessionmaker,
        model=model,
        worker=worker,
        name="ws-instance",
    )
    assert instance.id is not None

    await WorkerController(cfg=cast(Config, None))._reconcile(
        Event(
            type=EventType.UPDATED,
            data=worker,
            changed_fields={"state": (WorkerStateEnum.READY, WorkerStateEnum.NOT_READY)},
        )
    )

    refreshed_instance = await _get_instance(worker_instance_sessionmaker, instance.id)
    assert refreshed_instance is not None
    assert refreshed_instance.state == ModelInstanceStateEnum.RUNNING


@pytest.mark.asyncio
async def test_legacy_cleanup_path_deletes_reverse_probe_instances_after_grace_period(
    worker_instance_sessionmaker,
    monkeypatch,
):
    now = datetime.now(timezone.utc)
    cluster = await _create_cluster(worker_instance_sessionmaker, "legacy-cleaner")
    assert cluster.id is not None
    worker = await _create_worker(
        worker_instance_sessionmaker,
        cluster_id=cluster.id,
        name="legacy-cleaner-worker",
        heartbeat_time=now - timedelta(minutes=10),
        state=WorkerStateEnum.NOT_READY,
        reachability_mode=WorkerReachabilityModeEnum.REVERSE_PROBE,
        state_message="Heartbeat lost",
    )
    model = await _create_model(
        worker_instance_sessionmaker,
        cluster_id=cluster.id,
        name="legacy-cleaner-model",
    )
    instance = await _create_instance(
        worker_instance_sessionmaker,
        model=model,
        worker=worker,
        name="legacy-cleaner-instance",
    )
    assert instance.id is not None
    monkeypatch.setattr("gpustack.server.worker_instance_cleaner.envs.MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD", 60)

    await WorkerInstanceCleaner(interval=1)._cleanup_offline_worker_instances()

    assert await _get_instance(worker_instance_sessionmaker, instance.id) is None


@pytest.mark.asyncio
async def test_legacy_cleanup_path_preserves_ws_instances_during_transport_loss(
    worker_instance_sessionmaker,
    monkeypatch,
):
    now = datetime.now(timezone.utc)
    cluster = await _create_cluster(worker_instance_sessionmaker, "ws-cleaner")
    assert cluster.id is not None
    worker = await _create_worker(
        worker_instance_sessionmaker,
        cluster_id=cluster.id,
        name="ws-cleaner-worker",
        heartbeat_time=now - timedelta(minutes=10),
        state=WorkerStateEnum.NOT_READY,
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
        state_message="Outbound worker control websocket is unavailable.",
    )
    model = await _create_model(
        worker_instance_sessionmaker,
        cluster_id=cluster.id,
        name="ws-cleaner-model",
    )
    instance = await _create_instance(
        worker_instance_sessionmaker,
        model=model,
        worker=worker,
        name="ws-cleaner-instance",
    )
    assert instance.id is not None
    monkeypatch.setattr("gpustack.server.worker_instance_cleaner.envs.MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD", 60)

    await WorkerInstanceCleaner(interval=1)._cleanup_offline_worker_instances()

    refreshed_instance = await _get_instance(worker_instance_sessionmaker, instance.id)
    assert refreshed_instance is not None
    assert refreshed_instance.state == ModelInstanceStateEnum.RUNNING
