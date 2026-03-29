from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
from pydantic import ValidationError
import sqlalchemy as sa
from sqlmodel import SQLModel

from gpustack import envs
from gpustack.schemas.workers import (
    WorkerCommand,
    WorkerCommandAckMessage,
    WorkerCommandResultMessage,
    WorkerControlCommandStateEnum,
    WorkerCreate,
    WorkerHelloMessage,
    WorkerReachabilityCapabilities,
    WorkerReachabilityModeEnum,
    WorkerSession,
    WorkerStatusStored,
    WorkerStatus,
    SystemReserved,
)


def _worker_create(**kwargs) -> WorkerCreate:
    return WorkerCreate(
        name="worker-outbound-control",
        labels={"env": "test"},
        hostname="worker-host",
        ip="192.168.1.10",
        ifname="eth0",
        port=8080,
        worker_uuid="worker-uuid",
        cluster_id=1,
        status=WorkerStatus.get_default_status(),
        system_reserved=SystemReserved(ram=0, vram=0),
        **kwargs,
    )


def test_legacy_worker_registration_defaults_to_reverse_probe():
    worker = _worker_create()

    assert worker.capabilities is None
    assert worker.reachability_mode == WorkerReachabilityModeEnum.REVERSE_PROBE


def test_outbound_control_ws_defaults_from_reachability_mode_env(monkeypatch):
    monkeypatch.setattr(
        envs,
        "WORKER_DEFAULT_REACHABILITY_MODE",
        WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS.value,
    )

    worker = _worker_create(
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True)
    )

    assert worker.reachability_mode == WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS


def test_persisted_legacy_worker_revalidation_defaults_to_reverse_probe():
    persisted_worker = _worker_create().model_dump()

    revalidated_worker = WorkerStatusStored.model_validate(persisted_worker)

    assert revalidated_worker.reachability_mode == WorkerReachabilityModeEnum.REVERSE_PROBE


def test_persisted_outbound_capable_worker_preserves_stored_reachability_mode(
    monkeypatch,
):
    monkeypatch.setattr(
        envs,
        "WORKER_DEFAULT_REACHABILITY_MODE",
        WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS.value,
    )
    persisted_worker = _worker_create(
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
        reachability_mode=WorkerReachabilityModeEnum.REVERSE_PROBE,
    ).model_dump()

    revalidated_worker = WorkerStatusStored.model_validate(persisted_worker)

    assert revalidated_worker.reachability_mode == WorkerReachabilityModeEnum.REVERSE_PROBE


def test_invalid_reachability_mode_requires_explicit_capability():
    with pytest.raises(ValidationError):
        _worker_create(reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS)


def test_invalid_control_message_rejected_ack_requires_error_message():
    with pytest.raises(ValidationError):
        WorkerCommandAckMessage(
            session_id="session-1",
            command_id="command-1",
            accepted=False,
        )


def test_invalid_control_message_failed_result_requires_error_message():
    with pytest.raises(ValidationError):
        WorkerCommandResultMessage(
            session_id="session-1",
            command_id="command-1",
            state=WorkerControlCommandStateEnum.FAILED,
        )


def test_invalid_control_message_hello_mode_requires_capability():
    with pytest.raises(ValidationError):
        WorkerHelloMessage(
            session_id="session-1",
            worker_uuid="worker-uuid",
            reachability_mode=WorkerReachabilityModeEnum.REVERSE_HTTP,
        )


def test_migration_metadata_registers_worker_session_and_command_tables():
    table_names = SQLModel.metadata.tables.keys()

    assert WorkerSession.__tablename__ in table_names
    assert WorkerCommand.__tablename__ in table_names


def _load_migration_module(migration_file_name: str):
    migration_path = (
        Path(__file__).resolve().parents[2]
        / "gpustack"
        / "migrations"
        / "versions"
        / migration_file_name
    )
    spec = spec_from_file_location(f"test_{migration_path.stem}", migration_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_outbound_control_migration_preserves_existing_workers_and_defaults_legacy_mode(
    tmp_path: Path,
):
    migration = _load_migration_module(
        "2026_03_28_1400-bf4c3d9e21a1_add_worker_outbound_control_contracts.py"
    )
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'worker_outbound_control.db'}")
    metadata = sa.MetaData()
    workers = sa.Table(
        "workers",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("worker_uuid", sa.String(), nullable=False),
    )
    metadata.create_all(engine)

    try:
        with engine.begin() as conn:
            conn.execute(
                workers.insert(),
                [
                    {
                        "id": 1,
                        "name": "legacy-worker",
                        "worker_uuid": "legacy-worker-uuid",
                    }
                ],
            )

            context = MigrationContext.configure(conn)
            with Operations.context(context):
                migration.upgrade()

            inspector = sa.inspect(conn)
            rows = conn.execute(
                sa.text(
                    "SELECT id, name, worker_uuid, capabilities, reachability_mode "
                    "FROM workers ORDER BY id"
                )
            ).mappings().all()
            table_names = inspector.get_table_names()

        assert len(rows) == 1
        assert rows[0]["id"] == 1
        assert rows[0]["name"] == "legacy-worker"
        assert rows[0]["worker_uuid"] == "legacy-worker-uuid"
        assert rows[0]["capabilities"] is None
        assert (
            rows[0]["reachability_mode"]
            == WorkerReachabilityModeEnum.REVERSE_PROBE.value
        )
        assert "worker_sessions" in table_names
        assert "worker_commands" in table_names
    finally:
        engine.dispose()
