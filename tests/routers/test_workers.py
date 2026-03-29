from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.routes.workers import (
    get_worker,
    load_active_control_sessions,
    to_worker_public,
    update_worker_data,
)
from gpustack.schemas.workers import (
    WorkerReachabilityCapabilities,
    WorkerControlChannelEnum,
    WorkerControlPlaneHealthEnum,
    WorkerRuntimeHealthEnum,
    WorkerSession,
    WorkerSessionStateEnum,
    WorkerTransportHealthEnum,
    WorkerReachabilityModeEnum,
    WorkerReachabilityHealthEnum,
    Worker,
    WorkerCreate,
    WorkerStateEnum,
    WorkerTransportHealthEnum,
    WorkerStatus,
    Maintenance,
    SystemReserved,
)
from gpustack.schemas.clusters import Cluster, ClusterProvider


def test_update_worker_data_preserves_maintenance_mode():
    """
    Test that maintenance mode is preserved when a worker re-registers.
    This verifies the fix for the issue where workers automatically exit
    maintenance mode after restart.
    """
    # Create an existing worker with maintenance mode enabled
    existing_worker = Worker(
        id=1,
        name="test-worker",
        labels={"env": "test"},
        maintenance=Maintenance(enabled=True, message="Scheduled maintenance"),
        state=WorkerStateEnum.MAINTENANCE,
        cluster_id=1,
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="test-uuid-123",
        status=WorkerStatus.get_default_status(),
    )

    # Create a worker registration request without maintenance field
    # (simulating a worker restart/re-registration)
    worker_in = WorkerCreate(
        name="test-worker",
        labels={"env": "test", "new": "label"},
        maintenance=None,  # Not set during re-registration
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="test-uuid-123",
        cluster_id=1,
        status=WorkerStatus.get_default_status(),
        system_reserved=SystemReserved(ram=0, vram=0),
    )

    # Update the worker data
    updated_worker = update_worker_data(worker_in, existing=existing_worker)

    # Verify that maintenance mode is preserved
    assert updated_worker.maintenance is not None
    assert updated_worker.maintenance.enabled is True
    assert updated_worker.maintenance.message == "Scheduled maintenance"
    # State will be computed as MAINTENANCE because of compute_state()
    assert updated_worker.state == WorkerStateEnum.MAINTENANCE


def test_update_worker_data_can_disable_maintenance_mode():
    """
    Test that maintenance mode can be explicitly disabled when provided.
    """
    # Create an existing worker with maintenance mode enabled
    existing_worker = Worker(
        id=1,
        name="test-worker",
        labels={"env": "test"},
        maintenance=Maintenance(enabled=True, message="Scheduled maintenance"),
        state=WorkerStateEnum.MAINTENANCE,
        cluster_id=1,
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="test-uuid-123",
        status=WorkerStatus.get_default_status(),
    )

    # Create a worker update request with maintenance explicitly disabled
    worker_in = WorkerCreate(
        name="test-worker",
        labels={"env": "test"},
        maintenance=Maintenance(enabled=False, message=None),
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="test-uuid-123",
        cluster_id=1,
        status=WorkerStatus.get_default_status(),
        system_reserved=SystemReserved(ram=0, vram=0),
    )

    # Update the worker data
    updated_worker = update_worker_data(worker_in, existing=existing_worker)

    # Verify that maintenance mode is disabled
    assert updated_worker.maintenance is not None
    assert updated_worker.maintenance.enabled is False
    assert updated_worker.maintenance.message is None
    # State will be computed based on heartbeat, but maintenance is disabled
    # Since maintenance is disabled, the state won't be MAINTENANCE
    assert updated_worker.state != WorkerStateEnum.MAINTENANCE


def test_update_worker_data_new_worker_without_maintenance():
    """
    Test that a new worker can be created without maintenance mode.
    """
    # Create a new worker registration request
    worker_in = WorkerCreate(
        name="new-worker",
        labels={"env": "prod"},
        maintenance=None,
        hostname="new-host",
        ip="192.168.1.101",
        ifname="eth0",
        port=8080,
        worker_uuid="new-uuid-456",
        cluster_id=1,
        status=WorkerStatus.get_default_status(),
        system_reserved=SystemReserved(ram=0, vram=0),
    )

    # Create cluster for new worker
    cluster = Cluster(
        id=1,
        name="test-cluster",
        provider=ClusterProvider.Docker,
    )

    # Create a new worker (no existing worker)
    new_worker = update_worker_data(worker_in, existing=None, cluster=cluster)

    # Verify that the new worker is created without maintenance mode
    assert new_worker.maintenance is None
    # State may be NOT_READY due to missing heartbeat, but not MAINTENANCE
    assert new_worker.state != WorkerStateEnum.MAINTENANCE


def test_update_worker_data_preserves_labels_merge():
    """
    Test that labels are properly merged when updating a worker.
    """
    # Create an existing worker with some labels
    existing_worker = Worker(
        id=1,
        name="test-worker",
        labels={"env": "test", "region": "us-west"},
        maintenance=None,
        state=WorkerStateEnum.READY,
        cluster_id=1,
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="test-uuid-123",
        status=WorkerStatus.get_default_status(),
    )

    # Create a worker update with new labels
    worker_in = WorkerCreate(
        name="test-worker",
        labels={"env": "prod", "zone": "a"},  # env changes, zone is new
        maintenance=None,
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="test-uuid-123",
        cluster_id=1,
        status=WorkerStatus.get_default_status(),
        system_reserved=SystemReserved(ram=0, vram=0),
    )

    # Update the worker data
    updated_worker = update_worker_data(worker_in, existing=existing_worker)

    # Verify that labels are properly merged
    assert updated_worker.labels["env"] == "prod"  # Updated
    assert updated_worker.labels["region"] == "us-west"  # Preserved
    assert updated_worker.labels["zone"] == "a"  # New


def test_update_worker_data_preserves_legacy_outbound_reachability_mode_contract():
    existing_worker = Worker(
        id=1,
        name="test-worker",
        labels={"env": "test"},
        cluster_id=1,
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="test-uuid-123",
        status=WorkerStatus.get_default_status(),
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
    )

    worker_in = WorkerCreate(
        name="test-worker",
        labels={"env": "test", "new": "label"},
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="test-uuid-123",
        cluster_id=1,
        status=WorkerStatus.get_default_status(),
        system_reserved=SystemReserved(ram=0, vram=0),
    )

    updated_worker = update_worker_data(worker_in, existing=existing_worker)

    assert updated_worker.capabilities is not None
    assert updated_worker.capabilities.outbound_control_ws is True
    assert (
        updated_worker.reachability_mode
        == WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS
    )


def test_ws_capable_state_model():
    now = datetime.now(timezone.utc)
    active_session = WorkerSession(
        session_id="session-ws-active",
        worker_id=1,
        control_channel=WorkerControlChannelEnum.OUTBOUND_CONTROL_WS,
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        state=WorkerSessionStateEnum.ACTIVE,
        connected_at=now,
        last_seen_at=now,
    )

    ready_worker = Worker(
        id=1,
        name="ws-worker-ready",
        labels={},
        cluster_id=1,
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="ws-worker-ready-uuid",
        status=WorkerStatus.get_default_status(),
        heartbeat_time=now,
        unreachable=True,
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
    )
    ready_worker.created_at = now
    ready_worker.updated_at = now

    ready_worker.compute_state()
    ready_public = to_worker_public(ready_worker, False, active_session)

    assert ready_worker.state == WorkerStateEnum.READY
    assert ready_public.health.runtime == WorkerRuntimeHealthEnum.READY
    assert ready_public.health.transport == WorkerTransportHealthEnum.CONNECTED
    assert ready_public.health.control_plane == WorkerControlPlaneHealthEnum.AVAILABLE
    assert (
        ready_public.health.reachability
        == WorkerReachabilityHealthEnum.NOT_APPLICABLE
    )

    stale_worker = Worker(
        id=2,
        name="ws-worker-stale",
        labels={},
        cluster_id=1,
        hostname="test-host",
        ip="192.168.1.101",
        ifname="eth0",
        port=8081,
        worker_uuid="ws-worker-stale-uuid",
        status=WorkerStatus.get_default_status(),
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
    )
    stale_worker.created_at = now
    stale_worker.updated_at = now

    stale_worker.compute_state()
    stale_public = to_worker_public(stale_worker, False, active_session)

    assert stale_worker.state == WorkerStateEnum.NOT_READY
    assert stale_public.health.runtime == WorkerRuntimeHealthEnum.NOT_READY
    assert stale_public.health.transport == WorkerTransportHealthEnum.CONNECTED
    assert stale_public.health.control_plane == WorkerControlPlaneHealthEnum.AVAILABLE


@pytest.mark.asyncio
async def test_load_active_control_sessions_prefers_highest_valid_generation():
    now = datetime.now(timezone.utc)
    worker_id = 5
    preferred_session = WorkerSession(
        id=22,
        session_id="session-generation-2",
        worker_id=worker_id,
        generation=2,
        control_channel=WorkerControlChannelEnum.OUTBOUND_CONTROL_WS,
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        state=WorkerSessionStateEnum.ACTIVE,
        connected_at=now - timedelta(seconds=4),
        last_seen_at=now - timedelta(seconds=1),
        expires_at=now + timedelta(minutes=5),
    )
    expired_newer_session = WorkerSession(
        id=33,
        session_id="session-generation-3-expired",
        worker_id=worker_id,
        generation=3,
        control_channel=WorkerControlChannelEnum.OUTBOUND_CONTROL_WS,
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        state=WorkerSessionStateEnum.ACTIVE,
        connected_at=now - timedelta(minutes=10),
        last_seen_at=now - timedelta(minutes=10),
        expires_at=now - timedelta(seconds=1),
    )
    older_session = WorkerSession(
        id=11,
        session_id="session-generation-1",
        worker_id=worker_id,
        generation=1,
        control_channel=WorkerControlChannelEnum.OUTBOUND_CONTROL_WS,
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        state=WorkerSessionStateEnum.ACTIVE,
        connected_at=now - timedelta(seconds=8),
        last_seen_at=now - timedelta(seconds=2),
        expires_at=now + timedelta(minutes=5),
    )

    session = MagicMock()
    session.exec = AsyncMock(
        return_value=MagicMock(
            all=MagicMock(
                return_value=[preferred_session, expired_newer_session, older_session]
            )
        )
    )

    active_sessions = await load_active_control_sessions(session, [worker_id])

    session.exec.assert_awaited_once()
    assert list(active_sessions) == [worker_id]
    assert active_sessions[worker_id].session_id == preferred_session.session_id
    assert active_sessions[worker_id].generation == preferred_session.generation



def test_worker_state_dimensions_exposed():
    now = datetime.now(timezone.utc)
    active_session = WorkerSession(
        session_id="session-ws-active",
        worker_id=3,
        control_channel=WorkerControlChannelEnum.OUTBOUND_CONTROL_WS,
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        state=WorkerSessionStateEnum.ACTIVE,
        connected_at=now,
        last_seen_at=now,
    )

    nat_worker = Worker(
        id=3,
        name="ws-worker-nat",
        labels={},
        cluster_id=1,
        hostname="test-host",
        ip="192.168.1.102",
        ifname="eth0",
        port=8082,
        worker_uuid="ws-worker-nat-uuid",
        status=WorkerStatus.get_default_status(),
        heartbeat_time=now,
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
    )
    nat_worker.created_at = now
    nat_worker.updated_at = now

    nat_public = to_worker_public(nat_worker, False, active_session)

    assert nat_public.health.transport == WorkerTransportHealthEnum.CONNECTED
    assert nat_public.health.control_plane == WorkerControlPlaneHealthEnum.AVAILABLE
    assert nat_public.health.runtime == WorkerRuntimeHealthEnum.READY
    assert nat_public.health.reachability == WorkerReachabilityHealthEnum.NOT_APPLICABLE
    assert nat_public.reverse_http_available is False
    assert nat_public.reverse_http_unavailable_message is not None
    assert "reverse-only server-to-worker actions" in nat_public.reverse_http_unavailable_message

    legacy_worker = Worker(
        id=4,
        name="legacy-worker",
        labels={},
        cluster_id=1,
        hostname="legacy-host",
        ip="192.168.1.103",
        ifname="eth0",
        port=8083,
        worker_uuid="legacy-worker-uuid",
        status=WorkerStatus.get_default_status(),
        heartbeat_time=now,
        reachability_mode=WorkerReachabilityModeEnum.REVERSE_PROBE,
    )
    legacy_worker.created_at = now
    legacy_worker.updated_at = now

    legacy_public = to_worker_public(legacy_worker, False, None)

    assert legacy_public.health.transport == WorkerTransportHealthEnum.UNSUPPORTED
    assert legacy_public.health.control_plane == WorkerControlPlaneHealthEnum.UNSUPPORTED
    assert legacy_public.health.runtime == WorkerRuntimeHealthEnum.READY
    assert legacy_public.health.reachability == WorkerReachabilityHealthEnum.REACHABLE
    assert legacy_public.reverse_http_available is True
    assert legacy_public.reverse_http_unavailable_message is None


@pytest.mark.asyncio
async def test_get_worker_ignores_expired_active_control_session():
    now = datetime.now(timezone.utc)
    worker = Worker(
        id=7,
        name="ws-worker-expired-session",
        labels={},
        cluster_id=1,
        hostname="test-host",
        ip="192.168.1.107",
        ifname="eth0",
        port=8087,
        worker_uuid="ws-worker-expired-session-uuid",
        status=WorkerStatus.get_default_status(),
        heartbeat_time=now,
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
    )
    worker.created_at = now
    worker.updated_at = now

    stale_session = WorkerSession(
        session_id="session-expired-active-row",
        worker_id=cast(int, worker.id),
        control_channel=WorkerControlChannelEnum.OUTBOUND_CONTROL_WS,
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        state=WorkerSessionStateEnum.ACTIVE,
        connected_at=now - timedelta(minutes=10),
        last_seen_at=now - timedelta(minutes=10),
        expires_at=now - timedelta(seconds=1),
    )

    session = MagicMock()
    session.exec = AsyncMock(
        return_value=MagicMock(all=MagicMock(return_value=[stale_session]))
    )

    with patch(
        "gpustack.routes.workers.Worker.one_by_id",
        AsyncMock(return_value=worker),
    ):
        worker_public = await get_worker(
            cast(Any, SimpleNamespace(worker=None)),
            session,
            cast(int, worker.id),
        )

    session.exec.assert_awaited_once()
    assert worker_public.health.transport == WorkerTransportHealthEnum.DISCONNECTED
    assert (
        worker_public.health.control_plane
        == WorkerControlPlaneHealthEnum.UNAVAILABLE
    )
    assert worker_public.health.runtime == WorkerRuntimeHealthEnum.READY
