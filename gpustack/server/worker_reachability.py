from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional, cast

import sqlalchemy as sa
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack import envs
from gpustack.schemas.workers import (
    Worker,
    WorkerReachabilityModeEnum,
    WorkerStateEnum,
    WorkerSession,
    WorkerSessionStateEnum,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass
class WorkerReachabilityDecision:
    reverse_probe_required: bool
    reverse_probe_reason: str
    active_control_session: Optional[WorkerSession] = None
    transport_timed_out: bool = False
    transport_message: Optional[str] = None


def _control_session_order_key(worker_session: WorkerSession) -> tuple[int, datetime, int]:
    last_seen_at = worker_session.last_seen_at or worker_session.connected_at
    assert last_seen_at is not None
    return (worker_session.generation, last_seen_at, worker_session.id or 0)


def load_control_sessions_by_worker_id(
    worker_sessions: Iterable[WorkerSession],
    now: Optional[datetime] = None,
) -> Dict[int, WorkerSession]:
    current_time = now or utcnow()
    active_sessions: Dict[int, WorkerSession] = {}
    for worker_session in worker_sessions:
        if worker_session.state != WorkerSessionStateEnum.ACTIVE:
            continue
        if not is_control_session_usable(worker_session, current_time):
            continue
        if worker_session.worker_id is None:
            continue

        current_session = active_sessions.get(worker_session.worker_id)
        if current_session is None or _control_session_order_key(worker_session) > _control_session_order_key(current_session):
            active_sessions[worker_session.worker_id] = worker_session
    return active_sessions


async def fetch_active_control_sessions_by_worker_id(
    session: AsyncSession,
    worker_ids: list[int],
    now: Optional[datetime] = None,
) -> Dict[int, WorkerSession]:
    if not worker_ids:
        return {}

    worker_session_columns = cast(Any, sa.inspect(cast(Any, WorkerSession)).c)
    worker_id_column = cast(Any, worker_session_columns.worker_id)
    generation_column = cast(Any, worker_session_columns.generation)
    id_column = cast(Any, worker_session_columns.id)
    statement = select(WorkerSession).where(
        worker_id_column.in_(worker_ids),
        WorkerSession.state == WorkerSessionStateEnum.ACTIVE,
    ).order_by(
        worker_id_column,
        generation_column.desc(),
        id_column.desc(),
    )
    worker_sessions = list((await session.exec(statement)).all())
    return load_control_sessions_by_worker_id(worker_sessions, now=now)


def is_control_session_usable(
    worker_session: Optional[WorkerSession],
    now: Optional[datetime] = None,
) -> bool:
    if worker_session is None:
        return False

    current_time = now or utcnow()
    if worker_session.state != WorkerSessionStateEnum.ACTIVE:
        return False
    if worker_session.expires_at is not None and worker_session.expires_at < current_time:
        return False

    last_seen_at = worker_session.last_seen_at or worker_session.connected_at
    if last_seen_at is None:
        return False

    timeout_at = last_seen_at + timedelta(
        seconds=envs.WORKER_CONTROL_SESSION_LOSS_TIMEOUT_SECONDS
    )
    return current_time <= timeout_at


def evaluate_worker_reachability(
    worker: Worker,
    active_control_session: Optional[WorkerSession],
    now: Optional[datetime] = None,
) -> WorkerReachabilityDecision:
    current_time = now or utcnow()

    if worker.reachability_mode != WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS:
        worker.set_active_control_session(None)
        return WorkerReachabilityDecision(
            reverse_probe_required=True,
            reverse_probe_reason="applied:legacy_reverse_probe",
        )

    if active_control_session is not None:
        worker.set_active_control_session(active_control_session)
        worker.unreachable = False
        return WorkerReachabilityDecision(
            reverse_probe_required=False,
            reverse_probe_reason="bypassed:active_outbound_control_session",
            active_control_session=active_control_session,
        )

    worker.set_active_control_session(None)
    worker.unreachable = False

    transport_message = _build_transport_unavailable_message(worker, current_time)
    worker.state = WorkerStateEnum.NOT_READY
    worker.state_message = transport_message

    return WorkerReachabilityDecision(
        reverse_probe_required=True,
        reverse_probe_reason="applied:outbound_control_session_unavailable",
        transport_timed_out=True,
        transport_message=transport_message,
    )


def _build_transport_unavailable_message(
    worker: Worker,
    now: Optional[datetime] = None,
) -> str:
    current_time = now or utcnow()
    heartbeat_hint = "no recent control session"
    if worker.heartbeat_time is not None:
        heartbeat_hint = worker.heartbeat_time.isoformat()

    timeout_seconds = envs.WORKER_CONTROL_SESSION_LOSS_TIMEOUT_SECONDS
    return (
        "Outbound worker control websocket is unavailable or stale "
        f"(timeout: {timeout_seconds}s, worker heartbeat: {heartbeat_hint}, now: {current_time.isoformat()}). "
        "Reverse probe was applied while transport was unavailable, and the worker is blocked "
        "from new scheduling until the outbound control session reconnects."
    )
