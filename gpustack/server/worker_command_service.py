import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Optional
import uuid

from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack import envs
from gpustack.server.db import async_session
from gpustack.server.worker_control import worker_control_session_registry
from gpustack.server.worker_control_observability import (
    classify_result_failure_source,
    control_log_extra,
    record_command_ack_latency,
    record_command_failure,
    record_command_result_latency,
    record_replay_fallback,
    sanitize_log_value,
)
from gpustack.schemas.workers import (
    WorkerCommandMessage,
    WorkerCommand,
    WorkerControlChannelEnum,
    WorkerControlCommandStateEnum,
    WorkerErrorMessage,
    WorkerReachabilityModeEnum,
    WorkerSession,
    WorkerSessionStateEnum,
)


logger = logging.getLogger(__name__)


class WorkerSessionOpenSequencer:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._worker_locks: dict[int, asyncio.Lock] = {}

    async def lock_for(self, worker_id: int) -> asyncio.Lock:
        async with self._lock:
            worker_lock = self._worker_locks.get(worker_id)
            if worker_lock is None:
                worker_lock = asyncio.Lock()
                self._worker_locks[worker_id] = worker_lock
            return worker_lock


worker_session_open_sequencer = WorkerSessionOpenSequencer()


class WorkerCommandCreateSequencer:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._worker_locks: dict[int, asyncio.Lock] = {}

    async def lock_for(self, worker_id: int) -> asyncio.Lock:
        async with self._lock:
            worker_lock = self._worker_locks.get(worker_id)
            if worker_lock is None:
                worker_lock = asyncio.Lock()
                self._worker_locks[worker_id] = worker_lock
            return worker_lock


worker_command_create_sequencer = WorkerCommandCreateSequencer()


COMMAND_RECONCILE_NOW = "reconcile_now"
COMMAND_RECONCILE_MODEL_INSTANCE = "reconcile_model_instance"
COMMAND_SYNC_RUNTIME_STATE = "sync_runtime_state"
CONTROL_ERROR_REQUIRES_FULL_RECONCILE = {
    "full_reconcile_required",
    "replay_gap",
    "resume_rejected",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


class StaleWorkerSessionError(ValueError):
    pass


TERMINAL_COMMAND_STATES = {
    WorkerControlCommandStateEnum.SUCCEEDED,
    WorkerControlCommandStateEnum.FAILED,
    WorkerControlCommandStateEnum.CANCELLED,
    WorkerControlCommandStateEnum.EXPIRED,
    WorkerControlCommandStateEnum.TIMED_OUT,
    WorkerControlCommandStateEnum.SUPERSEDED,
}


@dataclass
class WorkerCommandReplayWindow:
    commands: list[WorkerCommand] = field(default_factory=list)
    requires_full_reconcile: bool = False
    full_reconcile_reason: Optional[str] = None
    replay_floor_sequence: int = 0
    replay_through_sequence: int = 0


class WorkerCommandService:
    def __init__(
        self,
        session: AsyncSession,
        replay_window_size: int = 1024,
    ):
        self.session = session
        self.replay_window_size = replay_window_size

    async def open_session(
        self,
        worker_id: int,
        *,
        control_channel: WorkerControlChannelEnum,
        reachability_mode: WorkerReachabilityModeEnum,
        session_id: Optional[str] = None,
        protocol_version: int = 1,
        details: Optional[dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> WorkerSession:
        now = now or utcnow()
        worker_lock = await worker_session_open_sequencer.lock_for(worker_id)
        async with worker_lock:
            generation = await self._next_session_generation(worker_id)
            expires_at = now + timedelta(seconds=envs.WORKER_CONTROL_SESSION_TTL_SECONDS)

            existing_session = None
            if session_id is not None:
                existing_session = await WorkerSession.one_by_field(
                    self.session, "session_id", session_id
                )
                if existing_session is not None and existing_session.worker_id != worker_id:
                    raise ValueError("session_id already belongs to a different worker")

            active_sessions = await WorkerSession.all_by_fields(
                self.session,
                fields={"worker_id": worker_id, "state": WorkerSessionStateEnum.ACTIVE},
            )
            for active_session in active_sessions:
                if existing_session is not None and active_session.id == existing_session.id:
                    continue
                active_session.state = WorkerSessionStateEnum.STALE
                active_session.disconnected_at = now
                active_session.last_seen_at = now
                self.session.add(active_session)

            if existing_session is None:
                worker_session = WorkerSession(
                    session_id=session_id or str(uuid.uuid4()),
                    worker_id=worker_id,
                    generation=generation,
                    control_channel=control_channel,
                    reachability_mode=reachability_mode,
                    state=WorkerSessionStateEnum.ACTIVE,
                    protocol_version=protocol_version,
                    connected_at=now,
                    last_seen_at=now,
                    disconnected_at=None,
                    expires_at=expires_at,
                    details=details,
                )
                self.session.add(worker_session)
            else:
                worker_session = existing_session
                worker_session.generation = generation
                worker_session.control_channel = control_channel
                worker_session.reachability_mode = reachability_mode
                worker_session.state = WorkerSessionStateEnum.ACTIVE
                worker_session.protocol_version = protocol_version
                worker_session.connected_at = now
                worker_session.last_seen_at = now
                worker_session.disconnected_at = None
                worker_session.expires_at = expires_at
                worker_session.details = details
                self.session.add(worker_session)

            await self.session.commit()
            await self.session.refresh(worker_session)
            return worker_session

    async def close_session(
        self,
        session_id: str,
        generation: int,
        *,
        state: WorkerSessionStateEnum = WorkerSessionStateEnum.CLOSED,
        now: Optional[datetime] = None,
    ) -> WorkerSession:
        now = now or utcnow()
        worker_session = await self._require_fresh_session(session_id, generation, now)
        worker_session.state = state
        worker_session.disconnected_at = now
        worker_session.last_seen_at = now
        self.session.add(worker_session)
        await self.session.commit()
        await self.session.refresh(worker_session)
        return worker_session

    async def touch_session(
        self,
        session_id: str,
        generation: int,
        *,
        replay_cursor: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> WorkerSession:
        now = now or utcnow()
        worker_session = await self._require_fresh_session(session_id, generation, now)
        worker_session.last_seen_at = now
        worker_session.expires_at = now + timedelta(
            seconds=envs.WORKER_CONTROL_SESSION_TTL_SECONDS
        )
        if replay_cursor is not None:
            worker_session.replay_cursor = max(worker_session.replay_cursor, replay_cursor)
        self.session.add(worker_session)
        await self.session.commit()
        await self.session.refresh(worker_session)
        return worker_session

    async def create_command(
        self,
        worker_id: int,
        *,
        command_type: str,
        payload: Optional[dict[str, Any]] = None,
        command_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        desired_worker_state_revision: Optional[int] = None,
        not_before: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
        now: Optional[datetime] = None,
    ) -> WorkerCommand:
        now = now or utcnow()
        worker_lock = await worker_command_create_sequencer.lock_for(worker_id)
        async with worker_lock:
            if idempotency_key is not None:
                existing_command = await self._get_command_by_idempotency_key(
                    worker_id, idempotency_key
                )
                if existing_command is not None:
                    return existing_command

            command = WorkerCommand(
                command_id=command_id or str(uuid.uuid4()),
                worker_id=worker_id,
                sequence=await self._next_command_sequence(worker_id),
                command_type=command_type,
                payload=payload or {},
                idempotency_key=idempotency_key,
                desired_worker_state_revision=desired_worker_state_revision,
                not_before=not_before,
                expires_at=expires_at
                or now + timedelta(seconds=envs.WORKER_CONTROL_COMMAND_TTL_SECONDS),
            )
            self.session.add(command)
            await self.session.commit()
            await self.session.refresh(command)
            return command

    async def mark_command_sent(
        self,
        command_id: str,
        *,
        session_id: str,
        session_generation: int,
        now: Optional[datetime] = None,
    ) -> WorkerCommand:
        now = now or utcnow()
        worker_session = await self._require_fresh_session(session_id, session_generation, now)
        command = await self._require_worker_command(command_id, worker_session.worker_id)

        if command.state in TERMINAL_COMMAND_STATES:
            return command

        if command.worker_session_id is not None:
            self._ensure_command_session_matches(command, worker_session)
        else:
            command.worker_session_id = worker_session.id
            command.worker_session_generation = worker_session.generation
            command.dispatched_at = now
            command.dispatch_attempts += 1
            worker_session.last_command_sequence = max(
                worker_session.last_command_sequence,
                command.sequence,
            )

        if command.acknowledged_at is not None:
            return command

        command.state = WorkerControlCommandStateEnum.SENT

        self.session.add(command)
        self.session.add(worker_session)
        await self.session.commit()
        await self.session.refresh(command)
        return command

    async def lease_command(
        self,
        command_id: str,
        *,
        session_id: str,
        session_generation: int,
        now: Optional[datetime] = None,
    ) -> WorkerCommand:
        now = now or utcnow()
        worker_session = await self._require_fresh_session(session_id, session_generation, now)
        command = await self._require_worker_command(command_id, worker_session.worker_id)

        if command.state in TERMINAL_COMMAND_STATES:
            return command

        command.worker_session_id = worker_session.id
        command.worker_session_generation = worker_session.generation
        command.state = WorkerControlCommandStateEnum.LEASED
        command.dispatched_at = now
        command.dispatch_attempts += 1
        worker_session.last_command_sequence = max(
            worker_session.last_command_sequence,
            command.sequence,
        )

        self.session.add(command)
        self.session.add(worker_session)
        await self.session.commit()
        await self.session.refresh(command)
        return command

    async def acknowledge_command(
        self,
        command_id: str,
        *,
        session_id: str,
        session_generation: int,
        accepted: bool = True,
        state: WorkerControlCommandStateEnum = WorkerControlCommandStateEnum.ACKNOWLEDGED,
        error_message: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> WorkerCommand:
        now = now or utcnow()
        worker_session = await self._require_fresh_session(session_id, session_generation, now)
        command = await self._require_worker_command(command_id, worker_session.worker_id)
        self._ensure_command_session_matches(command, worker_session)

        if command.state in TERMINAL_COMMAND_STATES:
            return command
        if command.acknowledged_at is not None:
            return command

        command.acknowledged_at = now
        if command.dispatched_at is not None:
            record_command_ack_latency(
                worker_id=worker_session.worker_id,
                command_type=command.command_type,
                accepted=accepted,
                latency_seconds=max(
                    0.0,
                    (now - command.dispatched_at).total_seconds(),
                ),
            )
        if accepted:
            command.state = state
        else:
            command.state = WorkerControlCommandStateEnum.FAILED
            command.completed_at = now
            if error_message is not None:
                command.error_message = error_message
            record_command_failure(
                worker_id=worker_session.worker_id,
                command_type=command.command_type,
                failure_source="control",
                stage="ack",
            )
            logger.warning(
                "Worker command acknowledgement rejected: %s",
                sanitize_log_value(error_message or "worker rejected command"),
                extra=control_log_extra(
                    worker_id=worker_session.worker_id,
                    session_id=worker_session.session_id,
                    session_generation=worker_session.generation,
                    command_id=command.command_id,
                    command_type=command.command_type,
                    failure_source="control",
                ),
            )

        if error_message is not None:
            command.error_message = error_message

        worker_session.last_acknowledged_command_sequence = max(
            worker_session.last_acknowledged_command_sequence,
            command.sequence,
        )
        worker_session.replay_cursor = max(worker_session.replay_cursor, command.sequence)

        self.session.add(command)
        self.session.add(worker_session)
        await self.session.commit()
        await self.session.refresh(command)
        return command

    async def record_command_result(
        self,
        command_id: str,
        *,
        session_id: str,
        session_generation: int,
        state: WorkerControlCommandStateEnum,
        result: Optional[dict[str, Any]] = None,
        error_message: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> WorkerCommand:
        now = now or utcnow()
        worker_session = await self._require_fresh_session(session_id, session_generation, now)
        command = await self._require_worker_command(command_id, worker_session.worker_id)
        self._ensure_command_session_matches(command, worker_session)

        if command.state in TERMINAL_COMMAND_STATES:
            return command

        command.state = state
        command.result = result
        command.error_message = error_message
        command.completed_at = now
        if command.acknowledged_at is None:
            command.acknowledged_at = now

        failure_source = classify_result_failure_source(state)
        if command.dispatched_at is not None:
            record_command_result_latency(
                worker_id=worker_session.worker_id,
                command_type=command.command_type,
                state=state,
                failure_source=failure_source,
                latency_seconds=max(
                    0.0,
                    (now - command.dispatched_at).total_seconds(),
                ),
            )
        if failure_source != "none":
            record_command_failure(
                worker_id=worker_session.worker_id,
                command_type=command.command_type,
                failure_source=failure_source,
                stage="result",
            )
            logger.warning(
                "Worker command completed with failure state %s: %s",
                state.value,
                sanitize_log_value(error_message or "worker runtime failure"),
                extra=control_log_extra(
                    worker_id=worker_session.worker_id,
                    session_id=worker_session.session_id,
                    session_generation=worker_session.generation,
                    command_id=command.command_id,
                    command_type=command.command_type,
                    failure_source=failure_source,
                ),
            )

        worker_session.last_acknowledged_command_sequence = max(
            worker_session.last_acknowledged_command_sequence,
            command.sequence,
        )
        worker_session.last_completed_command_sequence = max(
            worker_session.last_completed_command_sequence,
            command.sequence,
        )
        worker_session.replay_cursor = max(worker_session.replay_cursor, command.sequence)

        self.session.add(command)
        self.session.add(worker_session)
        await self.session.commit()
        await self.session.refresh(command)
        return command

    async def get_replay_window(
        self,
        *,
        session_id: str,
        session_generation: int,
        after_sequence: int,
        limit: int = 128,
        now: Optional[datetime] = None,
    ) -> WorkerCommandReplayWindow:
        now = now or utcnow()
        worker_session = await self._require_fresh_session(session_id, session_generation, now)
        latest_sequence = await self._latest_command_sequence(worker_session.worker_id)
        if latest_sequence == 0:
            return WorkerCommandReplayWindow()

        replay_floor = max(1, latest_sequence - self.replay_window_size + 1)
        if after_sequence < replay_floor - 1:
            worker_session.requires_full_reconcile = True
            worker_session.full_reconcile_reason = (
                f"Replay gap exhausted after sequence {after_sequence}; "
                f"replay floor is {replay_floor}"
            )
            record_replay_fallback(
                worker_id=worker_session.worker_id,
                reason="replay_gap_exhausted",
            )
            logger.warning(
                "Worker command replay fell back to full reconciliation after sequence %s.",
                after_sequence,
                extra=control_log_extra(
                    worker_id=worker_session.worker_id,
                    session_id=worker_session.session_id,
                    session_generation=worker_session.generation,
                    failure_source="control",
                ),
            )
            self.session.add(worker_session)
            await self.session.commit()
            await self.session.refresh(worker_session)
            return WorkerCommandReplayWindow(
                requires_full_reconcile=True,
                full_reconcile_reason=worker_session.full_reconcile_reason,
                replay_floor_sequence=replay_floor,
                replay_through_sequence=latest_sequence,
            )

        statement = (
            select(WorkerCommand)
            .where(WorkerCommand.worker_id == worker_session.worker_id)
            .where(WorkerCommand.sequence > after_sequence)
            .order_by("sequence")
            .limit(limit)
        )
        commands = [
            command
            for command in (await self.session.exec(statement)).all()
            if command.state not in TERMINAL_COMMAND_STATES
        ]
        worker_session.requires_full_reconcile = False
        worker_session.full_reconcile_reason = None
        self.session.add(worker_session)
        await self.session.commit()
        await self.session.refresh(worker_session)
        return WorkerCommandReplayWindow(
            commands=commands,
            replay_floor_sequence=replay_floor,
            replay_through_sequence=latest_sequence,
        )

    async def record_control_error(
        self,
        *,
        session_id: str,
        session_generation: int,
        error_code: str,
        error_message: str,
        now: Optional[datetime] = None,
    ) -> WorkerSession:
        now = now or utcnow()
        worker_session = await self._require_fresh_session(session_id, session_generation, now)
        details = dict(worker_session.details or {})
        details["last_control_error"] = {
            "error_code": error_code,
            "error_message": error_message,
            "recorded_at": now.isoformat(),
        }
        worker_session.details = details
        if error_code in CONTROL_ERROR_REQUIRES_FULL_RECONCILE:
            worker_session.requires_full_reconcile = True
            worker_session.full_reconcile_reason = error_message
            record_replay_fallback(
                worker_id=worker_session.worker_id,
                reason=error_code,
            )
        logger.warning(
            "Worker control error recorded: %s",
            sanitize_log_value(error_message),
            extra=control_log_extra(
                worker_id=worker_session.worker_id,
                session_id=worker_session.session_id,
                session_generation=worker_session.generation,
                failure_source="control",
                error_code=error_code,
            ),
        )
        self.session.add(worker_session)
        await self.session.commit()
        await self.session.refresh(worker_session)
        return worker_session

    async def prepare_recovery_commands(
        self,
        *,
        session_id: str,
        session_generation: int,
        resume_requested: bool,
        after_sequence: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> list[WorkerCommand]:
        now = now or utcnow()
        worker_session = await self._require_fresh_session(session_id, session_generation, now)

        expired_commands = await self._expire_outstanding_commands(
            worker_session.worker_id,
            now=now,
        )
        if expired_commands:
            timeout_reason = self._format_recovery_reason(
                "Command replay expired during control-session recovery",
                command_count=len(expired_commands),
            )
            record_replay_fallback(
                worker_id=worker_session.worker_id,
                reason="expired_commands",
            )
            await self._set_full_reconcile_required(worker_session, reason=timeout_reason)

        if not resume_requested:
            superseded_commands = await self._supersede_outstanding_commands(
                worker_session.worker_id,
                now=now,
                reason=(
                    "Worker started a newer control session before pending commands "
                    "completed"
                ),
            )
            if superseded_commands:
                worker_session = await self._require_fresh_session(
                    session_id, session_generation, now
                )
                restart_reason = self._format_recovery_reason(
                    "Worker restart superseded the previous control session",
                    command_count=len(superseded_commands),
                )
                record_replay_fallback(
                    worker_id=worker_session.worker_id,
                    reason="superseded_session",
                )
                await self._set_full_reconcile_required(
                    worker_session,
                    reason=restart_reason,
                )

        worker_session = await self._require_fresh_session(session_id, session_generation, now)
        if worker_session.requires_full_reconcile:
            return [
                await self._create_full_reconcile_command(
                    worker_session,
                    reason=worker_session.full_reconcile_reason
                    or "Full reconciliation required after worker control recovery",
                    now=now,
                )
            ]

        replay_after_sequence = (
            worker_session.replay_cursor if after_sequence is None else after_sequence
        )
        replay_window = await self.get_replay_window(
            session_id=session_id,
            session_generation=session_generation,
            after_sequence=replay_after_sequence,
            now=now,
        )

        if replay_window.requires_full_reconcile:
            worker_session = await self._require_fresh_session(session_id, session_generation, now)
            reason = replay_window.full_reconcile_reason or (
                "Replay history is unavailable; full reconciliation required"
            )
            await self._set_full_reconcile_required(worker_session, reason=reason)
            return [
                await self._create_full_reconcile_command(
                    worker_session,
                    reason=reason,
                    replay_floor_sequence=replay_window.replay_floor_sequence,
                    replay_through_sequence=replay_window.replay_through_sequence,
                    now=now,
                )
            ]

        return list(replay_window.commands)

    async def _require_fresh_session(
        self,
        session_id: str,
        session_generation: int,
        now: datetime,
    ) -> WorkerSession:
        worker_session = await WorkerSession.one_by_field(
            self.session, "session_id", session_id
        )
        if worker_session is None:
            raise StaleWorkerSessionError(f"Unknown worker session '{session_id}'")
        if worker_session.generation != session_generation:
            raise StaleWorkerSessionError(
                f"Stale worker session generation {session_generation} for '{session_id}'"
            )
        if worker_session.state != WorkerSessionStateEnum.ACTIVE:
            raise StaleWorkerSessionError(
                f"Worker session '{session_id}' is not active"
            )
        if worker_session.expires_at is not None and worker_session.expires_at < now:
            raise StaleWorkerSessionError(
                f"Worker session '{session_id}' expired at {worker_session.expires_at.isoformat()}"
            )

        active_session = await WorkerSession.one_by_fields(
            self.session,
            {"worker_id": worker_session.worker_id, "state": WorkerSessionStateEnum.ACTIVE},
        )
        if active_session is not None and active_session.generation != session_generation:
            raise StaleWorkerSessionError(
                f"Worker session '{session_id}' has been superseded by generation {active_session.generation}"
            )

        return worker_session

    async def _expire_outstanding_commands(
        self,
        worker_id: int,
        *,
        now: datetime,
    ) -> list[WorkerCommand]:
        statement = (
            select(WorkerCommand)
            .where(WorkerCommand.worker_id == worker_id)
            .order_by("sequence")
        )
        commands = [
            command
            for command in (await self.session.exec(statement)).all()
            if command.state not in TERMINAL_COMMAND_STATES
            and command.expires_at is not None
            and command.expires_at < now
        ]
        for command in commands:
            if command.worker_session_id is not None or command.dispatched_at is not None:
                command.state = WorkerControlCommandStateEnum.TIMED_OUT
                command.error_message = (
                    "Command timed out before worker recovery completed"
                )
            else:
                command.state = WorkerControlCommandStateEnum.EXPIRED
                command.error_message = (
                    "Command expired before it could be dispatched after recovery"
                )
            command.completed_at = now
            self.session.add(command)

        if commands:
            await self.session.commit()
            for command in commands:
                await self.session.refresh(command)
        return commands

    async def _supersede_outstanding_commands(
        self,
        worker_id: int,
        *,
        now: datetime,
        reason: str,
    ) -> list[WorkerCommand]:
        statement = (
            select(WorkerCommand)
            .where(WorkerCommand.worker_id == worker_id)
            .order_by("sequence")
        )
        commands = [
            command
            for command in (await self.session.exec(statement)).all()
            if command.state not in TERMINAL_COMMAND_STATES
        ]
        for command in commands:
            command.state = WorkerControlCommandStateEnum.SUPERSEDED
            command.error_message = reason
            command.completed_at = now
            self.session.add(command)

        if commands:
            await self.session.commit()
            for command in commands:
                await self.session.refresh(command)
        return commands

    async def _set_full_reconcile_required(
        self,
        worker_session: WorkerSession,
        *,
        reason: str,
    ) -> WorkerSession:
        worker_session.requires_full_reconcile = True
        worker_session.full_reconcile_reason = reason
        self.session.add(worker_session)
        await self.session.commit()
        await self.session.refresh(worker_session)
        return worker_session

    async def _create_full_reconcile_command(
        self,
        worker_session: WorkerSession,
        *,
        reason: str,
        replay_floor_sequence: Optional[int] = None,
        replay_through_sequence: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> WorkerCommand:
        idempotency_key = (
            f"{COMMAND_RECONCILE_NOW}:{worker_session.worker_id}:recover:"
            f"{worker_session.session_id}:{worker_session.generation}"
        )
        return await self.create_command(
            worker_session.worker_id,
            command_type=COMMAND_RECONCILE_NOW,
            payload={
                "reason": reason,
                "full_reconcile": True,
                **(
                    {"replay_floor_sequence": replay_floor_sequence}
                    if replay_floor_sequence is not None
                    else {}
                ),
                **(
                    {"replay_through_sequence": replay_through_sequence}
                    if replay_through_sequence is not None
                    else {}
                ),
            },
            idempotency_key=idempotency_key,
            now=now,
        )

    @staticmethod
    def _format_recovery_reason(base_reason: str, *, command_count: int) -> str:
        noun = "command" if command_count == 1 else "commands"
        return f"{base_reason} ({command_count} {noun})"

    def _ensure_command_session_matches(
        self,
        command: WorkerCommand,
        worker_session: WorkerSession,
    ):
        if command.worker_session_id is None:
            raise StaleWorkerSessionError(
                f"Worker command '{command.command_id}' has not been dispatched on a session"
            )
        if command.worker_session_id != worker_session.id:
            raise StaleWorkerSessionError(
                f"Worker command '{command.command_id}' belongs to a different session"
            )
        if command.worker_session_generation != worker_session.generation:
            raise StaleWorkerSessionError(
                f"Worker command '{command.command_id}' belongs to a different session generation"
            )

    async def _require_worker_command(
        self,
        command_id: str,
        worker_id: int,
    ) -> WorkerCommand:
        statement = (
            select(WorkerCommand)
            .execution_options(populate_existing=True)
            .where(WorkerCommand.command_id == command_id)
            .where(WorkerCommand.worker_id == worker_id)
        )
        command = (await self.session.exec(statement)).first()
        if command is None:
            raise ValueError(f"Unknown worker command '{command_id}'")
        return command

    async def _get_command_by_idempotency_key(
        self,
        worker_id: int,
        idempotency_key: str,
    ) -> Optional[WorkerCommand]:
        statement = (
            select(WorkerCommand)
            .where(WorkerCommand.worker_id == worker_id)
            .where(WorkerCommand.idempotency_key == idempotency_key)
        )
        return (await self.session.exec(statement)).first()

    async def _next_session_generation(self, worker_id: int) -> int:
        statement = select(func.max(WorkerSession.generation)).where(
            WorkerSession.worker_id == worker_id
        )
        current_generation = (await self.session.exec(statement)).one_or_none() or 0
        return current_generation + 1

    async def _next_command_sequence(self, worker_id: int) -> int:
        statement = select(func.max(WorkerCommand.sequence)).where(
            WorkerCommand.worker_id == worker_id
        )
        current_sequence = (await self.session.exec(statement)).one_or_none() or 0
        return current_sequence + 1

    async def _latest_command_sequence(self, worker_id: int) -> int:
        statement = select(func.max(WorkerCommand.sequence)).where(
            WorkerCommand.worker_id == worker_id
        )
        return (await self.session.exec(statement)).one_or_none() or 0


class WorkerCommandDispatchService:
    def __init__(
        self,
        *,
        session_factory=async_session,
        session_registry=worker_control_session_registry,
        replay_window_size: int = 1024,
    ):
        self._session_factory = session_factory
        self._session_registry = session_registry
        self._replay_window_size = replay_window_size

    async def emit_reconcile_for_model_instance(
        self,
        *,
        model_id: int,
        model_instance_id: int,
        worker_ids: list[int],
        idempotency_token: str,
    ) -> list[WorkerCommand]:
        payload = {
            "model_id": model_id,
            "model_instance_id": model_instance_id,
        }
        return await self._emit_commands(
            worker_ids=worker_ids,
            command_type=COMMAND_RECONCILE_MODEL_INSTANCE,
            payload=payload,
            idempotency_key_factory=lambda worker_id: (
                f"{COMMAND_RECONCILE_MODEL_INSTANCE}:{worker_id}:{model_instance_id}:{idempotency_token}"
            ),
        )

    async def emit_reconcile_now(
        self,
        *,
        worker_ids: list[int],
        model_id: Optional[int] = None,
        idempotency_token: str,
        reason: str,
        full_reconcile: bool = False,
        replay_floor_sequence: Optional[int] = None,
        replay_through_sequence: Optional[int] = None,
    ) -> list[WorkerCommand]:
        payload = {
            "reason": reason,
            "full_reconcile": full_reconcile,
        }
        if model_id is not None:
            payload["model_id"] = model_id
        if replay_floor_sequence is not None:
            payload["replay_floor_sequence"] = replay_floor_sequence
        if replay_through_sequence is not None:
            payload["replay_through_sequence"] = replay_through_sequence

        return await self._emit_commands(
            worker_ids=worker_ids,
            command_type=COMMAND_RECONCILE_NOW,
            payload=payload,
            idempotency_key_factory=lambda worker_id: (
                f"{COMMAND_RECONCILE_NOW}:{worker_id}:{idempotency_token}"
            ),
        )

    async def emit_sync_runtime_state(
        self,
        *,
        worker_id: int,
        idempotency_token: str,
        reason: str,
    ) -> list[WorkerCommand]:
        return await self._emit_commands(
            worker_ids=[worker_id],
            command_type=COMMAND_SYNC_RUNTIME_STATE,
            payload={"worker_id": worker_id, "reason": reason},
            idempotency_key_factory=lambda current_worker_id: (
                f"{COMMAND_SYNC_RUNTIME_STATE}:{current_worker_id}:{idempotency_token}"
            ),
        )

    async def dispatch_replay_window(
        self,
        *,
        session_id: str,
        session_generation: int,
        after_sequence: Optional[int] = None,
        resume_requested: bool = True,
    ) -> list[WorkerCommand]:
        async with self._session_factory() as session:
            service = WorkerCommandService(
                session=session,
                replay_window_size=self._replay_window_size,
            )
            worker_session = await service._require_fresh_session(
                session_id,
                session_generation,
                utcnow(),
            )
            commands = await service.prepare_recovery_commands(
                session_id=session_id,
                session_generation=session_generation,
                resume_requested=resume_requested,
                after_sequence=after_sequence,
                now=utcnow(),
            )

            dispatched_commands = []
            for command in commands:
                dispatched_commands.append(
                    await self._dispatch_command_record(
                        service,
                        command,
                        expected_session=worker_session,
                    )
                )
            return dispatched_commands

    async def _emit_commands(
        self,
        *,
        worker_ids: list[int],
        command_type: str,
        payload: dict[str, Any],
        idempotency_key_factory,
    ) -> list[WorkerCommand]:
        dispatched_commands: list[WorkerCommand] = []
        unique_worker_ids = sorted({worker_id for worker_id in worker_ids if worker_id is not None})
        if not unique_worker_ids:
            return dispatched_commands

        async with self._session_factory() as session:
            service = WorkerCommandService(
                session=session,
                replay_window_size=self._replay_window_size,
            )
            for worker_id in unique_worker_ids:
                command = await service.create_command(
                    worker_id,
                    command_type=command_type,
                    payload=payload,
                    idempotency_key=idempotency_key_factory(worker_id),
                )
                dispatched_commands.append(await self._dispatch_command_record(service, command))

        return dispatched_commands

    async def _dispatch_command_record(
        self,
        service: WorkerCommandService,
        command: WorkerCommand,
        *,
        expected_session: Optional[WorkerSession] = None,
    ) -> WorkerCommand:
        active_session = await self._session_registry.get_active_session(command.worker_id)
        if active_session is None:
            return command

        if expected_session is not None and (
            active_session.session_id != expected_session.session_id
            or active_session.generation != expected_session.generation
        ):
            return command

        try:
            command = await service.lease_command(
                command.command_id,
                session_id=active_session.session_id,
                session_generation=active_session.generation,
                now=utcnow(),
            )
        except StaleWorkerSessionError:
            logger.info(
                "Worker command dispatch skipped because the session became stale before leasing.",
                extra=control_log_extra(
                    worker_id=command.worker_id,
                    session_id=active_session.session_id,
                    session_generation=active_session.generation,
                    command_id=command.command_id,
                    command_type=command.command_type,
                    failure_source="control",
                ),
            )
            return command

        command_message = WorkerCommandMessage(
            session_id=active_session.session_id,
            protocol_version=active_session.protocol_version,
            sent_at=utcnow(),
            command_id=command.command_id,
            command_type=command.command_type,
            payload=command.payload,
            desired_worker_state_revision=command.desired_worker_state_revision,
        )

        try:
            await active_session.websocket.send_json(command_message.model_dump(mode="json"))
        except Exception as e:
            record_command_failure(
                worker_id=command.worker_id,
                command_type=command.command_type,
                failure_source="transport",
                stage="dispatch",
            )
            logger.warning(
                "Failed to dispatch worker command to active session: %s",
                sanitize_log_value(str(e)),
                extra=control_log_extra(
                    worker_id=command.worker_id,
                    session_id=active_session.session_id,
                    session_generation=active_session.generation,
                    command_id=command.command_id,
                    command_type=command.command_type,
                    failure_source="transport",
                ),
            )
            return command

        try:
            return await service.mark_command_sent(
                command.command_id,
                session_id=active_session.session_id,
                session_generation=active_session.generation,
                now=utcnow(),
            )
        except StaleWorkerSessionError:
            logger.info(
                "Worker command dispatch skipped because the session became stale.",
                extra=control_log_extra(
                    worker_id=command.worker_id,
                    session_id=active_session.session_id,
                    session_generation=active_session.generation,
                    command_id=command.command_id,
                    command_type=command.command_type,
                    failure_source="control",
                ),
            )
            return command

    async def send_control_error(
        self,
        *,
        worker_id: int,
        error_code: str,
        error_message: str,
    ) -> bool:
        active_session = await self._session_registry.get_active_session(worker_id)
        if active_session is None:
            return False

        await active_session.websocket.send_json(
            WorkerErrorMessage(
                session_id=active_session.session_id,
                protocol_version=active_session.protocol_version,
                sent_at=utcnow(),
                error_code=error_code,
                error_message=error_message,
            ).model_dump(mode="json")
        )
        return True
