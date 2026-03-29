import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
from typing import Any, Deque, Dict, Optional, Tuple
import uuid

from fastapi import WebSocket, status

from gpustack import envs
from gpustack.server.worker_control_observability import (
    record_session_connected,
    record_session_disconnected,
    record_session_replacement,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass
class WorkerControlSession:
    worker_id: int
    worker_uuid: str
    session_id: str
    generation: int
    protocol_version: int
    websocket: Any
    connected_at: datetime
    last_seen_at: datetime
    disconnected_at: Optional[datetime] = None
    replaced_by_generation: Optional[int] = None
    disconnect_reason: Optional[str] = None


class WorkerControlSessionRegistry:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._active_by_worker: Dict[int, WorkerControlSession] = {}
        self._sessions_by_key: Dict[Tuple[int, int], WorkerControlSession] = {}

    async def register(
        self,
        worker_id: int,
        worker_uuid: str,
        websocket: Any,
        generation: Optional[int] = None,
        session_id: Optional[str] = None,
        protocol_version: int = 1,
    ) -> WorkerControlSession:
        previous_session = None
        session_to_close = None
        should_record_connected = False
        async with self._lock:
            session_generation = generation
            if session_generation is None:
                active_generation = self._active_by_worker.get(worker_id)
                session_generation = 1 if active_generation is None else active_generation.generation + 1
            assert session_generation is not None
            now = utcnow()
            session_generation = int(session_generation)
            active_session = WorkerControlSession(
                worker_id=worker_id,
                worker_uuid=worker_uuid,
                session_id=session_id or str(uuid.uuid4()),
                generation=session_generation,
                protocol_version=protocol_version,
                websocket=websocket,
                connected_at=now,
                last_seen_at=now,
            )
            previous_session = self._active_by_worker.get(worker_id)

            should_replace_active = previous_session is None or (
                previous_session.generation < session_generation
            ) or (
                previous_session.generation == session_generation
                and previous_session.session_id == active_session.session_id
            )

            if should_replace_active:
                if previous_session is not None:
                    previous_session.replaced_by_generation = session_generation
                    previous_session.disconnect_reason = "replaced"
                    session_to_close = previous_session
                self._active_by_worker[worker_id] = active_session
                self._sessions_by_key[(worker_id, session_generation)] = active_session
                should_record_connected = True
            else:
                assert previous_session is not None
                active_session.replaced_by_generation = previous_session.generation
                active_session.disconnect_reason = "replaced"
                session_to_close = active_session

        if previous_session is not None and should_replace_active:
            record_session_replacement(worker_id)

        if should_record_connected:
            record_session_connected(
                worker_id,
                control_channel="outbound_control_ws",
            )

        if session_to_close is not None:
            await self._close_websocket(
                session_to_close.websocket,
                code=status.WS_1008_POLICY_VIOLATION,
                reason="replaced by newer worker control session",
            )

        return active_session

    async def touch(self, worker_id: int, generation: int) -> Optional[WorkerControlSession]:
        async with self._lock:
            session = self._sessions_by_key.get((worker_id, generation))
            if session is None:
                return None
            session.last_seen_at = utcnow()
            return session

    async def unregister(
        self,
        worker_id: int,
        generation: int,
        reason: str,
    ) -> Optional[WorkerControlSession]:
        async with self._lock:
            session = self._sessions_by_key.pop((worker_id, generation), None)
            if session is None:
                return None

            session.disconnected_at = utcnow()
            session.disconnect_reason = reason

            active_session = self._active_by_worker.get(worker_id)
            still_active = (
                active_session is not None and active_session.generation != generation
            )
            if active_session is not None and active_session.generation == generation:
                del self._active_by_worker[worker_id]

            record_session_disconnected(
                worker_id,
                reason=reason,
                still_active=still_active,
            )

            return session

    async def get_active_session(self, worker_id: int) -> Optional[WorkerControlSession]:
        async with self._lock:
            return self._active_by_worker.get(worker_id)

    async def get_session(
        self, worker_id: int, generation: int
    ) -> Optional[WorkerControlSession]:
        async with self._lock:
            return self._sessions_by_key.get((worker_id, generation))

    @staticmethod
    async def _close_websocket(websocket: Any, code: int, reason: str):
        close = getattr(websocket, "close", None)
        if close is None:
            return

        try:
            result = close(code=code, reason=reason)
        except TypeError:
            result = close(code=code)

        if inspect.isawaitable(result):
            await result


class WorkerControlMessageTooLarge(Exception):
    pass


class WorkerControlRateLimitExceeded(Exception):
    pass


class WorkerControlInboundLimiter:
    def __init__(self):
        self._received_at: Deque[float] = deque()
        self._window_seconds = envs.WORKER_CONTROL_WS_RATE_LIMIT_WINDOW_SECONDS
        self._max_messages = envs.WORKER_CONTROL_WS_RATE_LIMIT_MESSAGES
        self._max_message_bytes = envs.WORKER_CONTROL_WS_MAX_MESSAGE_BYTES

    def check(self, payload: str):
        size = len(payload.encode("utf-8"))
        if size > self._max_message_bytes:
            raise WorkerControlMessageTooLarge(
                f"Inbound websocket message exceeded {self._max_message_bytes} bytes"
            )

        now = asyncio.get_running_loop().time()
        while self._received_at and now - self._received_at[0] >= self._window_seconds:
            self._received_at.popleft()

        if len(self._received_at) >= self._max_messages:
            raise WorkerControlRateLimitExceeded(
                "Inbound websocket message rate limit exceeded"
            )

        self._received_at.append(now)


worker_control_session_registry = WorkerControlSessionRegistry()
