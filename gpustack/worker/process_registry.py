import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import tempfile
import threading
from typing import List, Optional, Union

import psutil
from pydantic import BaseModel, Field, ValidationError

from gpustack.config.config import Config
from gpustack.schemas.models import ModelInstanceStateEnum


DIRECT_PROCESS_RUNTIME_MODE = "direct_process"
DIRECT_PROCESS_UNHEALTHY_MESSAGE = "Inference server exited or unhealthy."
_PID_START_TIME_TOLERANCE_SECONDS = 1.0
_TRANSITIONAL_MODEL_INSTANCE_STATES = {
    ModelInstanceStateEnum.INITIALIZING,
    ModelInstanceStateEnum.DOWNLOADING,
    ModelInstanceStateEnum.STARTING,
}

logger = logging.getLogger(__name__)


def _getpgid(pid: int) -> Optional[int]:
    getpgid = getattr(os, "getpgid", None)
    if getpgid is None:
        return None

    try:
        return getpgid(pid)
    except ProcessLookupError:
        return None
    except OSError:
        return None


class DirectProcessRegistryEntry(BaseModel):
    model_instance_id: int
    deployment_name: str
    pid: int
    process_group_id: Optional[int] = None
    port: int
    log_path: str
    backend: str
    mode: str
    created_at: datetime
    updated_at: datetime
    pid_started_at: Optional[datetime] = None


class DirectProcessEntryStatus(str, Enum):
    LIVE = "live"
    MISSING = "missing"
    STALE = "stale"


class DirectProcessRegistryStatus(BaseModel):
    status: DirectProcessEntryStatus
    reason: Optional[str] = None
    entry: Optional[DirectProcessRegistryEntry] = None


class DirectProcessRuntimeState(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    EXITED = "exited"
    STALE = "stale"
    MISSING = "missing"


class DirectProcessStateTransition(BaseModel):
    runtime_state: DirectProcessRuntimeState
    next_state: Optional[ModelInstanceStateEnum] = None
    state_message: Optional[str] = None


class _DirectProcessRegistryData(BaseModel):
    entries: List[DirectProcessRegistryEntry] = Field(default_factory=list)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_process_group_id(pid: int) -> Optional[int]:
    if pid <= 0:
        return None

    process_group_id = _getpgid(pid)
    if process_group_id is not None:
        return process_group_id

    return pid


def get_process_started_at(pid: int) -> Optional[datetime]:
    if pid <= 0:
        return None

    try:
        return datetime.fromtimestamp(psutil.Process(pid).create_time(), tz=timezone.utc)
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return None


def build_direct_process_registry_entry(
    model_instance_id: int,
    deployment_name: str,
    pid: int,
    port: int,
    log_path: str,
    backend: str,
    mode: str = DIRECT_PROCESS_RUNTIME_MODE,
    process_group_id: Optional[int] = None,
    created_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None,
) -> DirectProcessRegistryEntry:
    created_time = created_at or utcnow()
    updated_time = updated_at or created_time

    return DirectProcessRegistryEntry(
        model_instance_id=model_instance_id,
        deployment_name=deployment_name,
        pid=pid,
        process_group_id=(
            process_group_id if process_group_id is not None else get_process_group_id(pid)
        ),
        port=port,
        log_path=log_path,
        backend=backend.value if isinstance(backend, Enum) else str(backend),
        mode=mode,
        created_at=created_time,
        updated_at=updated_time,
        pid_started_at=get_process_started_at(pid),
    )


def inspect_direct_process_entry(
    entry: Optional[DirectProcessRegistryEntry],
) -> DirectProcessRegistryStatus:
    if entry is None:
        return DirectProcessRegistryStatus(
            status=DirectProcessEntryStatus.MISSING,
            reason="registry_entry_missing",
        )

    if entry.pid <= 0:
        return DirectProcessRegistryStatus(
            status=DirectProcessEntryStatus.STALE,
            reason="invalid_pid",
            entry=entry,
        )

    try:
        process = psutil.Process(entry.pid)
        process_started_at = datetime.fromtimestamp(
            process.create_time(), tz=timezone.utc
        )
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return DirectProcessRegistryStatus(
            status=DirectProcessEntryStatus.STALE,
            reason="pid_not_running",
            entry=entry,
        )
    except (psutil.AccessDenied, ValueError):
        return DirectProcessRegistryStatus(
            status=DirectProcessEntryStatus.STALE,
            reason="pid_not_accessible",
            entry=entry,
        )

    if entry.pid_started_at is not None:
        started_delta = abs(
            (process_started_at - entry.pid_started_at).total_seconds()
        )
        if started_delta > _PID_START_TIME_TOLERANCE_SECONDS:
            return DirectProcessRegistryStatus(
                status=DirectProcessEntryStatus.STALE,
                reason="pid_reused",
                entry=entry,
            )

    if entry.process_group_id is not None:
        current_process_group_id = _getpgid(entry.pid)
        if (
            current_process_group_id is not None
            and current_process_group_id != entry.process_group_id
        ):
            return DirectProcessRegistryStatus(
                status=DirectProcessEntryStatus.STALE,
                reason="process_group_changed",
                entry=entry,
            )

    return DirectProcessRegistryStatus(
        status=DirectProcessEntryStatus.LIVE,
        reason="pid_running",
        entry=entry,
    )


def map_direct_process_state_transition(
    current_state: ModelInstanceStateEnum,
    entry_status: DirectProcessRegistryStatus,
    is_ready: bool,
) -> DirectProcessStateTransition:
    if entry_status.status == DirectProcessEntryStatus.MISSING:
        return DirectProcessStateTransition(
            runtime_state=DirectProcessRuntimeState.MISSING,
            next_state=(
                None
                if current_state == ModelInstanceStateEnum.ERROR
                else ModelInstanceStateEnum.ERROR
            ),
            state_message=(
                None
                if current_state == ModelInstanceStateEnum.ERROR
                else DIRECT_PROCESS_UNHEALTHY_MESSAGE
            ),
        )

    if entry_status.status == DirectProcessEntryStatus.STALE:
        return DirectProcessStateTransition(
            runtime_state=DirectProcessRuntimeState.STALE,
            next_state=(
                None
                if current_state == ModelInstanceStateEnum.ERROR
                else ModelInstanceStateEnum.ERROR
            ),
            state_message=(
                None
                if current_state == ModelInstanceStateEnum.ERROR
                else DIRECT_PROCESS_UNHEALTHY_MESSAGE
            ),
        )

    if is_ready:
        return DirectProcessStateTransition(
            runtime_state=DirectProcessRuntimeState.RUNNING,
            next_state=(
                None
                if current_state == ModelInstanceStateEnum.RUNNING
                else ModelInstanceStateEnum.RUNNING
            ),
            state_message=("" if current_state != ModelInstanceStateEnum.RUNNING else None),
        )

    if current_state in _TRANSITIONAL_MODEL_INSTANCE_STATES:
        return DirectProcessStateTransition(
            runtime_state=DirectProcessRuntimeState.STARTING,
        )

    if current_state == ModelInstanceStateEnum.RUNNING:
        return DirectProcessStateTransition(
            runtime_state=DirectProcessRuntimeState.STARTING,
            next_state=ModelInstanceStateEnum.STARTING,
            state_message="",
        )

    return DirectProcessStateTransition(
        runtime_state=DirectProcessRuntimeState.STARTING,
        next_state=ModelInstanceStateEnum.STARTING,
        state_message="",
    )


class DirectProcessRegistry:
    _lock = threading.Lock()

    def __init__(self, cfg_or_path: Union[Config, str, Path]):
        if isinstance(cfg_or_path, Config):
            self._path = (
                Path(str(cfg_or_path.data_dir))
                / "worker"
                / "direct_process_registry.json"
            )
        elif isinstance(cfg_or_path, Path):
            self._path = cfg_or_path
        else:
            self._path = Path(str(cfg_or_path))

        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def list_entries(self) -> List[DirectProcessRegistryEntry]:
        return self._read_entries()

    def get_by_model_instance_id(
        self, model_instance_id: int
    ) -> Optional[DirectProcessRegistryEntry]:
        return next(
            (
                entry
                for entry in self._read_entries()
                if entry.model_instance_id == model_instance_id
            ),
            None,
        )

    def get_by_deployment_name(
        self, deployment_name: str
    ) -> Optional[DirectProcessRegistryEntry]:
        return next(
            (
                entry
                for entry in self._read_entries()
                if entry.deployment_name == deployment_name
            ),
            None,
        )

    def inspect_by_model_instance_id(
        self, model_instance_id: int
    ) -> DirectProcessRegistryStatus:
        return inspect_direct_process_entry(self.get_by_model_instance_id(model_instance_id))

    def inspect_by_deployment_name(
        self, deployment_name: str
    ) -> DirectProcessRegistryStatus:
        return inspect_direct_process_entry(self.get_by_deployment_name(deployment_name))

    def upsert(
        self,
        entry: Optional[DirectProcessRegistryEntry] = None,
        *,
        model_instance_id: Optional[int] = None,
        deployment_name: Optional[str] = None,
        pid: Optional[int] = None,
        port: Optional[int] = None,
        log_path: Optional[str] = None,
        backend: Optional[str] = None,
        mode: str = DIRECT_PROCESS_RUNTIME_MODE,
        process_group_id: Optional[int] = None,
    ) -> DirectProcessRegistryEntry:
        if entry is None:
            if None in [
                model_instance_id,
                deployment_name,
                pid,
                port,
                log_path,
                backend,
            ]:
                raise ValueError(
                    "Direct-process registry upsert requires complete metadata."
                )
            assert model_instance_id is not None
            assert deployment_name is not None
            assert pid is not None
            assert port is not None
            assert log_path is not None
            assert backend is not None
            entry_model_instance_id = model_instance_id
            entry_deployment_name = deployment_name
            entry_pid = pid
            entry_port = port
            entry_log_path = log_path
            entry_backend = backend
            entry = build_direct_process_registry_entry(
                model_instance_id=entry_model_instance_id,
                deployment_name=entry_deployment_name,
                pid=entry_pid,
                port=entry_port,
                log_path=entry_log_path,
                backend=entry_backend,
                mode=mode,
                process_group_id=process_group_id,
            )

        with self._lock:
            entries = self._read_entries()
            existing = next(
                (
                    current
                    for current in entries
                    if current.model_instance_id == entry.model_instance_id
                    or current.deployment_name == entry.deployment_name
                ),
                None,
            )
            created_at = existing.created_at if existing else entry.created_at
            normalized_entry = entry.model_copy(
                update={
                    "created_at": created_at,
                    "updated_at": utcnow(),
                }
            )

            filtered_entries = [
                current
                for current in entries
                if current.model_instance_id != normalized_entry.model_instance_id
                and current.deployment_name != normalized_entry.deployment_name
            ]
            filtered_entries.append(normalized_entry)
            self._write_entries(filtered_entries)
            return normalized_entry

    def remove_by_model_instance_id(self, model_instance_id: int) -> Optional[DirectProcessRegistryEntry]:
        with self._lock:
            entries = self._read_entries()
            removed_entry = next(
                (
                    entry
                    for entry in entries
                    if entry.model_instance_id == model_instance_id
                ),
                None,
            )
            if removed_entry is None:
                return None

            self._write_entries(
                [
                    entry
                    for entry in entries
                    if entry.model_instance_id != model_instance_id
                ]
            )
            return removed_entry

    def remove_by_deployment_name(self, deployment_name: str) -> Optional[DirectProcessRegistryEntry]:
        with self._lock:
            entries = self._read_entries()
            removed_entry = next(
                (
                    entry
                    for entry in entries
                    if entry.deployment_name == deployment_name
                ),
                None,
            )
            if removed_entry is None:
                return None

            self._write_entries(
                [
                    entry
                    for entry in entries
                    if entry.deployment_name != deployment_name
                ]
            )
            return removed_entry

    def _read_entries(self) -> List[DirectProcessRegistryEntry]:
        if not self._path.exists():
            return []

        try:
            content = self._path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning(
                "Failed to read direct-process registry %s: %s",
                self._path,
                exc,
            )
            return []

        if not content:
            return []

        try:
            payload = json.loads(content)
            data = _DirectProcessRegistryData.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(
                "Ignoring invalid direct-process registry %s: %s",
                self._path,
                exc,
            )
            return []

        return data.entries

    def _write_entries(self, entries: List[DirectProcessRegistryEntry]) -> None:
        payload = _DirectProcessRegistryData(entries=entries).model_dump(mode="json")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self._path.parent,
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temp_path = Path(handle.name)

        temp_path.replace(self._path)
