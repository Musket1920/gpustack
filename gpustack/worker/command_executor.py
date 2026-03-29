import asyncio
from collections import OrderedDict
import logging
from typing import Any, Callable, Optional

from gpustack.client import ClientSet
from gpustack.schemas.model_files import ModelFileStateEnum
from gpustack.schemas.workers import (
    WorkerCommandAckMessage,
    WorkerCommandMessage,
    WorkerCommandResultMessage,
    WorkerControlCommandStateEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.server.worker_command_service import (
    COMMAND_RECONCILE_MODEL_INSTANCE,
    COMMAND_RECONCILE_NOW,
    COMMAND_SYNC_RUNTIME_STATE,
)
from gpustack.server.worker_control_observability import (
    control_log_extra,
    sanitize_log_value,
)

logger = logging.getLogger(__name__)


class WorkerCommandExecutor:
    def __init__(
        self,
        *,
        clientset_getter: Callable[[], ClientSet],
        serve_manager,
        worker_manager,
        model_file_manager,
        benchmark_manager,
        completed_command_cache_size: int = 256,
    ):
        self._clientset_getter = clientset_getter
        self._serve_manager = serve_manager
        self._worker_manager = worker_manager
        self._model_file_manager = model_file_manager
        self._benchmark_manager = benchmark_manager
        self._completed_command_cache_size = completed_command_cache_size
        self._inflight_commands: dict[str, asyncio.Task] = {}
        self._completed_commands: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @property
    def _clientset(self) -> ClientSet:
        return self._clientset_getter()

    async def stop(self):
        tasks = list(self._inflight_commands.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def handle_command(self, command: WorkerCommandMessage, websocket):
        error_message = self._validate_command(command)
        if error_message is not None:
            logger.warning(
                "Worker command validation rejected: %s",
                sanitize_log_value(error_message),
                extra=control_log_extra(
                    session_id=command.session_id,
                    command_id=command.command_id,
                    command_type=command.command_type,
                    failure_source="control",
                ),
            )
            await self._send_ack(
                websocket,
                command,
                accepted=False,
                error_message=error_message,
            )
            return

        completed = self._completed_commands.get(command.command_id)
        if completed is not None:
            await self._send_ack(websocket, command)
            await self._send_result(
                websocket,
                command,
                state=completed["state"],
                result=completed.get("result"),
                error_message=completed.get("error_message"),
            )
            return

        await self._send_ack(websocket, command)

        if command.command_id in self._inflight_commands:
            logger.debug(
                "Worker command already in flight; skipping duplicate execution.",
                extra=control_log_extra(
                    session_id=command.session_id,
                    command_id=command.command_id,
                    command_type=command.command_type,
                ),
            )
            return

        task = asyncio.create_task(self._execute_command(command, websocket))
        self._inflight_commands[command.command_id] = task
        task.add_done_callback(
            lambda _: self._inflight_commands.pop(command.command_id, None)
        )

    def _validate_command(self, command: WorkerCommandMessage) -> Optional[str]:
        payload = command.payload or {}
        if command.command_type == COMMAND_RECONCILE_MODEL_INSTANCE:
            model_instance_id = payload.get("model_instance_id")
            model_id = payload.get("model_id")
            if not isinstance(model_instance_id, int):
                return "reconcile_model_instance requires integer payload.model_instance_id"
            if not isinstance(model_id, int):
                return "reconcile_model_instance requires integer payload.model_id"
            return None

        if command.command_type == COMMAND_RECONCILE_NOW:
            reason = payload.get("reason")
            if reason is not None and not isinstance(reason, str):
                return "reconcile_now payload.reason must be a string when provided"
            return None

        if command.command_type == COMMAND_SYNC_RUNTIME_STATE:
            worker_id = payload.get("worker_id")
            reason = payload.get("reason")
            if not isinstance(worker_id, int):
                return "sync_runtime_state requires integer payload.worker_id"
            if reason is not None and not isinstance(reason, str):
                return "sync_runtime_state payload.reason must be a string when provided"
            return None

        return f"unsupported worker command type '{command.command_type}'"

    async def _execute_command(self, command: WorkerCommandMessage, websocket):
        try:
            result = await self._delegate_command(command)
            state = WorkerControlCommandStateEnum.SUCCEEDED
            error_message = None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(
                "Worker command execution failed: %s",
                sanitize_log_value(str(e)),
                extra=control_log_extra(
                    session_id=command.session_id,
                    command_id=command.command_id,
                    command_type=command.command_type,
                    failure_source="runtime",
                ),
            )
            result = None
            state = WorkerControlCommandStateEnum.FAILED
            error_message = str(e)

        self._remember_completed_command(
            command.command_id,
            state=state,
            result=result,
            error_message=error_message,
        )
        await self._send_result(
            websocket,
            command,
            state=state,
            result=result,
            error_message=error_message,
        )

    async def _delegate_command(self, command: WorkerCommandMessage) -> dict[str, Any]:
        if command.command_type == COMMAND_RECONCILE_MODEL_INSTANCE:
            return await self._delegate_reconcile_model_instance(command)
        if command.command_type in {
            COMMAND_RECONCILE_NOW,
            COMMAND_SYNC_RUNTIME_STATE,
        }:
            return await self._delegate_runtime_sync(command)
        raise ValueError(f"Unsupported worker command type: {command.command_type}")

    async def _delegate_reconcile_model_instance(
        self, command: WorkerCommandMessage
    ) -> dict[str, Any]:
        payload = command.payload
        model_instance = self._clientset.model_instances.get(payload["model_instance_id"])
        if model_instance.model_id != payload["model_id"]:
            raise ValueError(
                "reconcile_model_instance payload model_id does not match current model instance"
            )

        self._serve_manager._handle_model_instance_event(
            Event(type=EventType.UPDATED, data=model_instance)
        )
        return {
            "delegated": True,
            "delegated_to": ["serve_manager"],
            "model_id": payload["model_id"],
            "model_instance_id": payload["model_instance_id"],
        }

    async def _delegate_runtime_sync(
        self, command: WorkerCommandMessage
    ) -> dict[str, Any]:
        self._worker_manager.sync_worker_status()
        self._reconcile_pending_model_files()
        self._benchmark_manager.sync_benchmark_state()
        self._serve_manager.sync_model_instances_state()
        return {
            "delegated": True,
            "delegated_to": [
                "worker_manager",
                "model_file_manager",
                "benchmark_manager",
                "serve_manager",
            ],
            "reason": command.payload.get("reason"),
            "worker_id": command.payload.get("worker_id"),
            "full_reconcile": command.payload.get("full_reconcile"),
        }

    def _reconcile_pending_model_files(self):
        self._model_file_manager._prerun()
        model_files_page = self._clientset.model_files.list(
            params={"worker_id": self._model_file_manager._worker_id}
        )
        for model_file in model_files_page.items:
            if model_file.state != ModelFileStateEnum.DOWNLOADING:
                continue
            self._model_file_manager._create_download_task(model_file)

    def _remember_completed_command(
        self,
        command_id: str,
        *,
        state: WorkerControlCommandStateEnum,
        result: Optional[dict[str, Any]],
        error_message: Optional[str],
    ):
        self._completed_commands[command_id] = {
            "state": state,
            "result": result,
            "error_message": error_message,
        }
        self._completed_commands.move_to_end(command_id)
        while len(self._completed_commands) > self._completed_command_cache_size:
            self._completed_commands.popitem(last=False)

    async def _send_ack(
        self,
        websocket,
        command: WorkerCommandMessage,
        *,
        accepted: bool = True,
        error_message: Optional[str] = None,
    ):
        await websocket.send_json(
            WorkerCommandAckMessage(
                session_id=command.session_id,
                protocol_version=command.protocol_version,
                command_id=command.command_id,
                accepted=accepted,
                error_message=error_message,
            ).model_dump(mode="json")
        )

    async def _send_result(
        self,
        websocket,
        command: WorkerCommandMessage,
        *,
        state: WorkerControlCommandStateEnum,
        result: Optional[dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ):
        await websocket.send_json(
            WorkerCommandResultMessage(
                session_id=command.session_id,
                protocol_version=command.protocol_version,
                command_id=command.command_id,
                state=state,
                result=result,
                error_message=error_message,
            ).model_dump(mode="json")
        )
