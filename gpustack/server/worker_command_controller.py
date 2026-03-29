import asyncio
from datetime import datetime
import logging
from typing import Any, Optional, Sequence

from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.workers import Worker
from gpustack.server.bus import Event, EventType, event_bus
from gpustack.server.db import async_session
from gpustack.server.worker_command_service import WorkerCommandDispatchService


logger = logging.getLogger(__name__)


class WorkerCommandController:
    def __init__(self):
        self._dispatch_service = WorkerCommandDispatchService()

    async def start(self):
        await asyncio.gather(
            self._consume_topic(Model),
            self._consume_topic(ModelInstance),
            self._consume_topic(Worker),
        )

    async def _consume_topic(self, model_class):
        topic = model_class.__name__.lower()
        subscriber = event_bus.subscribe(topic)
        try:
            while True:
                event = await subscriber.receive()
                try:
                    await self._reconcile(model_class, event)
                except Exception as e:
                    logger.error(
                        "Failed to reconcile worker commands for %s event: %s",
                        topic,
                        e,
                    )
        finally:
            event_bus.unsubscribe(topic, subscriber)

    async def _reconcile(self, model_class, event: Event):
        if event.type == EventType.HEARTBEAT or event.data is None:
            return

        if model_class is Model:
            await self._handle_model_event(event)
            return
        if model_class is ModelInstance:
            await self._handle_model_instance_event(event)
            return
        if model_class is Worker:
            await self._handle_worker_event(event)

    async def _handle_model_event(self, event: Event):
        model: Model = event.data
        if model.id is None:
            return

        async with async_session() as session:
            instances = await ModelInstance.all_by_fields(
                session,
                fields={"model_id": model.id, "deleted_at": None},
            )

        worker_ids = _collect_worker_ids_from_instances(instances)
        if not worker_ids:
            return

        await self._dispatch_service.emit_reconcile_now(
            worker_ids=worker_ids,
            model_id=model.id,
            idempotency_token=_event_token(event, model.updated_at),
            reason=f"model_{event.type.name.lower()}",
        )

    async def _handle_model_instance_event(self, event: Event):
        model_instance: ModelInstance = event.data
        if model_instance.id is None:
            return

        worker_ids = _collect_model_instance_worker_ids(
            model_instance,
            event.changed_fields,
        )
        if not worker_ids:
            return

        await self._dispatch_service.emit_reconcile_for_model_instance(
            model_id=model_instance.model_id,
            model_instance_id=model_instance.id,
            worker_ids=worker_ids,
            idempotency_token=_event_token(event, model_instance.updated_at),
        )

    async def _handle_worker_event(self, event: Event):
        worker: Worker = event.data
        if worker.id is None:
            return

        if event.type == EventType.UPDATED:
            relevant_fields = {
                "state",
                "state_message",
                "maintenance",
                "unreachable",
                "reachability_mode",
                "proxy_mode",
            }
            if not any(field in (event.changed_fields or {}) for field in relevant_fields):
                return

        await self._dispatch_service.emit_sync_runtime_state(
            worker_id=worker.id,
            idempotency_token=_event_token(event, worker.updated_at),
            reason=f"worker_{event.type.name.lower()}",
        )


def _event_token(event: Event, updated_at: Optional[datetime]) -> str:
    timestamp = updated_at.isoformat() if updated_at is not None else "none"
    return f"{event.type.name.lower()}:{timestamp}"


def _history_value(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) == 0:
            return None
        if len(value) == 1:
            return value[0]
    return value


def _subordinate_worker_ids(distributed_servers: Any) -> set[int]:
    if distributed_servers is None:
        return set()

    subordinate_workers = getattr(distributed_servers, "subordinate_workers", None)
    if subordinate_workers is None and isinstance(distributed_servers, dict):
        subordinate_workers = distributed_servers.get("subordinate_workers")

    worker_ids = set()
    for worker in subordinate_workers or []:
        worker_id = (
            getattr(worker, "worker_id", None)
            if not isinstance(worker, dict)
            else worker.get("worker_id")
        )
        if worker_id is not None:
            worker_ids.add(worker_id)
    return worker_ids


def _collect_model_instance_worker_ids(
    model_instance: ModelInstance,
    changed_fields: Optional[dict[str, tuple[Any, Any]]] = None,
) -> list[int]:
    worker_ids = set()
    if model_instance.worker_id is not None:
        worker_ids.add(model_instance.worker_id)
    worker_ids.update(_subordinate_worker_ids(model_instance.distributed_servers))

    changed_fields = changed_fields or {}
    worker_id_change = changed_fields.get("worker_id")
    if worker_id_change is not None:
        old_worker_id = _history_value(worker_id_change[0])
        new_worker_id = _history_value(worker_id_change[1])
        if old_worker_id is not None:
            worker_ids.add(old_worker_id)
        if new_worker_id is not None:
            worker_ids.add(new_worker_id)

    distributed_change = changed_fields.get("distributed_servers")
    if distributed_change is not None:
        worker_ids.update(_subordinate_worker_ids(_history_value(distributed_change[0])))
        worker_ids.update(_subordinate_worker_ids(_history_value(distributed_change[1])))

    return sorted(worker_ids)


def _collect_worker_ids_from_instances(instances: Sequence[ModelInstance]) -> list[int]:
    worker_ids = set()
    for instance in instances:
        worker_ids.update(_collect_model_instance_worker_ids(instance))
    return sorted(worker_ids)
