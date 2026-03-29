import asyncio
import sys
from types import SimpleNamespace
import types
from typing import cast

import pytest

# Inject an fcntl stub before importing gpustack.worker modules so Windows test
# collection does not fail through gpustack.worker.__init__.
if "fcntl" not in sys.modules:
    _fcntl_stub = types.ModuleType("fcntl")
    _fcntl_stub.LOCK_EX = 1  # type: ignore[attr-defined]
    _fcntl_stub.LOCK_UN = 2  # type: ignore[attr-defined]
    _fcntl_stub.lockf = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    _fcntl_stub.flock = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["fcntl"] = _fcntl_stub

from gpustack.client import ClientSet
from gpustack.schemas.model_files import ModelFileStateEnum
from gpustack.schemas.workers import WorkerCommandMessage
from gpustack.server.bus import EventType
from gpustack.server.worker_command_service import (
    COMMAND_RECONCILE_MODEL_INSTANCE,
    COMMAND_RECONCILE_NOW,
)
from gpustack.worker.command_executor import WorkerCommandExecutor


class FakeWebSocket:
    def __init__(self):
        self.sent_json = []

    async def send_json(self, payload: dict):
        self.sent_json.append(payload)


class FakeModelInstancesAPI:
    def __init__(self, model_instance):
        self._model_instance = model_instance
        self.calls = []

    def get(self, model_instance_id: int):
        self.calls.append(model_instance_id)
        return self._model_instance


class FakeServeManager:
    def __init__(self):
        self.events = []
        self.sync_calls = 0

    def _handle_model_instance_event(self, event):
        self.events.append(event)

    def sync_model_instances_state(self):
        self.sync_calls += 1


class FakeWorkerManager:
    def __init__(self):
        self.sync_calls = 0

    def sync_worker_status(self):
        self.sync_calls += 1


class FakeModelFilesAPI:
    def __init__(self, items=None):
        self.items = list(items or [])
        self.calls = []

    def list(self, params):
        self.calls.append(params)
        return SimpleNamespace(items=list(self.items))


class FakeModelFileManager:
    def __init__(self, worker_id: int = 1):
        self._worker_id = worker_id
        self.prerun_calls = 0
        self.created_download_task_ids = []

    def reconcile_pending_downloads(self):
        raise AssertionError("reconcile_pending_downloads should not be called")

    def _prerun(self):
        self.prerun_calls += 1

    def _create_download_task(self, model_file):
        self.created_download_task_ids.append(model_file.id)


class FakeBenchmarkManager:
    def __init__(self):
        self.sync_calls = 0

    def sync_benchmark_state(self):
        self.sync_calls += 1


async def wait_for(predicate, timeout: float = 1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not satisfied before timeout")


def build_executor(model_instance):
    clientset = SimpleNamespace(
        model_instances=FakeModelInstancesAPI(model_instance),
        model_files=FakeModelFilesAPI(),
    )
    serve_manager = FakeServeManager()
    worker_manager = FakeWorkerManager()
    model_file_manager = FakeModelFileManager()
    benchmark_manager = FakeBenchmarkManager()
    executor = WorkerCommandExecutor(
        clientset_getter=lambda: cast(ClientSet, clientset),
        serve_manager=serve_manager,
        worker_manager=worker_manager,
        model_file_manager=model_file_manager,
        benchmark_manager=benchmark_manager,
    )
    return (
        executor,
        clientset,
        serve_manager,
        worker_manager,
        model_file_manager,
        benchmark_manager,
    )


def build_runtime_sync_executor(model_files=None):
    clientset = SimpleNamespace(
        model_instances=FakeModelInstancesAPI(SimpleNamespace(id=22, model_id=11)),
        model_files=FakeModelFilesAPI(model_files),
    )
    serve_manager = FakeServeManager()
    worker_manager = FakeWorkerManager()
    model_file_manager = FakeModelFileManager(worker_id=7)
    benchmark_manager = FakeBenchmarkManager()
    executor = WorkerCommandExecutor(
        clientset_getter=lambda: cast(ClientSet, clientset),
        serve_manager=serve_manager,
        worker_manager=worker_manager,
        model_file_manager=model_file_manager,
        benchmark_manager=benchmark_manager,
    )
    return (
        executor,
        clientset,
        serve_manager,
        worker_manager,
        model_file_manager,
        benchmark_manager,
    )


@pytest.mark.asyncio
async def test_reconcile_command_delegates_to_serve_manager():
    model_instance = SimpleNamespace(id=22, model_id=11)
    executor, clientset, serve_manager, *_ = build_executor(model_instance)
    websocket = FakeWebSocket()
    command = WorkerCommandMessage(
        session_id="session-1",
        command_id="cmd-1",
        command_type=COMMAND_RECONCILE_MODEL_INSTANCE,
        payload={"model_id": 11, "model_instance_id": 22},
    )

    await executor.handle_command(command, websocket)
    await wait_for(lambda: len(websocket.sent_json) == 2)

    assert clientset.model_instances.calls == [22]
    assert len(serve_manager.events) == 1
    assert serve_manager.events[0].type == EventType.UPDATED
    assert serve_manager.events[0].data is model_instance
    assert websocket.sent_json[0]["message_type"] == "command_ack"
    assert websocket.sent_json[0]["accepted"] is True
    assert websocket.sent_json[1]["message_type"] == "command_result"
    assert websocket.sent_json[1]["state"] == "succeeded"
    assert websocket.sent_json[1]["result"]["delegated_to"] == ["serve_manager"]

    await executor.stop()


@pytest.mark.asyncio
async def test_duplicate_command_is_idempotent():
    model_instance = SimpleNamespace(id=22, model_id=11)
    executor, clientset, serve_manager, *_ = build_executor(model_instance)
    websocket = FakeWebSocket()
    command = WorkerCommandMessage(
        session_id="session-1",
        command_id="cmd-duplicate",
        command_type=COMMAND_RECONCILE_MODEL_INSTANCE,
        payload={"model_id": 11, "model_instance_id": 22},
    )

    await executor.handle_command(command, websocket)
    await wait_for(lambda: len(websocket.sent_json) == 2)

    await executor.handle_command(command, websocket)
    await wait_for(lambda: len(websocket.sent_json) == 4)

    assert clientset.model_instances.calls == [22]
    assert len(serve_manager.events) == 1
    assert [message["message_type"] for message in websocket.sent_json] == [
        "command_ack",
        "command_result",
        "command_ack",
        "command_result",
    ]
    assert websocket.sent_json[1]["state"] == "succeeded"
    assert websocket.sent_json[3]["state"] == "succeeded"

    await executor.stop()


@pytest.mark.asyncio
async def test_reconcile_now_runtime_sync_is_idempotent():
    downloading_file = SimpleNamespace(id=91, state=ModelFileStateEnum.DOWNLOADING)
    ready_file = SimpleNamespace(id=92, state=ModelFileStateEnum.READY)
    (
        executor,
        clientset,
        serve_manager,
        worker_manager,
        model_file_manager,
        benchmark_manager,
    ) = build_runtime_sync_executor([downloading_file, ready_file])
    websocket = FakeWebSocket()
    command = WorkerCommandMessage(
        session_id="session-1",
        command_id="cmd-runtime-sync",
        command_type=COMMAND_RECONCILE_NOW,
        payload={"reason": "server recovery", "full_reconcile": True, "worker_id": 7},
    )

    await executor.handle_command(command, websocket)
    await wait_for(lambda: len(websocket.sent_json) == 2)

    await executor.handle_command(command, websocket)
    await wait_for(lambda: len(websocket.sent_json) == 4)

    assert worker_manager.sync_calls == 1
    assert serve_manager.sync_calls == 1
    assert benchmark_manager.sync_calls == 1
    assert model_file_manager.prerun_calls == 1
    assert model_file_manager.created_download_task_ids == [91]
    assert clientset.model_files.calls == [{"worker_id": 7}]
    assert websocket.sent_json[0]["message_type"] == "command_ack"
    assert websocket.sent_json[1]["message_type"] == "command_result"
    assert websocket.sent_json[1]["state"] == "succeeded"
    assert websocket.sent_json[1]["result"] == {
        "delegated": True,
        "delegated_to": [
            "worker_manager",
            "model_file_manager",
            "benchmark_manager",
            "serve_manager",
        ],
        "reason": "server recovery",
        "worker_id": 7,
        "full_reconcile": True,
    }
    assert websocket.sent_json[3]["result"] == websocket.sent_json[1]["result"]

    await executor.stop()
