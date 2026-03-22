import asyncio
import importlib
import sys
import types
from types import SimpleNamespace

import pytest


def _import_worker_module(module_name: str):
    fcntl_stub = types.ModuleType("fcntl")
    setattr(fcntl_stub, "LOCK_EX", 1)
    setattr(fcntl_stub, "LOCK_UN", 2)
    setattr(fcntl_stub, "lockf", lambda *args, **kwargs: None)
    setattr(fcntl_stub, "flock", lambda *args, **kwargs: None)
    original_fcntl = sys.modules.get("fcntl")
    sys.modules["fcntl"] = fcntl_stub
    try:
        return importlib.import_module(module_name)
    finally:
        if original_fcntl is None:
            sys.modules.pop("fcntl", None)
        else:
            sys.modules["fcntl"] = original_fcntl


worker_module = _import_worker_module("gpustack.worker.worker")


async def _noop_async():
    return None


@pytest.mark.asyncio
async def test_direct_process_stale_registry_cleanup_runs_on_worker_startup_and_shutdown(
    monkeypatch,
):
    worker = object.__new__(worker_module.Worker)
    reconcile_calls: list[str] = []

    worker._config = SimpleNamespace(
        reload_worker_config=lambda _default_config: None,
        worker_ip="127.0.0.1",
        worker_ifname="eth0",
        system_default_container_registry=None,
        gateway_mode=SimpleNamespace(value="none"),
    )
    worker._default_config = SimpleNamespace()
    worker._exporter_enabled = False
    worker._clientset = SimpleNamespace()
    worker._worker_id = 1
    worker._async_tasks = []
    worker._register = lambda: None
    worker.log_worker_config = lambda: None
    worker.worker_id = lambda: 1
    worker.worker_ip = lambda: "127.0.0.1"
    worker.worker_ifname = lambda: "eth0"
    worker.clientset = lambda: worker._clientset
    worker._serve_apis = _noop_async
    worker._serve_manager = SimpleNamespace(
        reconcile_stale_direct_process_registry=lambda: reconcile_calls.append(
            "reconcile"
        ),
        watch_models=_noop_async,
        watch_model_instances_event=_noop_async,
        watch_model_instances=_noop_async,
        sync_model_instances_state=lambda: None,
    )
    worker._benchmark_manager = SimpleNamespace(
        sync_benchmark_state=lambda: None,
        watch_benchmarks_event=_noop_async,
    )
    worker._worker_manager = SimpleNamespace(sync_worker_status=lambda: None)
    worker._workload_cleaner = SimpleNamespace(cleanup_orphan_workloads=lambda: None)
    worker._create_async_task = lambda coro: worker._async_tasks.append(
        asyncio.create_task(coro)
    )

    monkeypatch.setattr(worker_module, "add_signal_handlers_in_loop", lambda: None)
    monkeypatch.setattr(worker_module, "run_periodically_in_thread", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker_module,
        "InferenceBackendManager",
        lambda _clientset: SimpleNamespace(
            start_listener=_noop_async,
            get_backend_by_name=lambda _backend: None,
        ),
    )
    monkeypatch.setattr(
        worker_module.registration,
        "determine_default_registry",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        worker_module,
        "ModelFileManager",
        lambda **_kwargs: SimpleNamespace(watch_model_files=_noop_async),
    )
    monkeypatch.setattr(worker_module, "get_resource_injection_policy", lambda: "")

    await worker_module.Worker.start_async(worker)

    assert reconcile_calls == ["reconcile", "reconcile"]
