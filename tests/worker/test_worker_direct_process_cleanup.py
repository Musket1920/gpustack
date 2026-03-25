import asyncio
import importlib
import sys
import types
from pathlib import Path
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
bootstrap_manager_module = _import_worker_module("gpustack.worker.bootstrap_manager")


async def _noop_async():
    return None


def _make_bootstrap_manager(tmp_path: Path):
    return bootstrap_manager_module.BootstrapManager(
        SimpleNamespace(
            bootstrap_cache_dir=str(tmp_path / "cache" / "bootstrap"),
            bootstrap_workspace_dir=str(tmp_path / "data" / "bootstrap" / "workspace"),
            bootstrap_artifacts_dir=str(tmp_path / "data" / "bootstrap" / "artifacts"),
            bootstrap_manifests_dir=str(tmp_path / "data" / "bootstrap" / "manifests"),
            bootstrap_locks_dir=str(tmp_path / "data" / "bootstrap" / "locks"),
        )
    )


def _make_worker(reconcile_calls, task_events=None, reconcile_action=None):
    def _register():
        if task_events is not None:
            task_events.append("register")

    def _reload_worker_config(_default_config):
        if task_events is not None:
            task_events.append("reload")

    def _log_worker_config():
        if task_events is not None:
            task_events.append("log")

    def _reconcile_stale_direct_process_registry():
        reconcile_calls.append("reconcile")
        if task_events is not None:
            task_events.append("reconcile")
        if reconcile_action is not None:
            reconcile_action()

    def _sync_model_instances_state():
        return None

    def _sync_benchmark_state():
        return None

    def _sync_worker_status():
        return None

    def _cleanup_orphan_workloads():
        return None

    worker = object.__new__(worker_module.Worker)
    worker._config = SimpleNamespace(
        reload_worker_config=_reload_worker_config,
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
    worker._register = _register
    worker.log_worker_config = _log_worker_config
    worker.worker_id = lambda: 1
    worker.worker_ip = lambda: "127.0.0.1"
    worker.worker_ifname = lambda: "eth0"
    worker.clientset = lambda: worker._clientset
    worker._serve_apis = _noop_async
    worker._serve_manager = SimpleNamespace(
        reconcile_stale_direct_process_registry=_reconcile_stale_direct_process_registry,
        watch_models=_noop_async,
        watch_model_instances_event=_noop_async,
        watch_model_instances=_noop_async,
        sync_model_instances_state=_sync_model_instances_state,
    )
    worker._benchmark_manager = SimpleNamespace(
        sync_benchmark_state=_sync_benchmark_state,
        watch_benchmarks_event=_noop_async,
    )
    worker._worker_manager = SimpleNamespace(sync_worker_status=_sync_worker_status)
    worker._workload_cleaner = SimpleNamespace(
        cleanup_orphan_workloads=_cleanup_orphan_workloads
    )
    worker._create_async_task = lambda coro: worker._async_tasks.append(
        asyncio.create_task(coro)
    )
    return worker


@pytest.mark.asyncio
async def test_direct_process_stale_registry_cleanup_runs_on_worker_startup_and_shutdown(
    monkeypatch,
    tmp_path: Path,
):
    bootstrap_manager = _make_bootstrap_manager(tmp_path)
    runtime_roots = bootstrap_manager.prepare_runtime_roots(11, 22)
    sibling_runtime_roots = bootstrap_manager.prepare_runtime_roots(11, 23)
    (runtime_roots.workspace / "stale.txt").write_text("stale", encoding="utf-8")
    (sibling_runtime_roots.workspace / "keep.txt").write_text("keep", encoding="utf-8")

    reconcile_calls: list[str] = []
    lifecycle_events: list[str] = []
    periodic_calls: list[tuple[str, tuple[int, ...]]] = []
    worker = _make_worker(
        reconcile_calls,
        lifecycle_events,
        reconcile_action=lambda: bootstrap_manager.cleanup_runtime_roots(11, 22),
    )

    monkeypatch.setattr(worker_module, "add_signal_handlers_in_loop", lambda: None)
    monkeypatch.setattr(
        worker_module,
        "run_periodically_in_thread",
        lambda func, *args, **kwargs: periodic_calls.append((func.__name__, args)),
    )
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
    assert lifecycle_events[:4] == ["register", "reload", "log", "reconcile"]
    assert periodic_calls == [
        ("_heartbeat", (worker_module.envs.WORKER_HEARTBEAT_INTERVAL,)),
        ("_sync_worker_status", (worker_module.envs.WORKER_STATUS_SYNC_INTERVAL,)),
        (
            "_sync_model_instances_state",
            (worker_module.envs.MODEL_INSTANCE_HEALTH_CHECK_INTERVAL,),
        ),
        ("_cleanup_orphan_workloads", (120, 15)),
        ("_sync_benchmark_state", (3, 15)),
    ]
    assert lifecycle_events[-1] == "reconcile"
    assert not runtime_roots.workspace.exists()
    assert not runtime_roots.artifacts.exists()
    assert not runtime_roots.manifests.exists()
    assert not runtime_roots.locks.exists()
    assert sibling_runtime_roots.workspace.exists()
    assert sibling_runtime_roots.artifacts.exists()
    assert sibling_runtime_roots.manifests.exists()
    assert sibling_runtime_roots.locks.exists()


@pytest.mark.asyncio
async def test_direct_process_stale_registry_cleanup_runs_in_finally_when_startup_task_fails(
    monkeypatch,
    tmp_path: Path,
):
    bootstrap_manager = _make_bootstrap_manager(tmp_path)
    runtime_roots = bootstrap_manager.prepare_runtime_roots(11, 22)
    sibling_runtime_roots = bootstrap_manager.prepare_runtime_roots(11, 23)
    (runtime_roots.workspace / "stale.txt").write_text("stale", encoding="utf-8")
    (sibling_runtime_roots.workspace / "keep.txt").write_text("keep", encoding="utf-8")

    reconcile_calls: list[str] = []
    lifecycle_events: list[str] = []
    worker = _make_worker(
        reconcile_calls,
        lifecycle_events,
        reconcile_action=lambda: bootstrap_manager.cleanup_runtime_roots(11, 22),
    )

    async def _failing_watch_models():
        lifecycle_events.append("watch_models_started")
        raise RuntimeError("boom")

    worker._serve_manager.watch_models = _failing_watch_models

    monkeypatch.setattr(worker_module, "add_signal_handlers_in_loop", lambda: None)
    monkeypatch.setattr(
        worker_module, "run_periodically_in_thread", lambda *args, **kwargs: None
    )
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

    with pytest.raises(RuntimeError, match="boom"):
        await worker_module.Worker.start_async(worker)

    assert reconcile_calls == ["reconcile", "reconcile"]
    assert lifecycle_events[:4] == ["register", "reload", "log", "reconcile"]
    assert lifecycle_events[-1] == "reconcile"
    assert not runtime_roots.workspace.exists()
    assert not runtime_roots.artifacts.exists()
    assert not runtime_roots.manifests.exists()
    assert not runtime_roots.locks.exists()
    assert sibling_runtime_roots.workspace.exists()
    assert sibling_runtime_roots.artifacts.exists()
    assert sibling_runtime_roots.manifests.exists()
    assert sibling_runtime_roots.locks.exists()
