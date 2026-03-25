import contextlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from gpustack.config.config import Config
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    SourceEnum,
)
from tests.worker.fake_backend_fixture import SCRIPTS_DIR


WORKER_DIR = Path(__file__).resolve().parents[2] / "gpustack" / "worker"


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


def test_serve_manager_maps_llama_cpp_to_first_class_server():
    serve_manager_module = _import_worker_module("gpustack.worker.serve_manager")
    llama_cpp_module = _import_worker_module("gpustack.worker.backends.llama_cpp")

    assert (
        serve_manager_module._SERVER_CLASS_MAPPING[BackendEnum.LLAMA_CPP]
        is llama_cpp_module.LlamaCppServer
    )


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def make_config(tmp_path: Path, **kwargs) -> Config:
    return Config(
        token="test",
        jwt_secret_key="test",
        data_dir=str(tmp_path),
        server_url="http://127.0.0.1:30080",
        direct_process_mode=True,
        **kwargs,
    )


def make_model_instance(**kwargs) -> ModelInstance:
    defaults = {
        "id": 1,
        "name": "test-instance",
        "worker_id": 1,
        "worker_name": "worker-1",
        "worker_ip": "127.0.0.1",
        "model_id": 1,
        "model_name": "test-model",
        "state": ModelInstanceStateEnum.SCHEDULED,
        "source": SourceEnum.HUGGING_FACE,
        "huggingface_repo_id": "Qwen/Qwen2.5-7B-Instruct",
    }
    defaults.update(kwargs)
    return ModelInstance(**defaults)


def make_model(**kwargs) -> Model:
    defaults = {
        "id": 1,
        "name": "test-model",
        "source": SourceEnum.HUGGING_FACE,
        "huggingface_repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "backend": BackendEnum.VLLM,
        "backend_version": "0.8.0",
    }
    defaults.update(kwargs)
    return Model(**defaults)


class _ModelInstancesAPI:
    def __init__(self, items: list[ModelInstance]):
        self._items = items

    def list(self):
        return SimpleNamespace(items=self._items)


class _ClientSetStub:
    def __init__(self, items: list[ModelInstance]):
        self.headers = {}
        self.model_instances = _ModelInstancesAPI(items)


def _apply_model_instance_patch(model_instance: ModelInstance, patch: dict) -> None:
    for key, value in patch.items():
        if "." in key:
            continue
        setattr(model_instance, key, value)


@dataclass(frozen=True)
class _PreparedEnvironmentIdentityStub:
    backend: str
    backend_version: str
    recipe_id: str | None
    prepared_environment_id: str | None
    resolver_version: str
    python_identity: dict[str, str]

    def manifest_payload(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "recipe_id": self.recipe_id,
            "prepared_environment_id": self.prepared_environment_id,
            "resolver_version": self.resolver_version,
            "python_identity": self.python_identity,
        }


class _BootstrapManagerStub:
    PREPARED_CACHE_CONTEXT_FILENAME = "bootstrap-prepared-context.json"
    PREPARED_CACHE_RESOLVER_VERSION = "direct-process-bootstrap-v1"
    BOOTSTRAP_MANIFEST_FILENAME = "bootstrap-manifest.json"
    BOOTSTRAP_ARTIFACT_FILENAME = "bootstrap-artifact.json"
    BOOTSTRAP_CONTEXT_FILENAME = "bootstrap-context.txt"
    BOOTSTRAP_LOCK_FILENAME = "bootstrap.lock"
    PREPARED_CACHE_ARTIFACTS_DIRNAME = "artifacts"
    PREPARED_CACHE_PROVISIONING_FILENAME = "bootstrap-provisioning.json"
    PREPARED_CACHE_ENV_FILENAME = "prepared.env"
    PREPARED_CACHE_CONFIG_FILENAME = "prepared-config.json"
    PREPARED_CACHE_LAUNCH_FILENAME = "prepared-launch.sh"
    PREPARED_CACHE_EXECUTABLE_PROVENANCE_FILENAME = "executable-provenance.json"

    def __init__(self, calls: list[tuple[object, ...]], root: Path):
        self.calls = calls
        self._root = root

    def _prepared_cache_path(self, backend_name: str, backend_version: str) -> Path:
        return self._root / "prepared-cache" / backend_name / backend_version

    def prepared_cache_root(self, backend_name: str, backend_version: str):
        self.calls.append(("prepared_cache_root", backend_name, backend_version))
        return self._prepared_cache_path(backend_name, backend_version)

    def prepared_cache_context_path(self, backend_name: str, backend_version: str) -> Path:
        self.calls.append(("prepared_cache_context_path", backend_name, backend_version))
        return self._prepared_cache_path(backend_name, backend_version).joinpath(
            self.PREPARED_CACHE_CONTEXT_FILENAME
        )

    def prepared_cache_artifacts_root(
        self,
        backend_name: str,
        backend_version: str,
    ) -> Path:
        return self._prepared_cache_path(backend_name, backend_version).joinpath(
            self.PREPARED_CACHE_ARTIFACTS_DIRNAME
        )

    def prepared_cache_artifact_path(
        self,
        backend_name: str,
        backend_version: str,
        *parts: str,
    ) -> Path:
        path = self.prepared_cache_artifacts_root(backend_name, backend_version).joinpath(
            *parts
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def prepared_environment_identity(
        self,
        *,
        backend_name: str,
        backend_version: str,
        recipe_id: str | None,
        prepared_environment_id: str | None,
    ):
        self.calls.append(
            (
                "prepared_environment_identity",
                backend_name,
                backend_version,
                recipe_id,
                prepared_environment_id,
            )
        )
        return _PreparedEnvironmentIdentityStub(
            backend=backend_name,
            backend_version=backend_version,
            recipe_id=recipe_id,
            prepared_environment_id=prepared_environment_id,
            resolver_version=self.PREPARED_CACHE_RESOLVER_VERSION,
            python_identity={
                "implementation": "cpython",
                "version": sys.version.split()[0],
                "executable": sys.executable,
                "cache_tag": sys.implementation.cache_tag or "",
            },
        )

    def build_prepared_cache_record(
        self,
        identity,
        *,
        invalidation_state: str = "valid",
        invalidation_reason: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            (
                "build_prepared_cache_record",
                identity.backend,
                identity.backend_version,
                invalidation_state,
            )
        )
        manifest_payload = identity.manifest_payload()
        manifest_hash = hashlib.sha256(
            json.dumps(
                manifest_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            **manifest_payload,
            "manifest_hash": manifest_hash,
            "invalidation": {
                "state": invalidation_state,
                "reason": invalidation_reason,
            },
        }

    def prepare_prepared_cache_root(self, backend_name: str, backend_version: str):
        self.calls.append(("prepare_prepared_cache_root", backend_name, backend_version))
        path = self._prepared_cache_path(backend_name, backend_version)
        path.mkdir(parents=True, exist_ok=True)
        artifacts_root = self.prepared_cache_artifacts_root(backend_name, backend_version)
        artifacts_root.mkdir(parents=True, exist_ok=True)
        self.prepared_cache_context_path(backend_name, backend_version).write_text(
            json.dumps(
                {
                    "backend": backend_name,
                    "backend_version": backend_version,
                    "manifest_hash": "stub-manifest-hash",
                    "invalidation": {"state": "valid", "reason": None},
                }
            ),
            encoding="utf-8",
        )
        self.prepared_cache_artifact_path(
            backend_name,
            backend_version,
            self.PREPARED_CACHE_ENV_FILENAME,
        ).write_text("PATH=/prepared/bin:${PATH}\n", encoding="utf-8")
        self.prepared_cache_artifact_path(
            backend_name,
            backend_version,
            self.PREPARED_CACHE_LAUNCH_FILENAME,
        ).write_text("#!/bin/sh\nexec \"$@\"\n", encoding="utf-8")
        self.prepared_cache_artifact_path(
            backend_name,
            backend_version,
            self.PREPARED_CACHE_CONFIG_FILENAME,
        ).write_text(
            json.dumps(
                {
                    "backend": backend_name,
                    "backend_version": backend_version,
                    "prepared_cache_root": str(path),
                }
            ),
            encoding="utf-8",
        )
        self.prepared_cache_artifact_path(
            backend_name,
            backend_version,
            self.PREPARED_CACHE_EXECUTABLE_PROVENANCE_FILENAME,
        ).write_text(
            json.dumps(
                {
                    "state": "resolved",
                    "name": "vllm",
                    "prepared_path": "/prepared/bin/vllm",
                    "sha256": "sha256:stub",
                }
            ),
            encoding="utf-8",
        )
        self.prepared_cache_artifact_path(
            backend_name,
            backend_version,
            self.PREPARED_CACHE_PROVISIONING_FILENAME,
        ).write_text(
            json.dumps({"state": "reused", "backend": backend_name}),
            encoding="utf-8",
        )
        return path

    def prepare_runtime_roots(self, deployment_id: int, model_instance_id: int):
        self.calls.append(("prepare_runtime_roots", deployment_id, model_instance_id))
        return SimpleNamespace(
            workspace=self.workspace_path(deployment_id, model_instance_id),
            artifacts=self.artifacts_path(deployment_id, model_instance_id),
            manifests=self.manifests_path(deployment_id, model_instance_id),
            locks=self.locks_path(deployment_id, model_instance_id),
        )

    def cleanup_runtime_roots(self, deployment_id: int, model_instance_id: int):
        self.calls.append(("cleanup_runtime_roots", deployment_id, model_instance_id))

    def workspace_path(self, deployment_id: int, model_instance_id: int, *parts: str) -> Path:
        base = self._root / "workspace" / str(deployment_id) / str(model_instance_id)
        if parts:
            path = base / Path(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        path = base
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifacts_path(self, deployment_id: int, model_instance_id: int) -> Path:
        path = self._root / "artifacts" / str(deployment_id) / str(model_instance_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def manifests_path(self, deployment_id: int, model_instance_id: int) -> Path:
        path = self._root / "manifests" / str(deployment_id) / str(model_instance_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def locks_path(self, deployment_id: int, model_instance_id: int) -> Path:
        path = self._root / "locks" / str(deployment_id) / str(model_instance_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_path(self, deployment_id: int, model_instance_id: int, *parts: str) -> Path:
        path = self.artifacts_path(deployment_id, model_instance_id) / Path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def manifest_path(self, deployment_id: int, model_instance_id: int, *parts: str) -> Path:
        path = self.manifests_path(deployment_id, model_instance_id) / Path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def lock_path(self, deployment_id: int, model_instance_id: int, *parts: str) -> Path:
        path = self.locks_path(deployment_id, model_instance_id) / Path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _build_manager(
    serve_manager_module,
    tmp_path: Path,
    model_instance: ModelInstance,
    *,
    bootstrap_manager=None,
):
    cfg = make_config(tmp_path)
    model = make_model()
    clientset = _ClientSetStub([model_instance])
    manager = object.__new__(serve_manager_module.ServeManager)
    manager._config = cfg
    manager._serve_log_dir = str(tmp_path / "serve")
    Path(manager._serve_log_dir).mkdir(parents=True, exist_ok=True)
    manager._worker_id_getter = lambda: 1
    manager._clientset_getter = lambda: clientset
    manager._provisioning_processes = {}
    manager._error_model_instances = {}
    manager._model_cache_by_instance = {}
    manager._model_instance_by_instance_id = {}
    manager._assigned_ports = {}
    manager._direct_process_registry = serve_manager_module.DirectProcessRegistry(cfg)
    manager._bootstrap_manager = bootstrap_manager
    manager._inference_backend_manager = SimpleNamespace(
        get_backend_by_name=lambda _backend: SimpleNamespace(health_check_path="/v1/models")
    )

    updates: list[dict] = []

    def update_model_instance(_id: int, **kwargs):
        updates.append(kwargs)
        _apply_model_instance_patch(model_instance, kwargs)

    manager._update_model_instance = update_model_instance
    manager._get_model = lambda _mi: model
    manager._refresh_model = lambda _mi: model
    manager._update_model = lambda *_args, **_kwargs: None

    return manager, model, updates


class _ProvisioningProcess:
    def __init__(
        self,
        scenario: str,
        args: tuple,
        child_pid_path: Path | None = None,
        lifecycle_events: list[tuple[object, ...]] | None = None,
    ):
        self._scenario = scenario
        self._args = args
        self._child_pid_path = child_pid_path
        self._lifecycle_events = lifecycle_events
        self._wrapper: subprocess.Popen[str] | None = None
        self._real_process: subprocess.Popen[str] | None = None
        self._log_handle = None
        self.pid: int | None = None
        self.real_pid: int | None = None

    def start(self):
        model_instance, backend, _, log_file_path, _, _, _, _, registry_path = self._args
        if self._lifecycle_events is not None:
            self._lifecycle_events.append(("process_start", model_instance.id, backend))
        self._wrapper = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.2)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.pid = self._wrapper.pid

        if self._scenario == "ready":
            self._real_process = subprocess.Popen(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "fake_ready_server.py"),
                    "--port",
                    str(model_instance.port),
                    "--health-path",
                    "/v1/models",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        elif self._scenario == "failed_process":
            self._log_handle = open(log_file_path, "a", encoding="utf-8")
            self._real_process = subprocess.Popen(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "fake_early_exit.py"),
                    "--exit-code",
                    "17",
                    "--message",
                    "fake backend exited early",
                ],
                stdout=self._log_handle,
                stderr=self._log_handle,
                text=True,
            )
        else:
            raise ValueError(f"Unsupported scenario: {self._scenario}")

        self.real_pid = self._real_process.pid
        assert registry_path is not None
        registry = serve_manager_module.DirectProcessRegistry(registry_path)
        registry.upsert(
            model_instance_id=model_instance.id,
            deployment_name=model_instance.name,
            pid=self.real_pid,
            port=model_instance.port,
            log_path=log_file_path,
            backend=backend,
        )

    def is_alive(self) -> bool:
        return self._wrapper is not None and self._wrapper.poll() is None

    def join(self, timeout: float = 0):
        if self._wrapper is None:
            return
        if timeout <= 0:
            self._wrapper.poll()
            return
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._wrapper.wait(timeout=timeout)

    def cleanup(self):
        for process in (self._real_process, self._wrapper):
            if process and process.poll() is None:
                with contextlib.suppress(Exception):
                    psutil.Process(process.pid).kill()
                    process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()


class _ProcessFactory:
    def __init__(
        self,
        scenario: str,
        lifecycle_events: list[tuple[object, ...]] | None = None,
    ):
        self._scenario = scenario
        self._lifecycle_events = lifecycle_events
        self.instances: list[_ProvisioningProcess] = []

    def __call__(self, target, args):
        del target
        process = _ProvisioningProcess(
            self._scenario,
            args,
            lifecycle_events=self._lifecycle_events,
        )
        self.instances.append(process)
        return process


serve_manager_module = _import_worker_module("gpustack.worker.serve_manager")


def test_direct_process_start_ready_stop_tracks_serving_pid(monkeypatch, tmp_path: Path):
    model_instance = make_model_instance()
    manager, model, updates = _build_manager(serve_manager_module, tmp_path, model_instance)
    process_factory = _ProcessFactory("ready")

    monkeypatch.setattr(serve_manager_module, "get_meta_from_running_instance", lambda *_args: None)
    monkeypatch.setattr(serve_manager_module.multiprocessing, "Process", process_factory)
    monkeypatch.setattr(serve_manager_module.platform, "system", lambda: "linux")

    try:
        serve_manager_module.ServeManager._start_model_instance(manager, model_instance)

        assert updates[0]["pid"] is None
        assert _wait_until(lambda: not manager._is_provisioning(model_instance))
        assert _wait_until(
            lambda: serve_manager_module.is_ready(
                BackendEnum.VLLM,
                model_instance,
                "/v1/models",
                model,
            )
        )

        serve_manager_module.ServeManager.sync_model_instances_state(manager)

        assert process_factory.instances[0].real_pid is not None
        real_pid = process_factory.instances[0].real_pid
        assert model_instance.state == ModelInstanceStateEnum.RUNNING
        assert model_instance.pid == real_pid
        registry_entry = manager._direct_process_registry.get_by_model_instance_id(model_instance.id)
        assert registry_entry is not None
        assert registry_entry.pid == real_pid

        serve_manager_module.ServeManager._stop_model_instance(manager, model_instance)

        assert _wait_until(
            lambda: not psutil.pid_exists(real_pid),
            timeout=5,
        )
        assert manager._direct_process_registry.get_by_model_instance_id(model_instance.id) is None
    finally:
        for process in process_factory.instances:
            process.cleanup()


def test_direct_process_start_prepares_bootstrap_before_provisioning_launch(
    monkeypatch, tmp_path: Path
):
    model_instance = make_model_instance()
    bootstrap_calls: list[tuple[object, ...]] = []
    manager, model, _ = _build_manager(
        serve_manager_module,
        tmp_path,
        model_instance,
        bootstrap_manager=_BootstrapManagerStub(bootstrap_calls, tmp_path / "bootstrap-stub"),
    )
    process_factory = _ProcessFactory("ready", lifecycle_events=bootstrap_calls)

    monkeypatch.setattr(serve_manager_module, "get_meta_from_running_instance", lambda *_args: None)
    monkeypatch.setattr(serve_manager_module.multiprocessing, "Process", process_factory)
    monkeypatch.setattr(serve_manager_module.platform, "system", lambda: "linux")

    try:
        serve_manager_module.ServeManager._start_model_instance(manager, model_instance)

        assert bootstrap_calls[:6] == [
            ("cleanup_runtime_roots", model_instance.model_id, model_instance.id),
            (
                "prepared_environment_identity",
                str(BackendEnum.VLLM),
                model.backend_version,
                "vllm",
                f"vllm:{model.backend_version}",
            ),
            (
                "build_prepared_cache_record",
                str(BackendEnum.VLLM),
                model.backend_version,
                "valid",
            ),
            ("prepared_cache_root", str(BackendEnum.VLLM), model.backend_version),
            (
                "prepared_cache_context_path",
                str(BackendEnum.VLLM),
                model.backend_version,
            ),
            ("prepare_prepared_cache_root", str(BackendEnum.VLLM), model.backend_version),
        ]
        assert (
            bootstrap_calls.index(("prepare_runtime_roots", model_instance.model_id, model_instance.id))
            < bootstrap_calls.index(("process_start", model_instance.id, BackendEnum.VLLM))
        )
        assert _wait_until(lambda: not manager._is_provisioning(model_instance))
        assert _wait_until(
            lambda: serve_manager_module.is_ready(
                BackendEnum.VLLM,
                model_instance,
                "/v1/models",
                model,
            )
        )

        serve_manager_module.ServeManager.sync_model_instances_state(manager)

        entry = manager._direct_process_registry.get_by_model_instance_id(model_instance.id)
        assert entry is not None
        assert entry.pid == process_factory.instances[0].real_pid
    finally:
        serve_manager_module.ServeManager._stop_model_instance(manager, model_instance)
        for process in process_factory.instances:
            process.cleanup()


def test_direct_process_bootstrap_before_launch_writes_runtime_prepared_artifact_references(
    monkeypatch,
    tmp_path: Path,
):
    model_instance = make_model_instance()
    manager, _, _ = _build_manager(serve_manager_module, tmp_path, model_instance)
    manager._bootstrap_manager = serve_manager_module.BootstrapManager(manager._config)
    runtime_artifact_path = manager._bootstrap_manager.artifact_path(
        model_instance.model_id,
        model_instance.id,
        manager._bootstrap_manager.BOOTSTRAP_ARTIFACT_FILENAME,
    )
    observed: dict[str, object] = {}

    class _ObservingProcessFactory(_ProcessFactory):
        def __call__(self, target, args):
            observed["artifact_exists_before_process_object"] = runtime_artifact_path.exists()
            return super().__call__(target, args)

    process_factory = _ObservingProcessFactory("ready")

    monkeypatch.setattr(serve_manager_module, "get_meta_from_running_instance", lambda *_args: None)
    monkeypatch.setattr(
        serve_manager_module,
        "ensure_model_instance_direct_process_support",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(serve_manager_module.multiprocessing, "Process", process_factory)
    monkeypatch.setattr(serve_manager_module.platform, "system", lambda: "linux")

    try:
        serve_manager_module.ServeManager._start_model_instance(manager, model_instance)

        runtime_artifact = json.loads(runtime_artifact_path.read_text(encoding="utf-8"))
        assert observed["artifact_exists_before_process_object"] is True
        assert runtime_artifact["prepared_artifacts"]["prepared_cache_context"].endswith(
            manager._bootstrap_manager.PREPARED_CACHE_CONTEXT_FILENAME
        )
        assert runtime_artifact["prepared_artifacts"]["env"].endswith(
            manager._bootstrap_manager.PREPARED_CACHE_ENV_FILENAME
        )
        assert runtime_artifact["prepared_artifacts"]["config"].endswith(
            manager._bootstrap_manager.PREPARED_CACHE_CONFIG_FILENAME
        )
        assert runtime_artifact["prepared_artifacts"]["launch"].endswith(
            manager._bootstrap_manager.PREPARED_CACHE_LAUNCH_FILENAME
        )
        assert runtime_artifact["prepared_artifacts"]["executable_provenance"].endswith(
            manager._bootstrap_manager.PREPARED_CACHE_EXECUTABLE_PROVENANCE_FILENAME
        )
    finally:
        serve_manager_module.ServeManager._stop_model_instance(manager, model_instance)
        for process in process_factory.instances:
            process.cleanup()


def test_direct_process_failed_process_sets_error_from_registry(
    monkeypatch, tmp_path: Path
):
    model_instance = make_model_instance()
    manager, _, updates = _build_manager(serve_manager_module, tmp_path, model_instance)
    process_factory = _ProcessFactory("failed_process")

    monkeypatch.setattr(serve_manager_module, "get_meta_from_running_instance", lambda *_args: None)
    monkeypatch.setattr(serve_manager_module.multiprocessing, "Process", process_factory)
    monkeypatch.setattr(serve_manager_module.platform, "system", lambda: "linux")

    try:
        serve_manager_module.ServeManager._start_model_instance(manager, model_instance)

        assert updates[0]["pid"] is None
        assert _wait_until(lambda: not manager._is_provisioning(model_instance))
        assert _wait_until(
            lambda: process_factory.instances[0].real_pid is not None
            and not psutil.pid_exists(process_factory.instances[0].real_pid),
            timeout=5,
        )

        serve_manager_module.ServeManager.sync_model_instances_state(manager)

        assert model_instance.state == ModelInstanceStateEnum.ERROR
        assert model_instance.state_message == "Inference server exited or unhealthy."
        log_path = Path(manager._serve_log_dir) / f"{model_instance.id}.log"
        assert "fake backend exited early" in log_path.read_text(encoding="utf-8")
    finally:
        for process in process_factory.instances:
            process.cleanup()


def test_direct_process_live_but_not_ready_transitions_out_of_running(
    monkeypatch, tmp_path: Path
):
    model_instance = make_model_instance(
        state=ModelInstanceStateEnum.RUNNING,
        port=18082,
        ports=[18082],
        pid=99999,
    )
    manager, _, updates = _build_manager(serve_manager_module, tmp_path, model_instance)
    process_factory = _ProcessFactory("ready")

    monkeypatch.setattr(serve_manager_module, "get_meta_from_running_instance", lambda *_args: None)
    monkeypatch.setattr(serve_manager_module.multiprocessing, "Process", process_factory)
    monkeypatch.setattr(serve_manager_module.platform, "system", lambda: "linux")
    registry_entry = None

    try:
        serve_manager_module.ServeManager._start_model_instance(manager, model_instance)

        assert _wait_until(lambda: not manager._is_provisioning(model_instance))
        assert _wait_until(
            lambda: serve_manager_module.is_ready(
                BackendEnum.VLLM,
                model_instance,
                "/v1/models",
            )
        )
        serve_manager_module.ServeManager.sync_model_instances_state(manager)
        assert model_instance.state == ModelInstanceStateEnum.RUNNING

        registry_entry = manager._direct_process_registry.get_by_model_instance_id(
            model_instance.id
        )
        assert registry_entry is not None

        broken_port = registry_entry.port + 1000
        model_instance.port = broken_port
        model_instance.ports = [broken_port]
        serve_manager_module.ServeManager.sync_model_instances_state(manager)

        assert model_instance.state == ModelInstanceStateEnum.STARTING
        assert model_instance.state_message == ""
        assert updates[-1]["state"] == ModelInstanceStateEnum.STARTING
    finally:
        restored_port = registry_entry.port if registry_entry is not None else 18082
        model_instance.port = restored_port
        model_instance.ports = [restored_port]
        serve_manager_module.ServeManager._stop_model_instance(manager, model_instance)
        for process in process_factory.instances:
            process.cleanup()


def test_direct_process_stop_kills_process_tree_from_registry(tmp_path: Path):
    model_instance = make_model_instance(port=18080, ports=[18080], state=ModelInstanceStateEnum.RUNNING)
    manager, _, _ = _build_manager(serve_manager_module, tmp_path, model_instance)
    child_pid_path = tmp_path / "fake-child.pid"
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPTS_DIR / "fake_process_tree_parent.py"),
            "--child-pid-file",
            str(child_pid_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    try:
        assert _wait_until(child_pid_path.exists)
        child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())
        assert _wait_until(lambda: psutil.pid_exists(child_pid))

        manager._direct_process_registry.upsert(
            model_instance_id=model_instance.id,
            deployment_name=model_instance.name,
            pid=process.pid,
            port=model_instance.port,
            log_path=str(Path(manager._serve_log_dir) / f"{model_instance.id}.log"),
            backend=BackendEnum.VLLM,
        )

        serve_manager_module.ServeManager._stop_model_instance(manager, model_instance)

        assert _wait_until(lambda: process.poll() is not None, timeout=5)
        assert _wait_until(lambda: not psutil.pid_exists(child_pid), timeout=5)
        assert manager._direct_process_registry.get_by_model_instance_id(model_instance.id) is None
    finally:
        if process.poll() is None:
            with contextlib.suppress(Exception):
                psutil.Process(process.pid).kill()
                process.wait(timeout=5)


def test_direct_process_stop_uses_contract_signal_and_timeout(monkeypatch, tmp_path: Path):
    manager, _, _ = _build_manager(
        serve_manager_module,
        tmp_path,
        make_model_instance(state=ModelInstanceStateEnum.RUNNING),
    )
    entry = serve_manager_module.DirectProcessRegistryEntry(
        model_instance_id=1,
        worker_id=1,
        deployment_name="custom-stop-contract",
        runtime_name="serve",
        pid=4321,
        process_group_id=8765,
        port=18080,
        log_path=str(tmp_path / "serve" / "1.log"),
        backend=BackendEnum.CUSTOM,
        mode=serve_manager_module.DIRECT_PROCESS_RUNTIME_MODE,
        stop_signal="SIGINT",
        stop_timeout_seconds=7,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    observed: dict = {}

    class _FakeProcess:
        def children(self, recursive=False):
            return []

        def wait(self, timeout=None):
            observed["wait_timeout"] = timeout

    monkeypatch.setattr(serve_manager_module.platform, "system", lambda: "linux")
    monkeypatch.setattr(
        serve_manager_module.os,
        "killpg",
        lambda pgid, sig: observed.update({"process_group_id": pgid, "signal": sig}),
        raising=False,
    )
    monkeypatch.setattr(serve_manager_module.psutil, "Process", lambda _pid: _FakeProcess())
    monkeypatch.setattr(serve_manager_module.psutil, "pid_exists", lambda _pid: False)
    monkeypatch.setattr(
        serve_manager_module,
        "terminate_process_tree",
        lambda _pid: observed.update({"terminate_called": True}),
    )

    serve_manager_module.ServeManager._terminate_direct_process_entry(manager, entry)

    assert observed["process_group_id"] == 8765
    assert observed["signal"] == serve_manager_module.signal.SIGINT
    assert observed["wait_timeout"] == 7
    assert "terminate_called" not in observed


def test_direct_process_sync_marks_startup_timeout_as_error(tmp_path: Path):
    model_instance = make_model_instance(
        state=ModelInstanceStateEnum.STARTING,
        port=18083,
        ports=[18083],
        pid=1234,
    )
    manager, model, updates = _build_manager(serve_manager_module, tmp_path, model_instance)
    created_at = datetime.now(timezone.utc) - timedelta(seconds=10)

    manager._direct_process_registry.upsert(
        model_instance_id=model_instance.id,
        deployment_name=model_instance.name,
        pid=1234,
        port=model_instance.port,
        log_path=str(Path(manager._serve_log_dir) / f"{model_instance.id}.log"),
        backend=BackendEnum.CUSTOM,
        startup_timeout_seconds=5,
    )
    entry = manager._direct_process_registry.get_by_model_instance_id(model_instance.id)
    assert entry is not None
    timed_out_entry = entry.model_copy(
        update={
            "created_at": created_at,
            "updated_at": created_at,
        }
    )
    manager._direct_process_registry.inspect_by_model_instance_id = lambda _id: (  # type: ignore[method-assign]
        serve_manager_module.DirectProcessRegistryStatus(
            status=serve_manager_module.DirectProcessEntryStatus.LIVE,
            reason="pid_running",
            entry=timed_out_entry,
        )
    )

    manager._inference_backend_manager = SimpleNamespace(
        get_backend_by_name=lambda _backend: SimpleNamespace(health_check_path="/health")
    )
    manager._get_model = lambda _mi: model.model_copy(update={"backend": BackendEnum.CUSTOM})
    manager._refresh_model = manager._get_model

    serve_manager_module.ServeManager._sync_direct_process_model_instance(
        manager, model_instance
    )

    assert model_instance.state == ModelInstanceStateEnum.ERROR
    assert model_instance.state_message == "Inference server did not become ready within 5 seconds."
    assert updates[-1]["state"] == ModelInstanceStateEnum.ERROR


def test_direct_process_stale_registry_cleanup_kills_orphaned_process_group(
    tmp_path: Path,
):
    model_instance = make_model_instance(port=18081, ports=[18081])
    assert model_instance.id is not None
    manager, _, _ = _build_manager(serve_manager_module, tmp_path, model_instance)
    bootstrap_manager = serve_manager_module.BootstrapManager(manager._config)
    manager._bootstrap_manager = bootstrap_manager
    runtime_roots = bootstrap_manager.prepare_runtime_roots(
        model_instance.model_id,
        model_instance.id,
    )
    sibling_runtime_roots = bootstrap_manager.prepare_runtime_roots(
        model_instance.model_id,
        model_instance.id + 1,
    )
    (runtime_roots.workspace / "stale.txt").write_text("stale", encoding="utf-8")
    (sibling_runtime_roots.workspace / "sibling.txt").write_text(
        "keep",
        encoding="utf-8",
    )
    child_pid_path = tmp_path / "stale-child.pid"
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPTS_DIR / "fake_process_tree_parent.py"),
            "--child-pid-file",
            str(child_pid_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    try:
        assert _wait_until(child_pid_path.exists)
        child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())
        assert _wait_until(lambda: psutil.pid_exists(child_pid))

        manager._direct_process_registry.upsert(
            model_instance_id=model_instance.id,
            deployment_name=model_instance.name,
            pid=process.pid,
            port=model_instance.port,
            log_path=str(Path(manager._serve_log_dir) / f"{model_instance.id}.log"),
            backend=BackendEnum.VLLM,
        )

        serve_manager_module.ServeManager.reconcile_stale_direct_process_registry(
            manager
        )

        assert _wait_until(lambda: process.poll() is not None, timeout=5)
        assert _wait_until(lambda: not psutil.pid_exists(child_pid), timeout=5)
        assert manager._direct_process_registry.list_entries() == []
        assert not runtime_roots.workspace.exists()
        assert not runtime_roots.artifacts.exists()
        assert not runtime_roots.manifests.exists()
        assert not runtime_roots.locks.exists()
        assert sibling_runtime_roots.workspace.exists()
        assert sibling_runtime_roots.artifacts.exists()
        assert sibling_runtime_roots.manifests.exists()
        assert sibling_runtime_roots.locks.exists()
    finally:
        if process.poll() is None:
            with contextlib.suppress(Exception):
                psutil.Process(process.pid).kill()
                process.wait(timeout=5)


# ---------------------------------------------------------------------------
# Characterization: cleanup-and-recreate restart policy is locked
# ---------------------------------------------------------------------------

def test_direct_process_start_removes_stale_registry_entry_before_launch(
    monkeypatch, tmp_path: Path
):
    """Characterization: _start_model_instance removes any existing registry entry before
    launching a new process (cleanup-and-recreate policy, not reattach)."""
    model_instance = make_model_instance()
    manager, _, updates = _build_manager(serve_manager_module, tmp_path, model_instance)
    process_factory = _ProcessFactory("ready")

    # Pre-seed a stale registry entry for the same model instance
    manager._direct_process_registry.upsert(
        model_instance_id=model_instance.id,
        deployment_name=model_instance.name,
        pid=99999,  # non-existent PID
        port=19999,
        log_path=str(Path(manager._serve_log_dir) / f"{model_instance.id}.log"),
        backend=BackendEnum.VLLM,
    )
    assert manager._direct_process_registry.get_by_model_instance_id(model_instance.id) is not None

    monkeypatch.setattr(serve_manager_module, "get_meta_from_running_instance", lambda *_args: None)
    monkeypatch.setattr(serve_manager_module.multiprocessing, "Process", process_factory)
    monkeypatch.setattr(serve_manager_module.platform, "system", lambda: "linux")

    try:
        serve_manager_module.ServeManager._start_model_instance(manager, model_instance)

        # After start, the stale entry (pid=99999) must be gone; a new entry is written by the provisioner
        assert _wait_until(lambda: not manager._is_provisioning(model_instance))
        # The registry entry written by the provisioner must have the real PID, not the stale one
        assert _wait_until(
            lambda: serve_manager_module.is_ready(
                BackendEnum.VLLM,
                model_instance,
                "/v1/models",
                make_model(),
            )
        )
        serve_manager_module.ServeManager.sync_model_instances_state(manager)

        entry = manager._direct_process_registry.get_by_model_instance_id(model_instance.id)
        assert entry is not None
        assert entry.pid != 99999, "stale PID must have been replaced by cleanup-and-recreate"
    finally:
        for process in process_factory.instances:
            process.cleanup()


def test_direct_process_stop_removes_registry_entry(monkeypatch, tmp_path: Path):
    """Characterization: _stop_model_instance removes the registry entry after killing the process."""
    model_instance = make_model_instance()
    assert model_instance.id is not None
    manager, _, _ = _build_manager(serve_manager_module, tmp_path, model_instance)
    bootstrap_manager = serve_manager_module.BootstrapManager(manager._config)
    manager._bootstrap_manager = bootstrap_manager
    process_factory = _ProcessFactory("ready")

    runtime_roots = bootstrap_manager.prepare_runtime_roots(
        model_instance.model_id,
        model_instance.id,
    )
    sibling_runtime_roots = bootstrap_manager.prepare_runtime_roots(
        model_instance.model_id,
        model_instance.id + 1,
    )
    (runtime_roots.workspace / "stop.txt").write_text("remove", encoding="utf-8")
    (sibling_runtime_roots.workspace / "sibling.txt").write_text(
        "keep",
        encoding="utf-8",
    )

    monkeypatch.setattr(serve_manager_module, "get_meta_from_running_instance", lambda *_args: None)
    monkeypatch.setattr(serve_manager_module.multiprocessing, "Process", process_factory)
    monkeypatch.setattr(serve_manager_module.platform, "system", lambda: "linux")

    try:
        serve_manager_module.ServeManager._start_model_instance(manager, model_instance)
        assert _wait_until(lambda: not manager._is_provisioning(model_instance))
        assert _wait_until(
            lambda: serve_manager_module.is_ready(
                BackendEnum.VLLM,
                model_instance,
                "/v1/models",
                make_model(),
            )
        )
        serve_manager_module.ServeManager.sync_model_instances_state(manager)

        # Registry entry must exist before stop
        assert manager._direct_process_registry.get_by_model_instance_id(model_instance.id) is not None

        serve_manager_module.ServeManager._stop_model_instance(manager, model_instance)

        # Registry entry must be gone after stop
        assert manager._direct_process_registry.get_by_model_instance_id(model_instance.id) is None
        assert not runtime_roots.workspace.exists()
        assert not runtime_roots.artifacts.exists()
        assert not runtime_roots.manifests.exists()
        assert not runtime_roots.locks.exists()
        assert sibling_runtime_roots.workspace.exists()
        assert sibling_runtime_roots.artifacts.exists()
        assert sibling_runtime_roots.manifests.exists()
        assert sibling_runtime_roots.locks.exists()
    finally:
        for process in process_factory.instances:
            process.cleanup()


def test_direct_process_log_path_is_serve_dir_model_instance_id(
    monkeypatch, tmp_path: Path
):
    """Characterization: serve log path is <serve_log_dir>/<model_instance_id>.log."""
    model_instance = make_model_instance()
    manager, _, _ = _build_manager(serve_manager_module, tmp_path, model_instance)
    process_factory = _ProcessFactory("ready")

    monkeypatch.setattr(serve_manager_module, "get_meta_from_running_instance", lambda *_args: None)
    monkeypatch.setattr(serve_manager_module.multiprocessing, "Process", process_factory)
    monkeypatch.setattr(serve_manager_module.platform, "system", lambda: "linux")

    try:
        serve_manager_module.ServeManager._start_model_instance(manager, model_instance)
        assert _wait_until(lambda: not manager._is_provisioning(model_instance))
        assert _wait_until(
            lambda: serve_manager_module.is_ready(
                BackendEnum.VLLM,
                model_instance,
                "/v1/models",
                make_model(),
            )
        )
        serve_manager_module.ServeManager.sync_model_instances_state(manager)

        entry = manager._direct_process_registry.get_by_model_instance_id(model_instance.id)
        assert entry is not None
        # Normalize separators for cross-platform comparison
        expected_log_path = str(Path(manager._serve_log_dir) / f"{model_instance.id}.log")
        assert Path(entry.log_path) == Path(expected_log_path)
    finally:
        for process in process_factory.instances:
            process.cleanup()
