import contextlib
import importlib
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


def _build_manager(serve_manager_module, tmp_path: Path, model_instance: ModelInstance):
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
    def __init__(self, scenario: str, args: tuple, child_pid_path: Path | None = None):
        self._scenario = scenario
        self._args = args
        self._child_pid_path = child_pid_path
        self._wrapper: subprocess.Popen[str] | None = None
        self._real_process: subprocess.Popen[str] | None = None
        self._log_handle = None
        self.pid: int | None = None
        self.real_pid: int | None = None

    def start(self):
        model_instance, backend, _, log_file_path, _, _, _, _, registry_path = self._args
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
    def __init__(self, scenario: str):
        self._scenario = scenario
        self.instances: list[_ProvisioningProcess] = []

    def __call__(self, target, args):
        del target
        process = _ProvisioningProcess(self._scenario, args)
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


def test_direct_process_stale_registry_cleanup_kills_orphaned_process_group(
    tmp_path: Path,
):
    model_instance = make_model_instance(port=18081, ports=[18081])
    manager, _, _ = _build_manager(serve_manager_module, tmp_path, model_instance)
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
    finally:
        if process.poll() is None:
            with contextlib.suppress(Exception):
                psutil.Process(process.pid).kill()
                process.wait(timeout=5)
