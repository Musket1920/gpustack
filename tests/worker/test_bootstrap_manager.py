from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import types

from gpustack.config.config import Config
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    SourceEnum,
)
import pytest


def bootstrap_manager_module_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "gpustack"
        / "worker"
        / "bootstrap_manager.py"
    )


def load_bootstrap_manager_module(module_name: str = "test_bootstrap_manager_module") -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, bootstrap_manager_module_path())
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BootstrapManagerModule = load_bootstrap_manager_module()
BootstrapManager = BootstrapManagerModule.BootstrapManager


def import_worker_module(module_name: str):
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


serve_manager_module = import_worker_module("gpustack.worker.serve_manager")
runtime_bootstrap_manager_module = import_worker_module("gpustack.worker.bootstrap_manager")
worker_manager_module = import_worker_module("gpustack.worker.worker_manager")


def make_config(tmp_path: Path, **overrides) -> Config:
    defaults = dict(
        token="test-token",
        jwt_secret_key="test-jwt-secret",
        data_dir=str(tmp_path),
        server_url="http://127.0.0.1:30080",
        direct_process_mode=True,
    )
    defaults.update(overrides)
    return Config(**defaults)


def config_path(path_value: str | None) -> Path:
    assert path_value is not None
    return Path(path_value)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_host_bootstrap_request(
    *,
    backend: str = "BackendEnum.VLLM",
    backend_version: str = "0.8.0",
    recipe_id: str = "vllm-host-bootstrap",
    recipe_source: str = "https://bootstrap.example/recipes/vllm.json",
    input_source: str = "https://bootstrap.example/files/vllm.tar.gz",
    input_sha256: str = "1" * 64,
    actions: tuple[dict[str, object], ...] | None = None,
):
    action_specs = actions or (
        {"name": "install-package", "details": {"manager": "apt", "package": "demo"}},
        {"name": "place-binary", "details": {"path": "/usr/local/bin/vllm"}},
    )
    return BootstrapManagerModule.HostBootstrapRequest(
        backend=backend,
        backend_version=backend_version,
        recipe_id=recipe_id,
        recipe_source=recipe_source,
        inputs=(
            BootstrapManagerModule.HostBootstrapInput(
                name="archive",
                source=input_source,
                sha256=input_sha256,
            ),
        ),
        actions=tuple(
            BootstrapManagerModule.HostBootstrapAction(
                name=str(action_spec["name"]),
                details=action_spec.get("details") if isinstance(action_spec, dict) else None,
            )
            for action_spec in action_specs
        ),
    )


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


class _ModelInstancesAPI:
    def __init__(self, items: list[ModelInstance]):
        self._items = items

    def list(self):
        return types.SimpleNamespace(items=self._items)


class _ClientSetStub:
    def __init__(self, items: list[ModelInstance]):
        self.headers = {}
        self.model_instances = _ModelInstancesAPI(items)


def _apply_model_instance_patch(model_instance: ModelInstance, patch: dict) -> None:
    for key, value in patch.items():
        if "." in key:
            continue
        setattr(model_instance, key, value)


class _BootstrapManagerStub:
    def __init__(self):
        self.calls: list[tuple[object, ...]] = []

    def prepare_prepared_cache_root(self, backend_name: str, backend_version: str):
        self.calls.append(("prepare_prepared_cache_root", backend_name, backend_version))

    def prepare_runtime_roots(self, deployment_id: int, model_instance_id: int):
        self.calls.append(("prepare_runtime_roots", deployment_id, model_instance_id))

    def cleanup_runtime_roots(self, deployment_id: int, model_instance_id: int):
        self.calls.append(("cleanup_runtime_roots", deployment_id, model_instance_id))


class _NoopProcess:
    def __init__(self, args: tuple, lifecycle_events: list[tuple[object, ...]]):
        self._args = args
        self._lifecycle_events = lifecycle_events
        self.pid = 4321
        self.daemon = False

    def start(self):
        model_instance, backend, *_ = self._args
        self._lifecycle_events.append(("process_start", model_instance.id, backend))


class _ProcessFactory:
    def __init__(self, lifecycle_events: list[tuple[object, ...]]):
        self.lifecycle_events = lifecycle_events
        self.instances: list[_NoopProcess] = []

    def __call__(self, target, args):
        process = _NoopProcess(args, self.lifecycle_events)
        self.instances.append(process)
        return process


def build_reporting_manager(
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
    manager._inference_backend_manager = types.SimpleNamespace(
        get_backend_by_name=lambda _backend: types.SimpleNamespace(
            health_check_path="/v1/models"
        )
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


def test_bootstrap_directory_roots_follow_config_contract(tmp_path: Path):
    cfg = make_config(tmp_path)

    assert config_path(cfg.data_dir) == tmp_path
    assert config_path(cfg.cache_dir) == tmp_path / "cache"
    assert config_path(cfg.bootstrap_dir) == tmp_path / "bootstrap"
    assert config_path(cfg.bootstrap_cache_dir) == tmp_path / "cache" / "bootstrap"
    assert config_path(cfg.bootstrap_workspace_dir) == tmp_path / "bootstrap" / "workspaces"
    assert config_path(cfg.bootstrap_artifacts_dir) == tmp_path / "bootstrap" / "artifacts"
    assert config_path(cfg.bootstrap_manifests_dir) == tmp_path / "bootstrap" / "manifests"
    assert config_path(cfg.bootstrap_locks_dir) == tmp_path / "bootstrap" / "locks"


def test_bootstrap_cache_key_root_stays_separate_from_runtime_roots(tmp_path: Path):
    cfg = make_config(tmp_path)

    cache_root = config_path(cfg.bootstrap_cache_dir)
    runtime_roots = {
        config_path(cfg.bootstrap_workspace_dir),
        config_path(cfg.bootstrap_artifacts_dir),
        config_path(cfg.bootstrap_manifests_dir),
        config_path(cfg.bootstrap_locks_dir),
    }

    assert cache_root.parent == config_path(cfg.cache_dir)
    assert cache_root != config_path(cfg.bootstrap_dir)
    assert all(root.parent == config_path(cfg.bootstrap_dir) for root in runtime_roots)
    assert cache_root not in runtime_roots


def test_bootstrap_workspace_key_and_other_runtime_roots_are_distinct(tmp_path: Path):
    cfg = make_config(tmp_path)

    roots = {
        config_path(cfg.bootstrap_cache_dir),
        config_path(cfg.bootstrap_workspace_dir),
        config_path(cfg.bootstrap_artifacts_dir),
        config_path(cfg.bootstrap_manifests_dir),
        config_path(cfg.bootstrap_locks_dir),
    }

    assert len(roots) == 5
    assert config_path(cfg.bootstrap_workspace_dir).name == "workspaces"
    assert config_path(cfg.bootstrap_artifacts_dir).name == "artifacts"
    assert config_path(cfg.bootstrap_manifests_dir).name == "manifests"
    assert config_path(cfg.bootstrap_locks_dir).name == "locks"


def test_prepare_prepared_cache_root_is_idempotent_and_localized(tmp_path: Path):
    cfg = make_config(tmp_path)
    manager = BootstrapManager(cfg)

    prepared_cache_root = manager.prepared_cache_root("vllm", "0.8.5")
    runtime_roots = manager.runtime_roots(11, 22)

    assert not prepared_cache_root.exists()
    assert not runtime_roots.workspace.exists()
    assert not runtime_roots.artifacts.exists()
    assert not runtime_roots.manifests.exists()
    assert not runtime_roots.locks.exists()

    first = manager.prepare_prepared_cache_root("vllm", "0.8.5")
    second = manager.prepare_prepared_cache_root("vllm", "0.8.5")

    assert first == prepared_cache_root
    assert second == prepared_cache_root
    assert first.exists()
    assert first.is_dir()
    assert first.parent == config_path(cfg.bootstrap_cache_dir) / "vllm"

    assert not runtime_roots.workspace.exists()
    assert not runtime_roots.artifacts.exists()
    assert not runtime_roots.manifests.exists()
    assert not runtime_roots.locks.exists()


def test_prepare_runtime_roots_is_idempotent_and_separate_from_cache_root(tmp_path: Path):
    cfg = make_config(tmp_path)
    manager = BootstrapManager(cfg)

    prepared_cache_root = manager.prepared_cache_root("vllm", "0.8.5")
    expected_roots = manager.runtime_roots(11, 22)

    assert not prepared_cache_root.exists()
    assert not expected_roots.workspace.exists()
    assert not expected_roots.artifacts.exists()
    assert not expected_roots.manifests.exists()
    assert not expected_roots.locks.exists()

    first = manager.prepare_runtime_roots(11, 22)
    second = manager.prepare_runtime_roots(11, 22)

    assert first == expected_roots
    assert second == expected_roots
    assert first.workspace.exists()
    assert first.artifacts.exists()
    assert first.manifests.exists()
    assert first.locks.exists()
    assert first.workspace.parent == config_path(cfg.bootstrap_workspace_dir) / "deployment-11"
    assert first.artifacts.parent == config_path(cfg.bootstrap_artifacts_dir) / "deployment-11"
    assert first.manifests.parent == config_path(cfg.bootstrap_manifests_dir) / "deployment-11"
    assert first.locks.parent == config_path(cfg.bootstrap_locks_dir) / "deployment-11"
    assert first.workspace != first.artifacts
    assert first.workspace != first.manifests
    assert first.workspace != first.locks
    assert first.artifacts != prepared_cache_root
    assert not prepared_cache_root.exists()


def test_prepare_prepared_cache_root_for_cleans_failed_materialization(tmp_path: Path):
    cfg = make_config(tmp_path)
    manager = BootstrapManager(cfg)

    prepared_cache_root = manager.prepared_cache_root("vllm", "0.8.5")
    runtime_roots = manager.runtime_roots(11, 22)

    def materialize(root: Path):
        (root / "weights.bin").write_text("prepared")
        raise RuntimeError("cache prepare failed")

    with pytest.raises(RuntimeError, match="cache prepare failed"):
        manager.prepare_prepared_cache_root_for("vllm", "0.8.5", materialize)

    assert not prepared_cache_root.exists()
    assert not prepared_cache_root.parent.exists()
    assert config_path(cfg.bootstrap_cache_dir).exists()
    assert not runtime_roots.workspace.exists()
    assert not runtime_roots.artifacts.exists()
    assert not runtime_roots.manifests.exists()
    assert not runtime_roots.locks.exists()


def test_prepare_runtime_roots_for_cleans_failed_materialization(tmp_path: Path):
    cfg = make_config(tmp_path)
    manager = BootstrapManager(cfg)

    prepared_cache_root = manager.prepared_cache_root("vllm", "0.8.5")
    runtime_roots = manager.runtime_roots(11, 22)

    def materialize(roots):
        (roots.workspace / "launch.sh").write_text("run")
        (roots.artifacts / "artifact.txt").write_text("artifact")
        (roots.manifests / "manifest.json").write_text("{}")
        (roots.locks / "bootstrap.lock").write_text("locked")
        raise RuntimeError("runtime prepare failed")

    with pytest.raises(RuntimeError, match="runtime prepare failed"):
        manager.prepare_runtime_roots_for(11, 22, materialize)

    assert not runtime_roots.workspace.exists()
    assert not runtime_roots.artifacts.exists()
    assert not runtime_roots.manifests.exists()
    assert not runtime_roots.locks.exists()
    assert not runtime_roots.workspace.parent.exists()
    assert not runtime_roots.artifacts.parent.exists()
    assert not runtime_roots.manifests.parent.exists()
    assert not runtime_roots.locks.parent.exists()
    assert config_path(cfg.bootstrap_workspace_dir).exists()
    assert config_path(cfg.bootstrap_artifacts_dir).exists()
    assert config_path(cfg.bootstrap_manifests_dir).exists()
    assert config_path(cfg.bootstrap_locks_dir).exists()
    assert not prepared_cache_root.exists()


def test_prepare_prepared_cache_root_reuses_same_root_across_concurrent_calls(tmp_path: Path):
    cfg = make_config(tmp_path)
    manager = BootstrapManager(cfg)

    expected_root = manager.prepared_cache_root("vllm", "0.8.5")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: manager.prepare_prepared_cache_root("vllm", "0.8.5"),
                range(16),
            )
        )

    assert all(result == expected_root for result in results)
    assert expected_root.exists()
    assert expected_root.is_dir()


def test_prepare_prepared_cache_root_for_hides_partial_state_during_concurrency_materialization(
    tmp_path: Path,
):
    cfg = make_config(tmp_path)
    manager = BootstrapManager(cfg)
    expected_root = manager.prepared_cache_root("vllm", "0.8.5")
    started = threading.Event()
    release = threading.Event()
    calls: list[Path] = []

    def materialize(root: Path) -> Path:
        calls.append(root)
        (root / "artifact.txt").parent.mkdir(parents=True, exist_ok=True)
        (root / "artifact.txt").write_text("prepared", encoding="utf-8")
        started.set()
        release.wait(timeout=5)
        return root

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            manager.prepare_prepared_cache_root_for,
            "vllm",
            "0.8.5",
            materialize,
        )
        assert started.wait(timeout=5)
        assert not expected_root.exists()
        second = executor.submit(
            manager.prepare_prepared_cache_root_for,
            "vllm",
            "0.8.5",
            materialize,
        )
        time.sleep(0.05)
        assert not second.done()
        assert not expected_root.exists()
        release.set()
        first_result = first.result(timeout=5)
        second_result = second.result(timeout=5)

    assert first_result == expected_root
    assert second_result == expected_root
    assert calls and len(calls) == 1
    assert expected_root.exists()
    assert (expected_root / "artifact.txt").read_text(encoding="utf-8") == "prepared"


def test_prepare_runtime_roots_reuses_same_roots_across_concurrent_calls(tmp_path: Path):
    cfg = make_config(tmp_path)
    manager = BootstrapManager(cfg)

    expected_roots = manager.runtime_roots(11, 22)
    prepared_cache_root = manager.prepared_cache_root("vllm", "0.8.5")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: manager.prepare_runtime_roots(11, 22), range(16)))

    assert all(result == expected_roots for result in results)
    assert expected_roots.workspace.exists()
    assert expected_roots.artifacts.exists()
    assert expected_roots.manifests.exists()
    assert expected_roots.locks.exists()
    assert not prepared_cache_root.exists()


def test_cleanup_prepared_cache_root_is_scoped_and_idempotent(tmp_path: Path):
    cfg = make_config(tmp_path)
    manager = BootstrapManager(cfg)

    prepared_cache_root = manager.prepare_prepared_cache_root("vllm", "0.8.5")
    sibling_cache_root = manager.prepare_prepared_cache_root("sglang", "0.4.0")
    sentinel_file = prepared_cache_root / "weights.bin"
    sentinel_file.write_text("prepared")

    assert prepared_cache_root.exists()
    assert sibling_cache_root.exists()

    first = manager.cleanup_prepared_cache_root("vllm", "0.8.5")
    second = manager.cleanup_prepared_cache_root("vllm", "0.8.5")

    assert first == prepared_cache_root
    assert second == prepared_cache_root
    assert not prepared_cache_root.exists()
    assert not prepared_cache_root.parent.exists()
    assert sibling_cache_root.exists()
    assert config_path(cfg.bootstrap_cache_dir).exists()


def test_cleanup_runtime_roots_is_scoped_and_idempotent(tmp_path: Path):
    cfg = make_config(tmp_path)
    manager = BootstrapManager(cfg)

    runtime_roots = manager.prepare_runtime_roots(11, 22)
    sibling_runtime_roots = manager.prepare_runtime_roots(11, 23)
    (runtime_roots.workspace / "launch.sh").write_text("run")
    (runtime_roots.artifacts / "artifact.txt").write_text("artifact")
    (runtime_roots.manifests / "manifest.json").write_text("{}")
    (runtime_roots.locks / "bootstrap.lock").write_text("locked")

    first = manager.cleanup_runtime_roots(11, 22)
    second = manager.cleanup_runtime_roots(11, 22)

    assert first == runtime_roots
    assert second == runtime_roots
    assert not runtime_roots.workspace.exists()
    assert not runtime_roots.artifacts.exists()
    assert not runtime_roots.manifests.exists()
    assert not runtime_roots.locks.exists()
    assert sibling_runtime_roots.workspace.exists()
    assert sibling_runtime_roots.artifacts.exists()
    assert sibling_runtime_roots.manifests.exists()
    assert sibling_runtime_roots.locks.exists()
    assert config_path(cfg.bootstrap_workspace_dir).exists()
    assert config_path(cfg.bootstrap_artifacts_dir).exists()
    assert config_path(cfg.bootstrap_manifests_dir).exists()
    assert config_path(cfg.bootstrap_locks_dir).exists()


def test_bootstrap_manager_module_has_no_direct_process_registry_contract(monkeypatch):
    requested_imports: list[str] = []
    real_import = __import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        requested_imports.append(name)
        assert name != "gpustack.worker.process_registry"
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", tracking_import)

    module = load_bootstrap_manager_module("test_bootstrap_manager_contract_module")
    module_source = bootstrap_manager_module_path().read_text(encoding="utf-8")

    assert "DirectProcessRegistry" not in module_source
    assert "process_registry" not in module_source
    assert not hasattr(module, "DirectProcessRegistry")
    assert "gpustack.worker.process_registry" not in requested_imports


@pytest.mark.parametrize(
    ("prepared_cache_state", "expected_launch_message"),
    [
        (
            "initialized",
            "Prepared direct-process bootstrap environment; executable provenance: resolved; launching inference server (waiting for readiness).",
        ),
        (
            "reused",
            "Reusing direct-process bootstrap environment; executable provenance: resolved; launching inference server (waiting for readiness).",
        ),
        (
            "repaired",
            "Repaired direct-process bootstrap environment; executable provenance: resolved; launching inference server (waiting for readiness).",
        ),
    ],
)
def test_direct_process_bootstrap_reporting_distinguishes_preparing_from_launch_state(
    monkeypatch,
    tmp_path: Path,
    prepared_cache_state: str,
    expected_launch_message: str,
):
    model_instance = make_model_instance()
    bootstrap_manager = _BootstrapManagerStub()
    lifecycle_events: list[tuple[object, ...]] = []
    manager, _, updates = build_reporting_manager(
        tmp_path,
        model_instance,
        bootstrap_manager=bootstrap_manager,
    )

    monkeypatch.setattr(
        serve_manager_module,
        "ensure_model_instance_direct_process_support",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        serve_manager_module.multiprocessing,
        "Process",
        _ProcessFactory(lifecycle_events),
    )
    monkeypatch.setattr(
        manager,
        "_prepare_direct_process_bootstrap",
        lambda *_args, **_kwargs: serve_manager_module.DirectProcessBootstrapReport(
            prepared_cache_state=prepared_cache_state,
            executable_provenance={"state": "resolved"},
        ),
    )
    monkeypatch.setattr(
        manager,
        "_assign_ports",
        lambda mi, *_args, **_kwargs: setattr(mi, "port", 8000),
    )

    serve_manager_module.ServeManager._start_model_instance(manager, model_instance)

    assert updates[0]["state"] == ModelInstanceStateEnum.INITIALIZING
    assert updates[0]["state_message"] == (
        "Preparing direct-process bootstrap environment before launch."
    )
    assert updates[0]["pid"] is None
    assert updates[1]["state"] == ModelInstanceStateEnum.INITIALIZING
    assert updates[1]["state_message"] == expected_launch_message
    assert updates[1]["pid"] is None
    assert model_instance.state == ModelInstanceStateEnum.INITIALIZING
    assert model_instance.state != ModelInstanceStateEnum.RUNNING
    assert bootstrap_manager.calls == [("cleanup_runtime_roots", 1, 1)]
    assert lifecycle_events == [("process_start", model_instance.id, BackendEnum.VLLM)]


def test_direct_process_bootstrap_failure_reports_preparation_error_without_running_state(
    monkeypatch,
    tmp_path: Path,
):
    model_instance = make_model_instance()
    bootstrap_manager = _BootstrapManagerStub()
    manager, _, updates = build_reporting_manager(
        tmp_path,
        model_instance,
        bootstrap_manager=bootstrap_manager,
    )
    process_factory = _ProcessFactory([])

    monkeypatch.setattr(
        serve_manager_module,
        "ensure_model_instance_direct_process_support",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        serve_manager_module.multiprocessing,
        "Process",
        process_factory,
    )

    def fail_prepare(*_args, **_kwargs):
        raise RuntimeError("bootstrap cache missing")

    monkeypatch.setattr(manager, "_prepare_direct_process_bootstrap", fail_prepare)

    serve_manager_module.ServeManager._start_model_instance(manager, model_instance)

    assert updates[0]["state_message"] == (
        "Preparing direct-process bootstrap environment before launch."
    )
    assert updates[-1]["state"] == ModelInstanceStateEnum.ERROR
    assert updates[-1]["state_message"] == (
        "Failed to prepare direct-process bootstrap environment: bootstrap cache missing"
    )
    assert model_instance.state == ModelInstanceStateEnum.ERROR
    assert model_instance.state != ModelInstanceStateEnum.RUNNING
    assert model_instance.pid is None
    assert process_factory.instances == []
    assert bootstrap_manager.calls == [
        ("cleanup_runtime_roots", 1, 1),
        ("cleanup_runtime_roots", 1, 1),
    ]


def test_direct_process_bootstrap_identity_reuse_records_canonical_manifest(tmp_path: Path):
    model_instance = make_model_instance()
    manager, model, _ = build_reporting_manager(tmp_path, model_instance)

    reused = serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
        manager,
        model_instance,
        model,
        BackendEnum.VLLM,
    )

    assert reused.prepared_cache_state == "initialized"

    reused = serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
        manager,
        model_instance,
        model,
        BackendEnum.VLLM,
    )

    assert reused.prepared_cache_state == "reused"

    bootstrap_manager = manager._get_bootstrap_manager()
    prepared_context = json.loads(
        bootstrap_manager.prepared_cache_context_path("BackendEnum.VLLM", "0.8.0").read_text(
            encoding="utf-8"
        )
    )
    runtime_manifest = json.loads(
        bootstrap_manager.manifest_path(
            1,
            1,
            bootstrap_manager.BOOTSTRAP_MANIFEST_FILENAME,
        ).read_text(encoding="utf-8")
    )

    assert prepared_context["recipe_id"] == "vllm"
    assert prepared_context["prepared_environment_id"] == "vllm:0.8.0"
    assert prepared_context["backend_version"] == "0.8.0"
    assert prepared_context["resolver_version"] == bootstrap_manager.PREPARED_CACHE_RESOLVER_VERSION
    assert prepared_context["invalidation"] == {"state": "valid", "reason": None}
    assert prepared_context["python_identity"]["version"]
    assert prepared_context["manifest_hash"] == runtime_manifest["manifest_hash"]
    assert runtime_manifest["prepared_cache_state"] == "reused"
    assert runtime_manifest["launch_provenance"]["launch_state"] == "launching"
    assert runtime_manifest["launch_provenance"]["ready"] is False
    assert runtime_manifest["launch_provenance"]["executable_provenance"]["state"] == "not-declared"
    assert runtime_manifest["launch_provenance"]["provisioning_audit_artifact"].endswith(
        bootstrap_manager.PREPARED_CACHE_PROVISIONING_FILENAME
    )


def test_direct_process_bootstrap_manifest_mismatch_rejects_and_marks_invalid(
    tmp_path: Path,
):
    model_instance = make_model_instance()
    manager, model, _ = build_reporting_manager(tmp_path, model_instance)

    serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
        manager,
        model_instance,
        model,
        BackendEnum.VLLM,
    )

    bootstrap_manager = manager._get_bootstrap_manager()
    prepared_context_path = bootstrap_manager.prepared_cache_context_path(
        "BackendEnum.VLLM",
        "0.8.0",
    )
    prepared_context = json.loads(prepared_context_path.read_text(encoding="utf-8"))
    prepared_context["manifest_hash"] = "stale-hash"
    prepared_context_path.write_text(
        json.dumps(prepared_context, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
            manager,
            model_instance,
            model,
            BackendEnum.VLLM,
        )

    invalidated_context = json.loads(prepared_context_path.read_text(encoding="utf-8"))
    assert invalidated_context["invalidation"]["state"] == "invalid"
    assert "manifest hash mismatch" in invalidated_context["invalidation"]["reason"]


@pytest.mark.parametrize(
    "prepared_context_content",
    [
        pytest.param("{not-json", id="corrupted_manifest"),
        pytest.param(None, id="interrupted_prepare"),
    ],
)
def test_direct_process_bootstrap_invalidation_rejects_corrupted_or_interrupted_cache(
    tmp_path: Path,
    prepared_context_content: str | None,
):
    model_instance = make_model_instance()
    manager, model, _ = build_reporting_manager(tmp_path, model_instance)
    bootstrap_manager = manager._get_bootstrap_manager()

    prepared_cache_root = bootstrap_manager.prepare_prepared_cache_root(
        "BackendEnum.VLLM",
        "0.8.0",
    )
    prepared_context_path = bootstrap_manager.prepared_cache_context_path(
        "BackendEnum.VLLM",
        "0.8.0",
    )
    if prepared_context_content is not None:
        prepared_context_path.write_text(prepared_context_content, encoding="utf-8")
    else:
        assert prepared_cache_root.exists()
        assert not prepared_context_path.exists()

    with pytest.raises(RuntimeError, match="Prepared direct-process bootstrap cache is invalid"):
        serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
            manager,
            model_instance,
            model,
            BackendEnum.VLLM,
        )

    invalidated_context = json.loads(prepared_context_path.read_text(encoding="utf-8"))
    assert invalidated_context["invalidation"]["state"] == "invalid"
    assert invalidated_context["recipe_id"] == "vllm"
    assert invalidated_context["prepared_environment_id"] == "vllm:0.8.0"


def test_direct_process_bootstrap_materializes_venv_cache_and_generated_artifacts(
    monkeypatch,
    tmp_path: Path,
):
    model_instance = make_model_instance()
    manager, model, _ = build_reporting_manager(tmp_path, model_instance)
    executable_source = tmp_path / "fake-vllm"
    executable_source.write_text("#!/usr/bin/env python\nprint('ok')\n", encoding="utf-8")
    executable_hash = sha256_path(executable_source)
    install_calls: list[list[str]] = []

    monkeypatch.setenv(
        "GPUSTACK_BOOTSTRAP_PYTHON_DEPS",
        json.dumps(
            [
                {
                    "requirement": "demo-package==1.0.0",
                    "hashes": [f"sha256:{'1' * 64}"],
                }
            ]
        ),
    )
    monkeypatch.setenv(
        "GPUSTACK_BOOTSTRAP_EXECUTABLE",
        json.dumps(
            {
                "name": "vllm",
                "path": str(executable_source),
                "sha256": f"sha256:{executable_hash}",
            }
        ),
    )

    def fake_run(command, **kwargs):
        if "ensurepip" in command:
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        install_calls.append(command)
        assert "--require-hashes" in command
        return types.SimpleNamespace(returncode=0, stdout="installed", stderr="")

    monkeypatch.setattr(runtime_bootstrap_manager_module.subprocess, "run", fake_run)

    reused = serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
        manager,
        model_instance,
        model,
        BackendEnum.VLLM,
    )

    assert reused.prepared_cache_state == "initialized"
    assert reused.executable_provenance["state"] == "resolved"
    bootstrap_manager = manager._get_bootstrap_manager()
    prepared_root = bootstrap_manager.prepared_cache_root("BackendEnum.VLLM", "0.8.0")
    artifacts_root = bootstrap_manager.prepared_cache_artifacts_root(
        "BackendEnum.VLLM",
        "0.8.0",
    )
    requirements_lock = artifacts_root / bootstrap_manager.PREPARED_CACHE_REQUIREMENTS_FILENAME
    env_artifact = artifacts_root / bootstrap_manager.PREPARED_CACHE_ENV_FILENAME
    config_artifact = artifacts_root / bootstrap_manager.PREPARED_CACHE_CONFIG_FILENAME
    launch_artifact = artifacts_root / bootstrap_manager.PREPARED_CACHE_LAUNCH_FILENAME
    provenance_artifact = (
        artifacts_root
        / bootstrap_manager.PREPARED_CACHE_EXECUTABLE_PROVENANCE_FILENAME
    )
    provisioning_artifact = (
        artifacts_root / bootstrap_manager.PREPARED_CACHE_PROVISIONING_FILENAME
    )

    assert prepared_root.exists()
    assert bootstrap_manager.prepared_cache_venv_root("BackendEnum.VLLM", "0.8.0").exists()
    assert requirements_lock.exists()
    assert env_artifact.exists()
    assert config_artifact.exists()
    assert launch_artifact.exists()
    assert provenance_artifact.exists()
    assert provisioning_artifact.exists()
    assert install_calls

    executable_provenance = json.loads(provenance_artifact.read_text(encoding="utf-8"))
    provisioning_record = json.loads(provisioning_artifact.read_text(encoding="utf-8"))
    assert executable_provenance["state"] == "resolved"
    assert executable_provenance["sha256"] == f"sha256:{executable_hash}"
    assert Path(executable_provenance["prepared_path"]).exists()
    assert provisioning_record["artifacts"]["launch"] == str(launch_artifact)
    assert "demo-package==1.0.0" in requirements_lock.read_text(encoding="utf-8")

    install_calls.clear()
    reused = serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
        manager,
        model_instance,
        model,
        BackendEnum.VLLM,
    )
    assert reused.prepared_cache_state == "reused"
    assert reused.executable_provenance["state"] == "resolved"
    assert install_calls == []


def test_direct_process_bootstrap_materializes_consumed_artifact_metadata(
    monkeypatch,
    tmp_path: Path,
):
    model_instance = make_model_instance()
    manager, model, _ = build_reporting_manager(tmp_path, model_instance)
    monkeypatch.delenv("GPUSTACK_BOOTSTRAP_PYTHON_DEPS", raising=False)
    monkeypatch.delenv("GPUSTACK_BOOTSTRAP_EXECUTABLE", raising=False)

    reused = serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
        manager,
        model_instance,
        model,
        BackendEnum.VLLM,
    )

    assert reused.prepared_cache_state == "initialized"
    assert reused.executable_provenance["state"] == "not-declared"
    bootstrap_manager = manager._get_bootstrap_manager()
    prepared_config = json.loads(
        bootstrap_manager.prepared_cache_artifact_path(
            "BackendEnum.VLLM",
            "0.8.0",
            bootstrap_manager.PREPARED_CACHE_CONFIG_FILENAME,
        ).read_text(encoding="utf-8")
    )
    runtime_artifact = json.loads(
        bootstrap_manager.artifact_path(
            1,
            1,
            bootstrap_manager.BOOTSTRAP_ARTIFACT_FILENAME,
        ).read_text(encoding="utf-8")
    )
    assert prepared_config["env_artifact"].endswith(
        bootstrap_manager.PREPARED_CACHE_ENV_FILENAME
    )
    assert runtime_artifact["prepared_environment_id"] == "vllm:0.8.0"


def test_direct_process_artifact_consumption_resolves_runtime_and_prepared_launch_artifacts(
    tmp_path: Path,
):
    model_instance = make_model_instance()
    manager, model, _ = build_reporting_manager(tmp_path, model_instance)

    serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
        manager,
        model_instance,
        model,
        BackendEnum.VLLM,
    )

    bootstrap_manager = manager._get_bootstrap_manager()
    artifacts = bootstrap_manager.resolve_direct_process_launch_artifacts(
        deployment_id=model_instance.model_id,
        model_instance_id=model_instance.id,
        backend_name=str(BackendEnum.VLLM),
        backend_version=model.backend_version,
        recipe_id="vllm",
        prepared_environment_id="vllm:0.8.0",
    )

    assert artifacts.runtime_artifact["prepared_artifacts"]["launch"] == str(
        artifacts.prepared_launch_path
    )
    assert artifacts.runtime_artifact["prepared_artifacts"]["env"] == str(
        artifacts.prepared_env_path
    )
    assert artifacts.runtime_artifact["prepared_artifacts"]["config"] == str(
        artifacts.prepared_config_path
    )
    assert artifacts.runtime_artifact["prepared_artifacts"]["executable_provenance"] == str(
        artifacts.prepared_provenance_path
    )
    assert artifacts.prepared_config["env_artifact"] == str(artifacts.prepared_env_path)
    assert artifacts.prepared_config["executable_provenance"] == str(
        artifacts.prepared_provenance_path
    )
    assert artifacts.manifest_hash == artifacts.runtime_artifact["manifest_hash"]
    assert artifacts.manifest_hash == artifacts.prepared_context["manifest_hash"]


def test_direct_process_artifact_consumption_rejects_mismatched_runtime_identity(
    tmp_path: Path,
):
    model_instance = make_model_instance()
    manager, model, _ = build_reporting_manager(tmp_path, model_instance)

    serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
        manager,
        model_instance,
        model,
        BackendEnum.VLLM,
    )

    bootstrap_manager = manager._get_bootstrap_manager()
    runtime_artifact_path = bootstrap_manager.artifact_path(
        model_instance.model_id,
        model_instance.id,
        bootstrap_manager.BOOTSTRAP_ARTIFACT_FILENAME,
    )
    runtime_artifact = json.loads(runtime_artifact_path.read_text(encoding="utf-8"))
    runtime_artifact["prepared_environment_id"] = "vllm:stale"
    runtime_artifact_path.write_text(
        json.dumps(runtime_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="runtime bootstrap artifact prepared_environment_id mismatch",
    ):
        bootstrap_manager.resolve_direct_process_launch_artifacts(
            deployment_id=model_instance.model_id,
            model_instance_id=model_instance.id,
            backend_name=str(BackendEnum.VLLM),
            backend_version=model.backend_version,
            recipe_id="vllm",
            prepared_environment_id="vllm:0.8.0",
        )


def test_direct_process_bootstrap_rejects_unpinned_dependency_input(
    monkeypatch,
    tmp_path: Path,
):
    model_instance = make_model_instance()
    manager, model, _ = build_reporting_manager(tmp_path, model_instance)
    monkeypatch.setenv(
        "GPUSTACK_BOOTSTRAP_PYTHON_DEPS",
        json.dumps([{"requirement": "unsafe-package>=1.0", "hashes": []}]),
    )
    monkeypatch.delenv("GPUSTACK_BOOTSTRAP_EXECUTABLE", raising=False)

    with pytest.raises(RuntimeError, match="exact == pin|hash-pinned"):
        serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
            manager,
            model_instance,
            model,
            BackendEnum.VLLM,
        )

    bootstrap_manager = manager._get_bootstrap_manager()
    assert not bootstrap_manager.prepared_cache_root("BackendEnum.VLLM", "0.8.0").exists()


def test_direct_process_bootstrap_rejects_hash_mismatched_executable_input(
    monkeypatch,
    tmp_path: Path,
):
    model_instance = make_model_instance()
    manager, model, _ = build_reporting_manager(tmp_path, model_instance)
    executable_source = tmp_path / "fake-vllm"
    executable_source.write_text("#!/usr/bin/env python\nprint('ok')\n", encoding="utf-8")

    monkeypatch.delenv("GPUSTACK_BOOTSTRAP_PYTHON_DEPS", raising=False)
    monkeypatch.setenv(
        "GPUSTACK_BOOTSTRAP_EXECUTABLE",
        json.dumps(
            {
                "name": "vllm",
                "path": str(executable_source),
                "sha256": f"sha256:{'0' * 64}",
            }
        ),
    )

    with pytest.raises(RuntimeError, match="hash mismatch"):
        serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
            manager,
            model_instance,
            model,
            BackendEnum.VLLM,
        )

    bootstrap_manager = manager._get_bootstrap_manager()
    assert not bootstrap_manager.prepared_cache_root("BackendEnum.VLLM", "0.8.0").exists()


def test_direct_process_bootstrap_repairs_missing_prepared_artifact_and_records_audit(
    monkeypatch,
    tmp_path: Path,
):
    model_instance = make_model_instance()
    manager, model, _ = build_reporting_manager(tmp_path, model_instance)
    monkeypatch.delenv("GPUSTACK_BOOTSTRAP_PYTHON_DEPS", raising=False)
    monkeypatch.delenv("GPUSTACK_BOOTSTRAP_EXECUTABLE", raising=False)

    initial_report = serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
        manager,
        model_instance,
        model,
        BackendEnum.VLLM,
    )
    assert initial_report.prepared_cache_state == "initialized"

    bootstrap_manager = manager._get_bootstrap_manager()
    env_artifact = bootstrap_manager.prepared_cache_artifact_path(
        "BackendEnum.VLLM",
        "0.8.0",
        bootstrap_manager.PREPARED_CACHE_ENV_FILENAME,
    )
    env_artifact.unlink()

    repaired = serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
        manager,
        model_instance,
        model,
        BackendEnum.VLLM,
    )

    assert repaired.prepared_cache_state == "repaired"
    assert repaired.repair_reason == "prepared env artifact missing"
    provisioning_record = json.loads(
        bootstrap_manager.prepared_cache_artifact_path(
            "BackendEnum.VLLM",
            "0.8.0",
            bootstrap_manager.PREPARED_CACHE_PROVISIONING_FILENAME,
        ).read_text(encoding="utf-8")
    )
    audit_events = provisioning_record["audit_events"]
    assert [event["operation"] for event in audit_events[-2:]] == ["repair", "repair"]
    assert audit_events[-2]["outcome"] == "requested"
    assert audit_events[-2]["reason"] == "prepared env artifact missing"
    assert audit_events[-1]["outcome"] == "repaired"
    runtime_manifest = json.loads(
        bootstrap_manager.manifest_path(
            1,
            1,
            bootstrap_manager.BOOTSTRAP_MANIFEST_FILENAME,
        ).read_text(encoding="utf-8")
    )
    assert runtime_manifest["prepared_cache_state"] == "repaired"
    assert runtime_manifest["launch_provenance"]["repair_reason"] == (
        "prepared env artifact missing"
    )
    assert env_artifact.exists()


def test_direct_process_bootstrap_cleanup_records_audit_outcome(tmp_path: Path):
    model_instance = make_model_instance()
    manager, model, _ = build_reporting_manager(tmp_path, model_instance)

    serve_manager_module.ServeManager._prepare_direct_process_bootstrap(
        manager,
        model_instance,
        model,
        BackendEnum.VLLM,
    )

    bootstrap_manager = manager._get_bootstrap_manager()
    serve_manager_module.ServeManager._cleanup_direct_process_bootstrap(
        manager,
        mi=model_instance,
    )

    provisioning_record = json.loads(
        bootstrap_manager.prepared_cache_artifact_path(
            "BackendEnum.VLLM",
            "0.8.0",
            bootstrap_manager.PREPARED_CACHE_PROVISIONING_FILENAME,
        ).read_text(encoding="utf-8")
    )
    assert provisioning_record["audit_events"][-1]["operation"] == "cleanup"
    assert provisioning_record["audit_events"][-1]["outcome"] == "runtime-roots-removed"
    runtime_roots = bootstrap_manager.runtime_roots(1, 1)
    assert not runtime_roots.workspace.exists()
    assert not runtime_roots.artifacts.exists()
    assert not runtime_roots.manifests.exists()
    assert not runtime_roots.locks.exists()


def test_host_bootstrap_disabled_by_default_and_cannot_be_triggered(tmp_path: Path):
    cfg = make_config(
        tmp_path,
        host_bootstrap_allowed_recipe_sources=["https://bootstrap.example/recipes"],
    )
    manager = BootstrapManager(cfg)
    request = make_host_bootstrap_request()

    with pytest.raises(RuntimeError, match="disabled on this worker"):
        manager.execute_host_bootstrap(request)

    audit_payload = json.loads(
        manager.host_bootstrap_audit_path(
            backend_name=request.backend,
            backend_version=request.backend_version,
            recipe_id=request.recipe_id,
        ).read_text(encoding="utf-8")
    )
    assert audit_payload["mutation_performed"] is False
    assert audit_payload["audit_events"][-1]["outcome"] == "rejected"


def test_worker_manager_host_bootstrap_control_path_delegates_to_guarded_executor(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(BootstrapManagerModule.sys, "platform", "linux")
    monkeypatch.setattr(worker_manager_module.sys, "platform", "linux")
    cfg = make_config(
        tmp_path,
        enable_host_bootstrap=False,
        host_bootstrap_allowed_recipe_sources=["https://bootstrap.example/recipes"],
    )
    manager = object.__new__(worker_manager_module.WorkerManager)
    manager._cfg = cfg
    manager._bootstrap_manager = None
    request = make_host_bootstrap_request()

    with pytest.raises(RuntimeError, match="disabled on this worker"):
        manager.execute_host_bootstrap(request)

    audit_payload = json.loads(
        BootstrapManager(cfg)
        .host_bootstrap_audit_path(
            backend_name=request.backend,
            backend_version=request.backend_version,
            recipe_id=request.recipe_id,
        )
        .read_text(encoding="utf-8")
    )
    assert audit_payload["mutation_performed"] is False
    assert audit_payload["audit_events"][-1]["outcome"] == "rejected"


def test_host_bootstrap_dry_run_reports_actions_without_mutation(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(BootstrapManagerModule.sys, "platform", "linux")
    cfg = make_config(
        tmp_path,
        enable_host_bootstrap=True,
        host_bootstrap_dry_run=True,
        host_bootstrap_allowed_recipe_sources=["https://bootstrap.example/recipes"],
    )
    manager = BootstrapManager(cfg)
    request = make_host_bootstrap_request()
    mutation_calls: list[str] = []

    result = manager.execute_host_bootstrap(
        request,
        mutate_action=lambda action: mutation_calls.append(action.name),
    )

    assert result.dry_run is True
    assert result.mutation_performed is False
    assert mutation_calls == []
    assert [action["name"] for action in result.actions] == [
        "install-package",
        "place-binary",
    ]
    audit_payload = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit_payload["dry_run"] is True
    assert audit_payload["mutation_performed"] is False
    assert audit_payload["audit_events"][-1]["outcome"] == "dry-run"


def test_host_bootstrap_allowlist_rejects_untrusted_recipe_source(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(BootstrapManagerModule.sys, "platform", "linux")
    cfg = make_config(
        tmp_path,
        enable_host_bootstrap=True,
        host_bootstrap_allowed_recipe_sources=["https://trusted.example/recipes"],
    )
    manager = BootstrapManager(cfg)
    request = make_host_bootstrap_request(recipe_source="https://bootstrap.example/recipes/vllm.json")

    with pytest.raises(RuntimeError, match="not allowlisted"):
        manager.execute_host_bootstrap(request)

    audit_payload = json.loads(
        manager.host_bootstrap_audit_path(
            backend_name=request.backend,
            backend_version=request.backend_version,
            recipe_id=request.recipe_id,
        ).read_text(encoding="utf-8")
    )
    assert audit_payload["audit_events"][-1]["outcome"] == "rejected"
    assert "trusted.example/recipes" in audit_payload["audit_events"][-1]["details"]["allowlist"][0]


def test_host_bootstrap_hash_pinned_inputs_reject_hashless_input(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(BootstrapManagerModule.sys, "platform", "linux")
    cfg = make_config(
        tmp_path,
        enable_host_bootstrap=True,
        host_bootstrap_allowed_recipe_sources=["https://bootstrap.example/recipes"],
    )
    manager = BootstrapManager(cfg)
    request = make_host_bootstrap_request(input_sha256="")

    with pytest.raises(RuntimeError, match="Expected sha256 hash"):
        manager.execute_host_bootstrap(request)

    audit_payload = json.loads(
        manager.host_bootstrap_audit_path(
            backend_name=request.backend,
            backend_version=request.backend_version,
            recipe_id=request.recipe_id,
        ).read_text(encoding="utf-8")
    )
    assert audit_payload["audit_events"][-1]["outcome"] == "rejected"


def test_host_bootstrap_platform_guardrails_reject_non_linux_workers(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(BootstrapManagerModule.sys, "platform", "darwin")
    cfg = make_config(
        tmp_path,
        enable_host_bootstrap=True,
        host_bootstrap_allowed_recipe_sources=["https://bootstrap.example/recipes"],
    )
    manager = BootstrapManager(cfg)
    request = make_host_bootstrap_request()

    with pytest.raises(RuntimeError, match="only supported on Linux workers"):
        manager.execute_host_bootstrap(request)


def test_host_bootstrap_audit_records_completed_mutation_execution(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(BootstrapManagerModule.sys, "platform", "linux")
    cfg = make_config(
        tmp_path,
        enable_host_bootstrap=True,
        host_bootstrap_allowed_recipe_sources=["https://bootstrap.example/recipes"],
    )
    manager = BootstrapManager(cfg)
    request = make_host_bootstrap_request()
    mutation_calls: list[str] = []

    result = manager.execute_host_bootstrap(
        request,
        mutate_action=lambda action: mutation_calls.append(action.name),
    )

    assert result.dry_run is False
    assert result.mutation_performed is True
    assert mutation_calls == ["install-package", "place-binary"]
    audit_payload = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit_payload["mutation_performed"] is True
    assert [event["outcome"] for event in audit_payload["audit_events"][-2:]] == [
        "completed",
        "completed",
    ] or audit_payload["audit_events"][-1]["outcome"] == "completed"
    assert audit_payload["audit_events"][-1]["operation"] == "host-bootstrap"
