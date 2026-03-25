import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# Inject an fcntl stub before importing worker modules on Windows.
if "fcntl" not in sys.modules:
    _fcntl_stub = types.ModuleType("fcntl")
    _fcntl_stub.LOCK_EX = 1  # type: ignore[attr-defined]
    _fcntl_stub.LOCK_UN = 2  # type: ignore[attr-defined]
    _fcntl_stub.lockf = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    _fcntl_stub.flock = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["fcntl"] = _fcntl_stub

from gpustack.config.config import Config
from gpustack.schemas.models import (
    BackendEnum,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
    SourceEnum,
)


serve_manager_module = importlib.import_module("gpustack.worker.serve_manager")


def make_config(tmp_path: Path, worker_id: int = 1) -> Config:
    return Config(
        token="test",
        jwt_secret_key="test",
        data_dir=str(tmp_path / f"worker-{worker_id}"),
        log_dir=str(tmp_path / f"worker-{worker_id}" / "logs"),
        cache_dir=str(tmp_path / f"worker-{worker_id}" / "cache"),
        server_url="http://127.0.0.1:30080",
        direct_process_mode=True,
    )


def make_model() -> Model:
    return Model(
        id=1,
        name="test-model",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        backend=BackendEnum.ASCEND_MINDIE,
        backend_version="2.0",
    )


def make_model_instance() -> ModelInstance:
    return ModelInstance(
        id=1,
        name="distributed-mindie",
        worker_id=1,
        worker_name="leader",
        worker_ip="127.0.0.1",
        worker_ifname="lo",
        model_id=1,
        model_name="test-model",
        state=ModelInstanceStateEnum.STARTING,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        port=9000,
        ports=[9000, 9001],
        distributed_servers=DistributedServers(
            mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
            subordinate_workers=[
                ModelInstanceSubordinateWorker(
                    worker_id=2,
                    worker_name="follower",
                    worker_ip="127.0.0.2",
                    worker_ifname="eth0",
                    state=ModelInstanceStateEnum.STARTING,
                )
            ],
        ),
    )


def build_manager(tmp_path: Path, worker_id: int):
    cfg = make_config(tmp_path, worker_id=worker_id)
    assert cfg.log_dir is not None
    assert cfg.data_dir is not None
    Path(cfg.log_dir, "serve").mkdir(parents=True, exist_ok=True)
    Path(cfg.data_dir, "worker").mkdir(parents=True, exist_ok=True)
    manager = object.__new__(serve_manager_module.ServeManager)
    clientset = SimpleNamespace(headers={}, model_instances=SimpleNamespace())
    manager._config = cfg
    manager._serve_log_dir = str(Path(cfg.log_dir) / "serve")
    manager._worker_id_getter = lambda: worker_id
    manager._clientset_getter = lambda: clientset
    manager._provisioning_processes = {}
    manager._error_model_instances = {}
    manager._model_cache_by_instance = {}
    manager._model_instance_by_instance_id = {}
    manager._assigned_ports = {}
    manager._direct_process_registry = serve_manager_module.DirectProcessRegistry(cfg)
    manager._bootstrap_manager = None
    manager._inference_backend_manager = SimpleNamespace(
        get_backend_by_name=lambda _backend: SimpleNamespace(health_check_path="/v1/models")
    )
    manager._get_model = lambda _mi: make_model()
    manager._refresh_model = lambda _mi: make_model()
    manager._update_model = lambda *_args, **_kwargs: None
    updates = []
    manager._update_model_instance = lambda _id, **kwargs: updates.append(kwargs)
    return manager, updates


def _write_runtime_entry(
    registry,
    *,
    worker_id: int,
    model_instance_id: int,
    deployment_name: str,
    pid: int,
    port: int,
    log_path: str,
):
    registry.upsert(
        model_instance_id=model_instance_id,
        worker_id=worker_id,
        deployment_name=deployment_name,
        runtime_name="serve",
        pid=pid,
        process_group_id=None,
        port=port,
        log_path=log_path,
        backend=BackendEnum.ASCEND_MINDIE,
    )


def _prepare_bootstrap_runtime_roots(manager, model_instance: ModelInstance):
    assert model_instance.model_id is not None
    assert model_instance.id is not None
    bootstrap_manager = serve_manager_module.BootstrapManager(manager._config)
    roots = bootstrap_manager.prepare_runtime_roots(model_instance.model_id, model_instance.id)
    roots.workspace.joinpath("workspace.txt").write_text("workspace", encoding="utf-8")
    roots.artifacts.joinpath("artifact.txt").write_text("artifact", encoding="utf-8")
    roots.manifests.joinpath("manifest.txt").write_text("manifest", encoding="utf-8")
    roots.locks.joinpath("bootstrap.lock").write_text("lock", encoding="utf-8")
    return roots


def _assert_bootstrap_runtime_roots_removed(roots) -> None:
    assert not roots.workspace.exists()
    assert not roots.artifacts.exists()
    assert not roots.manifests.exists()
    assert not roots.locks.exists()


def test_distributed_direct_process_mindie_waits_for_follower_before_leader_ready(
    monkeypatch, tmp_path: Path
):
    model_instance = make_model_instance()
    assert model_instance.distributed_servers is not None
    assert model_instance.distributed_servers.subordinate_workers is not None
    leader_manager, leader_updates = build_manager(tmp_path, 1)
    follower_manager, follower_updates = build_manager(tmp_path, 2)

    _write_runtime_entry(
        leader_manager._direct_process_registry,
        worker_id=1,
        model_instance_id=1,
        deployment_name="distributed-mindie",
        pid=101,
        port=9000,
        log_path=str(Path(leader_manager._serve_log_dir) / "1.log"),
    )
    _write_runtime_entry(
        follower_manager._direct_process_registry,
        worker_id=2,
        model_instance_id=1,
        deployment_name="distributed-mindie-follower",
        pid=201,
        port=9001,
        log_path=str(Path(follower_manager._serve_log_dir) / "1.log"),
    )

    leader_entry = leader_manager._direct_process_registry.list_by_model_instance_id(1)[0]
    follower_entry = follower_manager._direct_process_registry.list_by_model_instance_id(1)[0]
    leader_manager._direct_process_registry.inspect_by_model_instance_id = (
        lambda _model_instance_id: SimpleNamespace(
            status=serve_manager_module.DirectProcessEntryStatus.LIVE,
            entry=leader_entry,
        )
    )
    follower_manager._direct_process_registry.inspect_by_model_instance_id = (
        lambda _model_instance_id: SimpleNamespace(
            status=serve_manager_module.DirectProcessEntryStatus.LIVE,
            entry=follower_entry,
        )
    )

    monkeypatch.setattr(
        serve_manager_module,
        "inspect_direct_process_entry",
        lambda entry: SimpleNamespace(
            status=serve_manager_module.DirectProcessEntryStatus.LIVE,
            entry=entry,
        ),
    )
    monkeypatch.setattr(serve_manager_module, "get_meta_from_running_instance", lambda *_args: None)
    monkeypatch.setattr(serve_manager_module, "is_ready", lambda *args, **kwargs: True)

    leader_manager._sync_direct_process_model_instance(model_instance)

    assert leader_updates[-1]["pid"] == 101
    assert "state" not in leader_updates[-1]

    follower_manager._sync_direct_process_model_instance(model_instance)
    assert (
        follower_updates[-1]["distributed_servers.subordinate_workers.0"].state
        == ModelInstanceStateEnum.RUNNING
    )

    model_instance.distributed_servers.subordinate_workers[0].state = (
        ModelInstanceStateEnum.RUNNING
    )
    leader_manager._sync_direct_process_model_instance(model_instance)

    assert leader_updates[-1]["state"] == ModelInstanceStateEnum.RUNNING
    assert leader_updates[-1]["pid"] == 101


def test_distributed_direct_process_mindie_subordinate_preflight_failure_rolls_back_and_cleans_up(
    monkeypatch, tmp_path: Path
):
    model_instance = make_model_instance()
    assert model_instance.distributed_servers is not None
    assert model_instance.distributed_servers.subordinate_workers is not None
    model_instance.state = ModelInstanceStateEnum.RUNNING
    model_instance.distributed_servers.subordinate_workers[0].state = (
        ModelInstanceStateEnum.ERROR
    )
    model_instance.distributed_servers.subordinate_workers[0].state_message = (
        "Direct-process MindIE host prerequisites not met: missing mindieservice_daemon"
    )
    leader_manager, leader_updates = build_manager(tmp_path, 1)
    leader_manager._model_instance_by_instance_id[model_instance.id] = model_instance
    runtime_roots = _prepare_bootstrap_runtime_roots(leader_manager, model_instance)

    _write_runtime_entry(
        leader_manager._direct_process_registry,
        worker_id=1,
        model_instance_id=1,
        deployment_name="distributed-mindie",
        pid=101,
        port=9000,
        log_path=str(Path(leader_manager._serve_log_dir) / "1.log"),
    )

    leader_entry = leader_manager._direct_process_registry.list_by_model_instance_id(1)[0]
    leader_manager._direct_process_registry.inspect_by_model_instance_id = (
        lambda _model_instance_id: SimpleNamespace(
            status=serve_manager_module.DirectProcessEntryStatus.LIVE,
            entry=leader_entry,
        )
    )

    terminated = []
    monkeypatch.setattr(
        serve_manager_module,
        "inspect_direct_process_entry",
        lambda entry: SimpleNamespace(
            status=serve_manager_module.DirectProcessEntryStatus.LIVE,
            entry=entry,
        ),
    )
    monkeypatch.setattr(
        leader_manager,
        "_terminate_direct_process_entry",
        lambda entry: terminated.append(entry.runtime_name),
    )

    leader_manager._sync_direct_process_model_instance(model_instance)

    assert terminated == ["serve"]
    assert leader_manager._direct_process_registry.list_by_model_instance_id(1) == []
    _assert_bootstrap_runtime_roots_removed(runtime_roots)
    assert "Distributed serving error in subordinate worker" in leader_updates[-1][
        "state_message"
    ]
    assert "MindIE host prerequisites not met" in leader_updates[-1]["state_message"]


def test_distributed_direct_process_mindie_missing_runtime_rolls_back_and_surfaces_runtime_failure(
    monkeypatch, tmp_path: Path
):
    model_instance = make_model_instance()
    leader_manager, leader_updates = build_manager(tmp_path, 1)
    leader_manager._model_instance_by_instance_id[model_instance.id] = model_instance
    runtime_roots = _prepare_bootstrap_runtime_roots(leader_manager, model_instance)

    _write_runtime_entry(
        leader_manager._direct_process_registry,
        worker_id=1,
        model_instance_id=1,
        deployment_name="distributed-mindie",
        pid=101,
        port=9000,
        log_path=str(Path(leader_manager._serve_log_dir) / "1.log"),
    )

    terminated = []
    monkeypatch.setattr(
        leader_manager,
        "_terminate_direct_process_entries",
        lambda entries: terminated.extend(entry.runtime_name for entry in entries),
    )

    leader_manager._sync_direct_process_model_instance(model_instance)

    assert terminated == ["serve"]
    assert leader_manager._direct_process_registry.list_by_model_instance_id(1) == []
    _assert_bootstrap_runtime_roots_removed(runtime_roots)
    assert leader_updates[-1]["pid"] == 101
    assert leader_updates[-1]["state"] == ModelInstanceStateEnum.ERROR
    assert leader_updates[-1]["state_message"] == (
        "Distributed MindIE runtime missing or unhealthy on the current worker."
    )
