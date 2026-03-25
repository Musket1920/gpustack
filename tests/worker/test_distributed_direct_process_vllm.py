import asyncio
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
route_logs = importlib.import_module("gpustack.routes.worker.logs")
worker_logs = importlib.import_module("gpustack.worker.logs")
LogOptions = worker_logs.LogOptions


def normalize_newlines(lines: list[str]) -> list[str]:
    return [line.replace("\r\n", "\n") for line in lines]


def make_config(tmp_path: Path, worker_id: int = 1) -> Config:
    return Config(
        token="test",
        jwt_secret_key="test",
        data_dir=str(tmp_path / f"worker-{worker_id}"),
        log_dir=str(tmp_path / f"worker-{worker_id}" / "logs"),
        cache_dir=str(tmp_path / f"worker-{worker_id}" / "cache"),
        server_url="http://127.0.0.1:30080",
        direct_process_mode=True,
        distributed_direct_process_vllm=True,
    )


def make_model() -> Model:
    return Model(
        id=1,
        name="test-model",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        backend=BackendEnum.VLLM,
        backend_version="0.8.0",
    )


def make_model_instance() -> ModelInstance:
    return ModelInstance(
        id=1,
        name="distributed-vllm",
        worker_id=1,
        worker_name="leader",
        worker_ip="127.0.0.1",
        worker_ifname="lo",
        model_id=1,
        model_name="test-model",
        state=ModelInstanceStateEnum.STARTING,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        port=8000,
        ports=[8000, 8100],
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


def build_manager(tmp_path: Path, worker_id: int, model_instance: ModelInstance):
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


def _write_runtime_entry(registry, *, model_instance_id: int, deployment_name: str, runtime_name: str, pid: int, port: int, log_path: str):
    registry.upsert(
        model_instance_id=model_instance_id,
        worker_id=1,
        deployment_name=deployment_name,
        runtime_name=runtime_name,
        pid=pid,
        process_group_id=None,
        port=port,
        log_path=log_path,
        backend=BackendEnum.VLLM,
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


def test_distributed_direct_process_vllm_start_ready_runtime(monkeypatch, tmp_path: Path):
    model_instance = make_model_instance()
    assert model_instance.distributed_servers is not None
    leader_manager, leader_updates = build_manager(tmp_path, 1, model_instance)
    follower_manager, follower_updates = build_manager(tmp_path, 2, model_instance)

    leader_log = str(Path(leader_manager._serve_log_dir) / "1.log")
    ray_head_log = str(Path(leader_manager._serve_log_dir) / "1.ray-head.log")
    follower_log = str(Path(follower_manager._serve_log_dir) / "1.log")

    _write_runtime_entry(
        leader_manager._direct_process_registry,
        model_instance_id=1,
        deployment_name="distributed-vllm",
        runtime_name="serve",
        pid=101,
        port=8000,
        log_path=leader_log,
    )
    _write_runtime_entry(
        leader_manager._direct_process_registry,
        model_instance_id=1,
        deployment_name="distributed-vllm-ray-head",
        runtime_name="ray-head",
        pid=102,
        port=8100,
        log_path=ray_head_log,
    )
    follower_manager._direct_process_registry.upsert(
        model_instance_id=1,
        worker_id=2,
        deployment_name="distributed-vllm-f0",
        runtime_name="ray-worker",
        pid=201,
        process_group_id=None,
        port=8100,
        log_path=follower_log,
        backend=BackendEnum.VLLM,
    )

    monkeypatch.setattr(
        serve_manager_module,
        "inspect_direct_process_entry",
        lambda entry: SimpleNamespace(status=serve_manager_module.DirectProcessEntryStatus.LIVE, entry=entry),
    )
    monkeypatch.setattr(serve_manager_module, "get_meta_from_running_instance", lambda *_args: None)
    monkeypatch.setattr(serve_manager_module, "is_ready", lambda *args, **kwargs: True)

    follower_manager._sync_direct_process_model_instance(model_instance)
    assert follower_updates[-1]["distributed_servers.subordinate_workers.0"].state == ModelInstanceStateEnum.RUNNING

    assert model_instance.distributed_servers.subordinate_workers is not None
    model_instance.distributed_servers.subordinate_workers[0].state = ModelInstanceStateEnum.RUNNING
    leader_manager._sync_direct_process_model_instance(model_instance)

    assert leader_updates[-1]["state"] == ModelInstanceStateEnum.RUNNING
    assert leader_updates[-1]["pid"] == 101


def test_distributed_direct_process_vllm_main_waits_for_subordinates_before_ready(
    monkeypatch, tmp_path: Path
):
    model_instance = make_model_instance()
    leader_manager, leader_updates = build_manager(tmp_path, 1, model_instance)

    _write_runtime_entry(
        leader_manager._direct_process_registry,
        model_instance_id=1,
        deployment_name="distributed-vllm",
        runtime_name="serve",
        pid=101,
        port=8000,
        log_path=str(Path(leader_manager._serve_log_dir) / "1.log"),
    )
    _write_runtime_entry(
        leader_manager._direct_process_registry,
        model_instance_id=1,
        deployment_name="distributed-vllm-ray-head",
        runtime_name="ray-head",
        pid=102,
        port=8100,
        log_path=str(Path(leader_manager._serve_log_dir) / "1.ray-head.log"),
    )

    monkeypatch.setattr(
        serve_manager_module,
        "inspect_direct_process_entry",
        lambda entry: SimpleNamespace(
            status=serve_manager_module.DirectProcessEntryStatus.LIVE,
            entry=entry,
        ),
    )
    monkeypatch.setattr(serve_manager_module, "is_ready", lambda *args, **kwargs: True)

    leader_manager._sync_direct_process_model_instance(model_instance)

    assert leader_updates[-1] == {"pid": 101}
    assert leader_manager._direct_process_registry.list_by_model_instance_id(1)


def test_distributed_direct_process_vllm_subordinate_failure_cleanup(monkeypatch, tmp_path: Path):
    model_instance = make_model_instance()
    assert model_instance.distributed_servers is not None
    assert model_instance.distributed_servers.subordinate_workers is not None
    model_instance.state = ModelInstanceStateEnum.RUNNING
    model_instance.distributed_servers.subordinate_workers[0].state = ModelInstanceStateEnum.ERROR
    model_instance.distributed_servers.subordinate_workers[0].state_message = "ray worker exited"
    leader_manager, leader_updates = build_manager(tmp_path, 1, model_instance)
    leader_manager._model_instance_by_instance_id[model_instance.id] = model_instance
    runtime_roots = _prepare_bootstrap_runtime_roots(leader_manager, model_instance)

    _write_runtime_entry(
        leader_manager._direct_process_registry,
        model_instance_id=1,
        deployment_name="distributed-vllm",
        runtime_name="serve",
        pid=101,
        port=8000,
        log_path=str(Path(leader_manager._serve_log_dir) / "1.log"),
    )
    _write_runtime_entry(
        leader_manager._direct_process_registry,
        model_instance_id=1,
        deployment_name="distributed-vllm-ray-head",
        runtime_name="ray-head",
        pid=102,
        port=8100,
        log_path=str(Path(leader_manager._serve_log_dir) / "1.ray-head.log"),
    )

    terminated = []
    monkeypatch.setattr(
        serve_manager_module,
        "inspect_direct_process_entry",
        lambda entry: SimpleNamespace(status=serve_manager_module.DirectProcessEntryStatus.LIVE, entry=entry),
    )
    monkeypatch.setattr(
        leader_manager,
        "_terminate_direct_process_entry",
        lambda entry: terminated.append(entry.runtime_name),
    )

    leader_manager._sync_direct_process_model_instance(model_instance)

    assert sorted(terminated) == ["ray-head", "serve"]
    assert leader_manager._direct_process_registry.list_by_model_instance_id(1) == []
    _assert_bootstrap_runtime_roots_removed(runtime_roots)
    assert "Distributed serving error in subordinate worker" in leader_updates[-1]["state_message"]


def test_distributed_direct_process_vllm_missing_runtime_rolls_back_and_errors(
    monkeypatch, tmp_path: Path
):
    model_instance = make_model_instance()
    leader_manager, leader_updates = build_manager(tmp_path, 1, model_instance)
    leader_manager._model_instance_by_instance_id[model_instance.id] = model_instance
    runtime_roots = _prepare_bootstrap_runtime_roots(leader_manager, model_instance)

    _write_runtime_entry(
        leader_manager._direct_process_registry,
        model_instance_id=1,
        deployment_name="distributed-vllm",
        runtime_name="serve",
        pid=101,
        port=8000,
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
    assert leader_updates[-1] == {
        "pid": 101,
        "state": ModelInstanceStateEnum.ERROR,
        "state_message": "Inference server exited or unhealthy.",
    }


def test_distributed_direct_process_vllm_prepare_failure_rolls_back_runtime_state(
    monkeypatch, tmp_path: Path
):
    model_instance = make_model_instance()
    leader_manager, leader_updates = build_manager(tmp_path, 1, model_instance)
    leader_manager._model_instance_by_instance_id[model_instance.id] = model_instance
    leader_manager._assigned_ports[model_instance.id] = [8000, 8100]
    leader_manager._model_cache_by_instance[model_instance.id] = {"cache": "warm"}
    leader_manager._error_model_instances[model_instance.id] = {"count": 1}

    stale_log = str(Path(leader_manager._serve_log_dir) / "1.stale.log")
    _write_runtime_entry(
        leader_manager._direct_process_registry,
        model_instance_id=1,
        deployment_name="distributed-vllm",
        runtime_name="serve",
        pid=101,
        port=8000,
        log_path=stale_log,
    )

    terminated = []

    def fail_prepare(mi, model, backend):
        _prepare_bootstrap_runtime_roots(leader_manager, mi)
        raise RuntimeError("bootstrap preparation failed")

    monkeypatch.setattr(
        serve_manager_module,
        "ensure_model_instance_direct_process_support",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(leader_manager, "_prepare_direct_process_bootstrap", fail_prepare)
    monkeypatch.setattr(
        leader_manager,
        "_terminate_direct_process_entries",
        lambda entries: terminated.extend(entry.runtime_name for entry in entries),
    )

    leader_manager._start_model_instance(model_instance)

    runtime_roots = serve_manager_module.BootstrapManager(
        leader_manager._config
    ).runtime_roots(model_instance.model_id, model_instance.id)

    assert terminated == ["serve"]
    assert leader_manager._direct_process_registry.list_by_model_instance_id(1) == []
    _assert_bootstrap_runtime_roots_removed(runtime_roots)
    assert leader_manager._assigned_ports == {}
    assert leader_manager._model_cache_by_instance == {}
    assert leader_manager._error_model_instances == {}
    assert leader_manager._provisioning_processes == {}
    assert leader_manager._model_instance_by_instance_id == {}
    assert leader_updates[-1] == {
        "state": ModelInstanceStateEnum.ERROR,
        "state_message": "Failed to prepare direct-process bootstrap environment: bootstrap preparation failed",
    }


@pytest.mark.asyncio
async def test_distributed_direct_process_vllm_logs_aggregate_local_runtime_files(tmp_path: Path):
    cfg = make_config(tmp_path, worker_id=1)
    assert cfg.log_dir is not None
    assert cfg.data_dir is not None
    Path(cfg.log_dir, "serve").mkdir(parents=True, exist_ok=True)
    Path(cfg.data_dir, "worker").mkdir(parents=True, exist_ok=True)

    main_log = Path(cfg.log_dir) / "serve" / "1.log"
    extra_log = Path(cfg.log_dir) / "serve" / "1.ray-head.log"
    main_log.write_text("serve-line\n", encoding="utf-8")
    extra_log.write_text("ray-line\n", encoding="utf-8")

    registry = serve_manager_module.DirectProcessRegistry(cfg)
    registry.upsert(
        model_instance_id=1,
        worker_id=1,
        deployment_name="distributed-vllm",
        runtime_name="serve",
        pid=101,
        process_group_id=None,
        port=8000,
        log_path=str(main_log),
        backend=BackendEnum.VLLM,
    )
    registry.upsert(
        model_instance_id=1,
        worker_id=1,
        deployment_name="distributed-vllm-ray-head",
        runtime_name="ray-head",
        pid=102,
        process_group_id=None,
        port=8100,
        log_path=str(extra_log),
        backend=BackendEnum.VLLM,
    )

    lines = [
        line
        async for line in route_logs.combined_log_generator(
            str(main_log),
            "",
            LogOptions(follow=False),
            "distributed-vllm",
            file_log_exists=True,
            file_only=True,
            extra_file_log_paths=route_logs._get_direct_process_log_paths(cfg, 1),
        )
    ]

    assert sorted(normalize_newlines(lines)) == ["ray-line\n", "serve-line\n"]
