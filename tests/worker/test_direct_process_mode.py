import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from gpustack.config.config import Config
from gpustack.schemas.benchmark import Benchmark, BenchmarkStateEnum
from gpustack.schemas.config import parse_base_model_to_env_vars
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

WORKER_DIR = Path(__file__).resolve().parents[2] / "gpustack" / "worker"


def _load_module_from_path(module_name: str, relative_path: str):
    module_path = WORKER_DIR / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


direct_process = _load_module_from_path(
    "tests.worker.direct_process_module", "direct_process.py"
)
DIRECT_PROCESS_MODE_UNSUPPORTED_BENCHMARK_MESSAGE = (
    direct_process.DIRECT_PROCESS_MODE_UNSUPPORTED_BENCHMARK_MESSAGE
)
DIRECT_PROCESS_MODE_UNSUPPORTED_DISTRIBUTED_MESSAGE = (
    direct_process.DIRECT_PROCESS_MODE_UNSUPPORTED_DISTRIBUTED_MESSAGE
)
DIRECT_PROCESS_MODE_UNSUPPORTED_PLATFORM_MESSAGE = (
    direct_process.DIRECT_PROCESS_MODE_UNSUPPORTED_PLATFORM_MESSAGE
)
DIRECT_PROCESS_MODE_UNSUPPORTED_SUBORDINATE_MESSAGE = (
    direct_process.DIRECT_PROCESS_MODE_UNSUPPORTED_SUBORDINATE_MESSAGE
)
ensure_benchmark_direct_process_support = (
    direct_process.ensure_benchmark_direct_process_support
)
ensure_model_instance_direct_process_support = (
    direct_process.ensure_model_instance_direct_process_support
)


def make_config(tmp_path, **kwargs) -> Config:
    return Config(
        token="test",
        jwt_secret_key="test",
        data_dir=str(tmp_path),
        server_url="http://127.0.0.1:30080",
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
    }
    defaults.update(kwargs)
    return Model(**defaults)


def make_benchmark(**kwargs) -> Benchmark:
    defaults = {
        "id": 1,
        "name": "test-benchmark",
        "cluster_id": 1,
        "model_instance_name": "test-instance",
        "worker_id": 1,
    }
    defaults.update(kwargs)
    return Benchmark(**defaults)


def test_direct_process_mode_config_env_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUSTACK_DIRECT_PROCESS_MODE", "true")

    cfg = make_config(tmp_path)

    assert getattr(cfg, "direct_process_mode") is True
    assert parse_base_model_to_env_vars(cfg)["GPUSTACK_DIRECT_PROCESS_MODE"] == "true"


def test_direct_process_mode_support_matrix_supported_vllm_linux_single_worker(
    monkeypatch, tmp_path
):
    cfg = make_config(tmp_path, direct_process_mode=True)
    mi = make_model_instance()

    monkeypatch.setattr(direct_process.platform, "system", lambda: "linux")

    ensure_model_instance_direct_process_support(
        cfg,
        mi,
        BackendEnum.VLLM,
        worker_id=1,
    )


@pytest.mark.parametrize(
    ("backend", "worker_id", "distributed_servers", "platform_name", "message"),
    [
        (
            BackendEnum.VLLM,
            1,
            None,
            "windows",
            DIRECT_PROCESS_MODE_UNSUPPORTED_PLATFORM_MESSAGE,
        ),
        (
            BackendEnum.SGLANG,
            1,
            None,
            "linux",
            "Direct process mode supports only the vLLM backend, got 'SGLang'.",
        ),
        (
            BackendEnum.VLLM,
            1,
            DistributedServers(
                mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
                subordinate_workers=[
                    ModelInstanceSubordinateWorker(worker_id=2, worker_ip="127.0.0.2")
                ],
            ),
            "linux",
            DIRECT_PROCESS_MODE_UNSUPPORTED_DISTRIBUTED_MESSAGE,
        ),
        (
            BackendEnum.VLLM,
            2,
            DistributedServers(
                mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
                subordinate_workers=[
                    ModelInstanceSubordinateWorker(worker_id=2, worker_ip="127.0.0.2")
                ],
            ),
            "linux",
            DIRECT_PROCESS_MODE_UNSUPPORTED_SUBORDINATE_MESSAGE,
        ),
    ],
    ids=["platform", "backend", "distributed", "subordinate"],
)
def test_direct_process_mode_unsupported_model_instance_validation(
    monkeypatch,
    tmp_path,
    backend,
    worker_id,
    distributed_servers,
    platform_name,
    message,
):
    cfg = make_config(tmp_path, direct_process_mode=True)
    mi = make_model_instance(distributed_servers=distributed_servers)

    monkeypatch.setattr(direct_process.platform, "system", lambda: platform_name)

    with pytest.raises(ValueError, match=message):
        ensure_model_instance_direct_process_support(
            cfg,
            mi,
            backend,
            worker_id=worker_id,
        )


def test_direct_process_mode_unsupported_benchmark(tmp_path):
    cfg = make_config(tmp_path, direct_process_mode=True)

    with pytest.raises(
        ValueError,
        match=DIRECT_PROCESS_MODE_UNSUPPORTED_BENCHMARK_MESSAGE,
    ):
        ensure_benchmark_direct_process_support(cfg)


def test_direct_process_mode_wrapper_without_subordinates_is_allowed(
    monkeypatch, tmp_path
):
    cfg = make_config(tmp_path, direct_process_mode=True)
    mi = make_model_instance(
        distributed_servers=DistributedServers(
            mode=DistributedServerCoordinateModeEnum.DELEGATED,
            subordinate_workers=[],
        )
    )

    monkeypatch.setattr(direct_process.platform, "system", lambda: "linux")

    ensure_model_instance_direct_process_support(
        cfg,
        mi,
        BackendEnum.VLLM,
        worker_id=1,
    )


@pytest.mark.asyncio
async def test_benchmark_rejected_before_launch_attempt_in_direct_process_mode(
    monkeypatch, tmp_path
):
    benchmark_manager_module = _import_worker_module("gpustack.worker.benchmark_manager")
    cfg = make_config(tmp_path, direct_process_mode=True)
    benchmark = make_benchmark()
    manager = object.__new__(benchmark_manager_module.BenchmarkManager)
    manager._config = cfg
    manager._benchmark_log_dir = str(tmp_path / "benchmarks")
    manager._provisioning_processes = {}

    updates = []

    async def fake_update_benchmark_state(benchmark_id: int, **kwargs):
        updates.append((benchmark_id, kwargs))

    manager._update_benchmark_state = fake_update_benchmark_state

    monkeypatch.setattr(direct_process.platform, "system", lambda: "linux")
    monkeypatch.setattr(
        benchmark_manager_module, "ensure_benchmark_direct_process_support", ensure_benchmark_direct_process_support
    )
    monkeypatch.setattr(
        benchmark_manager_module.multiprocessing,
        "Process",
        lambda *args, **kwargs: pytest.fail("benchmark launch should be rejected before process creation"),
    )

    await benchmark_manager_module.BenchmarkManager._start_benchmark(manager, benchmark)

    assert updates == [
        (
            benchmark.id,
            {
                "state": BenchmarkStateEnum.ERROR,
                "state_message": "Failed to start benchmark: "
                + DIRECT_PROCESS_MODE_UNSUPPORTED_BENCHMARK_MESSAGE,
            },
        )
    ]
    assert manager._provisioning_processes == {}
    assert not (tmp_path / "benchmarks" / f"{benchmark.id}.log").exists()


@pytest.mark.parametrize(
    ("worker_id", "message", "patch_key"),
    [
        (
            1,
            DIRECT_PROCESS_MODE_UNSUPPORTED_DISTRIBUTED_MESSAGE,
            None,
        ),
        (
            2,
            DIRECT_PROCESS_MODE_UNSUPPORTED_SUBORDINATE_MESSAGE,
            "distributed_servers.subordinate_workers.0",
        ),
    ],
    ids=["distributed_rejected", "subordinate_rejected"],
)
def test_distributed_rejected_before_launch_attempt_in_direct_process_mode(
    monkeypatch, tmp_path, worker_id, message, patch_key
):
    serve_manager_module = _import_worker_module("gpustack.worker.serve_manager")
    cfg = make_config(tmp_path, direct_process_mode=True)
    mi = make_model_instance(
        distributed_servers=DistributedServers(
            mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
            subordinate_workers=[
                ModelInstanceSubordinateWorker(worker_id=2, worker_ip="127.0.0.2")
            ],
        )
    )
    manager = object.__new__(serve_manager_module.ServeManager)
    manager._config = cfg
    manager._serve_log_dir = str(tmp_path / "serve")
    manager._provisioning_processes = {}
    manager._worker_id_getter = lambda: worker_id
    manager._get_model = lambda _: make_model()

    updates = []

    def fake_update_model_instance(model_instance_id: int, **kwargs):
        updates.append((model_instance_id, kwargs))

    manager._update_model_instance = fake_update_model_instance

    monkeypatch.setattr(direct_process.platform, "system", lambda: "linux")
    monkeypatch.setattr(
        serve_manager_module,
        "ensure_model_instance_direct_process_support",
        ensure_model_instance_direct_process_support,
    )
    monkeypatch.setattr(
        manager,
        "_assign_ports",
        lambda *args, **kwargs: pytest.fail("distributed direct-process launch should be rejected before port allocation"),
    )

    serve_manager_module.ServeManager._start_model_instance(manager, mi)

    assert manager._provisioning_processes == {}
    assert not (tmp_path / "serve" / f"{mi.id}.log").exists()
    assert len(updates) == 1
    updated_id, patch = updates[0]
    assert updated_id == mi.id
    if patch_key is None:
        assert patch == {
            "state": ModelInstanceStateEnum.ERROR,
            "state_message": f"Failed to start model instance: {message}",
        }
    else:
        assert list(patch.keys()) == [patch_key]
        subordinate = patch[patch_key]
        assert subordinate.state == ModelInstanceStateEnum.ERROR
        assert subordinate.state_message == f"Failed to start model instance: {message}"
