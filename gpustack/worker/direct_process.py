import importlib
import sys
import types
from pathlib import Path
from functools import lru_cache
from typing import FrozenSet

from gpustack.config.config import Config
from gpustack.schemas.models import BackendEnum, ModelInstance
from gpustack.utils import platform


DIRECT_PROCESS_MODE_UNSUPPORTED_PLATFORM_MESSAGE = (
    "Direct process mode is supported only on Linux workers."
)
DIRECT_PROCESS_MODE_UNSUPPORTED_DISTRIBUTED_MESSAGE = (
    "Direct process mode supports only single-worker launches; distributed workers are not supported."
)
DIRECT_PROCESS_MODE_UNSUPPORTED_DISTRIBUTED_BACKEND_MESSAGE = (
    "Direct process mode supports distributed launches only for contract-declared distributed backends."
)
DIRECT_PROCESS_MODE_DISTRIBUTED_DISABLED_MESSAGE = (
    "Distributed direct process mode is disabled on this worker."
)
DIRECT_PROCESS_MODE_UNSUPPORTED_SUBORDINATE_MESSAGE = (
    "Direct process mode supports only single-worker launches on the main worker; subordinate workers are not supported."
)
DIRECT_PROCESS_MODE_UNSUPPORTED_BENCHMARK_MESSAGE = (
    "Direct process mode does not support benchmarks."
)

_DIRECT_PROCESS_BACKEND_CONTRACTS: tuple[tuple[str, str, str], ...] = (
    (BackendEnum.VLLM, "vllm", "VLLMServer"),
    (BackendEnum.CUSTOM, "custom", "CustomServer"),
    (BackendEnum.SGLANG, "sglang", "SGLangServer"),
    (BackendEnum.VOX_BOX, "vox_box", "VoxBoxServer"),
    (BackendEnum.ASCEND_MINDIE, "ascend_mindie", "AscendMindIEServer"),
    (BackendEnum.LLAMA_CPP, "llama_cpp", "LlamaCppServer"),
)


@lru_cache
def _ensure_worker_backend_namespace_packages() -> None:
    worker_dir = Path(__file__).resolve().parent
    package_paths = {
        "gpustack.worker": worker_dir,
        "gpustack.worker.backends": worker_dir / "backends",
    }

    for package_name, package_path in package_paths.items():
        if package_name in sys.modules:
            continue

        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package


def _load_backend_server_class(module_name: str, class_name: str) -> type:
    _ensure_worker_backend_namespace_packages()
    module = importlib.import_module(f"gpustack.worker.backends.{module_name}")
    return getattr(module, class_name)


@lru_cache
def _get_direct_process_backend_contracts() -> dict[str, type]:
    return {
        backend: _load_backend_server_class(module_name, class_name)
        for backend, module_name, class_name in _DIRECT_PROCESS_BACKEND_CONTRACTS
    }


@lru_cache
def _get_direct_process_supported_backends() -> FrozenSet[str]:
    return frozenset(
        backend
        for backend, server_cls in _get_direct_process_backend_contracts().items()
        if server_cls.is_direct_process_ready()
    )


@lru_cache
def _get_direct_process_distributed_backends() -> FrozenSet[str]:
    return frozenset(
        backend
        for backend, server_cls in _get_direct_process_backend_contracts().items()
        if server_cls.is_direct_process_ready(distributed=True)
    )


def _is_distributed_direct_process_enabled(cfg: Config | None) -> bool:
    if cfg is None:
        return True
    return getattr(cfg, "distributed_direct_process_vllm", False)


def backend_supports_direct_process(backend: str) -> bool:
    """Check whether a backend supports single-worker direct-process mode."""
    return backend in _get_direct_process_supported_backends()


def backend_supports_distributed_direct_process(
    backend: str, cfg: Config | None = None
) -> bool:
    """Check whether a backend supports distributed (multi-worker) direct-process."""
    if not _is_distributed_direct_process_enabled(cfg):
        return False
    return backend in _get_direct_process_distributed_backends()


def get_direct_process_supported_backends() -> FrozenSet[str]:
    """Return the set of backends that support single-worker direct-process."""
    return _get_direct_process_supported_backends()


def get_direct_process_distributed_backends(
    cfg: Config | None = None,
) -> FrozenSet[str]:
    """Return the set of backends that support distributed direct-process."""
    if not _is_distributed_direct_process_enabled(cfg):
        return frozenset()
    return _get_direct_process_distributed_backends()


def ensure_model_instance_direct_process_support(
    cfg: Config,
    model_instance: ModelInstance,
    backend: str,
    worker_id: int,
) -> None:
    if not getattr(cfg, "direct_process_mode", False):
        return

    if platform.system() != "linux":
        raise ValueError(DIRECT_PROCESS_MODE_UNSUPPORTED_PLATFORM_MESSAGE)

    if not backend_supports_direct_process(backend):
        supported = ", ".join(sorted(get_direct_process_supported_backends()))
        raise ValueError(
            f"Direct process mode supports only these backends: {supported}; got '{backend}'."
        )

    subordinate_workers = []
    if model_instance.distributed_servers:
        subordinate_workers = model_instance.distributed_servers.subordinate_workers or []

    if subordinate_workers:
        distributed_backends = _get_direct_process_distributed_backends()
        if backend not in distributed_backends:
            supported = ", ".join(sorted(distributed_backends))
            raise ValueError(
                f"{DIRECT_PROCESS_MODE_UNSUPPORTED_DISTRIBUTED_BACKEND_MESSAGE} Supported backends: {supported}; got '{backend}'."
            )
        if not backend_supports_distributed_direct_process(backend, cfg):
            raise ValueError(DIRECT_PROCESS_MODE_DISTRIBUTED_DISABLED_MESSAGE)

    is_primary_worker = model_instance.worker_id == worker_id
    declared_subordinate_worker_ids = {
        subordinate_worker.worker_id for subordinate_worker in subordinate_workers
    }
    is_declared_subordinate_worker = worker_id in declared_subordinate_worker_ids

    if not is_primary_worker and not is_declared_subordinate_worker:
        raise ValueError(DIRECT_PROCESS_MODE_UNSUPPORTED_SUBORDINATE_MESSAGE)


def ensure_benchmark_direct_process_support(cfg: Config) -> None:
    if not getattr(cfg, "direct_process_mode", False):
        return

    raise ValueError(DIRECT_PROCESS_MODE_UNSUPPORTED_BENCHMARK_MESSAGE)
