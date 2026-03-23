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
    "Direct process mode supports distributed launches only for the vLLM backend."
)
DIRECT_PROCESS_MODE_DISTRIBUTED_VLLM_DISABLED_MESSAGE = (
    "Distributed vLLM direct process mode is disabled on this worker."
)
DIRECT_PROCESS_MODE_UNSUPPORTED_SUBORDINATE_MESSAGE = (
    "Direct process mode supports only single-worker launches on the main worker; subordinate workers are not supported."
)
DIRECT_PROCESS_MODE_UNSUPPORTED_BENCHMARK_MESSAGE = (
    "Direct process mode does not support benchmarks."
)

# ---------------------------------------------------------------------------
# Backend direct-process support registry
#
# Backends that support single-worker direct-process mode register here.
# The gatekeeper checks this set instead of hardcoding BackendEnum.VLLM.
# Future tasks (SGLang, VoxBox, MindIE, llama.cpp, custom) will add entries.
# ---------------------------------------------------------------------------
_DIRECT_PROCESS_SUPPORTED_BACKENDS: FrozenSet[str] = frozenset(
    {
        BackendEnum.VLLM,
        BackendEnum.CUSTOM,
        BackendEnum.SGLANG,
        BackendEnum.VOX_BOX,
        BackendEnum.ASCEND_MINDIE,
        BackendEnum.LLAMA_CPP,
    }
)

# ---------------------------------------------------------------------------
# Distributed direct-process support registry
#
# Backends that support multi-worker (distributed) direct-process register
# here. The worker feature flag decides whether vLLM is actually advertised.
# ---------------------------------------------------------------------------
_DIRECT_PROCESS_DISTRIBUTED_BACKENDS: FrozenSet[str] = frozenset({BackendEnum.VLLM})


def backend_supports_direct_process(backend: str) -> bool:
    """Check whether a backend supports single-worker direct-process mode."""
    return backend in _DIRECT_PROCESS_SUPPORTED_BACKENDS


def backend_supports_distributed_direct_process(
    backend: str, cfg: Config | None = None
) -> bool:
    """Check whether a backend supports distributed (multi-worker) direct-process."""
    if cfg is not None and not getattr(cfg, "distributed_direct_process_vllm", False):
        return False
    return backend in _DIRECT_PROCESS_DISTRIBUTED_BACKENDS


def get_direct_process_supported_backends() -> FrozenSet[str]:
    """Return the set of backends that support single-worker direct-process."""
    return _DIRECT_PROCESS_SUPPORTED_BACKENDS


def get_direct_process_distributed_backends(
    cfg: Config | None = None,
) -> FrozenSet[str]:
    """Return the set of backends that support distributed direct-process."""
    if cfg is not None and not getattr(cfg, "distributed_direct_process_vllm", False):
        return frozenset()
    return _DIRECT_PROCESS_DISTRIBUTED_BACKENDS


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
        supported = ", ".join(sorted(_DIRECT_PROCESS_SUPPORTED_BACKENDS))
        raise ValueError(
            f"Direct process mode supports only the {supported} backend, got '{backend}'."
        )

    subordinate_workers = []
    if model_instance.distributed_servers:
        subordinate_workers = model_instance.distributed_servers.subordinate_workers or []

    if model_instance.worker_id != worker_id:
        raise ValueError(DIRECT_PROCESS_MODE_UNSUPPORTED_SUBORDINATE_MESSAGE)

    if subordinate_workers:
        if backend != BackendEnum.VLLM:
            raise ValueError(
                DIRECT_PROCESS_MODE_UNSUPPORTED_DISTRIBUTED_BACKEND_MESSAGE
            )
        if not backend_supports_distributed_direct_process(backend, cfg):
            raise ValueError(DIRECT_PROCESS_MODE_DISTRIBUTED_VLLM_DISABLED_MESSAGE)


def ensure_benchmark_direct_process_support(cfg: Config) -> None:
    if not getattr(cfg, "direct_process_mode", False):
        return

    raise ValueError(DIRECT_PROCESS_MODE_UNSUPPORTED_BENCHMARK_MESSAGE)
