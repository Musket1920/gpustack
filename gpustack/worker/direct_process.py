from gpustack.config.config import Config
from gpustack.schemas.models import BackendEnum, ModelInstance
from gpustack.utils import platform


DIRECT_PROCESS_MODE_UNSUPPORTED_PLATFORM_MESSAGE = (
    "Direct process mode is supported only on Linux workers."
)
DIRECT_PROCESS_MODE_UNSUPPORTED_DISTRIBUTED_MESSAGE = (
    "Direct process mode supports only single-worker launches; distributed workers are not supported."
)
DIRECT_PROCESS_MODE_UNSUPPORTED_SUBORDINATE_MESSAGE = (
    "Direct process mode supports only single-worker launches on the main worker; subordinate workers are not supported."
)
DIRECT_PROCESS_MODE_UNSUPPORTED_BENCHMARK_MESSAGE = (
    "Direct process mode does not support benchmarks."
)


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

    if backend != BackendEnum.VLLM:
        raise ValueError(
            f"Direct process mode supports only the vLLM backend, got '{backend}'."
        )

    subordinate_workers = []
    if model_instance.distributed_servers:
        subordinate_workers = model_instance.distributed_servers.subordinate_workers or []

    if model_instance.worker_id != worker_id:
        raise ValueError(DIRECT_PROCESS_MODE_UNSUPPORTED_SUBORDINATE_MESSAGE)

    if subordinate_workers:
        raise ValueError(DIRECT_PROCESS_MODE_UNSUPPORTED_DISTRIBUTED_MESSAGE)


def ensure_benchmark_direct_process_support(cfg: Config) -> None:
    if not getattr(cfg, "direct_process_mode", False):
        return

    raise ValueError(DIRECT_PROCESS_MODE_UNSUPPORTED_BENCHMARK_MESSAGE)
