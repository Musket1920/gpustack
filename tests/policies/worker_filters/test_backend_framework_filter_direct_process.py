import sys
import types

# Inject an fcntl stub before importing any gpustack.policies module so that
# the import chain backend_framework_filter → vllm_resource_fit_selector →
# ascend_mindie_resource_fit_selector → gpustack.worker.__init__ →
# gpustack.utils.locks → fcntl does not fail on Windows.
if "fcntl" not in sys.modules:
    _fcntl_stub = types.ModuleType("fcntl")
    _fcntl_stub.LOCK_EX = 1  # type: ignore[attr-defined]
    _fcntl_stub.LOCK_UN = 2  # type: ignore[attr-defined]
    _fcntl_stub.lockf = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    _fcntl_stub.flock = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["fcntl"] = _fcntl_stub

from unittest.mock import patch
from types import SimpleNamespace

import pytest

from gpustack.policies.worker_filters.backend_framework_filter import (
    BackendFrameworkFilter,
    DIRECT_PROCESS_MODE_LABEL,
)
from gpustack.schemas.models import BackendEnum
from gpustack.schemas.workers import (
    DIRECT_PROCESS_BACKENDS_LABEL,
    DIRECT_PROCESS_BOOTSTRAP_BACKENDS_LABEL,
    DIRECT_PROCESS_DISTRIBUTED_BACKENDS_LABEL,
    OperatingSystemInfo,
)
from tests.fixtures.workers.fixtures import linux_nvidia_4_4080_16gx4
from tests.policies.worker_filters.test_backend_framework_filter import (
    create_model,
)


def _enable_direct_process(worker, backends=None):
    """Set direct-process mode labels on a worker.

    *backends* is a comma-separated string of backend names the worker
    advertises (e.g. ``"vLLM"``).  When ``None`` only the coarse mode
    label is set — the worker advertises no specific backend capability.
    """
    assert worker.labels is not None
    worker.labels[DIRECT_PROCESS_MODE_LABEL] = "true"
    if backends is not None:
        worker.labels[DIRECT_PROCESS_BACKENDS_LABEL] = backends


def _enable_distributed_direct_process(worker, backends="vLLM"):
    assert worker.labels is not None
    worker.labels[DIRECT_PROCESS_DISTRIBUTED_BACKENDS_LABEL] = backends


def _enable_bootstrap_ready_direct_process(worker, backends="vLLM"):
    assert worker.labels is not None
    worker.labels[DIRECT_PROCESS_BOOTSTRAP_BACKENDS_LABEL] = backends


# ── existing characterization tests (updated for capability labels) ──────


@pytest.mark.asyncio
async def test_direct_process_backend_version_vllm_linux_worker_bypasses_runner_filter():
    model = create_model(backend="vLLM", backend_version="0.13.0")
    worker = linux_nvidia_4_4080_16gx4()
    _enable_direct_process(worker, backends="vLLM")
    workers = [worker]

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.list_service_runners',
        side_effect=AssertionError("list_service_runners should not be called"),
    ):
        filtered_workers, messages = await BackendFrameworkFilter(model).filter(workers)

    assert len(filtered_workers) == 1
    assert filtered_workers[0].name == "host-4-4080"
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_direct_process_rejects_non_vllm_backend_with_clear_message():
    """Worker advertises vLLM but model requests SGLang — rejected with clear message."""
    model = create_model(backend=BackendEnum.SGLANG, backend_version="0.4.0")
    worker = linux_nvidia_4_4080_16gx4()
    _enable_direct_process(worker, backends="vLLM")
    workers = [worker]

    filtered_workers, messages = await BackendFrameworkFilter(model).filter(workers)

    assert len(filtered_workers) == 0
    assert len(messages) == 1
    assert "vLLM" in messages[0]
    assert "SGLang" in messages[0]


@pytest.mark.asyncio
async def test_direct_process_rejects_non_linux_worker_with_clear_message():
    model = create_model(backend="vLLM", backend_version="0.13.0")
    worker = linux_nvidia_4_4080_16gx4()
    worker.labels["os"] = "windows"
    _enable_direct_process(worker, backends="vLLM")

    filtered_workers, messages = await BackendFrameworkFilter(model).filter([worker])

    assert len(filtered_workers) == 0
    assert len(messages) == 1
    assert "Direct process mode is supported only on Linux workers." in messages[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_os_name", "label_os"),
    [
        pytest.param("Ubuntu", "linux", id="linux-label-overrides-linux-distro-name"),
        pytest.param("Linux", "windows", id="linux-status-name-wins-over-nonlinux-label"),
    ],
)
async def test_direct_process_accepts_linux_like_worker_platform_names(
    status_os_name,
    label_os,
):
    model = create_model(backend="vLLM", backend_version="0.13.0")
    worker = linux_nvidia_4_4080_16gx4()
    worker_status = worker.status
    assert worker_status is not None
    worker_status.os = OperatingSystemInfo(name=status_os_name, version="")
    worker.labels["os"] = label_os
    _enable_direct_process(worker, backends="vLLM")

    filtered_workers, messages = await BackendFrameworkFilter(model).filter([worker])

    assert len(filtered_workers) == 1
    assert filtered_workers[0].name == worker.name
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_direct_process_rejects_multi_worker_vllm_path_with_clear_message():
    model = create_model(
        backend="vLLM",
        backend_version="0.13.0",
        backend_parameters=["--tensor-parallel-size", "8"],
    )
    worker = linux_nvidia_4_4080_16gx4()
    _enable_direct_process(worker, backends="vLLM")
    workers = [worker]

    filtered_workers, messages = await BackendFrameworkFilter(model).filter(workers)

    assert len(filtered_workers) == 0
    assert len(messages) == 1
    assert "not bootstrap-ready on this worker" in messages[0]


@pytest.mark.asyncio
async def test_distributed_direct_process_vllm_bootstrap_ready_worker_passes_filter():
    model = create_model(
        backend="vLLM",
        backend_version="0.13.0",
        backend_parameters=["--tensor-parallel-size", "8"],
    )
    worker = linux_nvidia_4_4080_16gx4()
    _enable_direct_process(worker, backends="vLLM")
    _enable_bootstrap_ready_direct_process(worker)
    _enable_distributed_direct_process(worker)

    filtered_workers, messages = await BackendFrameworkFilter(model).filter([worker])

    assert len(filtered_workers) == 1
    assert len(messages) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "worker_backends", "backend_version"),
    [
        pytest.param("vLLM", "vLLM", "0.13.0", id="vllm"),
        pytest.param(BackendEnum.SGLANG, "SGLang", "0.4.0", id="sglang"),
        pytest.param("MindIE", "MindIE", None, id="mindie"),
    ],
)
async def test_distributed_direct_process_supported_backends_with_bootstrap_and_distributed_capabilities_pass_filter(
    backend,
    worker_backends,
    backend_version,
):
    model = create_model(
        backend=backend,
        backend_version=backend_version,
        backend_parameters=["--tensor-parallel-size", "8"],
    )
    worker = linux_nvidia_4_4080_16gx4()
    _enable_direct_process(worker, backends=worker_backends)
    _enable_bootstrap_ready_direct_process(worker, backends=worker_backends)
    _enable_distributed_direct_process(worker, backends=worker_backends)

    filtered_workers, messages = await BackendFrameworkFilter(model).filter([worker])

    assert len(filtered_workers) == 1
    assert filtered_workers[0].name == worker.name
    assert len(messages) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "worker_backends", "backend_version", "bootstrap_backends_label"),
    [
        pytest.param("vLLM", "vLLM", "0.13.0", "SGLang", id="vllm"),
        pytest.param(BackendEnum.SGLANG, "SGLang", "0.4.0", "vLLM", id="sglang"),
        pytest.param("MindIE", "MindIE", None, "vLLM", id="mindie"),
    ],
)
async def test_distributed_direct_process_supported_backends_without_bootstrap_ready_capability_rejected(
    backend,
    worker_backends,
    backend_version,
    bootstrap_backends_label,
):
    model = create_model(
        backend=backend,
        backend_version=backend_version,
        backend_parameters=["--tensor-parallel-size", "8"],
    )
    worker = linux_nvidia_4_4080_16gx4()
    _enable_direct_process(worker, backends=worker_backends)
    _enable_bootstrap_ready_direct_process(worker, backends=bootstrap_backends_label)
    _enable_distributed_direct_process(worker, backends=worker_backends)

    filtered_workers, messages = await BackendFrameworkFilter(model).filter([worker])

    assert len(filtered_workers) == 0
    assert len(messages) == 1
    assert "bootstrap" in messages[0].lower()
    assert worker.name in messages[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "backend",
        "worker_backends",
        "backend_version",
        "distributed_backends_label",
    ),
    [
        pytest.param("vLLM", "vLLM", "0.13.0", "SGLang", id="vllm"),
        pytest.param(BackendEnum.SGLANG, "SGLang", "0.4.0", "vLLM", id="sglang"),
        pytest.param("MindIE", "MindIE", None, "vLLM", id="mindie"),
    ],
)
async def test_distributed_direct_process_supported_backends_without_distributed_capability_rejected(
    backend,
    worker_backends,
    backend_version,
    distributed_backends_label,
):
    model = create_model(
        backend=backend,
        backend_version=backend_version,
        backend_parameters=["--tensor-parallel-size", "8"],
    )
    worker = linux_nvidia_4_4080_16gx4()
    _enable_direct_process(worker, backends=worker_backends)
    _enable_bootstrap_ready_direct_process(worker, backends=worker_backends)
    _enable_distributed_direct_process(worker, backends=distributed_backends_label)

    filtered_workers, messages = await BackendFrameworkFilter(model).filter([worker])

    assert len(filtered_workers) == 0
    assert len(messages) == 1
    assert "distributed" in messages[0].lower()
    assert worker.name in messages[0]


@pytest.mark.asyncio
async def test_distributed_direct_process_unsupported_backend_rejected_before_launch_filter():
    model = create_model(
        backend="VoxBox",
        distributed_inference_across_workers=True,
        backend_parameters=["--tensor-parallel-size", "8"],
    )
    worker = linux_nvidia_4_4080_16gx4()
    _enable_direct_process(worker, backends="vLLM")
    _enable_bootstrap_ready_direct_process(worker, backends="vLLM")
    _enable_distributed_direct_process(worker, backends="vLLM")

    filtered_workers, messages = await BackendFrameworkFilter(model).filter([worker])

    assert len(filtered_workers) == 0
    assert len(messages) == 1
    assert "supports only the vLLM backend" in messages[0]
    assert "VoxBox" in messages[0]
    assert worker.name in messages[0]


@pytest.mark.asyncio
async def test_distributed_direct_process_vllm_without_bootstrap_ready_capability_rejected():
    model = create_model(
        backend="vLLM",
        backend_version="0.13.0",
        backend_parameters=["--tensor-parallel-size", "8"],
    )
    worker = linux_nvidia_4_4080_16gx4()
    _enable_direct_process(worker, backends="vLLM")
    _enable_bootstrap_ready_direct_process(worker, backends="SGLang")
    _enable_distributed_direct_process(worker)

    filtered_workers, messages = await BackendFrameworkFilter(model).filter([worker])

    assert len(filtered_workers) == 0
    assert len(messages) == 1
    assert "bootstrap" in messages[0].lower()
    assert worker.name in messages[0]


@pytest.mark.asyncio
async def test_direct_process_filter_is_worker_scoped_for_mixed_fleets():
    model = create_model(backend="vLLM", backend_version="0.13.0")
    direct_process_worker = linux_nvidia_4_4080_16gx4()
    standard_worker = linux_nvidia_4_4080_16gx4()

    _enable_direct_process(direct_process_worker, backends="vLLM")
    standard_worker.name = "container-worker"

    with patch(
        "gpustack.policies.worker_filters.backend_framework_filter.list_service_runners",
        return_value=[],
    ), patch(
        "gpustack.policies.worker_filters.backend_framework_filter.async_session"
    ) as mock_async_session, patch(
        "gpustack.policies.worker_filters.backend_framework_filter.InferenceBackend.all",
        return_value=[],
    ):
        mock_async_session.return_value.__aenter__.return_value = SimpleNamespace()
        filtered_workers, messages = await BackendFrameworkFilter(model).filter(
            [direct_process_worker, standard_worker]
        )

    assert [worker.name for worker in filtered_workers] == ["host-4-4080"]
    assert len(messages) == 1
    assert "container-worker" in messages[0]


# ── new capability-based tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_capability_supported_backend_passes_filter():
    """Worker advertises vLLM capability — vLLM model passes the filter."""
    model = create_model(backend="vLLM", backend_version="0.13.0")
    worker = linux_nvidia_4_4080_16gx4()
    _enable_direct_process(worker, backends="vLLM")

    filtered_workers, messages = await BackendFrameworkFilter(model).filter([worker])

    assert len(filtered_workers) == 1
    assert filtered_workers[0].name == "host-4-4080"
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_capability_unsupported_backend_rejected():
    """Worker advertises vLLM but model requests MindIE — rejected."""
    model = create_model(backend="MindIE")
    worker = linux_nvidia_4_4080_16gx4()
    _enable_direct_process(worker, backends="vLLM")

    filtered_workers, messages = await BackendFrameworkFilter(model).filter([worker])

    assert len(filtered_workers) == 0
    assert len(messages) == 1
    assert "vLLM" in messages[0]
    assert "MindIE" in messages[0]


@pytest.mark.asyncio
async def test_capability_missing_backends_label_fails_safe():
    """Worker has direct-process mode enabled but no backends label — fails safe."""
    model = create_model(backend="vLLM", backend_version="0.13.0")
    worker = linux_nvidia_4_4080_16gx4()
    # Only set the mode label, no backends label
    _enable_direct_process(worker, backends=None)

    filtered_workers, messages = await BackendFrameworkFilter(model).filter([worker])

    assert len(filtered_workers) == 0
    assert len(messages) == 1
    assert "does not advertise direct-process support" in messages[0]
    assert "vLLM" in messages[0]


@pytest.mark.asyncio
async def test_capability_multi_backend_worker_passes_for_each():
    """Worker advertises multiple backends — model requesting any of them passes."""
    for backend_name in ["vLLM", "SGLang"]:
        model = create_model(backend=backend_name)
        worker = linux_nvidia_4_4080_16gx4()
        _enable_direct_process(worker, backends="SGLang,vLLM")

        filtered_workers, messages = await BackendFrameworkFilter(model).filter(
            [worker]
        )

        assert len(filtered_workers) == 1, (
            f"Expected {backend_name} to pass but got filtered out: {messages}"
        )
        assert len(messages) == 0


@pytest.mark.asyncio
async def test_capability_multi_backend_worker_rejects_unlisted():
    """Worker advertises SGLang,vLLM but model requests VoxBox — rejected."""
    model = create_model(backend="VoxBox")
    worker = linux_nvidia_4_4080_16gx4()
    _enable_direct_process(worker, backends="SGLang,vLLM")

    filtered_workers, messages = await BackendFrameworkFilter(model).filter([worker])

    assert len(filtered_workers) == 0
    assert len(messages) == 1
    assert "VoxBox" in messages[0]


@pytest.mark.asyncio
async def test_capability_disabled_mode_not_treated_as_direct_process():
    """Worker without direct-process mode label goes through standard filter path."""
    model = create_model(backend="vLLM", backend_version="0.13.0")
    worker = linux_nvidia_4_4080_16gx4()
    # No direct-process labels at all — standard container worker

    with patch(
        "gpustack.policies.worker_filters.backend_framework_filter.list_service_runners",
        return_value=[],
    ), patch(
        "gpustack.policies.worker_filters.backend_framework_filter.async_session"
    ) as mock_async_session, patch(
        "gpustack.policies.worker_filters.backend_framework_filter.InferenceBackend.all",
        return_value=[],
    ):
        mock_async_session.return_value.__aenter__.return_value = SimpleNamespace()
        filtered_workers, messages = await BackendFrameworkFilter(model).filter(
            [worker]
        )

    # Worker goes through standard (container) path, not direct-process path
    # It gets filtered by the standard runner check (no runners available)
    assert len(filtered_workers) == 0
    assert len(messages) == 1
    # Message should be about runner/version compatibility, not direct-process
    assert "direct-process" not in messages[0].lower()
