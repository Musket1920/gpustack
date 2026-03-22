from unittest.mock import patch
from types import SimpleNamespace

import pytest

from gpustack.policies.worker_filters.backend_framework_filter import (
    BackendFrameworkFilter,
    DIRECT_PROCESS_MODE_LABEL,
)
from gpustack.schemas.models import BackendEnum
from tests.fixtures.workers.fixtures import linux_nvidia_4_4080_16gx4
from tests.policies.worker_filters.test_backend_framework_filter import (
    create_model,
)


@pytest.mark.asyncio
async def test_direct_process_backend_version_vllm_linux_worker_bypasses_runner_filter(
):
    model = create_model(backend="vLLM", backend_version="0.13.0")
    worker = linux_nvidia_4_4080_16gx4()
    assert worker.labels is not None
    worker.labels[DIRECT_PROCESS_MODE_LABEL] = "true"
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
    model = create_model(backend=BackendEnum.SGLANG, backend_version="0.4.0")
    worker = linux_nvidia_4_4080_16gx4()
    assert worker.labels is not None
    worker.labels[DIRECT_PROCESS_MODE_LABEL] = "true"
    workers = [worker]

    filtered_workers, messages = await BackendFrameworkFilter(model).filter(workers)

    assert len(filtered_workers) == 0
    assert len(messages) == 1
    assert "supports only the vLLM backend" in messages[0]
    assert "SGLang" in messages[0]


@pytest.mark.asyncio
async def test_direct_process_rejects_non_linux_worker_with_clear_message():
    model = create_model(backend="vLLM", backend_version="0.13.0")
    worker = linux_nvidia_4_4080_16gx4()
    assert worker.labels is not None
    worker.labels["os"] = "windows"
    worker.labels[DIRECT_PROCESS_MODE_LABEL] = "true"

    filtered_workers, messages = await BackendFrameworkFilter(model).filter([worker])

    assert len(filtered_workers) == 0
    assert len(messages) == 1
    assert "Direct process mode is supported only on Linux workers." in messages[0]


@pytest.mark.asyncio
async def test_direct_process_rejects_multi_worker_vllm_path_with_clear_message(
):
    model = create_model(
        backend="vLLM",
        backend_version="0.13.0",
        backend_parameters=["--tensor-parallel-size", "8"],
    )
    worker = linux_nvidia_4_4080_16gx4()
    assert worker.labels is not None
    worker.labels[DIRECT_PROCESS_MODE_LABEL] = "true"
    workers = [worker]

    filtered_workers, messages = await BackendFrameworkFilter(model).filter(workers)

    assert len(filtered_workers) == 0
    assert len(messages) == 1
    assert "supports only single-worker launches" in messages[0]


@pytest.mark.asyncio
async def test_direct_process_filter_is_worker_scoped_for_mixed_fleets():
    model = create_model(backend="vLLM", backend_version="0.13.0")
    direct_process_worker = linux_nvidia_4_4080_16gx4()
    standard_worker = linux_nvidia_4_4080_16gx4()

    assert direct_process_worker.labels is not None
    direct_process_worker.labels[DIRECT_PROCESS_MODE_LABEL] = "true"
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
