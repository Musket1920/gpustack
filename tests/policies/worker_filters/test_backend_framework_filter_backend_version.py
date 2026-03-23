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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.policies.worker_filters.backend_framework_filter import (
    BackendFrameworkFilter,
)
from gpustack.schemas.inference_backend import VersionConfig
from tests.fixtures.workers.fixtures import linux_nvidia_4_4080_16gx4
from tests.policies.worker_filters.test_backend_framework_filter import (
    create_inference_backend,
    create_model,
)


@pytest.mark.asyncio
async def test_backend_version_container_mode_rejects_without_supported_runner(
    monkeypatch,
):
    model = create_model(backend="vLLM", backend_version="0.13.0")
    workers = [linux_nvidia_4_4080_16gx4()]
    backend = create_inference_backend(
        backend_name="vLLM",
        version_configs={
            "0.11.0": VersionConfig(
                image_name="test:0.11.0",
                run_command=None,
                entrypoint=None,
                built_in_frameworks=["cuda"],
                custom_framework="",
                env=None,
            )
        },
        is_built_in=True,
    )

    monkeypatch.delenv("GPUSTACK_DIRECT_PROCESS_MODE", raising=False)

    async def mock_session_exec(statement):
        mock_result = MagicMock()
        mock_result.first.return_value = backend
        return mock_result

    with patch(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session'
    ) as mock_async_session:
        mock_session = AsyncMock()
        mock_session.exec = mock_session_exec
        mock_async_session.return_value.__aenter__.return_value = mock_session

        with patch(
            'gpustack.policies.worker_filters.backend_framework_filter.list_service_runners',
            return_value=[],
        ):
            filtered_workers, messages = await BackendFrameworkFilter(model).filter(
                workers
            )

    assert len(filtered_workers) == 0
    assert len(messages) == 1
    assert "host-4-4080" in messages[0]
    assert "0.13.0" in messages[0]
