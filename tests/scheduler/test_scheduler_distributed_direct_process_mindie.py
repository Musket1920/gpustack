import sys
import types

# Inject an fcntl stub before importing scheduler/policy modules on Windows.
if "fcntl" not in sys.modules:
    _fcntl_stub = types.ModuleType("fcntl")
    _fcntl_stub.LOCK_EX = 1  # type: ignore[attr-defined]
    _fcntl_stub.LOCK_UN = 2  # type: ignore[attr-defined]
    _fcntl_stub.lockf = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    _fcntl_stub.flock = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["fcntl"] = _fcntl_stub

from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.scheduler.scheduler import _filter_unsupported_direct_process_candidates
import pytest

from gpustack.schemas.models import (
    BackendEnum,
    ComputedResourceClaim,
    ModelInstanceSubordinateWorker,
)
from tests.fixtures.workers.fixtures import (
    linux_ascend_1_910b_64gx8,
    linux_ascend_2_910b_64gx8,
)
from tests.utils.model import new_model


def _enable_direct_process(
    worker,
    *,
    backend: str,
    bootstrap: bool = False,
    distributed: bool = False,
):
    labels = worker.labels or {}
    labels["gpustack.direct-process-mode"] = "true"
    labels["gpustack.direct-process-backends"] = backend
    if bootstrap:
        labels["gpustack.direct-process-bootstrap-backends"] = backend
    if distributed:
        labels["gpustack.direct-process-distributed-backends"] = backend
    worker.labels = labels
    return worker


def _make_multi_worker_candidate(workers):
    main_worker, subordinate_worker = workers
    return ModelInstanceScheduleCandidate(
        worker=main_worker,
        gpu_type="cann",
        gpu_indexes=[gpu.index for gpu in main_worker.status.gpu_devices],
        computed_resource_claim=ComputedResourceClaim(vram={}, ram=0),
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=subordinate_worker.id,
                worker_name=subordinate_worker.name,
                worker_ip=subordinate_worker.ip,
                worker_ifname=subordinate_worker.ifname,
                total_gpus=len(subordinate_worker.status.gpu_devices),
                gpu_type="cann",
                gpu_indexes=[gpu.index for gpu in subordinate_worker.status.gpu_devices],
                computed_resource_claim=ComputedResourceClaim(vram={}, ram=0),
            )
        ],
    )


def test_distributed_direct_process_mindie_scheduler_allows_capable_workers():
    workers = [
        _enable_direct_process(
            linux_ascend_1_910b_64gx8(),
            backend="MindIE",
            bootstrap=True,
            distributed=True,
        ),
        _enable_direct_process(
            linux_ascend_2_910b_64gx8(),
            backend="MindIE",
            bootstrap=True,
            distributed=True,
        ),
    ]
    candidate = _make_multi_worker_candidate(workers)
    model = new_model(
        1,
        "distributed-mindie",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        backend=BackendEnum.ASCEND_MINDIE,
        distributed_inference_across_workers=True,
    )

    candidates, messages = _filter_unsupported_direct_process_candidates(
        model,
        workers,
        [candidate],
    )

    assert candidates == [candidate]
    assert messages == []


def test_distributed_direct_process_mindie_scheduler_requires_distributed_capability():
    workers = [
        _enable_direct_process(
            linux_ascend_1_910b_64gx8(),
            backend="MindIE",
            bootstrap=True,
            distributed=True,
        ),
        _enable_direct_process(
            linux_ascend_2_910b_64gx8(),
            backend="MindIE",
            bootstrap=True,
            distributed=False,
        ),
    ]
    candidate = _make_multi_worker_candidate(workers)
    model = new_model(
        1,
        "distributed-mindie",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        backend=BackendEnum.ASCEND_MINDIE,
        distributed_inference_across_workers=True,
    )

    candidates, messages = _filter_unsupported_direct_process_candidates(
        model,
        workers,
        [candidate],
    )

    assert candidates == []
    assert any("distributed MindIE support" in message for message in messages)
    assert any(workers[1].name in message for message in messages)


def test_distributed_direct_process_scheduler_rejects_unsupported_backend():
    workers = [
        _enable_direct_process(
            linux_ascend_1_910b_64gx8(),
            backend="VoxBox",
            bootstrap=True,
            distributed=True,
        ),
        _enable_direct_process(
            linux_ascend_2_910b_64gx8(),
            backend="VoxBox",
            bootstrap=True,
            distributed=True,
        ),
    ]
    candidate = _make_multi_worker_candidate(workers)
    model = new_model(
        1,
        "distributed-unsupported",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        backend=BackendEnum.VOX_BOX,
        distributed_inference_across_workers=True,
    )

    candidates, messages = _filter_unsupported_direct_process_candidates(
        model,
        workers,
        [candidate],
    )

    assert candidates == []
    assert any("contract-declared backends" in message for message in messages)
    assert any("VoxBox" in message for message in messages)
