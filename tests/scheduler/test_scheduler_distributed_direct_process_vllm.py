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
from gpustack.schemas.models import (
    BackendEnum,
    ComputedResourceClaim,
    ModelInstanceSubordinateWorker,
)
from tests.fixtures.workers.fixtures import (
    linux_nvidia_22_H100_80gx8,
    linux_nvidia_23_H100_80gx8,
)
from tests.utils.model import new_model


def _enable_direct_process(worker, *, backend: str, distributed: bool = False):
    labels = worker.labels or {}
    labels["gpustack.direct-process-mode"] = "true"
    labels["gpustack.direct-process-backends"] = backend
    if distributed:
        labels["gpustack.direct-process-distributed-backends"] = backend
    worker.labels = labels
    return worker


def _make_multi_worker_candidate(workers):
    main_worker, subordinate_worker = workers
    return ModelInstanceScheduleCandidate(
        worker=main_worker,
        gpu_type="cuda",
        gpu_indexes=[gpu.index for gpu in main_worker.status.gpu_devices],
        computed_resource_claim=ComputedResourceClaim(vram={}, ram=0),
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=subordinate_worker.id,
                worker_name=subordinate_worker.name,
                worker_ip=subordinate_worker.ip,
                worker_ifname=subordinate_worker.ifname,
                total_gpus=len(subordinate_worker.status.gpu_devices),
                gpu_type="cuda",
                gpu_indexes=[gpu.index for gpu in subordinate_worker.status.gpu_devices],
                computed_resource_claim=ComputedResourceClaim(vram={}, ram=0),
            )
        ],
    )


def test_distributed_direct_process_vllm_scheduler_requires_capable_workers():
    workers = [
        _enable_direct_process(
            linux_nvidia_22_H100_80gx8(), backend="vLLM", distributed=True
        ),
        _enable_direct_process(
            linux_nvidia_23_H100_80gx8(), backend="vLLM", distributed=False
        ),
    ]
    model = new_model(
        1,
        "distributed-vllm",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        backend=BackendEnum.VLLM,
        distributed_inference_across_workers=True,
    )

    candidates, messages = _filter_unsupported_direct_process_candidates(
        model,
        workers,
        [_make_multi_worker_candidate(workers)],
    )

    assert candidates == []
    assert any("distributed vLLM support" in message for message in messages)
    assert any(workers[1].name in message for message in messages)


def test_distributed_direct_process_non_vllm_rejected_in_scheduler():
    workers = [
        _enable_direct_process(linux_nvidia_22_H100_80gx8(), backend="SGLang"),
        _enable_direct_process(linux_nvidia_23_H100_80gx8(), backend="SGLang"),
    ]
    model = new_model(
        1,
        "distributed-sglang",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        backend=BackendEnum.SGLANG,
        distributed_inference_across_workers=True,
    )

    candidates, messages = _filter_unsupported_direct_process_candidates(
        model,
        workers,
        [_make_multi_worker_candidate(workers)],
    )

    assert candidates == []
    assert any("only for the vLLM backend" in message for message in messages)
