from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gpustack import envs
from gpustack.scheduler.evaluator import evaluate_model_metadata
from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.policies.worker_filters.status_filter import StatusFilter
from tests.utils.model import new_model
from gpustack.scheduler.scheduler import (
    _prepare_workers_for_new_placements,
    evaluate_pretrained_config,
    find_candidate,
)
from gpustack.schemas.models import CategoryEnum, BackendEnum, ComputedResourceClaim
from gpustack.schemas.workers import (
    Worker,
    WorkerControlChannelEnum,
    WorkerReachabilityCapabilities,
    WorkerReachabilityModeEnum,
    WorkerSession,
    WorkerSessionStateEnum,
    WorkerStateEnum,
    WorkerStatus,
)
from gpustack.server.worker_reachability import evaluate_worker_reachability
from tests.utils.mock import mock_async_session


@pytest.mark.parametrize(
    "case_name, model, expect_error, expect_error_match, expect_categories",
    [
        (
            "custom_code_without_trust_remote_code",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="microsoft/Phi-4-multimodal-instruct",
                backend=BackendEnum.VLLM,
                backend_parameters=[],
            ),
            ValueError,
            "The model contains custom code that must be executed to load correctly. If you trust the source, please pass the backend parameter `--trust-remote-code` to allow custom code to be run.",
            None,
        ),
        (
            "custom_code_with_trust_remote_code",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="microsoft/Phi-4-multimodal-instruct",
                backend=BackendEnum.VLLM,
                backend_parameters=["--trust-remote-code"],
            ),
            None,
            None,
            ["LLM"],
        ),
        (
            "unsupported_architecture",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="google-t5/t5-base",
                backend=BackendEnum.VLLM,
                backend_parameters=[],
            ),
            ValueError,
            "Unsupported architecture:",
            None,
        ),
        (
            "pass_unsupported_architecture_custom_backend",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="google-t5/t5-base",
                backend=BackendEnum.CUSTOM,
                backend_parameters=[],
            ),
            None,
            None,
            None,
        ),
        (
            "pass_unsupported_architecture_custom_backend_version",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="google-t5/t5-base",
                backend=BackendEnum.VLLM,
                backend_version="custom_version",
                backend_parameters=[],
            ),
            None,
            None,
            None,
        ),
        (
            "supported_architecture",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
                backend=BackendEnum.VLLM,
                backend_parameters=[],
            ),
            None,
            None,
            ["LLM"],
        ),
        (
            "pass_import_error_in_pretrained_config",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="deepseek-ai/DeepSeek-OCR",
                backend=BackendEnum.VLLM,
                backend_parameters=["--trust-remote-code"],
            ),
            None,
            None,
            ["LLM"],
        ),
        (
            "pass_image_model",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="Tongyi-MAI/Z-Image-Turbo",
                backend=BackendEnum.SGLANG,
                backend_parameters=[],
                categories=[CategoryEnum.IMAGE],
            ),
            None,
            None,
            ["IMAGE"],
        ),
    ],
)
@pytest.mark.asyncio
async def test_evaluate_pretrained_config(
    config, case_name, model, expect_error, expect_error_match, expect_categories
):
    try:
        if expect_error:
            with pytest.raises(expect_error, match=expect_error_match):
                await evaluate_pretrained_config(model)
        else:
            await evaluate_pretrained_config(model)
            if expect_categories:
                assert model.categories == [CategoryEnum[c] for c in expect_categories]
    except AssertionError as e:
        raise AssertionError(f"Test case '{case_name}' failed: {e}") from e


@pytest.mark.parametrize(
    "case_name, model, expect_compatible, expect_error_match",
    [
        (
            "unsupported_architecture",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="google-t5/t5-base",
                backend=BackendEnum.VLLM,
                backend_parameters=[],
            ),
            False,
            [
                "Unsupported architecture: ['T5ForConditionalGeneration']. To proceed with deployment, ensure the model is supported by backend, or deploy it using a custom backend version or custom backend."
            ],
        ),
        (
            "pass_evaluation_skip",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="google-t5/t5-base",
                backend=BackendEnum.VLLM,
                backend_parameters=[],
                env={"GPUSTACK_SKIP_MODEL_EVALUATION": "1"},
            ),
            True,
            [],
        ),
    ],
)
@pytest.mark.asyncio
async def test_evaluate_model_metadata(
    config, case_name, model, expect_compatible, expect_error_match
):
    try:
        actual_compatible, actual_error = await evaluate_model_metadata(
            config, model, []
        )
        assert (
            actual_compatible == expect_compatible
        ), f"Expected compatibility: {expect_compatible}, but got: {actual_compatible}. Error: {actual_error}"
        assert (
            expect_error_match == actual_error
        ), f"Expected error message: {expect_error_match}, but got: {actual_error}"
    except AssertionError as e:
        raise AssertionError(f"Test case '{case_name}' failed: {e}") from e


@pytest.mark.asyncio
async def test_transport_timeout_blocks_scheduling(monkeypatch):
    monkeypatch.setattr(envs, "WORKER_CONTROL_SESSION_LOSS_TIMEOUT_SECONDS", 30)

    worker = Worker(
        id=1,
        name="ws-worker-stale",
        labels={},
        cluster_id=1,
        hostname="test-host",
        ip="192.168.1.100",
        ifname="eth0",
        port=8080,
        worker_uuid="ws-worker-stale-uuid",
        status=WorkerStatus.get_default_status(),
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
    )

    decision = evaluate_worker_reachability(worker, None)
    worker.compute_state()

    candidates, messages = await StatusFilter(
        new_model(1, "test-model", 1, huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
    ).filter([worker])

    assert decision.reverse_probe_required is True
    assert decision.transport_timed_out is True
    assert decision.transport_message is not None
    assert worker.state == WorkerStateEnum.NOT_READY
    assert worker.state_message == decision.transport_message
    assert candidates == []
    assert messages == ["Matched 0/1 workers by READY status."]


@pytest.mark.asyncio
async def test_ws_worker_not_schedulable_when_disconnected(config, monkeypatch):
    monkeypatch.setattr(envs, "WORKER_CONTROL_SESSION_LOSS_TIMEOUT_SECONDS", 30)

    ws_worker = Worker(
        id=1,
        name="ws-worker-disconnected",
        labels={},
        cluster_id=1,
        hostname="ws-host",
        ip="192.168.1.10",
        ifname="eth0",
        port=8080,
        worker_uuid="ws-worker-disconnected-uuid",
        state=WorkerStateEnum.READY,
        status=WorkerStatus.get_default_status(),
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
    )
    legacy_worker = Worker(
        id=2,
        name="legacy-worker-ready",
        labels={},
        cluster_id=1,
        hostname="legacy-host",
        ip="192.168.1.11",
        ifname="eth0",
        port=8080,
        worker_uuid="legacy-worker-ready-uuid",
        state=WorkerStateEnum.READY,
        status=WorkerStatus.get_default_status(),
        reachability_mode=WorkerReachabilityModeEnum.REVERSE_PROBE,
    )
    model = new_model(
        1,
        "test-model",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend=BackendEnum.VLLM,
    )

    class FakeSelector:
        def __init__(self, *_args, **_kwargs):
            pass

        async def select_candidates(self, workers):
            return [
                ModelInstanceScheduleCandidate(
                    worker=worker,
                    gpu_indexes=[0],
                    computed_resource_claim=ComputedResourceClaim(ram=1, vram={0: 1}),
                    gpu_type="cuda",
                    score=1,
                )
                for worker in workers
            ]

        def get_messages(self):
            return []

    async def _no_active_sessions(_session, _worker_ids, now=None):
        return {}

    async def _score_identity(self, candidates):
        return candidates

    async def _filter_identity(self, workers):
        return workers, []

    with (
        patch.dict("sys.modules", {"fcntl": MagicMock()}),
        patch(
            "gpustack.scheduler.scheduler.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.scheduler.scheduler.fetch_active_control_sessions_by_worker_id",
            new=_no_active_sessions,
        ),
        patch(
            "gpustack.policies.candidate_selectors.VLLMResourceFitSelector",
            FakeSelector,
        ),
        patch(
            "gpustack.scheduler.scheduler.CandidateScoreChain.score",
            new=_score_identity,
        ),
        patch(
            "gpustack.scheduler.scheduler.WorkerFilterChain.filter",
            new=_filter_identity,
        ),
    ):
        candidate, messages = await find_candidate(
            config,
            model,
            [ws_worker, legacy_worker],
            [],
        )

    assert candidate is not None
    assert candidate.worker.name == "legacy-worker-ready"
    assert ws_worker.state == WorkerStateEnum.NOT_READY
    assert ws_worker.should_block_new_placements() is True
    assert messages == []


@pytest.mark.asyncio
async def test_ws_worker_schedulable_with_active_control_session(config, monkeypatch):
    ws_worker = Worker(
        id=1,
        name="ws-worker-nat-manageable",
        labels={},
        cluster_id=1,
        hostname="ws-host",
        ip="192.168.1.10",
        ifname="eth0",
        port=8080,
        worker_uuid="ws-worker-nat-manageable-uuid",
        status=WorkerStatus.get_default_status(),
        heartbeat_time=datetime.now(timezone.utc),
        capabilities=WorkerReachabilityCapabilities(outbound_control_ws=True),
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
    )
    active_session = WorkerSession(
        session_id="session-active-nat",
        worker_id=1,
        generation=1,
        control_channel=WorkerControlChannelEnum.OUTBOUND_CONTROL_WS,
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        state=WorkerSessionStateEnum.ACTIVE,
        connected_at=datetime.now(timezone.utc),
        last_seen_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    decision = evaluate_worker_reachability(ws_worker, active_session)
    ws_worker.compute_state()

    assert decision.reverse_probe_required is False
    assert ws_worker.state == WorkerStateEnum.READY
    assert ws_worker.should_block_new_placements() is False

    async def _active_sessions(_session, _worker_ids, now=None):
        return {ws_worker.id: active_session}

    with (
        patch(
            "gpustack.scheduler.scheduler.async_session",
            return_value=mock_async_session(),
        ),
        patch(
            "gpustack.scheduler.scheduler.fetch_active_control_sessions_by_worker_id",
            new=_active_sessions,
        ),
    ):
        prepared_workers = await _prepare_workers_for_new_placements([ws_worker])

    assert prepared_workers == [ws_worker]
