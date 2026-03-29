import sys
import types

import pytest

sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))

from gpustack import envs
from gpustack.schemas.workers import WorkerReachabilityModeEnum
from gpustack.worker.control_client import (
    worker_control_capabilities,
    worker_control_reachability_mode,
    worker_control_session_reachability_mode,
)


def test_legacy_mode_flag_disables_outbound_control(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "WORKER_CONTROL_ROLLOUT_MODE", "legacy_only")
    monkeypatch.setattr(envs, "WORKER_CONTROL_WS_ENABLED", False)
    monkeypatch.setattr(envs, "WORKER_REVERSE_HTTP_ENABLED", True)

    capabilities = worker_control_capabilities()

    assert capabilities.outbound_control_ws is False
    assert worker_control_reachability_mode() == WorkerReachabilityModeEnum.REVERSE_PROBE
    assert (
        worker_control_session_reachability_mode()
        == WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS
    )


def test_hybrid_mode_flag_prefers_legacy_reachability(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "WORKER_CONTROL_ROLLOUT_MODE", "hybrid")
    monkeypatch.setattr(envs, "WORKER_CONTROL_WS_ENABLED", True)
    monkeypatch.setattr(envs, "WORKER_REVERSE_HTTP_ENABLED", True)

    capabilities = worker_control_capabilities()

    assert capabilities.outbound_control_ws is True
    assert capabilities.reverse_http is True
    assert worker_control_reachability_mode() == WorkerReachabilityModeEnum.REVERSE_PROBE


def test_ws_preferred_mode_flag_prefers_outbound_control(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "WORKER_CONTROL_ROLLOUT_MODE", "ws_preferred")
    monkeypatch.setattr(envs, "WORKER_CONTROL_WS_ENABLED", True)
    monkeypatch.setattr(envs, "WORKER_REVERSE_HTTP_ENABLED", False)

    capabilities = worker_control_capabilities()

    assert capabilities.outbound_control_ws is True
    assert capabilities.reverse_http is False
    assert (
        worker_control_reachability_mode()
        == WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS
    )


def test_ws_preferred_can_advertise_reverse_http_when_available(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "WORKER_CONTROL_ROLLOUT_MODE", "ws_preferred")
    monkeypatch.setattr(envs, "WORKER_CONTROL_WS_ENABLED", True)
    monkeypatch.setattr(envs, "WORKER_REVERSE_HTTP_ENABLED", True)

    capabilities = worker_control_capabilities()

    assert capabilities.outbound_control_ws is True
    assert capabilities.reverse_http is True
    assert (
        worker_control_reachability_mode()
        == WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS
    )
