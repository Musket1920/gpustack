import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# Inject an fcntl stub before importing any gpustack.worker module so that
# gpustack.worker.__init__ → gpustack.utils.locks → fcntl does not fail on
# Windows where fcntl is unavailable.
if "fcntl" not in sys.modules:
    _fcntl_stub = types.ModuleType("fcntl")
    _fcntl_stub.LOCK_EX = 1  # type: ignore[attr-defined]
    _fcntl_stub.LOCK_UN = 2  # type: ignore[attr-defined]
    _fcntl_stub.lockf = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    _fcntl_stub.flock = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["fcntl"] = _fcntl_stub

from gpustack.config.config import Config
from gpustack.schemas.workers import (
    DirectProcessCapabilities,
    DIRECT_PROCESS_BACKENDS_LABEL,
    DIRECT_PROCESS_DISTRIBUTED_BACKENDS_LABEL,
    DIRECT_PROCESS_CUSTOM_CONTRACT_LABEL,
)
from gpustack.worker.worker_manager import (
    DIRECT_PROCESS_MODE_LABEL,
    WorkerManager,
)


def make_config(tmp_path: Path, direct_process_mode: bool) -> Config:
    return Config(
        token="test",
        jwt_secret_key="test",
        data_dir=str(tmp_path),
        server_url="http://127.0.0.1:30080",
        direct_process_mode=direct_process_mode,
        distributed_direct_process_vllm=False,
        worker_name="worker-1",
    )


# ---------------------------------------------------------------------------
# Existing characterization tests (preserved from task 1)
# ---------------------------------------------------------------------------


def test_builtin_labels_include_direct_process_signal_when_enabled(tmp_path: Path):
    manager = object.__new__(WorkerManager)
    manager._cfg = make_config(tmp_path, direct_process_mode=True)

    labels = manager._ensure_builtin_labels()

    assert labels[DIRECT_PROCESS_MODE_LABEL] == "true"


def test_builtin_labels_omit_direct_process_signal_when_disabled(tmp_path: Path):
    manager = object.__new__(WorkerManager)
    manager._cfg = make_config(tmp_path, direct_process_mode=False)

    labels = manager._ensure_builtin_labels()

    assert DIRECT_PROCESS_MODE_LABEL not in labels


# ---------------------------------------------------------------------------
# Task 3: Capability advertisement — label production
# ---------------------------------------------------------------------------


class TestDirectProcessCapabilityLabels:
    """Verify that _ensure_builtin_labels() populates per-backend capability
    labels when direct_process_mode is enabled."""

    def test_direct_process_enabled_includes_backends_label(self, tmp_path: Path):
        manager = object.__new__(WorkerManager)
        manager._cfg = make_config(tmp_path, direct_process_mode=True)

        labels = manager._ensure_builtin_labels()

        # The backends label must be present and contain at least vLLM
        assert DIRECT_PROCESS_BACKENDS_LABEL in labels
        backends = labels[DIRECT_PROCESS_BACKENDS_LABEL].split(",")
        assert "vLLM" in backends

    def test_direct_process_disabled_omits_capability_labels(self, tmp_path: Path):
        manager = object.__new__(WorkerManager)
        manager._cfg = make_config(tmp_path, direct_process_mode=False)

        labels = manager._ensure_builtin_labels()

        assert DIRECT_PROCESS_BACKENDS_LABEL not in labels
        assert DIRECT_PROCESS_DISTRIBUTED_BACKENDS_LABEL not in labels
        assert DIRECT_PROCESS_CUSTOM_CONTRACT_LABEL not in labels

    def test_distributed_backends_label_empty_when_flag_disabled(
        self, tmp_path: Path
    ):
        """The distributed label is omitted until the feature flag is enabled."""
        manager = object.__new__(WorkerManager)
        manager._cfg = make_config(tmp_path, direct_process_mode=True)

        labels = manager._ensure_builtin_labels()

        assert DIRECT_PROCESS_DISTRIBUTED_BACKENDS_LABEL not in labels

    def test_custom_contract_label_present_after_task_6(self, tmp_path: Path):
        """Custom contract support is enabled (task 6 implemented)."""
        manager = object.__new__(WorkerManager)
        manager._cfg = make_config(tmp_path, direct_process_mode=True)

        labels = manager._ensure_builtin_labels()

        assert labels[DIRECT_PROCESS_CUSTOM_CONTRACT_LABEL] == "true"

    def test_capability_labels_coexist_with_mode_label(self, tmp_path: Path):
        """The coarse mode label and the fine-grained capability labels must
        both be present when direct-process mode is enabled."""
        manager = object.__new__(WorkerManager)
        manager._cfg = make_config(tmp_path, direct_process_mode=True)

        labels = manager._ensure_builtin_labels()

        assert labels[DIRECT_PROCESS_MODE_LABEL] == "true"
        assert DIRECT_PROCESS_BACKENDS_LABEL in labels

    def test_backends_label_sorted_deterministic(self, tmp_path: Path):
        """When multiple backends are supported, the label value must be
        sorted for deterministic comparison."""
        fake_single = frozenset({"vLLM", "SGLang", "VoxBox"})
        manager = object.__new__(WorkerManager)
        manager._cfg = make_config(tmp_path, direct_process_mode=True)

        with patch(
            "gpustack.worker.worker_manager.get_direct_process_supported_backends",
            return_value=fake_single,
        ):
            labels = manager._ensure_builtin_labels()

        assert labels[DIRECT_PROCESS_BACKENDS_LABEL] == "SGLang,VoxBox,vLLM"

    def test_distributed_backends_label_present_when_registry_nonempty(
        self, tmp_path: Path
    ):
        """The feature flag enables distributed vLLM capability labels."""
        manager = object.__new__(WorkerManager)
        manager._cfg = make_config(tmp_path, direct_process_mode=True)
        manager._cfg.distributed_direct_process_vllm = True

        labels = manager._ensure_builtin_labels()

        assert labels[DIRECT_PROCESS_DISTRIBUTED_BACKENDS_LABEL] == "vLLM"


# ---------------------------------------------------------------------------
# Task 3: DirectProcessCapabilities — from_labels parsing
# ---------------------------------------------------------------------------


class TestDirectProcessCapabilitiesFromLabels:
    """Verify that DirectProcessCapabilities.from_labels() correctly parses
    worker labels into a structured capability object."""

    def test_none_labels_returns_disabled(self):
        caps = DirectProcessCapabilities.from_labels(None)
        assert caps.enabled is False
        assert caps.single_worker_backends == []
        assert caps.distributed_backends == []
        assert caps.custom_contract_support is False

    def test_empty_labels_returns_disabled(self):
        caps = DirectProcessCapabilities.from_labels({})
        assert caps.enabled is False

    def test_mode_label_false_returns_disabled(self):
        labels = {DIRECT_PROCESS_MODE_LABEL: "false"}
        caps = DirectProcessCapabilities.from_labels(labels)
        assert caps.enabled is False
        assert caps.single_worker_backends == []

    def test_mode_label_true_without_backends_returns_enabled_empty(self):
        labels = {DIRECT_PROCESS_MODE_LABEL: "true"}
        caps = DirectProcessCapabilities.from_labels(labels)
        assert caps.enabled is True
        assert caps.single_worker_backends == []
        assert caps.distributed_backends == []

    def test_full_capability_labels_parsed(self):
        labels = {
            DIRECT_PROCESS_MODE_LABEL: "true",
            DIRECT_PROCESS_BACKENDS_LABEL: "SGLang,VoxBox,vLLM",
            DIRECT_PROCESS_DISTRIBUTED_BACKENDS_LABEL: "vLLM",
            DIRECT_PROCESS_CUSTOM_CONTRACT_LABEL: "true",
        }
        caps = DirectProcessCapabilities.from_labels(labels)
        assert caps.enabled is True
        assert caps.single_worker_backends == ["SGLang", "VoxBox", "vLLM"]
        assert caps.distributed_backends == ["vLLM"]
        assert caps.custom_contract_support is True

    def test_backends_with_whitespace_trimmed(self):
        labels = {
            DIRECT_PROCESS_MODE_LABEL: "1",
            DIRECT_PROCESS_BACKENDS_LABEL: " vLLM , SGLang ",
        }
        caps = DirectProcessCapabilities.from_labels(labels)
        assert caps.single_worker_backends == ["vLLM", "SGLang"]

    def test_mode_label_yes_accepted(self):
        labels = {DIRECT_PROCESS_MODE_LABEL: "yes"}
        caps = DirectProcessCapabilities.from_labels(labels)
        assert caps.enabled is True

    def test_mode_label_on_accepted(self):
        labels = {DIRECT_PROCESS_MODE_LABEL: "on"}
        caps = DirectProcessCapabilities.from_labels(labels)
        assert caps.enabled is True

    def test_mode_label_1_accepted(self):
        labels = {DIRECT_PROCESS_MODE_LABEL: "1"}
        caps = DirectProcessCapabilities.from_labels(labels)
        assert caps.enabled is True


# ---------------------------------------------------------------------------
# Task 3: DirectProcessCapabilities — convenience queries
# ---------------------------------------------------------------------------


class TestDirectProcessCapabilitiesQueries:
    """Verify supports_backend() and supports_distributed_backend() helpers."""

    def test_supports_backend_true(self):
        caps = DirectProcessCapabilities(
            enabled=True,
            single_worker_backends=["vLLM", "SGLang"],
        )
        assert caps.supports_backend("vLLM") is True
        assert caps.supports_backend("SGLang") is True

    def test_supports_backend_false_for_unlisted(self):
        caps = DirectProcessCapabilities(
            enabled=True,
            single_worker_backends=["vLLM"],
        )
        assert caps.supports_backend("MindIE") is False

    def test_supports_backend_false_when_disabled(self):
        caps = DirectProcessCapabilities(
            enabled=False,
            single_worker_backends=["vLLM"],
        )
        assert caps.supports_backend("vLLM") is False

    def test_supports_distributed_backend_true(self):
        caps = DirectProcessCapabilities(
            enabled=True,
            distributed_backends=["vLLM"],
        )
        assert caps.supports_distributed_backend("vLLM") is True

    def test_supports_distributed_backend_false_for_unlisted(self):
        caps = DirectProcessCapabilities(
            enabled=True,
            distributed_backends=["vLLM"],
        )
        assert caps.supports_distributed_backend("SGLang") is False

    def test_supports_distributed_backend_false_when_disabled(self):
        caps = DirectProcessCapabilities(
            enabled=False,
            distributed_backends=["vLLM"],
        )
        assert caps.supports_distributed_backend("vLLM") is False


# ---------------------------------------------------------------------------
# Task 3: DirectProcessCapabilities — to_labels round-trip
# ---------------------------------------------------------------------------


class TestDirectProcessCapabilitiesToLabels:
    """Verify to_labels() produces the expected flat label dict."""

    def test_empty_capabilities_produce_no_labels(self):
        caps = DirectProcessCapabilities(enabled=True)
        assert caps.to_labels() == {}

    def test_single_backends_produce_sorted_label(self):
        caps = DirectProcessCapabilities(
            enabled=True,
            single_worker_backends=["VoxBox", "vLLM", "SGLang"],
        )
        labels = caps.to_labels()
        assert labels[DIRECT_PROCESS_BACKENDS_LABEL] == "SGLang,VoxBox,vLLM"

    def test_distributed_backends_produce_label(self):
        caps = DirectProcessCapabilities(
            enabled=True,
            distributed_backends=["vLLM"],
        )
        labels = caps.to_labels()
        assert labels[DIRECT_PROCESS_DISTRIBUTED_BACKENDS_LABEL] == "vLLM"

    def test_custom_contract_produces_label(self):
        caps = DirectProcessCapabilities(
            enabled=True,
            custom_contract_support=True,
        )
        labels = caps.to_labels()
        assert labels[DIRECT_PROCESS_CUSTOM_CONTRACT_LABEL] == "true"

    def test_round_trip_from_labels_to_labels(self):
        """Labels produced by to_labels() should round-trip through from_labels()."""
        original = DirectProcessCapabilities(
            enabled=True,
            single_worker_backends=["SGLang", "vLLM"],
            distributed_backends=["vLLM"],
            custom_contract_support=True,
        )
        labels = original.to_labels()
        # Add the mode label (not produced by to_labels)
        labels[DIRECT_PROCESS_MODE_LABEL] = "true"

        restored = DirectProcessCapabilities.from_labels(labels)
        assert restored.enabled is True
        assert sorted(restored.single_worker_backends) == ["SGLang", "vLLM"]
        assert restored.distributed_backends == ["vLLM"]
        assert restored.custom_contract_support is True


# ---------------------------------------------------------------------------
# Task 3: Negative / fail-safe cases
# ---------------------------------------------------------------------------


class TestDirectProcessCapabilitiesFailSafe:
    """Missing or malformed capability data must fail safe (no capability)."""

    def test_missing_mode_label_means_no_capability(self):
        """A worker without the mode label has no direct-process capability,
        even if backend labels are somehow present."""
        labels = {
            DIRECT_PROCESS_BACKENDS_LABEL: "vLLM",
        }
        caps = DirectProcessCapabilities.from_labels(labels)
        assert caps.enabled is False
        assert caps.single_worker_backends == []

    def test_garbage_mode_label_means_no_capability(self):
        labels = {DIRECT_PROCESS_MODE_LABEL: "maybe"}
        caps = DirectProcessCapabilities.from_labels(labels)
        assert caps.enabled is False

    def test_empty_backends_string_means_no_backends(self):
        labels = {
            DIRECT_PROCESS_MODE_LABEL: "true",
            DIRECT_PROCESS_BACKENDS_LABEL: "",
        }
        caps = DirectProcessCapabilities.from_labels(labels)
        assert caps.enabled is True
        assert caps.single_worker_backends == []

    def test_comma_only_backends_string_means_no_backends(self):
        labels = {
            DIRECT_PROCESS_MODE_LABEL: "true",
            DIRECT_PROCESS_BACKENDS_LABEL: ",,,",
        }
        caps = DirectProcessCapabilities.from_labels(labels)
        assert caps.enabled is True
        assert caps.single_worker_backends == []

    def test_supports_backend_on_default_instance_is_false(self):
        """Default-constructed capabilities should deny everything."""
        caps = DirectProcessCapabilities()
        assert caps.supports_backend("vLLM") is False
        assert caps.supports_distributed_backend("vLLM") is False
        assert caps.custom_contract_support is False
