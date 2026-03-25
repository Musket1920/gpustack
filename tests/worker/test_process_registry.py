import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest

from gpustack.config.config import Config
from gpustack.schemas.models import BackendEnum, ModelInstanceStateEnum
from tests.worker.fake_backend_fixture import fake_backend_fixture


PROCESS_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "gpustack"
    / "worker"
    / "process_registry.py"
)
process_registry_spec = importlib.util.spec_from_file_location(
    "tests.worker.process_registry_module",
    PROCESS_REGISTRY_PATH,
)
assert process_registry_spec is not None
assert process_registry_spec.loader is not None
process_registry = importlib.util.module_from_spec(process_registry_spec)
process_registry_spec.loader.exec_module(process_registry)
DIRECT_PROCESS_RUNTIME_MODE = process_registry.DIRECT_PROCESS_RUNTIME_MODE
DIRECT_PROCESS_UNHEALTHY_MESSAGE = process_registry.DIRECT_PROCESS_UNHEALTHY_MESSAGE
DirectProcessEntryStatus = process_registry.DirectProcessEntryStatus
DirectProcessRegistry = process_registry.DirectProcessRegistry
DirectProcessRuntimeState = process_registry.DirectProcessRuntimeState
get_process_group_id = process_registry.get_process_group_id
map_direct_process_state_transition = process_registry.map_direct_process_state_transition
utcnow = process_registry.utcnow


def make_config(tmp_path: Path) -> Config:
    return Config(
        token="test",
        jwt_secret_key="test",
        data_dir=str(tmp_path),
        server_url="http://127.0.0.1:30080",
    )


def test_process_registry_live_entry_persists_and_removes(
    fake_backend_fixture,
    tmp_path: Path,
):
    cfg = make_config(tmp_path)
    registry = DirectProcessRegistry(cfg)
    handle = fake_backend_fixture.start_ready_server()
    log_path = tmp_path / "serve" / "1.log"

    stored = registry.upsert(
        model_instance_id=1,
        deployment_name="test-instance",
        pid=handle.process.pid,
        process_group_id=get_process_group_id(handle.process.pid),
        port=handle.port,
        log_path=str(log_path),
        backend=BackendEnum.VLLM,
        mode=DIRECT_PROCESS_RUNTIME_MODE,
    )

    reloaded_registry = DirectProcessRegistry(cfg)
    reloaded = reloaded_registry.get_by_model_instance_id(1)
    liveness = reloaded_registry.inspect_by_deployment_name("test-instance")
    transition = map_direct_process_state_transition(
        ModelInstanceStateEnum.STARTING,
        liveness,
        is_ready=True,
    )

    assert reloaded is not None
    assert reloaded.model_instance_id == 1
    assert reloaded.deployment_name == "test-instance"
    assert reloaded.pid == handle.process.pid
    assert reloaded.process_group_id == stored.process_group_id
    assert reloaded.port == handle.port
    assert reloaded.log_path == str(log_path)
    assert reloaded.backend == BackendEnum.VLLM
    assert reloaded.mode == DIRECT_PROCESS_RUNTIME_MODE
    assert reloaded.updated_at >= reloaded.created_at
    assert liveness.status == DirectProcessEntryStatus.LIVE
    assert transition.runtime_state == DirectProcessRuntimeState.RUNNING
    assert transition.next_state == ModelInstanceStateEnum.RUNNING
    assert transition.state_message == ""

    removed = reloaded_registry.remove_by_model_instance_id(1)

    assert removed is not None
    assert DirectProcessRegistry(cfg).get_by_model_instance_id(1) is None


def test_process_registry_stale_entry_maps_to_error(fake_backend_fixture, tmp_path: Path):
    cfg = make_config(tmp_path)
    registry = DirectProcessRegistry(cfg)
    handle = fake_backend_fixture.start_ready_server()

    registry.upsert(
        model_instance_id=2,
        deployment_name="stale-instance",
        pid=handle.process.pid,
        process_group_id=get_process_group_id(handle.process.pid),
        port=handle.port,
        log_path=str(tmp_path / "serve" / "2.log"),
        backend=BackendEnum.VLLM,
        mode=DIRECT_PROCESS_RUNTIME_MODE,
    )

    handle.terminate()
    handle.process.wait(timeout=5)

    stale_entry = registry.inspect_by_model_instance_id(2)
    transition = map_direct_process_state_transition(
        ModelInstanceStateEnum.RUNNING,
        stale_entry,
        is_ready=False,
    )

    assert stale_entry.status == DirectProcessEntryStatus.STALE
    assert stale_entry.reason == "pid_not_running"
    assert transition.runtime_state == DirectProcessRuntimeState.STALE
    assert transition.next_state == ModelInstanceStateEnum.ERROR
    assert transition.state_message == DIRECT_PROCESS_UNHEALTHY_MESSAGE


def test_process_registry_live_entry_not_ready_keeps_transitional_state(
    fake_backend_fixture,
    tmp_path: Path,
):
    registry = DirectProcessRegistry(make_config(tmp_path))
    handle = fake_backend_fixture.start_ready_server()

    registry.upsert(
        model_instance_id=3,
        deployment_name="starting-instance",
        pid=handle.process.pid,
        process_group_id=get_process_group_id(handle.process.pid),
        port=handle.port,
        log_path=str(tmp_path / "serve" / "3.log"),
        backend=BackendEnum.VLLM,
        mode=DIRECT_PROCESS_RUNTIME_MODE,
    )

    live_entry = registry.inspect_by_model_instance_id(3)
    transition = map_direct_process_state_transition(
        ModelInstanceStateEnum.DOWNLOADING,
        live_entry,
        is_ready=False,
    )

    assert live_entry.status == DirectProcessEntryStatus.LIVE
    assert transition.runtime_state == DirectProcessRuntimeState.STARTING
    assert transition.next_state is None
    assert transition.state_message is None


def test_process_registry_ignores_corrupt_json(tmp_path: Path):
    registry = DirectProcessRegistry(make_config(tmp_path))
    registry.path.write_text('{"entries": [', encoding="utf-8")

    assert registry.list_entries() == []
    assert registry.get_by_model_instance_id(1) is None


def test_process_registry_ignores_invalid_entry_payload(tmp_path: Path):
    registry = DirectProcessRegistry(make_config(tmp_path))
    registry.path.write_text('{"entries": [{"model_instance_id": 1}]}', encoding="utf-8")

    assert registry.list_entries() == []


def test_process_registry_live_running_not_ready_transitions_out_of_running(
    fake_backend_fixture,
    tmp_path: Path,
):
    registry = DirectProcessRegistry(make_config(tmp_path))
    handle = fake_backend_fixture.start_ready_server()

    registry.upsert(
        model_instance_id=4,
        deployment_name="running-instance",
        pid=handle.process.pid,
        process_group_id=get_process_group_id(handle.process.pid),
        port=handle.port,
        log_path=str(tmp_path / "serve" / "4.log"),
        backend=BackendEnum.VLLM,
        mode=DIRECT_PROCESS_RUNTIME_MODE,
    )

    live_entry = registry.inspect_by_model_instance_id(4)
    transition = map_direct_process_state_transition(
        ModelInstanceStateEnum.RUNNING,
        live_entry,
        is_ready=False,
    )

    assert live_entry.status == DirectProcessEntryStatus.LIVE
    assert transition.runtime_state == DirectProcessRuntimeState.STARTING
    assert transition.next_state == ModelInstanceStateEnum.STARTING
    assert transition.state_message == ""


def test_process_registry_startup_timeout_maps_to_error(tmp_path: Path):
    registry = DirectProcessRegistry(make_config(tmp_path))
    created_at = utcnow() - timedelta(seconds=10)
    entry = registry.upsert(
        model_instance_id=11,
        deployment_name="startup-timeout-instance",
        pid=1,
        port=18080,
        log_path=str(tmp_path / "serve" / "11.log"),
        backend=BackendEnum.CUSTOM,
        startup_timeout_seconds=5,
    )
    timed_out_entry = entry.model_copy(
        update={
            "created_at": created_at,
            "updated_at": created_at,
        }
    )
    status = process_registry.DirectProcessRegistryStatus(
        status=DirectProcessEntryStatus.LIVE,
        reason="pid_running",
        entry=timed_out_entry,
    )

    transition = map_direct_process_state_transition(
        ModelInstanceStateEnum.STARTING,
        status,
        is_ready=False,
        now=utcnow(),
    )

    assert transition.runtime_state == DirectProcessRuntimeState.STARTING
    assert transition.next_state == ModelInstanceStateEnum.ERROR
    assert transition.state_message == "Inference server did not become ready within 5 seconds."


def test_process_registry_persists_runtime_control_fields(tmp_path: Path):
    registry = DirectProcessRegistry(make_config(tmp_path))

    entry = registry.upsert(
        model_instance_id=12,
        deployment_name="custom-runtime-contract-instance",
        pid=1,
        port=18081,
        log_path=str(tmp_path / "serve" / "12.log"),
        backend=BackendEnum.CUSTOM,
        startup_timeout_seconds=120,
        stop_signal="SIGINT",
        stop_timeout_seconds=9,
    )
    reloaded = DirectProcessRegistry(make_config(tmp_path)).get_by_model_instance_id(12)

    assert entry.startup_timeout_seconds == 120
    assert entry.stop_signal == "SIGINT"
    assert entry.stop_timeout_seconds == 9
    assert reloaded is not None
    assert reloaded.startup_timeout_seconds == 120
    assert reloaded.stop_signal == "SIGINT"
    assert reloaded.stop_timeout_seconds == 9


# ---------------------------------------------------------------------------
# Characterization: constant values and registry contract are locked
# ---------------------------------------------------------------------------

def test_process_registry_constant_values_are_locked():
    """Characterization: runtime mode and unhealthy message strings must not silently change."""
    assert DIRECT_PROCESS_RUNTIME_MODE == "direct_process"
    assert DIRECT_PROCESS_UNHEALTHY_MESSAGE == "Inference server exited or unhealthy."


def test_process_registry_backend_stored_as_string(fake_backend_fixture, tmp_path: Path):
    """Characterization: backend enum is serialized to string in registry entries."""
    registry = DirectProcessRegistry(make_config(tmp_path))
    handle = fake_backend_fixture.start_ready_server()

    entry = registry.upsert(
        model_instance_id=10,
        deployment_name="backend-string-instance",
        pid=handle.process.pid,
        process_group_id=get_process_group_id(handle.process.pid),
        port=handle.port,
        log_path=str(tmp_path / "serve" / "10.log"),
        backend=BackendEnum.VLLM,
        mode=DIRECT_PROCESS_RUNTIME_MODE,
    )

    # Backend must be stored as a plain string, not an enum object
    assert isinstance(entry.backend, str)
    assert entry.backend == "vLLM"

    # Reload from disk and verify string is preserved
    reloaded = DirectProcessRegistry(make_config(tmp_path)).get_by_model_instance_id(10)
    assert reloaded is not None
    assert isinstance(reloaded.backend, str)
    assert reloaded.backend == "vLLM"

    handle.terminate()
    handle.process.wait(timeout=5)


def test_process_registry_missing_entry_maps_to_error(tmp_path: Path):
    """Characterization: missing registry entry transitions to ERROR state."""
    registry = DirectProcessRegistry(make_config(tmp_path))

    missing_status = registry.inspect_by_model_instance_id(999)
    transition = map_direct_process_state_transition(
        ModelInstanceStateEnum.STARTING,
        missing_status,
        is_ready=False,
    )

    assert missing_status.status == DirectProcessEntryStatus.MISSING
    assert transition.runtime_state == DirectProcessRuntimeState.MISSING
    assert transition.next_state == ModelInstanceStateEnum.ERROR
    assert transition.state_message == DIRECT_PROCESS_UNHEALTHY_MESSAGE


def test_process_registry_missing_entry_already_error_no_transition(tmp_path: Path):
    """Characterization: missing entry when already in ERROR state does not re-transition."""
    registry = DirectProcessRegistry(make_config(tmp_path))

    missing_status = registry.inspect_by_model_instance_id(998)
    transition = map_direct_process_state_transition(
        ModelInstanceStateEnum.ERROR,
        missing_status,
        is_ready=False,
    )

    assert missing_status.status == DirectProcessEntryStatus.MISSING
    assert transition.runtime_state == DirectProcessRuntimeState.MISSING
    assert transition.next_state is None
    assert transition.state_message is None


def test_process_registry_upsert_requires_complete_metadata(tmp_path: Path):
    """Characterization: upsert with missing fields raises ValueError."""
    registry = DirectProcessRegistry(make_config(tmp_path))

    with pytest.raises(ValueError, match="requires complete metadata"):
        registry.upsert(
            model_instance_id=1,
            deployment_name="incomplete",
            # pid, port, log_path, backend all missing
        )


def test_process_registry_registry_path_is_under_worker_dir(tmp_path: Path):
    """Characterization: registry file lives at data_dir/worker/direct_process_registry.json."""
    cfg = make_config(tmp_path)
    registry = DirectProcessRegistry(cfg)

    expected_path = tmp_path / "worker" / "direct_process_registry.json"
    assert registry.path == expected_path


def test_process_registry_reused_deployment_name_replaces_prior_mapping(
    fake_backend_fixture,
    tmp_path: Path,
):
    """Characterization: a deployment name is a singleton mapping to the newest direct process entry."""
    registry = DirectProcessRegistry(make_config(tmp_path))
    first_handle = fake_backend_fixture.start_ready_server()
    second_handle = fake_backend_fixture.start_ready_server()

    registry.upsert(
        model_instance_id=21,
        deployment_name="shared-deployment",
        pid=first_handle.process.pid,
        process_group_id=get_process_group_id(first_handle.process.pid),
        port=first_handle.port,
        log_path=str(tmp_path / "serve" / "21.log"),
        backend=BackendEnum.VLLM,
        mode=DIRECT_PROCESS_RUNTIME_MODE,
    )
    replacement = registry.upsert(
        model_instance_id=22,
        deployment_name="shared-deployment",
        pid=second_handle.process.pid,
        process_group_id=get_process_group_id(second_handle.process.pid),
        port=second_handle.port,
        log_path=str(tmp_path / "serve" / "22.log"),
        backend=BackendEnum.VLLM,
        mode=DIRECT_PROCESS_RUNTIME_MODE,
    )

    reloaded_registry = DirectProcessRegistry(make_config(tmp_path))
    entries = reloaded_registry.list_entries()

    assert [entry.model_instance_id for entry in entries] == [22]
    assert reloaded_registry.get_by_model_instance_id(21) is None
    assert reloaded_registry.remove_by_model_instance_id(21) is None
    assert reloaded_registry.get_by_model_instance_id(22) is not None
    assert reloaded_registry.get_by_deployment_name("shared-deployment") == replacement


def test_process_registry_remove_clears_deployment_name_lookup(fake_backend_fixture, tmp_path: Path):
    """Characterization: removing the active entry clears both model-instance and deployment-name access paths."""
    registry = DirectProcessRegistry(make_config(tmp_path))
    handle = fake_backend_fixture.start_ready_server()

    registry.upsert(
        model_instance_id=23,
        deployment_name="cleanup-deployment",
        pid=handle.process.pid,
        process_group_id=get_process_group_id(handle.process.pid),
        port=handle.port,
        log_path=str(tmp_path / "serve" / "23.log"),
        backend=BackendEnum.VLLM,
        mode=DIRECT_PROCESS_RUNTIME_MODE,
    )

    removed = registry.remove_by_model_instance_id(23)
    reloaded_registry = DirectProcessRegistry(make_config(tmp_path))
    deployment_status = reloaded_registry.inspect_by_deployment_name("cleanup-deployment")

    assert removed is not None
    assert removed.model_instance_id == 23
    assert reloaded_registry.get_by_model_instance_id(23) is None
    assert reloaded_registry.get_by_deployment_name("cleanup-deployment") is None
    assert deployment_status.status == DirectProcessEntryStatus.MISSING
