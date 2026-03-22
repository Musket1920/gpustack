import importlib.util
from pathlib import Path

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
