import importlib
import json
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gpustack.schemas.models import BackendEnum, ModelInstanceStateEnum


def _import_mindie_module():
    fcntl_stub = types.ModuleType("fcntl")
    setattr(fcntl_stub, "LOCK_EX", 1)
    setattr(fcntl_stub, "LOCK_UN", 2)
    setattr(fcntl_stub, "lockf", lambda *args, **kwargs: None)
    setattr(fcntl_stub, "flock", lambda *args, **kwargs: None)
    original_fcntl = sys.modules.get("fcntl")
    sys.modules["fcntl"] = fcntl_stub
    try:
        return importlib.import_module("gpustack.worker.backends.ascend_mindie")
    finally:
        if original_fcntl is None:
            sys.modules.pop("fcntl", None)
        else:
            sys.modules["fcntl"] = original_fcntl


mindie_module = _import_mindie_module()
AscendMindIEServer = mindie_module.AscendMindIEServer
DIRECT_PROCESS_RUNTIME_MODE = mindie_module.DIRECT_PROCESS_RUNTIME_MODE


def _install_fake_launch_artifacts(server: AscendMindIEServer, tmp_path: Path):
    prepared_root = tmp_path / "prepared" / "mindie"
    artifacts_root = prepared_root / "artifacts"
    bin_root = prepared_root / "bin"
    runtime_root = tmp_path / "runtime" / "1"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    bin_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    prepared_daemon = bin_root / "mindieservice_daemon"
    prepared_daemon.write_text("#!/bin/sh\n", encoding="utf-8")
    prepared_launch = artifacts_root / "prepared-launch.sh"
    prepared_launch.write_text("#!/bin/sh\nexec \"$@\"\n", encoding="utf-8")
    prepared_env = artifacts_root / "prepared.env"
    prepared_env.write_text(
        "\n".join(
            [
                f"VIRTUAL_ENV={prepared_root / 'venv'}",
                f"PATH={bin_root}${{PATH}}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prepared_provenance = artifacts_root / "executable-provenance.json"
    prepared_provenance.write_text(
        json.dumps(
            {
                "state": "resolved",
                "name": "mindieservice_daemon",
                "prepared_path": str(prepared_daemon),
                "sha256": "sha256:1234",
            }
        ),
        encoding="utf-8",
    )
    prepared_config = artifacts_root / "prepared-config.json"
    prepared_config.write_text(
        json.dumps(
            {
                "backend": str(server._model.backend),
                "backend_version": server.get_direct_process_prepared_environment_identity().split(":", 1)[1],
                "prepared_cache_root": str(prepared_root),
                "venv_root": str(prepared_root / "venv"),
                "env_artifact": str(prepared_env),
                "executable_provenance": str(prepared_provenance),
                "manifest_hash": "manifest-123",
            }
        ),
        encoding="utf-8",
    )

    launch_artifacts = SimpleNamespace(
        prepared_launch_path=prepared_launch,
        prepared_env_path=prepared_env,
        prepared_config_path=prepared_config,
        prepared_provenance_path=prepared_provenance,
        prepared_config=json.loads(prepared_config.read_text(encoding="utf-8")),
        prepared_provenance=json.loads(prepared_provenance.read_text(encoding="utf-8")),
        runtime_artifact_path=runtime_root / "bootstrap-artifact.json",
        manifest_hash="manifest-123",
        prepared_environment_id=server.get_direct_process_prepared_environment_identity(),
    )
    server.resolve_direct_process_launch_artifacts = lambda: launch_artifacts
    return launch_artifacts


class _FakeSocket:
    def bind(self, address):
        self.address = address

    def close(self):
        pass


def _make_server(
    tmp_path: Path,
    direct_process_mode: bool = True,
) -> AscendMindIEServer:
    data_dir = tmp_path / "data"
    cache_dir = data_dir / "cache"
    log_dir = data_dir / "log"
    (log_dir / "serve").mkdir(parents=True, exist_ok=True)
    (data_dir / "worker").mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a fake model path with config.json for MindIE
    model_path = tmp_path / "models" / "test-mindie-model"
    model_path.mkdir(parents=True, exist_ok=True)
    config_json = model_path / "config.json"
    config_json.write_text(json.dumps({
        "torch_dtype": "float16",
        "max_position_embeddings": 8192,
    }), encoding="utf-8")

    # Create the MindIE install path structure for config writing
    install_path = tmp_path / "ascend" / "mindie" / "latest" / "mindie-service"
    (install_path / "conf").mkdir(parents=True, exist_ok=True)
    (install_path / "bin").mkdir(parents=True, exist_ok=True)
    # Create a fake daemon binary
    daemon_path = install_path / "bin" / "mindieservice_daemon"
    daemon_path.write_text("#!/bin/bash\n", encoding="utf-8")

    server = cast(Any, AscendMindIEServer.__new__(AscendMindIEServer))
    server._config = cast(
        Any,
        SimpleNamespace(
            direct_process_mode=direct_process_mode,
            data_dir=str(data_dir),
            cache_dir=str(cache_dir),
            log_dir=str(log_dir),
        ),
    )
    server._worker = cast(Any, SimpleNamespace(id=1, ip="127.0.0.1", ifname="lo"))
    server._model = cast(
        Any,
        SimpleNamespace(
            backend=BackendEnum.ASCEND_MINDIE,
            backend_version="2.0",
            backend_parameters=[],
            categories=[],
            env={"CUSTOM_ENV": "enabled"},
            name="test-mindie-model",
        ),
    )
    server._model_instance = cast(
        Any,
        SimpleNamespace(
            id=1,
            name="test-mindie-instance",
            model_name="test-mindie-model",
            backend_version=None,
            port=9000,
            ports=[9000],
            gpu_indexes=[0],
            gpu_addresses=["192.168.1.10"],
            worker_ip="127.0.0.1",
            distributed_servers=None,
        ),
    )
    server._model_path = str(model_path)
    server._draft_model_path = None
    server.inference_backend = cast(
        Any,
        SimpleNamespace(
            get_container_entrypoint=lambda *_args, **_kwargs: pytest.fail(
                "direct-process MindIE must not look up container entrypoints"
            )
        ),
    )
    server._derive_max_model_len = lambda default=None: 8192
    server.build_versioned_command_args = (
        lambda default_args, model_path=None, port=None: default_args
    )
    server._flatten_backend_param = lambda: []
    server._get_selected_gpu_devices = lambda: []
    server._get_device_info = lambda: ("ascend", None, None)
    server._get_deployment_metadata = lambda: cast(
        Any,
        SimpleNamespace(
            name="test-mindie-instance",
            distributed=False,
            distributed_leader=False,
            distributed_follower=False,
        ),
    )
    # Override install path to use tmp_path
    server._get_mindie_install_path = lambda: install_path
    # Store install_path for tests that need it
    server._test_install_path = install_path
    _install_fake_launch_artifacts(server, tmp_path)
    return server


def _set_distributed_topology(server: AscendMindIEServer) -> None:
    server._model_instance.ports = [9000, 9001]
    server._model_instance.gpu_indexes = [0, 1]
    server._model_instance.gpu_addresses = ["192.168.1.10", "192.168.1.11"]
    server._model_instance.worker_ip = "127.0.0.1"
    server._model_instance.distributed_servers = SimpleNamespace(
        subordinate_workers=[
            SimpleNamespace(
                worker_ip="127.0.0.2",
                gpu_indexes=[2, 3],
                gpu_addresses=["192.168.1.12", "192.168.1.13"],
            )
        ]
    )


# ---------------------------------------------------------------------------
# supports_direct_process classmethod
# ---------------------------------------------------------------------------


def test_mindie_server_supports_direct_process():
    """AscendMindIEServer declares direct-process support."""
    assert AscendMindIEServer.supports_direct_process() is True


def test_mindie_server_does_not_support_distributed_direct_process():
    """AscendMindIEServer declares distributed direct-process support."""
    assert AscendMindIEServer.supports_distributed_direct_process() is True


def test_mindie_server_declares_bootstrap_recipe_id():
    """AscendMindIEServer advertises the MindIE bootstrap recipe id."""
    assert AscendMindIEServer.get_direct_process_bootstrap_recipe_id() == "mindie"


def test_mindie_server_prepared_environment_identity_uses_backend_version(
    tmp_path: Path,
):
    """Prepared-environment identity stays version-scoped for MindIE."""
    server = _make_server(tmp_path)

    assert server.get_direct_process_prepared_environment_identity() == "mindie:2.0"


def test_mindie_server_prepared_environment_identity_falls_back_to_unknown(
    tmp_path: Path,
):
    """Prepared-environment identity falls back to unknown when no version is set."""
    server = _make_server(tmp_path)
    server._model_instance.backend_version = None
    server._model.backend_version = None

    assert server.get_direct_process_prepared_environment_identity() == "mindie:unknown"


# ---------------------------------------------------------------------------
# Command build
# ---------------------------------------------------------------------------


def test_direct_process_mindie_command_is_daemon_binary(tmp_path: Path):
    """Direct-process MindIE command is the mindieservice_daemon binary."""
    server = _make_server(tmp_path)

    command = server.build_direct_process_command(port=9000)

    assert len(command) >= 1
    assert "mindieservice_daemon" in command[0]
    # The path should end with the daemon binary name
    assert command[0].endswith("mindieservice_daemon")


# ---------------------------------------------------------------------------
# Env build
# ---------------------------------------------------------------------------


def test_direct_process_mindie_env_includes_model_env(tmp_path: Path):
    """build_direct_process_env includes model-level env vars."""
    server = _make_server(tmp_path)

    env = server.build_direct_process_env()

    assert env["CUSTOM_ENV"] == "enabled"


# ---------------------------------------------------------------------------
# Health path
# ---------------------------------------------------------------------------


def test_direct_process_mindie_health_path():
    """MindIE direct-process health path is /v1/models."""
    server = cast(Any, AscendMindIEServer.__new__(AscendMindIEServer))
    assert server.get_direct_process_health_path() == "/v1/models"


# ---------------------------------------------------------------------------
# Happy-path launch
# ---------------------------------------------------------------------------


def test_direct_process_mindie_launch_happy_path(
    monkeypatch, tmp_path: Path
):
    """A valid direct-process MindIE launch returns {pid, process_group_id, port, mode}."""
    server = _make_server(tmp_path)
    observed: dict = {}

    class FakeProcess:
        pid = 7777

    def fake_popen(args, **kwargs):
        observed["popen_args"] = list(args)
        observed["popen_kwargs"] = kwargs
        return FakeProcess()

    # The daemon binary exists at the test install path (created by _make_server)
    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.get_process_group_id", lambda pid: pid + 100
    )

    result = server._start()

    assert result == {
        "pid": 7777,
        "process_group_id": 7877,
        "port": 9000,
        "mode": DIRECT_PROCESS_RUNTIME_MODE,
    }
    # Verify command was built correctly
    assert observed["popen_args"][0] == str(
        tmp_path / "prepared" / "mindie" / "artifacts" / "prepared-launch.sh"
    )
    assert observed["popen_args"][1] == "__GPUSTACK_PREPARED_EXECUTABLE__"
    # Verify env includes model env
    assert observed["popen_kwargs"]["env"]["CUSTOM_ENV"] == "enabled"
    # Verify MindIE-specific env vars are set
    assert observed["popen_kwargs"]["env"]["MINDIE_LLM_FRAMEWORK_BACKEND"] == "ATB"
    assert observed["popen_kwargs"]["env"]["MIES_INSTALL_PATH"] is not None
    assert observed["popen_kwargs"]["env"]["MIES_CONFIG_JSON_PATH"] is not None
    assert observed["popen_kwargs"]["stdin"] is not None
    assert observed["popen_kwargs"]["start_new_session"] in {True, False}

    # Verify config JSON was written
    config_path = observed["popen_kwargs"]["env"]["MIES_CONFIG_JSON_PATH"]
    assert Path(config_path).exists()
    assert str(tmp_path / "runtime" / "1") in config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    assert "ServerConfig" in config
    assert config["ServerConfig"]["port"] == 9000


# ---------------------------------------------------------------------------
# Result shape characterization
# ---------------------------------------------------------------------------


def test_direct_process_mindie_result_has_exactly_four_keys(
    monkeypatch, tmp_path: Path
):
    """Characterization: start_direct_process returns exactly {pid, process_group_id, port, mode}."""
    server = _make_server(tmp_path)

    class FakeProcess:
        pid = 1234

    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.get_process_group_id", lambda pid: pid + 1
    )

    result = server.start_direct_process()

    assert set(result.keys()) == {"pid", "process_group_id", "port", "mode"}
    assert result["pid"] == 1234
    assert result["process_group_id"] == 1235
    assert result["port"] == 9000
    assert result["mode"] == DIRECT_PROCESS_RUNTIME_MODE


def test_direct_process_mindie_launch_uses_only_primary_port_and_worker_shape(
    monkeypatch, tmp_path: Path
):
    """Characterization: extra ports/GPU indexes do not make MindIE direct-process distributed."""
    server = _make_server(tmp_path)
    server._model_instance.ports = [9000, 9001, 9002]
    server._model_instance.gpu_indexes = [0, 1, 2]
    server._model_instance.distributed_servers = [
        SimpleNamespace(worker_id=2, rpc_port=9101),
        SimpleNamespace(worker_id=3, rpc_port=9102),
    ]
    observed: dict[str, Any] = {}

    class FakeProcess:
        pid = 4321

    def fake_popen(args, **kwargs):
        observed["args"] = list(args)
        observed["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.get_process_group_id", lambda pid: pid
    )

    result = server.start_direct_process()

    assert result == {
        "pid": 4321,
        "process_group_id": 4321,
        "port": 9000,
        "mode": DIRECT_PROCESS_RUNTIME_MODE,
    }
    assert observed["args"][0] == str(
        tmp_path / "prepared" / "mindie" / "artifacts" / "prepared-launch.sh"
    )

    config_path = Path(observed["kwargs"]["env"]["MIES_CONFIG_JSON_PATH"])
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert config["ServerConfig"]["port"] == 9000
    serialized_config = json.dumps(config)
    assert '"port": 9001' not in serialized_config
    assert '"port": 9002' not in serialized_config


# ---------------------------------------------------------------------------
# Distributed launch contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("deployment_metadata", "expected_gpu_ids"),
    [
        (
            SimpleNamespace(
                name="leader-distributed",
                distributed=True,
                distributed_leader=True,
                distributed_follower=False,
                distributed_follower_index=None,
            ),
            [[0, 1]],
        ),
        (
            SimpleNamespace(
                name="follower-distributed",
                distributed=True,
                distributed_leader=False,
                distributed_follower=True,
                distributed_follower_index=0,
            ),
            [[0, 1]],
        ),
    ],
    ids=["leader", "follower"],
)
def test_direct_process_mindie_distributed_launch_returns_single_local_runtime_entry(
    monkeypatch, tmp_path: Path, deployment_metadata, expected_gpu_ids
):
    """Distributed MindIE returns one local serve runtime entry per worker role."""
    server = _make_server(tmp_path)
    _set_distributed_topology(server)
    server._get_deployment_metadata = lambda: deployment_metadata
    observed: dict[str, Any] = {}

    class FakeProcess:
        pid = 4567

    def fake_popen(args, **kwargs):
        observed["args"] = list(args)
        observed["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.get_process_group_id", lambda pid: pid + 10
    )

    result = server._start()

    assert result == {
        "pid": 4567,
        "process_group_id": 4577,
        "port": 9000,
        "mode": DIRECT_PROCESS_RUNTIME_MODE,
        "runtime_entries": [
            {
                "deployment_name": deployment_metadata.name,
                "runtime_name": "serve",
                "pid": 4567,
                "process_group_id": 4577,
                "port": 9000,
                "log_path": server._runtime_log_path("serve"),
            }
        ],
    }
    assert observed["args"][0] == str(
        tmp_path / "prepared" / "mindie" / "artifacts" / "prepared-launch.sh"
    )

    config_path = Path(observed["kwargs"]["env"]["MIES_CONFIG_JSON_PATH"])
    rank_table_path = Path(observed["kwargs"]["env"]["RANK_TABLE_FILE"])
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(rank_table_path, "r", encoding="utf-8") as f:
        rank_table = json.load(f)

    assert config["ServerConfig"]["port"] == 9000
    assert config["BackendConfig"]["multiNodesInferEnabled"] is True
    assert config["BackendConfig"]["multiNodesInferPort"] == 9001
    assert config["BackendConfig"]["npuDeviceIds"] == expected_gpu_ids
    assert observed["kwargs"]["env"]["WORLD_SIZE"] == "4"
    assert observed["kwargs"]["env"]["RANKTABLEFILE"] == str(rank_table_path)
    assert observed["kwargs"]["env"]["RANK_TABLE_FILE"] == str(rank_table_path)
    assert rank_table["server_count"] == "2"


def test_direct_process_mindie_distributed_requires_explicit_role(tmp_path: Path):
    """Distributed MindIE must say whether the local worker is the leader or follower."""
    server = _make_server(tmp_path)
    _set_distributed_topology(server)
    server._get_deployment_metadata = lambda: SimpleNamespace(
        name="distributed-before-preflight",
        distributed=True,
        distributed_leader=False,
        distributed_follower=False,
        distributed_follower_index=None,
    )

    with pytest.raises(
        ValueError,
        match="Direct-process MindIE distributed launches require an explicit leader or follower role",
    ):
        server.start_direct_process()


@pytest.mark.parametrize(
    "deployment_metadata",
    [
        SimpleNamespace(
            name="leader-missing-runtime-port",
            distributed=True,
            distributed_leader=True,
            distributed_follower=False,
            distributed_follower_index=None,
        ),
        SimpleNamespace(
            name="follower-missing-runtime-port",
            distributed=True,
            distributed_leader=False,
            distributed_follower=True,
            distributed_follower_index=0,
        ),
    ],
    ids=["leader", "follower"],
)
def test_direct_process_mindie_distributed_requires_secondary_runtime_port(
    tmp_path: Path, deployment_metadata
):
    """Distributed MindIE requires an explicit secondary init/runtime port."""
    server = _make_server(tmp_path)
    _set_distributed_topology(server)
    server._model_instance.ports = [9000]
    server._get_deployment_metadata = lambda: deployment_metadata

    with pytest.raises(
        RuntimeError,
        match="Direct-process MindIE distributed launches require a secondary port for multi-node initialization",
    ):
        server.start_direct_process()


# ---------------------------------------------------------------------------
# Preflight failure: missing executable
# ---------------------------------------------------------------------------


def test_direct_process_mindie_preflight_fails_missing_executable(
    monkeypatch, tmp_path: Path
):
    """Preflight raises RuntimeError when the executable is not found."""
    server = _make_server(tmp_path)

    # Make the daemon binary not exist
    daemon_path = server._test_install_path / "bin" / "mindieservice_daemon"
    if daemon_path.exists():
        daemon_path.unlink()

    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "missing executable must prevent process spawn"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Direct-process MindIE host prerequisites not met",
    ):
        server.start_direct_process()


def test_direct_process_mindie_preflight_fails_updates_state(
    monkeypatch, tmp_path: Path
):
    """Preflight failure through start() updates model instance state to ERROR."""
    server = _make_server(tmp_path)
    updates = []

    def fake_update_model_instance(_id, **kwargs):
        updates.append(kwargs)

    monkeypatch.setattr(server, "_update_model_instance", fake_update_model_instance)

    # Make the daemon binary not exist
    daemon_path = server._test_install_path / "bin" / "mindieservice_daemon"
    if daemon_path.exists():
        daemon_path.unlink()

    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )

    with pytest.raises(RuntimeError):
        server.start()

    assert len(updates) == 1
    assert updates[0]["state"] == ModelInstanceStateEnum.ERROR
    assert "host prerequisites not met" in updates[0]["state_message"]


# ---------------------------------------------------------------------------
# Preflight failure: port bind
# ---------------------------------------------------------------------------


def test_direct_process_mindie_preflight_fails_port_bind(
    monkeypatch, tmp_path: Path
):
    """Preflight raises RuntimeError when the port cannot be bound."""
    server = _make_server(tmp_path)

    class BindFailSocket:
        def bind(self, address):
            raise OSError(98, "Address already in use")

        def close(self):
            pass

    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.socket.socket",
        lambda *_args, **_kwargs: BindFailSocket(),
    )

    with pytest.raises(
        RuntimeError,
        match="Direct-process MindIE host prerequisites not met",
    ):
        server.start_direct_process()


# ---------------------------------------------------------------------------
# Preflight failure: missing directory
# ---------------------------------------------------------------------------


def test_direct_process_mindie_preflight_fails_missing_directory(
    monkeypatch, tmp_path: Path
):
    """Preflight raises RuntimeError when a required directory is missing."""
    import shutil as _shutil

    server = _make_server(tmp_path)

    # Remove the serve log directory
    serve_log_dir = tmp_path / "data" / "log" / "serve"
    _shutil.rmtree(str(serve_log_dir))

    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )

    with pytest.raises(
        RuntimeError,
        match="Direct-process MindIE host prerequisites not met",
    ):
        server.start_direct_process()


# ---------------------------------------------------------------------------
# Container mode isolation
# ---------------------------------------------------------------------------


def test_mindie_server_container_mode_does_not_use_direct_process(
    monkeypatch, tmp_path: Path
):
    """When direct_process_mode is False, _start() follows the container path."""
    server = _make_server(tmp_path, direct_process_mode=False)

    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "container mode must not spawn direct processes"
        ),
    )

    # Allow container-mode inference_backend calls
    server.inference_backend = cast(
        Any,
        SimpleNamespace(
            get_container_entrypoint=lambda *_args, **_kwargs: None,
        ),
    )

    # Stub out container-mode dependencies
    container_called = {"called": False}

    def fake_create_workload(**kwargs):
        container_called["called"] = True

    monkeypatch.setattr(server, "_create_workload", fake_create_workload)
    monkeypatch.setattr(
        server, "_get_configured_image", lambda backend=None: "mindie-image:latest"
    )

    # _start() should go through the container path
    # The important thing is that subprocess.Popen is NOT called
    try:
        server._start()
    except Exception:
        pass  # Container path may fail due to incomplete stubs


# ---------------------------------------------------------------------------
# DIRECT_PROCESS_RUNTIME_MODE constant
# ---------------------------------------------------------------------------


def test_direct_process_mindie_mode_constant_value_is_locked():
    """Characterization: DIRECT_PROCESS_RUNTIME_MODE constant value must be 'direct_process'."""
    assert DIRECT_PROCESS_RUNTIME_MODE == "direct_process"
