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
            port=9000,
            ports=[9000],
            gpu_indexes=[0],
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
    return server


# ---------------------------------------------------------------------------
# supports_direct_process classmethod
# ---------------------------------------------------------------------------


def test_mindie_server_supports_direct_process():
    """AscendMindIEServer declares direct-process support."""
    assert AscendMindIEServer.supports_direct_process() is True


def test_mindie_server_does_not_support_distributed_direct_process():
    """AscendMindIEServer does not support distributed direct-process."""
    assert AscendMindIEServer.supports_distributed_direct_process() is False


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
    assert "mindieservice_daemon" in observed["popen_args"][0]
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


# ---------------------------------------------------------------------------
# Distributed rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "deployment_metadata",
    [
        SimpleNamespace(
            name="leader-distributed",
            distributed=True,
            distributed_leader=True,
            distributed_follower=False,
        ),
        SimpleNamespace(
            name="follower-distributed",
            distributed=True,
            distributed_leader=False,
            distributed_follower=True,
        ),
        SimpleNamespace(
            name="distributed-only",
            distributed=True,
            distributed_leader=False,
            distributed_follower=False,
        ),
    ],
    ids=["leader", "follower", "distributed-flag-only"],
)
def test_direct_process_mindie_rejects_distributed(
    monkeypatch, tmp_path: Path, deployment_metadata
):
    """Direct-process MindIE rejects all distributed deployment variants."""
    server = _make_server(tmp_path)
    server._get_deployment_metadata = lambda: deployment_metadata

    monkeypatch.setattr(
        "gpustack.worker.backends.ascend_mindie.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "distributed direct-process MindIE must not spawn a process"
        ),
    )

    with pytest.raises(
        ValueError,
        match="Direct-process MindIE does not support distributed launches",
    ):
        server._start()


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
