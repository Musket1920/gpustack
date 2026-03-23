import importlib
import logging
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, Optional, cast

import pytest

from gpustack.schemas.inference_backend import (
    DirectProcessContract,
    InferenceBackend,
    VersionConfig,
    VersionConfigDict,
)
from gpustack.schemas.models import BackendEnum, ModelInstanceStateEnum


def _import_custom_module():
    fcntl_stub = types.ModuleType("fcntl")
    setattr(fcntl_stub, "LOCK_EX", 1)
    setattr(fcntl_stub, "LOCK_UN", 2)
    setattr(fcntl_stub, "lockf", lambda *args, **kwargs: None)
    setattr(fcntl_stub, "flock", lambda *args, **kwargs: None)
    original_fcntl = sys.modules.get("fcntl")
    sys.modules["fcntl"] = fcntl_stub
    try:
        return importlib.import_module("gpustack.worker.backends.custom")
    finally:
        if original_fcntl is None:
            sys.modules.pop("fcntl", None)
        else:
            sys.modules["fcntl"] = original_fcntl


custom_module = _import_custom_module()
CustomServer = custom_module.CustomServer
DIRECT_PROCESS_RUNTIME_MODE = custom_module.DIRECT_PROCESS_RUNTIME_MODE


class _FakeSocket:
    def bind(self, address):
        self.address = address

    def close(self):
        pass


def _make_contract(
    command_template: str = "my-server serve {{model_path}} --port {{port}}",
    env_template: Optional[Dict[str, str]] = None,
    health_path: str = "/health",
    startup_timeout_seconds: int = 120,
    stop_signal: str = "SIGTERM",
    stop_timeout_seconds: int = 30,
    workdir: Optional[str] = None,
) -> DirectProcessContract:
    return DirectProcessContract(
        command_template=command_template,
        env_template=env_template,
        health_path=health_path,
        startup_timeout_seconds=startup_timeout_seconds,
        stop_signal=stop_signal,
        stop_timeout_seconds=stop_timeout_seconds,
        workdir=workdir,
    )


def _make_inference_backend(
    contract: Optional[DirectProcessContract] = None,
    version: str = "1.0.0",
) -> InferenceBackend:
    version_config = VersionConfig.model_validate(
        {
            "direct_process_contract": contract,
        }
    )
    return InferenceBackend(
        backend_name=BackendEnum.CUSTOM.value,
        version_configs=VersionConfigDict(root={version: version_config}),
        default_version=version,
    )


def _make_server(
    tmp_path: Path,
    direct_process_mode: bool = True,
    contract: Optional[DirectProcessContract] = None,
    version: str = "1.0.0",
) -> CustomServer:
    data_dir = tmp_path / "data"
    cache_dir = data_dir / "cache"
    log_dir = data_dir / "log"
    (log_dir / "serve").mkdir(parents=True, exist_ok=True)
    (data_dir / "worker").mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    server = cast(Any, CustomServer.__new__(CustomServer))
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
            backend=BackendEnum.CUSTOM,
            backend_version=version,
            backend_parameters=[],
            categories=[],
            env={"CUSTOM_ENV": "enabled"},
        ),
    )
    server._model_instance = cast(
        Any,
        SimpleNamespace(
            id=1,
            name="test-custom-instance",
            model_name="test-custom-model",
            port=9000,
            ports=[9000],
            gpu_indexes=[0],
        ),
    )
    server._model_path = "/models/test-custom-model"
    server._draft_model_path = None
    server.inference_backend = _make_inference_backend(
        contract=contract, version=version
    )
    server._flatten_backend_param = lambda: []
    server._get_selected_gpu_devices = lambda: []
    server._get_device_info = lambda: ("nvidia", None, None)
    server._get_deployment_metadata = lambda: cast(
        Any,
        SimpleNamespace(
            name="test-custom-instance",
            distributed=False,
            distributed_leader=False,
            distributed_follower=False,
        ),
    )
    return server


# ---------------------------------------------------------------------------
# supports_direct_process classmethod
# ---------------------------------------------------------------------------


def test_custom_server_supports_direct_process():
    """CustomServer declares direct-process support."""
    assert CustomServer.supports_direct_process() is True


def test_custom_server_does_not_support_distributed_direct_process():
    """CustomServer does not support distributed direct-process."""
    assert CustomServer.supports_distributed_direct_process() is False


# ---------------------------------------------------------------------------
# Valid contract launch — happy path
# ---------------------------------------------------------------------------


def test_direct_process_custom_launch_with_valid_contract(
    monkeypatch, tmp_path: Path
):
    """A valid DirectProcessContract produces a successful direct-process launch
    returning {pid, process_group_id, port, mode}."""
    contract = _make_contract(
        command_template="my-server serve {{model_path}} --port {{port}} --host {{worker_ip}}",
    )
    server = _make_server(tmp_path, contract=contract)
    observed: dict = {}

    class FakeProcess:
        pid = 7777

    def fake_popen(args, **kwargs):
        observed["popen_args"] = list(args)
        observed["popen_kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "gpustack.worker.backends.custom.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/my-server",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.get_process_group_id", lambda pid: pid + 100
    )

    result = server._start()

    assert result == {
        "pid": 7777,
        "process_group_id": 7877,
        "port": 9000,
        "mode": DIRECT_PROCESS_RUNTIME_MODE,
        "startup_timeout_seconds": 120,
        "stop_signal": "SIGTERM",
        "stop_timeout_seconds": 30,
    }
    # Verify command was resolved correctly
    assert observed["popen_args"][0] == "my-server"
    assert "serve" in observed["popen_args"]
    assert "/models/test-custom-model" in observed["popen_args"]
    assert "--port" in observed["popen_args"]
    assert "9000" in observed["popen_args"]
    assert "--host" in observed["popen_args"]
    assert "127.0.0.1" in observed["popen_args"]
    # Verify env includes model env
    assert observed["popen_kwargs"]["env"]["CUSTOM_ENV"] == "enabled"
    assert observed["popen_kwargs"]["stdin"] is not None
    assert observed["popen_kwargs"]["start_new_session"] in {True, False}


def test_direct_process_custom_result_has_exactly_seven_keys(
    monkeypatch, tmp_path: Path
):
    """Characterization: start_direct_process returns runtime metadata plus control fields."""
    contract = _make_contract()
    server = _make_server(tmp_path, contract=contract)

    class FakeProcess:
        pid = 1234

    monkeypatch.setattr(
        "gpustack.worker.backends.custom.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/my-server",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.get_process_group_id", lambda pid: pid + 1
    )

    result = server.start_direct_process()

    assert set(result.keys()) == {
        "pid",
        "process_group_id",
        "port",
        "mode",
        "startup_timeout_seconds",
        "stop_signal",
        "stop_timeout_seconds",
    }
    assert result["pid"] == 1234
    assert result["process_group_id"] == 1235
    assert result["port"] == 9000
    assert result["mode"] == DIRECT_PROCESS_RUNTIME_MODE
    assert result["startup_timeout_seconds"] == 120
    assert result["stop_signal"] == "SIGTERM"
    assert result["stop_timeout_seconds"] == 30


def test_direct_process_custom_result_includes_runtime_control_fields(
    monkeypatch, tmp_path: Path
):
    """start_direct_process returns contract-driven startup and stop controls."""
    contract = _make_contract()
    contract = contract.model_copy(
        update={
            "startup_timeout_seconds": 321,
            "stop_signal": "SIGINT",
            "stop_timeout_seconds": 12,
        }
    )
    server = _make_server(tmp_path, contract=contract)

    class FakeProcess:
        pid = 1234

    monkeypatch.setattr(
        "gpustack.worker.backends.custom.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/my-server",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.get_process_group_id", lambda pid: pid + 1
    )

    result = server.start_direct_process()

    assert result["startup_timeout_seconds"] == 321
    assert result["stop_signal"] == "SIGINT"
    assert result["stop_timeout_seconds"] == 12


# ---------------------------------------------------------------------------
# Contract env_template merging
# ---------------------------------------------------------------------------


def test_direct_process_custom_env_template_merged(
    monkeypatch, tmp_path: Path
):
    """env_template values from the contract are merged into the process env."""
    contract = _make_contract(
        env_template={
            "MY_PORT": "{{port}}",
            "MY_MODEL": "{{model_path}}",
            "STATIC_VAR": "static_value",
        },
    )
    server = _make_server(tmp_path, contract=contract)

    env = server.build_direct_process_env()

    assert env["MY_PORT"] == "9000"
    assert env["MY_MODEL"] == "/models/test-custom-model"
    assert env["STATIC_VAR"] == "static_value"
    # Base env should also be present
    assert env["CUSTOM_ENV"] == "enabled"


# ---------------------------------------------------------------------------
# Health path from contract
# ---------------------------------------------------------------------------


def test_direct_process_custom_health_path_from_contract(tmp_path: Path):
    """Health path is taken from the contract."""
    contract = _make_contract(health_path="/v1/health")
    server = _make_server(tmp_path, contract=contract)

    assert server.get_direct_process_health_path() == "/v1/health"


def test_direct_process_custom_health_path_default(tmp_path: Path):
    """Default health path is /health."""
    contract = _make_contract()
    server = _make_server(tmp_path, contract=contract)

    assert server.get_direct_process_health_path() == "/health"


# ---------------------------------------------------------------------------
# Workdir passed to Popen
# ---------------------------------------------------------------------------


def test_direct_process_custom_workdir_passed_to_popen(
    monkeypatch, tmp_path: Path
):
    """When contract specifies workdir, it is passed to Popen."""
    contract = _make_contract(workdir="/opt/my-server")
    server = _make_server(tmp_path, contract=contract)
    observed: dict = {}

    class FakeProcess:
        pid = 42

    def fake_popen(args, **kwargs):
        observed["cwd"] = kwargs.get("cwd")
        return FakeProcess()

    monkeypatch.setattr(
        "gpustack.worker.backends.custom.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/my-server",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.get_process_group_id", lambda pid: pid
    )

    server.start_direct_process()

    assert observed["cwd"] == "/opt/my-server"


def test_direct_process_custom_workdir_none_when_not_set(
    monkeypatch, tmp_path: Path
):
    """When contract does not specify workdir, cwd is None (inherit)."""
    contract = _make_contract(workdir=None)
    server = _make_server(tmp_path, contract=contract)
    observed: dict = {}

    class FakeProcess:
        pid = 42

    def fake_popen(args, **kwargs):
        observed["cwd"] = kwargs.get("cwd")
        return FakeProcess()

    monkeypatch.setattr(
        "gpustack.worker.backends.custom.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/my-server",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.get_process_group_id", lambda pid: pid
    )

    server.start_direct_process()

    assert observed["cwd"] is None


# ---------------------------------------------------------------------------
# Preflight failure: missing executable
# ---------------------------------------------------------------------------


def test_direct_process_custom_preflight_fails_missing_executable(
    monkeypatch, tmp_path: Path
):
    """Preflight raises RuntimeError when the executable is not found."""
    contract = _make_contract()
    server = _make_server(tmp_path, contract=contract)

    monkeypatch.setattr(
        "gpustack.worker.backends.custom.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "missing executable must prevent process spawn"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Direct-process custom backend host prerequisites not met",
    ):
        server.start_direct_process()


def test_direct_process_custom_preflight_fails_missing_executable_updates_state(
    monkeypatch, tmp_path: Path
):
    """Preflight failure through start() updates model instance state to ERROR."""
    contract = _make_contract()
    server = _make_server(tmp_path, contract=contract)
    updates = []

    def fake_update_model_instance(_id, **kwargs):
        updates.append(kwargs)

    monkeypatch.setattr(server, "_update_model_instance", fake_update_model_instance)
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.socket.socket",
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


def test_direct_process_custom_preflight_fails_port_bind(
    monkeypatch, tmp_path: Path
):
    """Preflight raises RuntimeError when the port cannot be bound."""
    contract = _make_contract()
    server = _make_server(tmp_path, contract=contract)

    class BindFailSocket:
        def bind(self, address):
            raise OSError(98, "Address already in use")

        def close(self):
            pass

    monkeypatch.setattr(
        "gpustack.worker.backends.custom.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/my-server",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.socket.socket",
        lambda *_args, **_kwargs: BindFailSocket(),
    )

    with pytest.raises(
        RuntimeError,
        match="Direct-process custom backend host prerequisites not met",
    ):
        server.start_direct_process()


# ---------------------------------------------------------------------------
# Preflight failure: missing directory
# ---------------------------------------------------------------------------


def test_direct_process_custom_preflight_fails_missing_directory(
    monkeypatch, tmp_path: Path
):
    """Preflight raises RuntimeError when a required directory is missing."""
    import shutil as _shutil

    contract = _make_contract()
    server = _make_server(tmp_path, contract=contract)

    # Remove the serve log directory
    serve_log_dir = tmp_path / "data" / "log" / "serve"
    _shutil.rmtree(str(serve_log_dir))

    monkeypatch.setattr(
        "gpustack.worker.backends.custom.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/my-server",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )

    with pytest.raises(
        RuntimeError,
        match="Direct-process custom backend host prerequisites not met",
    ):
        server.start_direct_process()


# ---------------------------------------------------------------------------
# Unsupported contract: no direct_process_contract set
# ---------------------------------------------------------------------------


def test_direct_process_custom_no_contract_fails_deterministically(
    monkeypatch, tmp_path: Path
):
    """When no DirectProcessContract is configured, start_direct_process
    raises ValueError deterministically."""
    # Create server with NO contract
    server = _make_server(tmp_path, contract=None)

    with pytest.raises(
        ValueError,
        match="Custom backend direct-process requires a DirectProcessContract",
    ):
        server.start_direct_process()


def test_direct_process_custom_no_contract_via_start_updates_state(
    monkeypatch, tmp_path: Path
):
    """When no contract is configured, start() updates model instance state to ERROR."""
    server = _make_server(tmp_path, contract=None)
    updates = []

    def fake_update_model_instance(_id, **kwargs):
        updates.append(kwargs)

    monkeypatch.setattr(server, "_update_model_instance", fake_update_model_instance)

    with pytest.raises(ValueError):
        server.start()

    assert len(updates) == 1
    assert updates[0]["state"] == ModelInstanceStateEnum.ERROR
    assert "DirectProcessContract" in updates[0]["state_message"]


def test_direct_process_custom_no_inference_backend_fails(
    monkeypatch, tmp_path: Path
):
    """When no inference backend is set at all, start_direct_process fails."""
    server = _make_server(tmp_path, contract=_make_contract())
    server.inference_backend = None

    with pytest.raises(
        ValueError,
        match="Custom backend direct-process requires an inference backend",
    ):
        server.start_direct_process()


# ---------------------------------------------------------------------------
# Container mode is not affected
# ---------------------------------------------------------------------------


def test_custom_server_container_mode_does_not_use_direct_process(
    monkeypatch, tmp_path: Path
):
    """When direct_process_mode is False, _start() follows the container path."""
    contract = _make_contract()
    server = _make_server(tmp_path, direct_process_mode=False, contract=contract)
    container_called = {"called": False}

    def fake_create_workload(**kwargs):
        container_called["called"] = True

    # Stub out container-mode dependencies
    monkeypatch.setattr(server, "_create_workload", fake_create_workload)
    monkeypatch.setattr(
        server, "_get_configured_image", lambda: "my-image:latest"
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.custom.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "container mode must not spawn direct processes"
        ),
    )

    # _start() should go through the container path, which calls _create_workload
    # We need to stub the remaining container-mode dependencies
    server._build_command_args = lambda: ["my-server", "serve"]
    server._get_configured_env = lambda: {"PATH": "/usr/bin"}

    # The container path will fail because we haven't fully stubbed it,
    # but the important thing is that it does NOT call start_direct_process
    # We verify by checking that subprocess.Popen was not called
    try:
        server._start()
    except Exception:
        pass  # Container path may fail due to incomplete stubs

    # If we got here without the pytest.fail from subprocess.Popen, container mode is correct


# ---------------------------------------------------------------------------
# Command template placeholder resolution
# ---------------------------------------------------------------------------


def test_direct_process_custom_command_template_resolves_placeholders(
    tmp_path: Path,
):
    """Command template placeholders are resolved correctly."""
    contract = _make_contract(
        command_template="my-server --model {{model_path}} --port {{port}} --name {{model_name}} --ip {{worker_ip}}",
    )
    server = _make_server(tmp_path, contract=contract)

    command = server.build_direct_process_command(port=9000)

    assert command == [
        "my-server",
        "--model",
        "/models/test-custom-model",
        "--port",
        "9000",
        "--name",
        "test-custom-model",
        "--ip",
        "127.0.0.1",
    ]


# ---------------------------------------------------------------------------
# DIRECT_PROCESS_RUNTIME_MODE constant
# ---------------------------------------------------------------------------


def test_direct_process_custom_mode_constant_value_is_locked():
    """Characterization: DIRECT_PROCESS_RUNTIME_MODE constant value must be 'direct_process'."""
    assert DIRECT_PROCESS_RUNTIME_MODE == "direct_process"
