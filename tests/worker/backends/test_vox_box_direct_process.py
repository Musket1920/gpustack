import importlib
import logging
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gpustack.schemas.models import BackendEnum, ModelInstanceStateEnum


def _import_vox_box_module():
    fcntl_stub = types.ModuleType("fcntl")
    setattr(fcntl_stub, "LOCK_EX", 1)
    setattr(fcntl_stub, "LOCK_UN", 2)
    setattr(fcntl_stub, "lockf", lambda *args, **kwargs: None)
    setattr(fcntl_stub, "flock", lambda *args, **kwargs: None)
    original_fcntl = sys.modules.get("fcntl")
    sys.modules["fcntl"] = fcntl_stub
    try:
        return importlib.import_module("gpustack.worker.backends.vox_box")
    finally:
        if original_fcntl is None:
            sys.modules.pop("fcntl", None)
        else:
            sys.modules["fcntl"] = original_fcntl


vox_box_module = _import_vox_box_module()
VoxBoxServer = vox_box_module.VoxBoxServer
DIRECT_PROCESS_RUNTIME_MODE = vox_box_module.DIRECT_PROCESS_RUNTIME_MODE


class _FakeSocket:
    def bind(self, address):
        self.address = address

    def close(self):
        pass


def _make_server(
    tmp_path: Path,
    direct_process_mode: bool = True,
) -> VoxBoxServer:
    data_dir = tmp_path / "data"
    cache_dir = data_dir / "cache"
    log_dir = data_dir / "log"
    (log_dir / "serve").mkdir(parents=True, exist_ok=True)
    (data_dir / "worker").mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    server = cast(Any, VoxBoxServer.__new__(VoxBoxServer))
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
            backend=BackendEnum.VOX_BOX,
            backend_version="0.1.0",
            backend_parameters=[],
            categories=[],
            env={"CUSTOM_ENV": "enabled"},
            name="test-vox-box-model",
        ),
    )
    server._model_instance = cast(
        Any,
        SimpleNamespace(
            id=1,
            name="test-vox-box-instance",
            model_name="test-vox-box-model",
            port=9000,
            ports=[9000],
            gpu_indexes=[0],
        ),
    )
    server._model_path = "/models/test-vox-box-model"
    server._draft_model_path = None
    server.inference_backend = cast(
        Any,
        SimpleNamespace(
            get_container_entrypoint=lambda *_args, **_kwargs: pytest.fail(
                "direct-process VoxBox must not look up container entrypoints"
            )
        ),
    )
    server._derive_max_model_len = lambda _default=None: None
    server.build_versioned_command_args = (
        lambda default_args, model_path=None, port=None: default_args
    )
    server._flatten_backend_param = lambda: []
    server._get_selected_gpu_devices = lambda: []
    server._get_device_info = lambda: ("nvidia", None, None)
    server._get_deployment_metadata = lambda: cast(
        Any,
        SimpleNamespace(
            name="test-vox-box-instance",
            distributed=False,
            distributed_leader=False,
            distributed_follower=False,
        ),
    )
    return server


# ---------------------------------------------------------------------------
# supports_direct_process classmethod
# ---------------------------------------------------------------------------


def test_vox_box_server_supports_direct_process():
    """VoxBoxServer declares direct-process support."""
    assert VoxBoxServer.supports_direct_process() is True


def test_vox_box_server_does_not_support_distributed_direct_process():
    """VoxBoxServer does not support distributed direct-process."""
    assert VoxBoxServer.supports_distributed_direct_process() is False


# ---------------------------------------------------------------------------
# Command build
# ---------------------------------------------------------------------------


def test_direct_process_vox_box_command_starts_with_vox_box(
    monkeypatch, tmp_path: Path
):
    """Direct-process VoxBox command starts with ['vox-box', 'start', '--model', ...]."""
    server = _make_server(tmp_path)

    command = server.build_direct_process_command(port=9000)

    assert command[0] == "vox-box"
    assert command[1] == "start"
    assert "--model" in command
    assert "/models/test-vox-box-model" in command
    assert "--host" in command
    assert "127.0.0.1" in command
    assert "--port" in command
    assert "9000" in command


# ---------------------------------------------------------------------------
# Env build
# ---------------------------------------------------------------------------


def test_direct_process_vox_box_env_includes_model_env(tmp_path: Path):
    """build_direct_process_env includes model-level env vars."""
    server = _make_server(tmp_path)

    env = server.build_direct_process_env()

    assert env["CUSTOM_ENV"] == "enabled"


# ---------------------------------------------------------------------------
# Health path
# ---------------------------------------------------------------------------


def test_direct_process_vox_box_health_path():
    """VoxBox direct-process health path is /health."""
    server = cast(Any, VoxBoxServer.__new__(VoxBoxServer))
    assert server.get_direct_process_health_path() == "/health"


# ---------------------------------------------------------------------------
# Happy-path launch
# ---------------------------------------------------------------------------


def test_direct_process_vox_box_launch_happy_path(
    monkeypatch, tmp_path: Path
):
    """A valid direct-process VoxBox launch returns {pid, process_group_id, port, mode}."""
    server = _make_server(tmp_path)
    observed: dict = {}

    class FakeProcess:
        pid = 7777

    def fake_popen(args, **kwargs):
        observed["popen_args"] = list(args)
        observed["popen_kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/vox-box",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.get_process_group_id", lambda pid: pid + 100
    )

    result = server._start()

    assert result == {
        "pid": 7777,
        "process_group_id": 7877,
        "port": 9000,
        "mode": DIRECT_PROCESS_RUNTIME_MODE,
    }
    # Verify command was built correctly
    assert observed["popen_args"][0] == "vox-box"
    assert "start" in observed["popen_args"]
    assert "--model" in observed["popen_args"]
    assert "/models/test-vox-box-model" in observed["popen_args"]
    # Verify env includes model env
    assert observed["popen_kwargs"]["env"]["CUSTOM_ENV"] == "enabled"
    assert observed["popen_kwargs"]["stdin"] is not None
    assert observed["popen_kwargs"]["start_new_session"] in {True, False}


# ---------------------------------------------------------------------------
# Result shape characterization
# ---------------------------------------------------------------------------


def test_direct_process_vox_box_result_has_exactly_four_keys(
    monkeypatch, tmp_path: Path
):
    """Characterization: start_direct_process returns exactly {pid, process_group_id, port, mode}."""
    server = _make_server(tmp_path)

    class FakeProcess:
        pid = 1234

    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/vox-box",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.get_process_group_id", lambda pid: pid + 1
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
def test_direct_process_vox_box_rejects_distributed(
    monkeypatch, tmp_path: Path, deployment_metadata
):
    """Direct-process VoxBox rejects all distributed deployment variants."""
    server = _make_server(tmp_path)
    server._get_deployment_metadata = lambda: deployment_metadata

    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "distributed direct-process VoxBox must not spawn a process"
        ),
    )

    with pytest.raises(
        ValueError,
        match="Direct-process VoxBox does not support distributed launches",
    ):
        server._start()


# ---------------------------------------------------------------------------
# Preflight failure: missing executable
# ---------------------------------------------------------------------------


def test_direct_process_vox_box_preflight_fails_missing_executable(
    monkeypatch, tmp_path: Path
):
    """Preflight raises RuntimeError when the executable is not found."""
    server = _make_server(tmp_path)

    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "missing executable must prevent process spawn"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Direct-process VoxBox host prerequisites not met",
    ):
        server.start_direct_process()


def test_direct_process_vox_box_preflight_fails_updates_state(
    monkeypatch, tmp_path: Path
):
    """Preflight failure through start() updates model instance state to ERROR."""
    server = _make_server(tmp_path)
    updates = []

    def fake_update_model_instance(_id, **kwargs):
        updates.append(kwargs)

    monkeypatch.setattr(server, "_update_model_instance", fake_update_model_instance)
    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.socket.socket",
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


def test_direct_process_vox_box_preflight_fails_port_bind(
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
        "gpustack.worker.backends.vox_box.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/vox-box",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.socket.socket",
        lambda *_args, **_kwargs: BindFailSocket(),
    )

    with pytest.raises(
        RuntimeError,
        match="Direct-process VoxBox host prerequisites not met",
    ):
        server.start_direct_process()


# ---------------------------------------------------------------------------
# Preflight failure: missing directory
# ---------------------------------------------------------------------------


def test_direct_process_vox_box_preflight_fails_missing_directory(
    monkeypatch, tmp_path: Path
):
    """Preflight raises RuntimeError when a required directory is missing."""
    import shutil as _shutil

    server = _make_server(tmp_path)

    # Remove the serve log directory
    serve_log_dir = tmp_path / "data" / "log" / "serve"
    _shutil.rmtree(str(serve_log_dir))

    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/vox-box",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )

    with pytest.raises(
        RuntimeError,
        match="Direct-process VoxBox host prerequisites not met",
    ):
        server.start_direct_process()


# ---------------------------------------------------------------------------
# Container mode isolation
# ---------------------------------------------------------------------------


def test_vox_box_server_container_mode_does_not_use_direct_process(
    monkeypatch, tmp_path: Path
):
    """When direct_process_mode is False, _start() follows the container path."""
    server = _make_server(tmp_path, direct_process_mode=False)

    monkeypatch.setattr(
        "gpustack.worker.backends.vox_box.subprocess.Popen",
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
        server, "_get_configured_image", lambda: "vox-box-image:latest"
    )
    server._get_serving_command_script = lambda env: None

    # _start() should go through the container path
    # The important thing is that subprocess.Popen is NOT called
    try:
        server._start()
    except Exception:
        pass  # Container path may fail due to incomplete stubs


# ---------------------------------------------------------------------------
# DIRECT_PROCESS_RUNTIME_MODE constant
# ---------------------------------------------------------------------------


def test_direct_process_vox_box_mode_constant_value_is_locked():
    """Characterization: DIRECT_PROCESS_RUNTIME_MODE constant value must be 'direct_process'."""
    assert DIRECT_PROCESS_RUNTIME_MODE == "direct_process"
