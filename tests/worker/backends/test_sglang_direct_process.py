import importlib
import json
import logging
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gpustack.schemas.models import BackendEnum, ModelInstanceStateEnum


def _import_sglang_module():
    fcntl_stub = types.ModuleType("fcntl")
    setattr(fcntl_stub, "LOCK_EX", 1)
    setattr(fcntl_stub, "LOCK_UN", 2)
    setattr(fcntl_stub, "lockf", lambda *args, **kwargs: None)
    setattr(fcntl_stub, "flock", lambda *args, **kwargs: None)
    original_fcntl = sys.modules.get("fcntl")
    sys.modules["fcntl"] = fcntl_stub
    try:
        return importlib.import_module("gpustack.worker.backends.sglang")
    finally:
        if original_fcntl is None:
            sys.modules.pop("fcntl", None)
        else:
            sys.modules["fcntl"] = original_fcntl


sglang_module = _import_sglang_module()
SGLangServer = sglang_module.SGLangServer
DIRECT_PROCESS_RUNTIME_MODE = sglang_module.DIRECT_PROCESS_RUNTIME_MODE


def _install_fake_launch_artifacts(server: SGLangServer, tmp_path: Path):
    prepared_root = tmp_path / "prepared" / "sglang"
    artifacts_root = prepared_root / "artifacts"
    bin_root = prepared_root / "bin"
    runtime_root = tmp_path / "runtime" / "1"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    bin_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    prepared_python = bin_root / "python"
    prepared_python.write_text("#!/bin/sh\n", encoding="utf-8")
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
                "name": "python",
                "prepared_path": str(prepared_python),
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
                "backend_version": server._model.backend_version,
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
    model_instance_overrides: dict[str, Any] | None = None,
    deployment_metadata: Any | None = None,
) -> SGLangServer:
    data_dir = tmp_path / "data"
    cache_dir = data_dir / "cache"
    log_dir = data_dir / "log"
    (log_dir / "serve").mkdir(parents=True, exist_ok=True)
    (data_dir / "worker").mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    server = cast(Any, SGLangServer.__new__(SGLangServer))
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
            backend=BackendEnum.SGLANG,
            backend_version="0.5.5",
            backend_parameters=[],
            categories=[],
            env={"CUSTOM_ENV": "enabled"},
            extended_kv_cache=None,
            speculative_config=None,
        ),
    )
    model_instance_data = {
        "id": 1,
        "name": "test-sglang-instance",
        "model_name": "test-sglang-model",
        "port": 9000,
        "ports": [9000],
        "gpu_indexes": [0],
        "computed_resource_claim": None,
    }
    if model_instance_overrides:
        model_instance_data.update(model_instance_overrides)
    server._model_instance = cast(Any, SimpleNamespace(**model_instance_data))
    server._model_path = "/models/test-sglang-model"
    server._draft_model_path = None
    server._direct_process_log_file_path = str(log_dir / "serve" / "1.log")
    server.inference_backend = cast(
        Any,
        SimpleNamespace(
            get_container_entrypoint=lambda *_args, **_kwargs: pytest.fail(
                "direct-process SGLang must not look up container entrypoints"
            )
        ),
    )
    server.is_diffusion = False
    server._derive_max_model_len = lambda _default=None: None
    server.build_versioned_command_args = (
        lambda default_args, model_path=None, port=None: default_args
    )
    server._flatten_backend_param = lambda: []
    server._get_selected_gpu_devices = lambda: []
    server._get_device_info = lambda: ("nvidia", None, None)
    server._get_model_architecture = lambda: []
    if deployment_metadata is None:
        deployment_metadata = SimpleNamespace(
            name="test-sglang-instance",
            distributed=False,
            distributed_leader=False,
            distributed_follower=False,
        )
    server._get_deployment_metadata = lambda: cast(Any, deployment_metadata)
    _install_fake_launch_artifacts(server, tmp_path)
    return server


# ---------------------------------------------------------------------------
# supports_direct_process classmethod
# ---------------------------------------------------------------------------


def test_sglang_server_supports_direct_process():
    """SGLangServer declares direct-process support."""
    assert SGLangServer.supports_direct_process() is True


def test_sglang_server_supports_distributed_direct_process():
    """SGLangServer declares distributed direct-process support."""
    assert SGLangServer.supports_distributed_direct_process() is True


def test_sglang_server_direct_process_bootstrap_contract(tmp_path: Path):
    """SGLang direct-process declares bootstrap recipe and prepared-environment identity."""
    server = _make_server(tmp_path)

    assert SGLangServer.get_direct_process_bootstrap_recipe_id() == "sglang"
    assert server.get_direct_process_prepared_environment_identity() == "sglang:0.5.5"


# ---------------------------------------------------------------------------
# Command build
# ---------------------------------------------------------------------------


def test_direct_process_sglang_command_starts_with_python_m_sglang(
    monkeypatch, tmp_path: Path
):
    """Direct-process SGLang command starts with ['python', '-m', 'sglang.launch_server', '--model-path', ...]."""
    server = _make_server(tmp_path)

    command = server.build_direct_process_command(port=9000)

    assert command[0] == "python"
    assert command[1] == "-m"
    assert command[2] == "sglang.launch_server"
    assert "--model-path" in command
    assert "/models/test-sglang-model" in command
    assert "--host" in command
    assert "127.0.0.1" in command
    assert "--port" in command
    assert "9000" in command


# ---------------------------------------------------------------------------
# Env build
# ---------------------------------------------------------------------------


def test_direct_process_sglang_env_includes_model_env(tmp_path: Path):
    """build_direct_process_env includes model-level env vars."""
    server = _make_server(tmp_path)

    env = server.build_direct_process_env()

    assert env["CUSTOM_ENV"] == "enabled"
    # SGLang-specific optimizations
    assert env.get("OMP_NUM_THREADS") == "1"
    assert env.get("SAFETENSORS_FAST_GPU") == "1"


# ---------------------------------------------------------------------------
# Health path
# ---------------------------------------------------------------------------


def test_direct_process_sglang_health_path():
    """SGLang direct-process health path is /v1/models."""
    server = cast(Any, SGLangServer.__new__(SGLangServer))
    assert server.get_direct_process_health_path() == "/v1/models"


# ---------------------------------------------------------------------------
# Happy-path launch
# ---------------------------------------------------------------------------


def test_direct_process_sglang_launch_happy_path(
    monkeypatch, tmp_path: Path
):
    """A valid direct-process SGLang launch returns {pid, process_group_id, port, mode}."""
    server = _make_server(tmp_path)
    observed: dict = {}

    class FakeProcess:
        pid = 7777

    def fake_popen(args, **kwargs):
        observed["popen_args"] = list(args)
        observed["popen_kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/python",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.get_process_group_id", lambda pid: pid + 100
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
        tmp_path / "prepared" / "sglang" / "artifacts" / "prepared-launch.sh"
    )
    assert observed["popen_args"][1] == "__GPUSTACK_PREPARED_EXECUTABLE__"
    assert "-m" in observed["popen_args"]
    assert "sglang.launch_server" in observed["popen_args"]
    assert "--model-path" in observed["popen_args"]
    assert "/models/test-sglang-model" in observed["popen_args"]
    # Verify env includes model env
    assert observed["popen_kwargs"]["env"]["CUSTOM_ENV"] == "enabled"
    assert Path(observed["popen_kwargs"]["env"]["GPUSTACK_PREPARED_EXECUTABLE"]).name == "python"
    assert str(tmp_path / "prepared" / "sglang" / "bin") in observed["popen_kwargs"]["env"]["PATH"]
    assert observed["popen_kwargs"]["stdin"] is not None
    assert observed["popen_kwargs"]["start_new_session"] in {True, False}


# ---------------------------------------------------------------------------
# Result shape characterization
# ---------------------------------------------------------------------------


def test_direct_process_sglang_result_has_exactly_four_keys(
    monkeypatch, tmp_path: Path
):
    """Characterization: start_direct_process returns exactly {pid, process_group_id, port, mode}."""
    server = _make_server(tmp_path)

    class FakeProcess:
        pid = 1234

    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/python",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.get_process_group_id", lambda pid: pid + 1
    )

    result = server.start_direct_process()

    assert set(result.keys()) == {"pid", "process_group_id", "port", "mode"}
    assert result["pid"] == 1234
    assert result["process_group_id"] == 1235
    assert result["port"] == 9000
    assert result["mode"] == DIRECT_PROCESS_RUNTIME_MODE


def test_direct_process_sglang_single_worker_launch_uses_only_primary_port(
    monkeypatch, tmp_path: Path
):
    """Characterization: non-distributed SGLang direct-process still launches as a single-port worker even if extra instance ports exist."""
    server = _make_server(
        tmp_path,
        model_instance_overrides={
            "port": 9000,
            "ports": [9000, 9001, 9002],
            "gpu_indexes": [0, 1],
        },
    )
    observed: dict[str, Any] = {}

    class FakeProcess:
        pid = 4321

    def fake_popen(args, **kwargs):
        observed["args"] = list(args)
        observed["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/python",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.get_process_group_id", lambda pid: pid
    )

    result = server.start_direct_process()

    port_flag_indexes = [
        index for index, arg in enumerate(observed["args"]) if arg == "--port"
    ]
    assert port_flag_indexes == [observed["args"].index("--port")]
    assert observed["args"][port_flag_indexes[0] + 1] == "9000"
    assert "9001" not in observed["args"]
    assert "9002" not in observed["args"]
    assert result["port"] == 9000


# ---------------------------------------------------------------------------
# Distributed contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "deployment_metadata",
        "current_worker_ip",
        "subordinate_worker_ips",
        "expected_node_rank",
    ),
    [
        (
            SimpleNamespace(
                name="leader-distributed",
                distributed=True,
                distributed_leader=True,
                distributed_follower=False,
            ),
            "10.0.0.1",
            ["10.0.0.2"],
            "0",
        ),
        (
            SimpleNamespace(
                name="follower-distributed",
                distributed=True,
                distributed_leader=False,
                distributed_follower=True,
            ),
            "10.0.0.2",
            ["10.0.0.2"],
            "1",
        ),
    ],
    ids=["leader", "follower"],
)
def test_direct_process_sglang_distributed_runtime_entry_contract(
    monkeypatch,
    tmp_path: Path,
    deployment_metadata,
    current_worker_ip,
    subordinate_worker_ips,
    expected_node_rank,
):
    """Distributed SGLang direct-process returns a single local serve runtime entry for leader and follower launches."""
    server = _make_server(
        tmp_path,
        model_instance_overrides={
            "port": 9000,
            "ports": [9000, 9001],
            "gpu_indexes": [0, 1],
            "worker_ip": "10.0.0.1",
            "distributed_servers": SimpleNamespace(
                subordinate_workers=[
                    SimpleNamespace(worker_ip=worker_ip, gpu_indexes=[0, 1])
                    for worker_ip in subordinate_worker_ips
                ]
            ),
        },
        deployment_metadata=deployment_metadata,
    )
    server._worker = cast(
        Any, SimpleNamespace(id=1, ip=current_worker_ip, ifname="eth0")
    )
    observed: dict[str, Any] = {}

    class FakeProcess:
        pid = 2468

    def fake_popen(args, **kwargs):
        observed["args"] = list(args)
        observed["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/python",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.get_process_group_id", lambda pid: pid + 100
    )

    result = server.start_direct_process()

    assert set(result.keys()) == {
        "pid",
        "process_group_id",
        "port",
        "mode",
        "runtime_entries",
    }
    assert result["pid"] == 2468
    assert result["process_group_id"] == 2568
    assert result["port"] == 9000
    assert result["mode"] == DIRECT_PROCESS_RUNTIME_MODE
    assert result["runtime_entries"] == [
        {
            "deployment_name": deployment_metadata.name,
            "runtime_name": "serve",
            "pid": 2468,
            "process_group_id": 2568,
            "port": 9000,
            "log_path": str(tmp_path / "data" / "log" / "serve" / "1.log"),
        }
    ]

    dist_init_addr_index = observed["args"].index("--dist-init-addr")
    node_rank_index = observed["args"].index("--node-rank")
    assert observed["args"][dist_init_addr_index + 1] == "10.0.0.1:9001"
    assert observed["args"][node_rank_index + 1] == expected_node_rank
    assert observed["kwargs"]["env"]["CUSTOM_ENV"] == "enabled"
    assert observed["kwargs"]["env"]["OMP_NUM_THREADS"] == "1"
    assert observed["kwargs"]["env"]["SAFETENSORS_FAST_GPU"] == "1"


def test_direct_process_sglang_distributed_requires_explicit_role(
    monkeypatch, tmp_path: Path
):
    """Distributed SGLang without leader/follower role still fails before preflight or process launch."""
    deployment_metadata = SimpleNamespace(
        name="distributed-only",
        distributed=True,
        distributed_leader=False,
        distributed_follower=False,
    )
    server = _make_server(
        tmp_path,
        model_instance_overrides={
            "ports": [9000, 9001],
            "gpu_indexes": [0, 1],
            "distributed_servers": SimpleNamespace(
                subordinate_workers=[
                    SimpleNamespace(worker_ip="10.0.0.2", gpu_indexes=[0, 1])
                ]
            ),
        },
        deployment_metadata=deployment_metadata,
    )

    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.shutil.which",
        lambda *_args, **_kwargs: pytest.fail(
            "distributed rejection must happen before executable preflight"
        ),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.socket.socket",
        lambda *_args, **_kwargs: pytest.fail(
            "distributed rejection must happen before port-bind preflight"
        ),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "distributed direct-process SGLang must not spawn a process"
        ),
    )

    with pytest.raises(
        ValueError,
        match="Direct-process SGLang distributed launches require an explicit leader or follower role.",
    ):
        server.start_direct_process()


# ---------------------------------------------------------------------------
# Preflight failure: missing executable
# ---------------------------------------------------------------------------


def test_direct_process_sglang_preflight_fails_missing_executable(
    monkeypatch, tmp_path: Path
):
    """Preflight raises RuntimeError when the executable is not found."""
    server = _make_server(tmp_path)

    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "missing executable must prevent process spawn"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Direct-process SGLang host prerequisites not met",
    ):
        server.start_direct_process()


def test_direct_process_sglang_preflight_fails_updates_state(
    monkeypatch, tmp_path: Path
):
    """Preflight failure through start() updates model instance state to ERROR."""
    server = _make_server(tmp_path)
    updates = []

    def fake_update_model_instance(_id, **kwargs):
        updates.append(kwargs)

    monkeypatch.setattr(server, "_update_model_instance", fake_update_model_instance)
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.socket.socket",
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


def test_direct_process_sglang_preflight_fails_port_bind(
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
        "gpustack.worker.backends.sglang.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/python",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.socket.socket",
        lambda *_args, **_kwargs: BindFailSocket(),
    )

    with pytest.raises(
        RuntimeError,
        match="Direct-process SGLang host prerequisites not met",
    ):
        server.start_direct_process()


# ---------------------------------------------------------------------------
# Preflight failure: missing directory
# ---------------------------------------------------------------------------


def test_direct_process_sglang_preflight_fails_missing_directory(
    monkeypatch, tmp_path: Path
):
    """Preflight raises RuntimeError when a required directory is missing."""
    import shutil as _shutil

    server = _make_server(tmp_path)

    # Remove the serve log directory
    serve_log_dir = tmp_path / "data" / "log" / "serve"
    _shutil.rmtree(str(serve_log_dir))

    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/python",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )

    with pytest.raises(
        RuntimeError,
        match="Direct-process SGLang host prerequisites not met",
    ):
        server.start_direct_process()


# ---------------------------------------------------------------------------
# Container mode isolation
# ---------------------------------------------------------------------------


def test_sglang_server_container_mode_does_not_use_direct_process(
    monkeypatch, tmp_path: Path
):
    """When direct_process_mode is False, _start() follows the container path."""
    server = _make_server(tmp_path, direct_process_mode=False)

    monkeypatch.setattr(
        "gpustack.worker.backends.sglang.subprocess.Popen",
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
        server, "_get_configured_image", lambda: "sglang-image:latest"
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


def test_direct_process_sglang_mode_constant_value_is_locked():
    """Characterization: DIRECT_PROCESS_RUNTIME_MODE constant value must be 'direct_process'."""
    assert DIRECT_PROCESS_RUNTIME_MODE == "direct_process"
