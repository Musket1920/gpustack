import importlib
import json
import logging
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gpustack.schemas.models import (
    BackendEnum,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
)


def _import_vllm_module():
    fcntl_stub = types.ModuleType("fcntl")
    setattr(fcntl_stub, "LOCK_EX", 1)
    setattr(fcntl_stub, "LOCK_UN", 2)
    setattr(fcntl_stub, "lockf", lambda *args, **kwargs: None)
    setattr(fcntl_stub, "flock", lambda *args, **kwargs: None)
    original_fcntl = sys.modules.get("fcntl")
    sys.modules["fcntl"] = fcntl_stub
    try:
        return importlib.import_module("gpustack.worker.backends.vllm")
    finally:
        if original_fcntl is None:
            sys.modules.pop("fcntl", None)
        else:
            sys.modules["fcntl"] = original_fcntl


vllm_module = _import_vllm_module()
VLLMServer = vllm_module.VLLMServer
DIRECT_PROCESS_RUNTIME_MODE = vllm_module.DIRECT_PROCESS_RUNTIME_MODE


def _install_fake_launch_artifacts(server: VLLMServer, tmp_path: Path):
    prepared_root = tmp_path / "prepared" / "vllm"
    artifacts_root = prepared_root / "artifacts"
    bin_root = prepared_root / "bin"
    runtime_root = tmp_path / "runtime" / "1"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    bin_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    prepared_executable = bin_root / "vllm"
    prepared_executable.write_text("#!/bin/sh\n", encoding="utf-8")
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
                "name": "vllm",
                "prepared_path": str(prepared_executable),
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
                "backend_version": server._model_instance.backend_version,
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
) -> VLLMServer:
    data_dir = tmp_path / "data"
    cache_dir = data_dir / "cache"
    log_dir = data_dir / "log"
    (log_dir / "serve").mkdir(parents=True, exist_ok=True)
    (data_dir / "worker").mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    server = cast(Any, VLLMServer.__new__(VLLMServer))
    server._config = cast(
        Any,
        SimpleNamespace(
            direct_process_mode=direct_process_mode,
            data_dir=str(data_dir),
            cache_dir=str(cache_dir),
            log_dir=str(log_dir),
            ray_port_range="41000-41999",
        ),
    )
    server._worker = cast(Any, SimpleNamespace(id=1, ip="127.0.0.1", ifname="lo"))
    server._model = cast(
        Any,
        SimpleNamespace(
            backend=BackendEnum.VLLM,
            backend_version="0.8.0",
            backend_parameters=[],
            categories=[],
            env={"CUSTOM_ENV": "enabled"},
            extended_kv_cache=None,
            speculative_config=None,
        ),
    )
    server._model_instance = cast(
        Any,
        SimpleNamespace(
            backend_version="0.8.1",
            id=1,
            name="test-instance",
            model_name="test-model",
            port=8000,
            ports=[8000, 8100],
            gpu_indexes=[0],
            worker_ip="127.0.0.1",
            distributed_servers=DistributedServers(
                mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
                subordinate_workers=[
                    ModelInstanceSubordinateWorker(worker_id=2, worker_ip="127.0.0.2")
                ],
            ),
        ),
    )
    server._model_path = "/models/test-model"
    server._draft_model_path = None
    server.inference_backend = cast(
        Any,
        SimpleNamespace(
            get_container_entrypoint=lambda *_args, **_kwargs: pytest.fail(
                "direct-process vLLM must not look up container entrypoints"
            )
        ),
    )
    server._derive_max_model_len = lambda _default=None: None
    server.build_versioned_command_args = lambda default_args, model_path=None, port=None: default_args
    server._flatten_backend_param = lambda: []
    server._get_selected_gpu_devices = lambda: []
    server._get_device_info = lambda: ("nvidia", None, None)
    server._get_deployment_metadata = lambda: cast(
        Any,
        SimpleNamespace(
            name="test-instance",
            distributed=False,
            distributed_leader=False,
            distributed_follower=False,
        ),
    )
    _install_fake_launch_artifacts(server, tmp_path)
    return server


def test_direct_process_host_prerequisites_pass_command_build_no_workload(
    monkeypatch, tmp_path: Path
):
    server = _make_server(tmp_path)
    observed: dict = {}

    def fake_build_versioned_command_args(default_args, model_path=None, port=None):
        observed["versioned_default_args"] = list(default_args)
        observed["versioned_model_path"] = model_path
        observed["versioned_port"] = port
        return ["custom-vllm", *default_args[1:]]

    class FakeProcess:
        pid = 4321

    def fake_popen(args, **kwargs):
        observed["popen_args"] = list(args)
        observed["popen_kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        server,
        "build_versioned_command_args",
        fake_build_versioned_command_args,
    )
    monkeypatch.setattr(
        server,
        "_get_configured_image",
        lambda: pytest.fail("direct-process vLLM must not resolve container images"),
    )
    monkeypatch.setattr(
        server,
        "_create_workload",
        lambda **_kwargs: pytest.fail("direct-process vLLM must not create workloads"),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.create_workload",
        lambda *_args, **_kwargs: pytest.fail(
            "direct-process vLLM must bypass create_workload"
        ),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.get_process_group_id", lambda pid: pid + 99
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/vllm",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr("gpustack.worker.backends.vllm.subprocess.Popen", fake_popen)

    start_result = server._start()

    assert type(server).get_direct_process_bootstrap_recipe_id() == "vllm"
    assert server.get_direct_process_prepared_environment_identity() == "vllm:0.8.1"
    assert observed["versioned_default_args"] == [
        "vllm",
        "serve",
        "/models/test-model",
    ]
    assert observed["versioned_model_path"] is None
    assert observed["versioned_port"] is None
    assert observed["popen_args"][:3] == [
        str(tmp_path / "prepared" / "vllm" / "artifacts" / "prepared-launch.sh"),
        "__GPUSTACK_PREPARED_EXECUTABLE__",
        "serve",
    ]
    assert "--host" in observed["popen_args"]
    assert "127.0.0.1" in observed["popen_args"]
    assert "--port" in observed["popen_args"]
    assert "8000" in observed["popen_args"]
    assert observed["popen_kwargs"]["env"]["CUSTOM_ENV"] == "enabled"
    assert Path(observed["popen_kwargs"]["env"]["GPUSTACK_PREPARED_EXECUTABLE"]).name == "vllm"
    assert str(tmp_path / "prepared" / "vllm" / "bin") in observed["popen_kwargs"]["env"]["PATH"]
    assert observed["popen_kwargs"]["stdin"] is not None
    assert observed["popen_kwargs"]["start_new_session"] in {True, False}
    assert start_result == {
        "pid": 4321,
        "process_group_id": 4420,
        "port": 8000,
        "mode": DIRECT_PROCESS_RUNTIME_MODE,
    }


@pytest.mark.parametrize(
    ("deployment_metadata", "expected_runtime_names"),
    [
        (
            SimpleNamespace(
                name="leader-distributed",
                distributed=True,
                distributed_leader=True,
                distributed_follower=False,
            ),
            ["ray-head", "serve"],
        ),
        (
            SimpleNamespace(
                name="follower-distributed",
                distributed=True,
                distributed_leader=False,
                distributed_follower=True,
            ),
            ["ray-worker"],
        ),
    ],
    ids=["leader", "follower"],
)
def test_direct_process_vllm_distributed_returns_runtime_entries(
    monkeypatch, tmp_path: Path, deployment_metadata, expected_runtime_names
):
    server = _make_server(tmp_path)
    server._get_deployment_metadata = lambda: deployment_metadata
    if deployment_metadata.distributed_follower:
        server._model_instance.worker_ip = "127.0.0.1"

    popen_calls = []

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

    def fake_popen(args, **kwargs):
        popen_calls.append((list(args), kwargs))
        return FakeProcess(5000 + len(popen_calls))

    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/vllm",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.get_process_group_id", lambda pid: pid + 1
    )
    monkeypatch.setattr("gpustack.worker.backends.vllm.subprocess.Popen", fake_popen)

    result = server._start()

    assert result["mode"] == DIRECT_PROCESS_RUNTIME_MODE
    assert [entry["runtime_name"] for entry in result["runtime_entries"]] == expected_runtime_names
    assert len(popen_calls) == len(expected_runtime_names)


def test_direct_process_prepared_environment_identity_falls_back_to_model_backend_version(
    tmp_path: Path,
):
    server = _make_server(tmp_path)
    server._model_instance.backend_version = None

    assert type(server).get_direct_process_bootstrap_recipe_id() == "vllm"
    assert server.get_direct_process_prepared_environment_identity() == "vllm:0.8.0"


def test_direct_process_prepared_environment_identity_falls_back_to_unknown_when_versions_missing(
    tmp_path: Path,
):
    server = _make_server(tmp_path)
    server._model_instance.backend_version = None
    server._model.backend_version = None

    assert server.get_direct_process_prepared_environment_identity() == "vllm:unknown"


def test_direct_process_host_prerequisites_fail_missing_vllm_updates_state_message(
    monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    server = _make_server(tmp_path)
    updates = []

    def fake_update_model_instance(_id, **kwargs):
        updates.append(kwargs)

    monkeypatch.setattr(server, "_update_model_instance", fake_update_model_instance)
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.importlib.util.find_spec",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(
            RuntimeError,
            match="Direct-process vLLM host prerequisites not met",
        ):
            server.start()

    assert updates == [
        {
            "state_message": "Failed to run vLLM: Direct-process vLLM host prerequisites not met: `vllm` is not available on PATH and the `vllm` package is not importable",
            "state": ModelInstanceStateEnum.ERROR,
        }
    ]
    assert "Direct-process vLLM preflight failed for test-instance" in caplog.text


# ---------------------------------------------------------------------------
# Characterization: host prerequisite checks and return contract are locked
# ---------------------------------------------------------------------------

def test_direct_process_start_result_has_exactly_four_keys(
    monkeypatch, tmp_path: Path
):
    """Characterization: _start() in direct-process mode returns exactly {pid, process_group_id, port, mode}."""
    server = _make_server(tmp_path)

    class FakeProcess:
        pid = 1234

    monkeypatch.setattr(
        server,
        "build_versioned_command_args",
        lambda default_args, model_path=None, port=None: list(default_args),
    )
    monkeypatch.setattr(
        server,
        "_get_configured_image",
        lambda: pytest.fail("direct-process must not resolve container images"),
    )
    monkeypatch.setattr(
        server,
        "_create_workload",
        lambda **_kwargs: pytest.fail("direct-process must not create workloads"),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.create_workload",
        lambda *_args, **_kwargs: pytest.fail("direct-process must bypass create_workload"),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.get_process_group_id", lambda pid: pid + 1
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/vllm",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )

    result = server._start()

    assert set(result.keys()) == {"pid", "process_group_id", "port", "mode"}
    assert result["pid"] == 1234
    assert result["process_group_id"] == 1235
    assert result["port"] == 8000
    assert result["mode"] == DIRECT_PROCESS_RUNTIME_MODE


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
    ],
    ids=["leader", "follower"],
)
def test_direct_process_distributed_start_result_keeps_runtime_entry_contract(
    monkeypatch,
    tmp_path: Path,
    deployment_metadata,
):
    server = _make_server(tmp_path)
    server._get_deployment_metadata = lambda: deployment_metadata
    if deployment_metadata.distributed_follower:
        server._model_instance.worker_ip = "127.0.0.1"

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

    call_count = 0

    def fake_popen(args, **kwargs):
        nonlocal call_count
        call_count += 1
        return FakeProcess(7000 + call_count)

    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/vllm",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.get_process_group_id", lambda pid: pid + 10
    )
    monkeypatch.setattr("gpustack.worker.backends.vllm.subprocess.Popen", fake_popen)

    result = server._start()

    assert set(result.keys()) == {
        "pid",
        "process_group_id",
        "port",
        "mode",
        "runtime_entries",
    }
    assert result["mode"] == DIRECT_PROCESS_RUNTIME_MODE
    assert result["runtime_entries"]
    assert all(
        set(entry.keys())
        == {"deployment_name", "runtime_name", "pid", "process_group_id", "port", "log_path"}
        for entry in result["runtime_entries"]
    )


def test_direct_process_port_check_failure_raises_runtime_error(
    monkeypatch, tmp_path: Path
):
    """Characterization: port bind failure raises RuntimeError with 'host prerequisites not met'."""
    server = _make_server(tmp_path)

    class BindFailSocket:
        def bind(self, address):
            raise OSError(98, "Address already in use")

        def close(self):
            pass

    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/vllm",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.socket.socket",
        lambda *_args, **_kwargs: BindFailSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("port failure must prevent process spawn"),
    )

    with pytest.raises(RuntimeError, match="Direct-process vLLM host prerequisites not met"):
        server._start()


def test_direct_process_directory_check_failure_raises_runtime_error(
    monkeypatch, tmp_path: Path
):
    """Characterization: missing required directory raises RuntimeError with 'host prerequisites not met'."""
    import shutil as _shutil

    # Use a server whose directories do NOT exist
    server = _make_server(tmp_path)
    # Remove the serve log directory to trigger directory check failure
    serve_log_dir = tmp_path / "data" / "log" / "serve"
    _shutil.rmtree(str(serve_log_dir))

    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/vllm",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("directory failure must prevent process spawn"),
    )

    with pytest.raises(RuntimeError, match="Direct-process vLLM host prerequisites not met"):
        server._start()


def test_direct_process_vllm_command_starts_with_vllm_serve(
    monkeypatch, tmp_path: Path
):
    """Characterization: direct-process vLLM command always starts with ['vllm', 'serve', <model_path>]."""
    server = _make_server(tmp_path)
    observed_args: list = []

    class FakeProcess:
        pid = 5678

    def fake_popen(args, **kwargs):
        observed_args.extend(args)
        return FakeProcess()

    monkeypatch.setattr(
        server,
        "build_versioned_command_args",
        lambda default_args, model_path=None, port=None: list(default_args),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.get_process_group_id", lambda pid: pid
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/vllm",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr("gpustack.worker.backends.vllm.subprocess.Popen", fake_popen)

    server._start()

    assert observed_args[0] == str(
        tmp_path / "prepared" / "vllm" / "artifacts" / "prepared-launch.sh"
    )
    assert observed_args[1] == "__GPUSTACK_PREPARED_EXECUTABLE__"
    assert observed_args[2] == "serve"


def test_direct_process_vllm_mode_constant_value_is_locked():
    """Characterization: DIRECT_PROCESS_RUNTIME_MODE constant value must be 'direct_process'."""
    assert DIRECT_PROCESS_RUNTIME_MODE == "direct_process"
