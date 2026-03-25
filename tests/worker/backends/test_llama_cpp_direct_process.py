import importlib
import json
import logging
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gpustack.schemas.models import BackendEnum, ModelInstanceStateEnum


fcntl_stub = types.ModuleType("fcntl")
setattr(fcntl_stub, "LOCK_EX", 1)
setattr(fcntl_stub, "LOCK_UN", 2)
setattr(fcntl_stub, "lockf", lambda *args, **kwargs: None)
setattr(fcntl_stub, "flock", lambda *args, **kwargs: None)


def _import_llama_cpp_module():
    original_fcntl = sys.modules.get("fcntl")
    sys.modules["fcntl"] = fcntl_stub
    try:
        return importlib.import_module("gpustack.worker.backends.llama_cpp")
    finally:
        if original_fcntl is None:
            sys.modules.pop("fcntl", None)
        else:
            sys.modules["fcntl"] = original_fcntl


llama_cpp_module = _import_llama_cpp_module()
LlamaCppServer = llama_cpp_module.LlamaCppServer
DIRECT_PROCESS_RUNTIME_MODE = llama_cpp_module.DIRECT_PROCESS_RUNTIME_MODE


def _install_fake_launch_artifacts(server: LlamaCppServer, tmp_path: Path):
    prepared_root = tmp_path / "prepared" / "llama.cpp"
    artifacts_root = prepared_root / "artifacts"
    bin_root = prepared_root / "bin"
    runtime_root = tmp_path / "runtime" / "1"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    bin_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    prepared_executable = bin_root / "llama-server"
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
                "name": "llama-server",
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
        prepared_provenance=json.loads(
            prepared_provenance.read_text(encoding="utf-8")
        ),
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
) -> LlamaCppServer:
    data_dir = tmp_path / "data"
    cache_dir = data_dir / "cache"
    log_dir = data_dir / "log"
    (log_dir / "serve").mkdir(parents=True, exist_ok=True)
    (data_dir / "worker").mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    server = cast(Any, LlamaCppServer.__new__(LlamaCppServer))
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
            backend=BackendEnum.LLAMA_CPP,
            backend_version="cpu",
            backend_parameters=["--ctx-size=4096", "--threads", "8"],
            categories=[],
            env={"CUSTOM_ENV": "enabled"},
            run_command=None,
            image_name=None,
        ),
    )
    server._model_instance = cast(
        Any,
        SimpleNamespace(
            backend_version="cpu",
            id=1,
            model_id=11,
            name="test-llama-cpp-instance",
            model_name="test-llama-cpp-model",
            port=9010,
            ports=[9010],
            gpu_indexes=[0],
        ),
    )
    server._model_path = "/models/test-model.gguf"
    server._draft_model_path = None
    server.inference_backend = cast(
        Any,
        SimpleNamespace(
            get_container_entrypoint=lambda *_args, **_kwargs: pytest.fail(
                "direct-process llama.cpp must not look up container entrypoints"
            )
        ),
    )
    server._flatten_backend_param = lambda: ["--ctx-size=4096", "--threads", "8"]
    server._get_selected_gpu_devices = lambda: []
    server._get_device_info = lambda: ("nvidia", None, None)
    server._get_deployment_metadata = lambda: cast(
        Any,
        SimpleNamespace(
            name="test-llama-cpp-instance",
            distributed=False,
            distributed_leader=False,
            distributed_follower=False,
        ),
    )
    _install_fake_launch_artifacts(server, tmp_path)
    return server


def test_llama_cpp_server_supports_direct_process():
    assert LlamaCppServer.supports_direct_process() is True


def test_llama_cpp_server_does_not_support_distributed_direct_process():
    assert LlamaCppServer.supports_distributed_direct_process() is False


def test_llama_cpp_server_direct_process_bootstrap_contract(tmp_path: Path):
    server = _make_server(tmp_path)

    assert LlamaCppServer.get_direct_process_bootstrap_recipe_id() == "llama.cpp"
    assert server.get_direct_process_prepared_environment_identity() == "llama.cpp:cpu"


def test_direct_process_llama_cpp_launch_command_uses_llama_server_and_health_contract(
    tmp_path: Path,
):
    server = _make_server(tmp_path)

    command = server.build_direct_process_command(port=9010)

    assert command[:3] == ["llama-server", "-m", "/models/test-model.gguf"]
    assert "--host" in command
    assert "127.0.0.1" in command
    assert "--port" in command
    assert "9010" in command
    assert "--alias" in command
    assert "test-llama-cpp-model" in command
    assert "--ctx-size=4096" in command
    assert server.get_direct_process_health_path() == "/health"


def test_direct_process_llama_cpp_launch_happy_path(
    monkeypatch, tmp_path: Path
):
    server = _make_server(tmp_path)
    observed: dict = {}

    class FakeProcess:
        pid = 7777

    def fake_popen(args, **kwargs):
        observed["popen_args"] = list(args)
        observed["popen_kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "gpustack.worker.backends.llama_cpp.shutil.which",
        lambda *_args, **_kwargs: "/usr/bin/llama-server",
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.llama_cpp.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.llama_cpp.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.llama_cpp.get_process_group_id", lambda pid: pid + 100
    )
    monkeypatch.setattr(
        server,
        "_get_configured_image",
        lambda: pytest.fail("direct-process llama.cpp must not resolve container images"),
    )
    monkeypatch.setattr(
        server,
        "_create_workload",
        lambda **_kwargs: pytest.fail("direct-process llama.cpp must not create workloads"),
    )

    result = server._start()

    assert result == {
        "pid": 7777,
        "process_group_id": 7877,
        "port": 9010,
        "mode": DIRECT_PROCESS_RUNTIME_MODE,
    }
    assert observed["popen_args"][:3] == [
        str(tmp_path / "prepared" / "llama.cpp" / "artifacts" / "prepared-launch.sh"),
        "__GPUSTACK_PREPARED_EXECUTABLE__",
        "-m",
    ]
    assert "/models/test-model.gguf" in observed["popen_args"]
    assert "--host" in observed["popen_args"]
    assert "127.0.0.1" in observed["popen_args"]
    assert "--port" in observed["popen_args"]
    assert "9010" in observed["popen_args"]
    assert observed["popen_kwargs"]["env"]["CUSTOM_ENV"] == "enabled"
    assert (
        Path(observed["popen_kwargs"]["env"]["GPUSTACK_PREPARED_EXECUTABLE"]).name
        == "llama-server"
    )
    assert str(tmp_path / "prepared" / "llama.cpp" / "bin") in observed[
        "popen_kwargs"
    ]["env"]["PATH"]
    assert observed["popen_kwargs"]["stdin"] is not None
    assert observed["popen_kwargs"]["start_new_session"] in {True, False}


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
def test_direct_process_llama_cpp_distributed_rejected(
    monkeypatch, tmp_path: Path, deployment_metadata
):
    server = _make_server(tmp_path)
    server._get_deployment_metadata = lambda: deployment_metadata

    monkeypatch.setattr(
        "gpustack.worker.backends.llama_cpp.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "distributed_rejected llama.cpp must not spawn a process"
        ),
    )

    with pytest.raises(
        ValueError,
        match="Direct-process llama\\.cpp does not support distributed launches; only single-worker direct-process is implemented",
    ):
        server._start()


def test_direct_process_llama_cpp_preflight_failure_updates_error_state(
    monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    server = _make_server(tmp_path)
    updates = []

    def fake_update_model_instance(_id, **kwargs):
        updates.append(kwargs)

    monkeypatch.setattr(server, "_update_model_instance", fake_update_model_instance)
    monkeypatch.setattr(
        "gpustack.worker.backends.llama_cpp.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.llama_cpp.socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(),
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(
            RuntimeError,
            match="Direct-process llama\\.cpp host prerequisites not met",
        ):
            server.start()

    assert updates == [
        {
            "state_message": "Failed to run llama.cpp: Direct-process llama.cpp host prerequisites not met: `llama-server` is not available on PATH or does not exist",
            "state": ModelInstanceStateEnum.ERROR,
        }
    ]
    assert "Direct-process llama.cpp preflight failed for test-llama-cpp-instance" in caplog.text
