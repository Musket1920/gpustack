import importlib
import logging
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gpustack.schemas.models import BackendEnum, ModelInstanceStateEnum


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
            id=1,
            name="test-instance",
            model_name="test-model",
            port=8000,
            ports=[8000],
            gpu_indexes=[0],
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

    assert observed["versioned_default_args"] == [
        "vllm",
        "serve",
        "/models/test-model",
    ]
    assert observed["versioned_model_path"] is None
    assert observed["versioned_port"] is None
    assert observed["popen_args"][:3] == [
        "custom-vllm",
        "serve",
        "/models/test-model",
    ]
    assert "--host" in observed["popen_args"]
    assert "127.0.0.1" in observed["popen_args"]
    assert "--port" in observed["popen_args"]
    assert "8000" in observed["popen_args"]
    assert observed["popen_kwargs"]["env"]["CUSTOM_ENV"] == "enabled"
    assert observed["popen_kwargs"]["stdin"] is not None
    assert observed["popen_kwargs"]["start_new_session"] in {True, False}
    assert start_result == {
        "pid": 4321,
        "process_group_id": 4420,
        "port": 8000,
        "mode": DIRECT_PROCESS_RUNTIME_MODE,
    }


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
def test_direct_process_vllm_unsupported_variants(
    monkeypatch, tmp_path: Path, deployment_metadata
):
    server = _make_server(tmp_path)
    server._get_deployment_metadata = lambda: deployment_metadata

    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "unsupported direct-process vLLM variants must not spawn a process"
        ),
    )
    monkeypatch.setattr(
        "gpustack.worker.backends.vllm.create_workload",
        lambda *_args, **_kwargs: pytest.fail(
            "unsupported direct-process vLLM variants must not create workloads"
        ),
    )

    with pytest.raises(
        ValueError,
        match="Direct-process vLLM does not support distributed or Ray launches",
    ):
        server._start()


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
