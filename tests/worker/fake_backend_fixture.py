import contextlib
from pathlib import Path
import socket
import subprocess
import sys
import time

import psutil
import pytest

from gpustack.schemas.models import ModelInstance, ModelInstanceStateEnum, SourceEnum
from gpustack.utils.process import terminate_process_tree


SCRIPTS_DIR = Path(__file__).with_name("fake_backend_scripts")


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class FakeBackendProcess:
    def __init__(self, process: subprocess.Popen[str], port: int | None = None):
        self.process = process
        self.port = port

    def wait_for_child_pid(self, child_pid_path: Path, timeout: float = 5.0) -> int:
        if not _wait_until(child_pid_path.exists, timeout=timeout):
            raise TimeoutError(f"Timed out waiting for {child_pid_path}")

        child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())
        if not _wait_until(lambda: psutil.pid_exists(child_pid), timeout=timeout):
            raise TimeoutError(f"Timed out waiting for child pid {child_pid}")

        return child_pid

    def terminate(self) -> None:
        if self.process.poll() is None:
            terminate_process_tree(self.process.pid)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=5)


class FakeBackendHarness:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.processes: list[FakeBackendProcess] = []

    def make_model_instance(self, port: int) -> ModelInstance:
        return ModelInstance(
            id=1,
            name="fake-backend-instance",
            worker_id=1,
            worker_name="worker-1",
            worker_ip="127.0.0.1",
            model_id=1,
            model_name="fake-backend-model",
            state=ModelInstanceStateEnum.INITIALIZING,
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
            port=port,
            ports=[port],
        )

    def start_ready_server(self, health_path: str = "/v1/models") -> FakeBackendProcess:
        port = _pick_free_port()
        handle = self._start_process(
            SCRIPTS_DIR / "fake_ready_server.py",
            "--port",
            str(port),
            "--health-path",
            health_path,
            port=port,
        )
        if not _wait_until(lambda: handle.process.poll() is None, timeout=2):
            raise RuntimeError("fake ready server exited before readiness check")
        return handle

    def start_early_exit(
        self, exit_code: int = 17, message: str = "fake backend exited early"
    ) -> FakeBackendProcess:
        return self._start_process(
            SCRIPTS_DIR / "fake_early_exit.py",
            "--exit-code",
            str(exit_code),
            "--message",
            message,
        )

    def start_process_tree(self) -> tuple[FakeBackendProcess, Path]:
        child_pid_path = self.workspace / "fake-child.pid"
        handle = self._start_process(
            SCRIPTS_DIR / "fake_process_tree_parent.py",
            "--child-pid-file",
            str(child_pid_path),
        )
        return handle, child_pid_path

    def cleanup(self) -> None:
        for handle in reversed(self.processes):
            handle.terminate()
        self.processes.clear()

    def _start_process(
        self, script_path: Path, *args: str, port: int | None = None
    ) -> FakeBackendProcess:
        process = subprocess.Popen(
            [sys.executable, str(script_path), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        handle = FakeBackendProcess(process=process, port=port)
        self.processes.append(handle)
        return handle


@pytest.fixture
def fake_backend_fixture(tmp_path: Path):
    harness = FakeBackendHarness(tmp_path)
    yield harness
    harness.cleanup()
