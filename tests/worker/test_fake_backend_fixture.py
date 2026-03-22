import contextlib
import time

import psutil
import pytest

from gpustack.utils import network
from gpustack.utils.process import terminate_process_tree
from tests.worker.fake_backend_fixture import fake_backend_fixture


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_fake_backend_fixture_ready_localhost_health(fake_backend_fixture):
    handle = fake_backend_fixture.start_ready_server()
    model_instance = fake_backend_fixture.make_model_instance(handle.port)
    health_check_url = f"http://{model_instance.worker_ip}:{model_instance.port}/v1/models"

    assert await network.is_url_reachable(
        health_check_url,
        timeout_in_second=5,
        retry_interval_in_second=1,
    )


def test_fake_backend_fixture_early_fail(fake_backend_fixture):
    handle = fake_backend_fixture.start_early_exit(
        exit_code=17,
        message="fake backend exited early",
    )

    _, stderr = handle.process.communicate(timeout=5)

    assert handle.process.returncode == 17
    assert "fake backend exited early" in stderr


def test_fake_backend_fixture_parent_child_process_tree(fake_backend_fixture):
    handle, child_pid_path = fake_backend_fixture.start_process_tree()
    child_pid = handle.wait_for_child_pid(child_pid_path)

    parent = psutil.Process(handle.process.pid)
    children = {process.pid for process in parent.children(recursive=True)}

    assert child_pid in children

    terminate_process_tree(handle.process.pid)

    assert _wait_until(lambda: handle.process.poll() is not None, timeout=5)
    assert _wait_until(lambda: not psutil.pid_exists(child_pid), timeout=5)

    with contextlib.suppress(psutil.NoSuchProcess):
        assert not parent.is_running()
