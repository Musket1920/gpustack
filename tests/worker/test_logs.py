import asyncio
import importlib
import sys
from typing import List, Union
import types
import pytest

def _import_with_fcntl_stub(module_name: str):
    fcntl_stub = types.ModuleType("fcntl")
    setattr(fcntl_stub, "LOCK_EX", 1)
    setattr(fcntl_stub, "LOCK_UN", 2)
    setattr(fcntl_stub, "lockf", lambda *args, **kwargs: None)
    setattr(fcntl_stub, "flock", lambda *args, **kwargs: None)
    original_fcntl = sys.modules.get("fcntl")
    sys.modules["fcntl"] = fcntl_stub
    try:
        return importlib.import_module(module_name)
    finally:
        if original_fcntl is None:
            sys.modules.pop("fcntl", None)
        else:
            sys.modules["fcntl"] = original_fcntl


route_logs = _import_with_fcntl_stub("gpustack.routes.worker.logs")
worker_logs = _import_with_fcntl_stub("gpustack.worker.logs")
LogOptions = worker_logs.LogOptions
log_generator = worker_logs.log_generator


@pytest.fixture
def sample_log_file(tmp_path):
    log_content = "line1\nline2\nline3\nline4\nline5\n"
    log_file = tmp_path / "test.log"
    log_file.write_text(log_content)
    return log_file


@pytest.fixture
def large_log_file(tmp_path):
    # Create a log file with 2KB in two lines
    log_content = "line" * 256 + "\n" + "line" * 256 + "\n"
    log_file = tmp_path / "large_test.log"
    log_file.write_text(log_content)
    return log_file


def normalize_newlines(data: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(data, str):
        return data.replace("\r\n", "\n")
    elif isinstance(data, list):
        return [line.replace("\r\n", "\n") for line in data]


@pytest.mark.asyncio
async def test_log_generator_default(sample_log_file):
    options = LogOptions()
    log_path = str(sample_log_file)

    result = normalize_newlines(
        [line async for line in log_generator(log_path, options)]
    )
    assert result == [
        "line1\n",
        "line2\n",
        "line3\n",
        "line4\n",
        "line5\n",
    ]


@pytest.mark.asyncio
async def test_log_generator_tail(sample_log_file):
    options = LogOptions(tail=2)
    log_path = str(sample_log_file)

    result = normalize_newlines(
        [line async for line in log_generator(log_path, options)]
    )
    assert result == ["line4\n", "line5\n"]


@pytest.mark.asyncio
async def test_log_generator_follow(sample_log_file):
    options = LogOptions(follow=True)
    log_path = str(sample_log_file)

    generator = log_generator(log_path, options)
    result = []
    async for line in generator:
        result.append(line)
        if len(result) == 5:
            break
    assert normalize_newlines(result) == [
        "line1\n",
        "line2\n",
        "line3\n",
        "line4\n",
        "line5\n",
    ]

    # Append a new line to the log file
    with open(log_path, "a") as file:
        file.write("line6\n")
    try:
        line6 = await asyncio.wait_for(generator.__anext__(), timeout=1)
        assert normalize_newlines(line6) == "line6\n"
    except StopAsyncIteration:
        pytest.fail("Expected a new line in the log file")


@pytest.mark.asyncio
async def test_log_generator_empty_file(tmp_path):
    empty_file = tmp_path / "empty.log"
    empty_file.touch()
    options = LogOptions(tail=0)

    result = [line async for line in log_generator(empty_file, options)]
    assert result == []


@pytest.mark.asyncio
async def test_log_generator_tail_larger_than_file(sample_log_file):
    options = LogOptions(tail=10)
    log_path = str(sample_log_file)

    result = normalize_newlines(
        [line async for line in log_generator(log_path, options)]
    )
    assert result == ["line1\n", "line2\n", "line3\n", "line4\n", "line5\n"]


@pytest.mark.asyncio
async def test_log_generator_tail_large_file(large_log_file):
    options = LogOptions(tail=1)
    log_path = str(large_log_file)

    result = normalize_newlines(
        [line async for line in log_generator(log_path, options)]
    )
    assert result == ["line" * 256 + "\n"]


@pytest.mark.asyncio
async def test_log_generator_tail_larger_than_large_file(large_log_file):
    options = LogOptions(tail=3)
    log_path = str(large_log_file)

    result = normalize_newlines(
        [line async for line in log_generator(log_path, options)]
    )
    assert result == ["line" * 256 + "\n", "line" * 256 + "\n"]


@pytest.mark.asyncio
async def test_direct_process_logs_stream_files_only(monkeypatch, tmp_path):
    log_file = tmp_path / "serve.log"
    log_file.write_text("line1\n", encoding="utf-8")
    container_calls: list[dict] = []

    def record_logs_workload(**kwargs):
        container_calls.append(kwargs)
        return "container should not be read"

    monkeypatch.setattr(route_logs, "logs_workload", record_logs_workload)

    generator = route_logs.combined_log_generator(
        str(log_file),
        "",
        LogOptions(follow=True),
        "test-instance",
        file_log_exists=True,
        file_only=True,
    )

    first_line = await asyncio.wait_for(generator.__anext__(), timeout=1)
    assert normalize_newlines(first_line) == "line1\n"

    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write("line2\n")

    second_line = await asyncio.wait_for(generator.__anext__(), timeout=1)
    assert normalize_newlines(second_line) == "line2\n"
    assert container_calls == []


@pytest.mark.asyncio
async def test_container_mode_logs_still_probe_container_stream(monkeypatch, tmp_path):
    log_file = tmp_path / "serve.log"
    log_file.write_text("file-line\n", encoding="utf-8")
    container_calls: list[dict] = []

    def fake_logs_workload(**kwargs):
        container_calls.append(kwargs)
        return "container-line\n"

    monkeypatch.setattr(route_logs, "logs_workload", fake_logs_workload)

    result = normalize_newlines(
        [
            line
            async for line in route_logs.combined_log_generator(
                str(log_file),
                "",
                LogOptions(follow=False),
                "test-instance",
                file_log_exists=True,
                file_only=False,
            )
        ]
    )

    assert result == ["file-line\n", "container-line\n"]
    assert container_calls == [
        {"name": "test-instance", "tail": 1, "follow": False},
        {"name": "test-instance", "tail": -1, "follow": False},
    ]


# ---------------------------------------------------------------------------
# Characterization: file-log routing contract is locked
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_direct_process_file_only_no_file_raises_not_found(monkeypatch, tmp_path):
    """Characterization: file_only=True with no log file raises NotFoundException, not silent empty."""
    container_calls: list[dict] = []

    def record_logs_workload(**kwargs):
        container_calls.append(kwargs)
        return "should not be called"

    monkeypatch.setattr(route_logs, "logs_workload", record_logs_workload)

    # Import NotFoundException from the api module
    from gpustack.api.exceptions import NotFoundException

    with pytest.raises(NotFoundException):
        async for _ in route_logs.combined_log_generator(
            str(tmp_path / "nonexistent.log"),
            "",
            LogOptions(follow=False),
            "test-instance",
            file_log_exists=False,
            file_only=True,
        ):
            pass

    assert container_calls == [], "container must not be probed when file_only=True"


@pytest.mark.asyncio
async def test_direct_process_file_only_never_calls_container(monkeypatch, tmp_path):
    """Characterization: file_only=True never calls logs_workload regardless of content."""
    log_file = tmp_path / "direct.log"
    log_file.write_text("direct-line\n", encoding="utf-8")
    container_calls: list[dict] = []

    def record_logs_workload(**kwargs):
        container_calls.append(kwargs)
        return "should not be called"

    monkeypatch.setattr(route_logs, "logs_workload", record_logs_workload)

    result = normalize_newlines(
        [
            line
            async for line in route_logs.combined_log_generator(
                str(log_file),
                "",
                LogOptions(follow=False),
                "test-instance",
                file_log_exists=True,
                file_only=True,
            )
        ]
    )

    assert result == ["direct-line\n"]
    assert container_calls == [], "container must not be probed when file_only=True"


@pytest.mark.asyncio
async def test_direct_process_bootstrap_logs_stay_file_only_without_implying_readiness(
    monkeypatch, tmp_path
):
    log_file = tmp_path / "bootstrap.log"
    bootstrap_lines = [
        "Preparing direct-process bootstrap environment before launch.\n",
        "Prepared direct-process bootstrap environment; executable provenance: resolved; launching inference server (waiting for readiness).\n",
        "Failed to prepare direct-process bootstrap environment: bootstrap cache missing\n",
    ]
    log_file.write_text("".join(bootstrap_lines), encoding="utf-8")
    readiness_probes: list[str] = []
    container_calls: list[dict] = []

    def record_container_ready(name: str):
        readiness_probes.append(name)
        return True

    def record_logs_workload(**kwargs):
        container_calls.append(kwargs)
        return "container should not be read"

    monkeypatch.setattr(route_logs, "is_container_logs_ready", record_container_ready)
    monkeypatch.setattr(route_logs, "logs_workload", record_logs_workload)

    result = normalize_newlines(
        [
            line
            async for line in route_logs.combined_log_generator(
                str(log_file),
                "",
                LogOptions(follow=False),
                "test-instance",
                file_log_exists=True,
                file_only=True,
            )
        ]
    )

    assert result == bootstrap_lines
    assert readiness_probes == []
    assert container_calls == []
    assert all("healthy" not in line.lower() for line in result)
    assert all("running" not in line.lower() for line in result)


@pytest.mark.asyncio
async def test_log_generator_file_path_is_canonical_source_for_direct_process(tmp_path):
    """Characterization: log_generator reads from the exact file path provided."""
    log_file = tmp_path / "serve" / "42.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("instance-42-line\n", encoding="utf-8")

    options = LogOptions()
    result = normalize_newlines(
        [line async for line in log_generator(str(log_file), options)]
    )

    assert result == ["instance-42-line\n"]
