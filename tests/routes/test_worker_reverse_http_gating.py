from types import SimpleNamespace
from typing import Any, cast
import sys
import types

import pytest
from fastapi import Response
from starlette.requests import Request

sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))

from gpustack.api.exceptions import ConflictException
from gpustack.client.worker_filesystem_client import (
    WorkerFilesystemClient,
    WorkerReverseHTTPUnsupportedError,
)
from gpustack.routes import benchmarks, model_instances, openai
from gpustack.schemas.benchmark import BenchmarkStateEnum
from gpustack.schemas.models import BackendEnum, ModelInstanceStateEnum
from gpustack.schemas.workers import (
    Worker,
    WorkerReachabilityCapabilities,
    WorkerReachabilityModeEnum,
    WorkerStatus,
)
from gpustack.worker.logs import LogOptions


class FailIfCalledClient:
    def get(self, *args, **kwargs):
        raise AssertionError("worker HTTP client should not be called")


class FakeResponseContext:
    def __init__(self, *, status_code: int = 200, text: str = "legacy-log-line\n"):
        self.status = status_code
        self.headers = {}
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self) -> str:
        return self._text


class RecordingClient:
    def __init__(self):
        self.calls: list[str] = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return FakeResponseContext()


def _build_request(client) -> Request:
    state = SimpleNamespace(http_client=client, http_client_no_proxy=client)
    app = SimpleNamespace(state=state)
    scope = {
        "type": "http",
        "app": app,
        "method": "GET",
        "path": "/model_instances/1/logs",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 8000),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def _build_openai_request(path: str = "/v1/chat/completions") -> Request:
    state = SimpleNamespace()
    app = SimpleNamespace(state=state)
    scope = {
        "type": "http",
        "app": app,
        "method": "POST",
        "path": path,
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 8000),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def _build_benchmark_request(client) -> Request:
    state = SimpleNamespace(http_client=client, http_client_no_proxy=client)
    app = SimpleNamespace(state=state)
    scope = {
        "type": "http",
        "app": app,
        "method": "GET",
        "path": "/benchmarks/1/logs",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 8000),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def _build_worker(
    *,
    reachability_mode: WorkerReachabilityModeEnum,
    reverse_http: bool,
) -> Worker:
    return Worker(
        id=1,
        name="worker-1",
        labels={},
        cluster_id=1,
        hostname="worker-host",
        ip="10.0.0.10",
        advertise_address="10.0.0.10",
        ifname="eth0",
        port=10150,
        worker_uuid="worker-uuid-1",
        status=WorkerStatus.get_default_status(),
        capabilities=WorkerReachabilityCapabilities(
            outbound_control_ws=(
                reachability_mode == WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS
            ),
            reverse_http=reverse_http,
        ),
        reachability_mode=reachability_mode,
    )


def _build_model_instance() -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        worker_id=1,
        name="instance-1",
        state=ModelInstanceStateEnum.RUNNING,
        model_files=[],
    )


def _build_openai_instance() -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        worker_id=1,
        worker_ip="10.0.0.10",
        port=8001,
    )


def _build_model(*, backend: BackendEnum = BackendEnum.VLLM) -> SimpleNamespace:
    return SimpleNamespace(
        id=11,
        name="resolved-model",
        backend=backend,
        env=None,
    )


def _build_user() -> SimpleNamespace:
    return SimpleNamespace(id=99, is_admin=False)


def _build_benchmark() -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        worker_id=1,
        name="benchmark-1",
        state=BenchmarkStateEnum.RUNNING,
    )


@pytest.mark.asyncio
async def test_nat_worker_reverse_only_route(monkeypatch: pytest.MonkeyPatch):
    worker = _build_worker(
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        reverse_http=False,
    )
    request = _build_request(FailIfCalledClient())

    async def fake_fetch_model_instance(session, id):
        return _build_model_instance()

    async def fake_fetch_worker(session, worker_id):
        return worker

    monkeypatch.setattr(model_instances, "fetch_model_instance", fake_fetch_model_instance)
    monkeypatch.setattr(model_instances, "fetch_worker", fake_fetch_worker)
    monkeypatch.setattr(model_instances, "use_proxy_env_for_url", lambda url: False)

    with pytest.raises(ConflictException) as exc_info:
        await model_instances.get_serving_logs(
            request,
            cast(Any, None),
            1,
            LogOptions(follow=False),
        )

    assert "reverse-http is not available" in exc_info.value.message


@pytest.mark.asyncio
async def test_legacy_worker_reverse_route_still_works(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = _build_worker(
        reachability_mode=WorkerReachabilityModeEnum.REVERSE_PROBE,
        reverse_http=True,
    )
    client = RecordingClient()
    request = _build_request(client)

    async def fake_fetch_model_instance(session, id):
        return _build_model_instance()

    async def fake_fetch_worker(session, worker_id):
        return worker

    monkeypatch.setattr(model_instances, "fetch_model_instance", fake_fetch_model_instance)
    monkeypatch.setattr(model_instances, "fetch_worker", fake_fetch_worker)
    monkeypatch.setattr(model_instances, "use_proxy_env_for_url", lambda url: False)

    response = await model_instances.get_serving_logs(
        request,
        cast(Any, None),
        1,
        LogOptions(follow=False),
    )

    assert response.status_code == 200
    assert bytes(response.body).decode() == "legacy-log-line\n"
    assert client.calls == [
        "http://10.0.0.10:10150/serveLogs/1?tail=-1&follow=False&model_instance_name=instance-1"
    ]


@pytest.mark.asyncio
async def test_ws_worker_with_reverse_http_capability_can_use_reverse_route(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = _build_worker(
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        reverse_http=True,
    )
    client = RecordingClient()
    request = _build_request(client)

    async def fake_fetch_model_instance(session, id):
        return _build_model_instance()

    async def fake_fetch_worker(session, worker_id):
        return worker

    monkeypatch.setattr(model_instances, "fetch_model_instance", fake_fetch_model_instance)
    monkeypatch.setattr(model_instances, "fetch_worker", fake_fetch_worker)
    monkeypatch.setattr(model_instances, "use_proxy_env_for_url", lambda url: False)

    response = await model_instances.get_serving_logs(
        request,
        cast(Any, None),
        1,
        LogOptions(follow=False),
    )

    assert response.status_code == 200
    assert bytes(response.body).decode() == "legacy-log-line\n"
    assert client.calls == [
        "http://10.0.0.10:10150/serveLogs/1?tail=-1&follow=False&model_instance_name=instance-1"
    ]


@pytest.mark.asyncio
async def test_benchmark_log_nat_worker_reverse_only_route(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = _build_worker(
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        reverse_http=False,
    )
    request = _build_benchmark_request(FailIfCalledClient())

    async def fake_benchmark_one_by_id(session, benchmark_id):
        return _build_benchmark()

    async def fake_worker_one_by_id(session, worker_id):
        return worker

    monkeypatch.setattr(benchmarks.Benchmark, "one_by_id", fake_benchmark_one_by_id)
    monkeypatch.setattr(benchmarks.Worker, "one_by_id", fake_worker_one_by_id)
    monkeypatch.setattr(benchmarks, "use_proxy_env_for_url", lambda url: False)

    with pytest.raises(ConflictException) as exc_info:
        await benchmarks.get_benchmark_logs(
            request,
            cast(Any, None),
            1,
            LogOptions(follow=False),
        )

    assert "reverse-http is not available" in exc_info.value.message
    assert "streaming benchmark logs" in exc_info.value.message


@pytest.mark.asyncio
async def test_benchmark_log_legacy_worker_reverse_route_still_works(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = _build_worker(
        reachability_mode=WorkerReachabilityModeEnum.REVERSE_PROBE,
        reverse_http=True,
    )
    client = RecordingClient()
    request = _build_benchmark_request(client)

    async def fake_benchmark_one_by_id(session, benchmark_id):
        return _build_benchmark()

    async def fake_worker_one_by_id(session, worker_id):
        return worker

    monkeypatch.setattr(benchmarks.Benchmark, "one_by_id", fake_benchmark_one_by_id)
    monkeypatch.setattr(benchmarks.Worker, "one_by_id", fake_worker_one_by_id)
    monkeypatch.setattr(benchmarks, "use_proxy_env_for_url", lambda url: False)

    response = await benchmarks.get_benchmark_logs(
        request,
        cast(Any, None),
        1,
        LogOptions(follow=False),
    )

    assert response.status_code == 200
    assert bytes(response.body).decode() == "legacy-log-line\n"
    assert client.calls == [
        "http://10.0.0.10:10150/benchmark_logs/1?tail=-1&follow=False&benchmark_name=benchmark-1"
    ]


@pytest.mark.asyncio
async def test_unsupported_nat_route_message():
    worker = _build_worker(
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        reverse_http=False,
    )
    client = WorkerFilesystemClient()

    try:
        with pytest.raises(
            WorkerReverseHTTPUnsupportedError,
            match="reverse-only server-to-worker operation is unsupported",
        ) as exc_info:
            await client.path_exists(worker, "/models")
    finally:
        await client.close()

    assert "checking worker filesystem paths" in str(exc_info.value)
    assert "reachability mode 'outbound_control_ws'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openai_nat_worker_reverse_only_route(monkeypatch: pytest.MonkeyPatch):
    worker = _build_worker(
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        reverse_http=False,
    )
    request = _build_openai_request()
    request.state.api_key = None

    class FakeUserService:
        def __init__(self, session):
            self.session = session

        async def model_allowed_for_user(self, **kwargs):
            return True

    class FakeModelRouteService:
        def __init__(self, session):
            self.session = session

        async def get_model_ids_by_model_route_name(self, model_name):
            return [_build_model()]

    class FakeWorkerService:
        def __init__(self, session):
            self.session = session

        async def get_by_id(self, worker_id):
            return worker

    async def fake_parse_request_body(request):
        return "route-model", False, {"model": "route-model"}, None

    async def fake_get_running_instance(session, model_id):
        return _build_openai_instance()

    async def fail_handle_standard_request(*args, **kwargs):
        raise AssertionError("OpenAI reverse proxy should not be attempted")

    monkeypatch.setattr(openai, "UserService", FakeUserService)
    monkeypatch.setattr(openai, "ModelRouteService", FakeModelRouteService)
    monkeypatch.setattr(openai, "WorkerService", FakeWorkerService)
    monkeypatch.setattr(openai, "parse_request_body", fake_parse_request_body)
    monkeypatch.setattr(openai, "get_running_instance", fake_get_running_instance)
    monkeypatch.setattr(openai, "handle_standard_request", fail_handle_standard_request)

    with pytest.raises(ConflictException) as exc_info:
        await openai.proxy_request_by_model(
            request,
            cast(Any, _build_user()),
            cast(Any, None),
        )

    assert "reverse-http is not available" in exc_info.value.message
    assert "proxying OpenAI-compatible requests" in exc_info.value.message


@pytest.mark.asyncio
async def test_openai_legacy_worker_reverse_route_still_works(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = _build_worker(
        reachability_mode=WorkerReachabilityModeEnum.REVERSE_PROBE,
        reverse_http=True,
    )
    request = _build_openai_request()
    request.state.api_key = None
    captured: dict[str, Any] = {}

    class FakeUserService:
        def __init__(self, session):
            self.session = session

        async def model_allowed_for_user(self, **kwargs):
            return True

    class FakeModelRouteService:
        def __init__(self, session):
            self.session = session

        async def get_model_ids_by_model_route_name(self, model_name):
            return [_build_model()]

    class FakeWorkerService:
        def __init__(self, session):
            self.session = session

        async def get_by_id(self, worker_id):
            return worker

    async def fake_parse_request_body(request):
        return "route-model", False, {"model": "route-model"}, None

    async def fake_get_running_instance(session, model_id):
        return _build_openai_instance()

    async def fake_handle_standard_request(
        request,
        url,
        body_json,
        form_data,
        extra_headers=None,
    ):
        captured["url"] = url
        captured["body_json"] = body_json
        captured["extra_headers"] = extra_headers
        return Response(status_code=200, content=b"ok")

    monkeypatch.setattr(openai, "UserService", FakeUserService)
    monkeypatch.setattr(openai, "ModelRouteService", FakeModelRouteService)
    monkeypatch.setattr(openai, "WorkerService", FakeWorkerService)
    monkeypatch.setattr(openai, "parse_request_body", fake_parse_request_body)
    monkeypatch.setattr(openai, "get_running_instance", fake_get_running_instance)
    monkeypatch.setattr(openai, "handle_standard_request", fake_handle_standard_request)

    response = await openai.proxy_request_by_model(
        request,
        cast(Any, _build_user()),
        cast(Any, None),
    )

    assert response.status_code == 200
    assert captured["url"] == "http://10.0.0.10:10150/proxy/v1/chat/completions"
    assert captured["body_json"] == {"model": "resolved-model"}
    assert captured["extra_headers"] == {
        "X-Target-Port": "8001",
        "Authorization": "Bearer None",
    }


@pytest.mark.asyncio
async def test_reverse_http_capability_allows_reverse_route_even_in_ws_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = _build_worker(
        reachability_mode=WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
        reverse_http=True,
    )
    client = RecordingClient()
    request = _build_request(client)

    async def fake_fetch_model_instance(session, id):
        return _build_model_instance()

    async def fake_fetch_worker(session, worker_id):
        return worker

    monkeypatch.setattr(model_instances, "fetch_model_instance", fake_fetch_model_instance)
    monkeypatch.setattr(model_instances, "fetch_worker", fake_fetch_worker)
    monkeypatch.setattr(model_instances, "use_proxy_env_for_url", lambda url: False)

    response = await model_instances.get_serving_logs(
        request,
        cast(Any, None),
        1,
        LogOptions(follow=False),
    )

    assert response.status_code == 200
    assert bytes(response.body).decode() == "legacy-log-line\n"
    assert client.calls == [
        "http://10.0.0.10:10150/serveLogs/1?tail=-1&follow=False&model_instance_name=instance-1"
    ]


@pytest.mark.asyncio
async def test_reverse_http_capability_false_blocks_reverse_route_even_in_legacy_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = _build_worker(
        reachability_mode=WorkerReachabilityModeEnum.REVERSE_PROBE,
        reverse_http=False,
    )
    request = _build_request(FailIfCalledClient())

    async def fake_fetch_model_instance(session, id):
        return _build_model_instance()

    async def fake_fetch_worker(session, worker_id):
        return worker

    monkeypatch.setattr(model_instances, "fetch_model_instance", fake_fetch_model_instance)
    monkeypatch.setattr(model_instances, "fetch_worker", fake_fetch_worker)
    monkeypatch.setattr(model_instances, "use_proxy_env_for_url", lambda url: False)

    with pytest.raises(ConflictException) as exc_info:
        await model_instances.get_serving_logs(
            request,
            cast(Any, None),
            1,
            LogOptions(follow=False),
        )

    assert "reverse-http is not available" in exc_info.value.message
    assert "reachability mode 'reverse_probe'" in exc_info.value.message
