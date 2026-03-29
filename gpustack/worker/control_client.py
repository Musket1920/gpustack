import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import random
from typing import Any, Callable, Optional
from urllib.parse import urlparse, urlunparse

import aiohttp

from gpustack import envs
from gpustack.client.generated_http_client import default_versioned_prefix
from gpustack.config.config import Config
from gpustack.config.registration import read_worker_token
from gpustack.schemas.workers import (
    WorkerCommandMessage,
    WorkerControlMessageBase,
    WorkerControlMessageTypeEnum,
    WorkerErrorMessage,
    WorkerHelloMessage,
    WorkerPingMessage,
    WorkerPongMessage,
    WorkerReachabilityCapabilities,
    WorkerReachabilityModeEnum,
)
from gpustack.server.worker_control_observability import (
    control_log_extra,
    record_session_reconnect,
    sanitize_log_value,
)
from gpustack.utils.network import use_proxy_env_for_url

logger = logging.getLogger(__name__)


def worker_control_capabilities() -> WorkerReachabilityCapabilities:
    return WorkerReachabilityCapabilities(
        outbound_control_ws=envs.WORKER_CONTROL_WS_ENABLED,
        reverse_http=envs.WORKER_REVERSE_HTTP_ENABLED,
    )


def worker_control_reachability_mode() -> WorkerReachabilityModeEnum:
    if envs.WORKER_CONTROL_ROLLOUT_MODE == "ws_preferred":
        return WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS
    return WorkerReachabilityModeEnum.REVERSE_PROBE


def worker_control_session_reachability_mode() -> WorkerReachabilityModeEnum:
    return WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS


def worker_control_ws_url(server_url: str) -> str:
    parsed = urlparse(server_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/") + default_versioned_prefix + "/workers/control/ws"
    return urlunparse(parsed._replace(scheme=scheme, path=path, params="", query="", fragment=""))


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass
class WorkerControlSessionSnapshot:
    session_generation: int = 0
    session_id: Optional[str] = None
    protocol_version: int = 1
    connected_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    advertised_session_id: Optional[str] = None
    reachability_mode: WorkerReachabilityModeEnum = field(
        default_factory=worker_control_session_reachability_mode
    )
    capabilities: WorkerReachabilityCapabilities = field(
        default_factory=worker_control_capabilities
    )


class WorkerControlConnectionClosed(Exception):
    def __init__(self, *, code: Optional[int], reason: Optional[str] = None):
        self.code = code
        self.reason = reason or "worker control websocket closed"
        super().__init__(self.reason)


class WorkerControlAuthError(Exception):
    pass


class WorkerControlClient:
    def __init__(
        self,
        cfg: Config,
        worker_id_getter: Callable[[], int],
        worker_uuid_getter: Callable[[], str],
        capabilities_getter: Callable[[], WorkerReachabilityCapabilities] = worker_control_capabilities,
        reachability_mode_getter: Callable[
            [], WorkerReachabilityModeEnum
        ] = worker_control_reachability_mode,
        *,
        session_factory: Optional[Callable[[], Any]] = None,
        ping_interval_seconds: Optional[float] = None,
        reconnect_initial_delay_seconds: float = 1.0,
        reconnect_max_delay_seconds: float = 30.0,
        reconnect_jitter_ratio: float = 0.2,
        max_consecutive_auth_failures: int = 5,
        command_executor=None,
    ):
        self._cfg = cfg
        self._worker_id_getter = worker_id_getter
        self._worker_uuid_getter = worker_uuid_getter
        self._capabilities_getter = capabilities_getter
        self._reachability_mode_getter = reachability_mode_getter
        self._session_factory = session_factory or self._build_session
        self._ping_interval_seconds = ping_interval_seconds or max(
            5.0, envs.WORKER_CONTROL_WS_HEARTBEAT_TIMEOUT_SECONDS / 3
        )
        self._reconnect_initial_delay_seconds = reconnect_initial_delay_seconds
        self._reconnect_max_delay_seconds = reconnect_max_delay_seconds
        self._reconnect_jitter_ratio = reconnect_jitter_ratio
        self._max_consecutive_auth_failures = max_consecutive_auth_failures
        self._command_executor = command_executor
        self._stop_event = asyncio.Event()
        self._session_lock = asyncio.Lock()
        self._client_session: Any = None
        self._websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session_state = WorkerControlSessionSnapshot()
        self._connected_event = asyncio.Event()

    @property
    def session_state(self) -> WorkerControlSessionSnapshot:
        return self._session_state

    @property
    def connected_event(self) -> asyncio.Event:
        return self._connected_event

    def set_command_executor(self, command_executor):
        self._command_executor = command_executor

    async def start(self):
        if self._worker_id_getter() is None:
            raise RuntimeError("worker control websocket requires worker registration")

        consecutive_auth_failures = 0
        consecutive_failures = 0

        while not self._stop_event.is_set():
            try:
                await self._connect_and_run()
                consecutive_auth_failures = 0
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except WorkerControlAuthError as e:
                consecutive_auth_failures += 1
                consecutive_failures += 1
                logger.warning(
                    "Worker control websocket authentication failed (attempt %s/%s): %s",
                    consecutive_auth_failures,
                    self._max_consecutive_auth_failures,
                    sanitize_log_value(str(e)),
                    extra=control_log_extra(
                        worker_id=self._worker_id_getter(),
                        session_id=self._session_state.session_id,
                        session_generation=self._session_state.session_generation or None,
                        failure_source="control",
                    ),
                )
                if consecutive_auth_failures >= self._max_consecutive_auth_failures:
                    logger.error(
                        "Worker control websocket giving up after %s consecutive authentication failures.",
                        consecutive_auth_failures,
                        extra=control_log_extra(
                            worker_id=self._worker_id_getter(),
                            session_id=self._session_state.session_id,
                            session_generation=self._session_state.session_generation or None,
                            failure_source="control",
                        ),
                    )
                    break
            except Exception as e:
                consecutive_failures += 1
                logger.warning(
                    "Worker control websocket disconnected: %s",
                    sanitize_log_value(str(e)),
                    extra=control_log_extra(
                        worker_id=self._worker_id_getter(),
                        session_id=self._session_state.session_id,
                        session_generation=self._session_state.session_generation or None,
                        failure_source="transport",
                    ),
                )

            if self._stop_event.is_set():
                break

            await self._sleep(self._compute_backoff_delay(consecutive_failures))

    async def stop(self):
        self._stop_event.set()
        if self._command_executor is not None:
            await self._command_executor.stop()
        await self._close_websocket()
        await self._close_client_session()

    async def _connect_and_run(self):
        token = read_worker_token(str(self._cfg.data_dir))
        if not token:
            raise WorkerControlAuthError("worker token unavailable")

        session = await self._get_client_session()
        websocket_url = worker_control_ws_url(self._cfg.get_server_url())
        headers = {"Authorization": f"Bearer {token}"}
        connect_kwargs = {
            "headers": headers,
            "autoping": False,
            "heartbeat": None,
        }
        ssl = self._ws_ssl_argument(self._cfg.get_server_url())
        if ssl is not None:
            connect_kwargs["ssl"] = ssl

        try:
            websocket = await session.ws_connect(websocket_url, **connect_kwargs)
        except aiohttp.WSServerHandshakeError as e:
            if e.status in {401, 403}:
                raise WorkerControlAuthError(str(e)) from e
            raise

        self._websocket = websocket
        self._connected_event.clear()
        try:
            await self._send_hello(websocket)
            await self._receive_server_hello(websocket)
            self._connected_event.set()
            await self._run_connected_session(websocket)
        finally:
            self._connected_event.clear()
            await self._close_websocket(websocket)

    async def _run_connected_session(self, websocket: aiohttp.ClientWebSocketResponse):
        ping_task = asyncio.create_task(self._ping_loop(websocket))
        receive_task = asyncio.create_task(self._receive_loop(websocket))
        tasks = {ping_task, receive_task}
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in pending:
                task.cancel()
            for task in pending:
                with suppress(asyncio.CancelledError):
                    await task

            for task in done:
                task.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            for task in tasks:
                with suppress(asyncio.CancelledError):
                    await task

    async def _receive_loop(self, websocket: aiohttp.ClientWebSocketResponse):
        while not self._stop_event.is_set():
            message = await websocket.receive()
            if message.type == aiohttp.WSMsgType.TEXT:
                parsed = self._parse_control_message(message.data)
                self._touch_session()

                if isinstance(parsed, WorkerPingMessage):
                    await self._send_pong(websocket)
                    continue

                if isinstance(parsed, WorkerPongMessage):
                    continue

                if isinstance(parsed, WorkerErrorMessage):
                    logger.warning(
                        "Worker control websocket server error: %s (%s)",
                        parsed.error_code,
                        sanitize_log_value(parsed.error_message),
                        extra=control_log_extra(
                            worker_id=self._worker_id_getter(),
                            session_id=self._session_state.session_id,
                            session_generation=self._session_state.session_generation or None,
                            failure_source="control",
                            error_code=parsed.error_code,
                        ),
                    )
                    continue

                if isinstance(parsed, WorkerCommandMessage):
                    if self._command_executor is None:
                        logger.warning(
                            "Ignoring worker control command %s because no command executor is configured.",
                            parsed.command_id,
                        )
                        continue
                    await self._command_executor.handle_command(parsed, websocket)
                    continue

                if isinstance(parsed, WorkerHelloMessage):
                    self._update_session_state(parsed)
                    continue

            if message.type == aiohttp.WSMsgType.ERROR:
                raise message.data or websocket.exception() or WorkerControlConnectionClosed(code=None)

            if message.type in {
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
            }:
                raise WorkerControlConnectionClosed(
                    code=websocket.close_code,
                    reason=getattr(message, "extra", None),
                )

            if message.type == aiohttp.WSMsgType.PING:
                await websocket.pong()
                continue

            if message.type == aiohttp.WSMsgType.PONG:
                continue

            raise WorkerControlConnectionClosed(
                code=websocket.close_code,
                reason=f"unsupported websocket message type: {message.type}",
            )

    async def _receive_server_hello(self, websocket: aiohttp.ClientWebSocketResponse):
        try:
            parsed = await self._receive_next_message(websocket)
        except WorkerControlConnectionClosed as e:
            if e.code == 1008:
                raise WorkerControlAuthError(e.reason) from e
            raise

        if not isinstance(parsed, WorkerHelloMessage):
            raise WorkerControlConnectionClosed(
                code=1008,
                reason="worker control websocket server did not send hello",
            )

        self._update_session_state(parsed)

    async def _receive_next_message(
        self, websocket: aiohttp.ClientWebSocketResponse
    ) -> WorkerControlMessageBase:
        message = await websocket.receive()
        if message.type != aiohttp.WSMsgType.TEXT:
            if message.type in {
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
            }:
                raise WorkerControlConnectionClosed(
                    code=websocket.close_code,
                    reason=getattr(message, "extra", None),
                )
            raise WorkerControlConnectionClosed(
                code=websocket.close_code,
                reason=f"unexpected websocket message type: {message.type}",
            )

        parsed = self._parse_control_message(message.data)
        self._touch_session()
        return parsed

    async def _send_hello(self, websocket: aiohttp.ClientWebSocketResponse):
        hello = WorkerHelloMessage(
            session_id=self._session_state.session_id,
            sent_at=utcnow(),
            worker_uuid=self._worker_uuid_getter(),
            capabilities=self._capabilities_getter(),
            reachability_mode=worker_control_session_reachability_mode(),
        )
        self._session_state.advertised_session_id = hello.session_id
        await websocket.send_json(hello.model_dump(mode="json"))

    async def _send_ping(self, websocket: aiohttp.ClientWebSocketResponse):
        await websocket.send_json(
            WorkerPingMessage(
                session_id=self._session_state.session_id,
                protocol_version=self._session_state.protocol_version,
                sent_at=utcnow(),
            ).model_dump(mode="json")
        )

    async def _send_pong(self, websocket: aiohttp.ClientWebSocketResponse):
        await websocket.send_json(
            WorkerPongMessage(
                session_id=self._session_state.session_id,
                protocol_version=self._session_state.protocol_version,
                sent_at=utcnow(),
            ).model_dump(mode="json")
        )

    async def _ping_loop(self, websocket: aiohttp.ClientWebSocketResponse):
        while not self._stop_event.is_set():
            await self._sleep(self._ping_interval_seconds)
            if self._stop_event.is_set():
                break
            await self._send_ping(websocket)

    def _update_session_state(self, hello: WorkerHelloMessage):
        prior_session_id = self._session_state.session_id
        self._session_state.session_generation += 1
        self._session_state.session_id = hello.session_id
        self._session_state.protocol_version = hello.protocol_version
        self._session_state.connected_at = utcnow()
        self._session_state.last_seen_at = self._session_state.connected_at
        self._session_state.capabilities = hello.capabilities
        self._session_state.reachability_mode = (
            hello.reachability_mode or worker_control_session_reachability_mode()
        )
        if prior_session_id is not None:
            record_session_reconnect(self._worker_id_getter())
            logger.info(
                "Worker control websocket reconnected.",
                extra=control_log_extra(
                    worker_id=self._worker_id_getter(),
                    session_id=self._session_state.session_id,
                    session_generation=self._session_state.session_generation,
                ),
            )
        else:
            logger.info(
                "Worker control websocket connected.",
                extra=control_log_extra(
                    worker_id=self._worker_id_getter(),
                    session_id=self._session_state.session_id,
                    session_generation=self._session_state.session_generation,
                ),
            )

    def _touch_session(self):
        self._session_state.last_seen_at = utcnow()

    def _parse_control_message(self, payload: str):
        base = WorkerControlMessageBase.model_validate_json(payload)
        if base.message_type == WorkerControlMessageTypeEnum.HELLO:
            return WorkerHelloMessage.model_validate_json(payload)
        if base.message_type == WorkerControlMessageTypeEnum.PING:
            return WorkerPingMessage.model_validate_json(payload)
        if base.message_type == WorkerControlMessageTypeEnum.PONG:
            return WorkerPongMessage.model_validate_json(payload)
        if base.message_type == WorkerControlMessageTypeEnum.ERROR:
            return WorkerErrorMessage.model_validate_json(payload)
        if base.message_type == WorkerControlMessageTypeEnum.COMMAND:
            return WorkerCommandMessage.model_validate_json(payload)
        return base

    async def _get_client_session(self) -> Any:
        async with self._session_lock:
            if self._client_session is None or self._client_session.closed:
                self._client_session = self._session_factory()
            return self._client_session

    def _build_session(self) -> aiohttp.ClientSession:
        connector = aiohttp.TCPConnector(limit=envs.TCP_CONNECTOR_LIMIT, force_close=True)
        return aiohttp.ClientSession(
            connector=connector,
            trust_env=use_proxy_env_for_url(self._cfg.get_server_url()),
        )

    async def _close_client_session(self):
        async with self._session_lock:
            session = self._client_session
            self._client_session = None
        if session is not None and not session.closed:
            await session.close()

    async def _close_websocket(
        self, websocket: Optional[aiohttp.ClientWebSocketResponse] = None
    ):
        ws = websocket or self._websocket
        if websocket is None:
            self._websocket = None
        if ws is not None and not ws.closed:
            await ws.close()

    def _compute_backoff_delay(self, attempt: int) -> float:
        bounded = min(
            self._reconnect_max_delay_seconds,
            self._reconnect_initial_delay_seconds * (2 ** max(attempt - 1, 0)),
        )
        jitter = bounded * self._reconnect_jitter_ratio * random.random()
        return bounded + jitter

    async def _sleep(self, delay: float):
        await asyncio.sleep(delay)

    @staticmethod
    def _ws_ssl_argument(server_url: str):
        parsed = urlparse(server_url)
        if parsed.hostname == "127.0.0.1" and parsed.scheme == "https":
            return False
        return None
