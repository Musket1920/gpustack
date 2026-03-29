import asyncio
import logging
from typing import Union

from fastapi import APIRouter, WebSocket, WebSocketException, status
from pydantic import ValidationError
from sqlalchemy.orm import selectinload
from starlette.websockets import WebSocketDisconnect

from gpustack import envs
from gpustack.api.auth import get_user_from_api_token
from gpustack.schemas.users import User, UserRole
from gpustack.schemas.workers import (
    WorkerCommandAckMessage,
    WorkerCommandResultMessage,
    WorkerControlChannelEnum,
    WorkerControlMessageBase,
    WorkerControlMessageTypeEnum,
    WorkerErrorMessage,
    WorkerHelloMessage,
    WorkerPingMessage,
    WorkerPongMessage,
    WorkerReachabilityModeEnum,
)
from gpustack.server.db import async_session
from gpustack.server.worker_control import (
    WorkerControlInboundLimiter,
    WorkerControlMessageTooLarge,
    WorkerControlRateLimitExceeded,
    WorkerControlSession,
    WorkerControlSessionRegistry,
    utcnow,
)
from gpustack.server.worker_command_service import (
    WorkerCommandDispatchService,
    WorkerCommandService,
    StaleWorkerSessionError,
)
from gpustack.server.worker_control_observability import (
    control_log_extra,
    sanitize_log_value,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _websocket_access_token(websocket: WebSocket) -> str | None:
    authorization = websocket.headers.get("authorization")
    if authorization:
        scheme, _, credentials = authorization.partition(" ")
        if scheme.lower() == "bearer" and credentials:
            return credentials.strip()

    return websocket.headers.get("x-api-key")


async def authenticate_worker_websocket(websocket: WebSocket) -> User:
    token = _websocket_access_token(websocket)
    if not token:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid authentication credentials",
        )

    try:
        async with async_session() as session:
            authed_user, _ = await get_user_from_api_token(session, token)
            if authed_user is None or authed_user.id is None:
                raise WebSocketException(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason="Invalid authentication credentials",
                )

            user = await User.one_by_id(
                session,
                authed_user.id,
                options=[selectinload(getattr(User, "worker"))],
            )
            if (
                user is None
                or not user.is_active
                or not user.is_system
                or user.role != UserRole.Worker
                or user.worker is None
            ):
                raise WebSocketException(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason="Invalid authentication credentials",
                )

            return user
    except WebSocketException:
        raise
    except Exception as e:
        logger.exception("Failed to authenticate worker websocket")
        raise WebSocketException(
            code=status.WS_1011_INTERNAL_ERROR,
            reason=f"Failed to authenticate worker websocket: {e}",
        )


def _registry(websocket: WebSocket) -> WorkerControlSessionRegistry:
    return websocket.app.state.worker_control_session_registry


async def _receive_text_message(websocket: WebSocket, limiter: WorkerControlInboundLimiter) -> str:
    message = await asyncio.wait_for(
        websocket.receive(),
        timeout=envs.WORKER_CONTROL_WS_HEARTBEAT_TIMEOUT_SECONDS,
    )
    message_type = message.get("type")
    if message_type == "websocket.disconnect":
        raise WebSocketDisconnect(
            code=message.get("code", status.WS_1000_NORMAL_CLOSURE),
            reason=message.get("reason"),
        )

    payload = message.get("text")
    if payload is None:
        raise WebSocketException(
            code=status.WS_1003_UNSUPPORTED_DATA,
            reason="Binary websocket frames are not supported",
        )

    limiter.check(payload)
    return payload


def _parse_message(
    payload: str,
) -> Union[
    WorkerHelloMessage,
    WorkerPingMessage,
    WorkerPongMessage,
    WorkerCommandAckMessage,
    WorkerCommandResultMessage,
    WorkerErrorMessage,
]:
    base_message = WorkerControlMessageBase.model_validate_json(payload)
    if base_message.message_type == WorkerControlMessageTypeEnum.HELLO:
        return WorkerHelloMessage.model_validate_json(payload)
    if base_message.message_type == WorkerControlMessageTypeEnum.PING:
        return WorkerPingMessage.model_validate_json(payload)
    if base_message.message_type == WorkerControlMessageTypeEnum.PONG:
        return WorkerPongMessage.model_validate_json(payload)
    if base_message.message_type == WorkerControlMessageTypeEnum.COMMAND_ACK:
        return WorkerCommandAckMessage.model_validate_json(payload)
    if base_message.message_type == WorkerControlMessageTypeEnum.COMMAND_RESULT:
        return WorkerCommandResultMessage.model_validate_json(payload)
    if base_message.message_type == WorkerControlMessageTypeEnum.ERROR:
        return WorkerErrorMessage.model_validate_json(payload)

    raise ValueError(
        f"Unsupported inbound control message type: {base_message.message_type.value}"
    )


async def _close_socket(websocket: WebSocket, code: int, reason: str):
    try:
        await websocket.close(code=code, reason=reason)
    except RuntimeError:
        logger.debug("worker control websocket already closed")


async def _receive_hello(websocket: WebSocket, limiter: WorkerControlInboundLimiter) -> WorkerHelloMessage:
    payload = await _receive_text_message(websocket, limiter)
    message = _parse_message(payload)
    if not isinstance(message, WorkerHelloMessage):
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="The first websocket control message must be hello",
        )
    return message


def _build_server_hello(
    worker_session: WorkerControlSession,
    hello_message: WorkerHelloMessage,
) -> WorkerHelloMessage:
    return WorkerHelloMessage(
        session_id=worker_session.session_id,
        protocol_version=worker_session.protocol_version,
        sent_at=utcnow(),
        worker_uuid=worker_session.worker_uuid,
        capabilities=hello_message.capabilities,
        reachability_mode=hello_message.reachability_mode,
    )


@router.websocket("/workers/control/ws")
async def worker_control_websocket(websocket: WebSocket):
    user = await authenticate_worker_websocket(websocket)
    worker = user.worker
    if worker is None or worker.id is None:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Authenticated worker is missing worker metadata",
        )
    await websocket.accept()

    session = None
    disconnect_reason = "disconnected"
    limiter = WorkerControlInboundLimiter()
    dispatch_service = WorkerCommandDispatchService()

    try:
        hello_message = await _receive_hello(websocket, limiter)
        if hello_message.worker_uuid != worker.worker_uuid:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Authenticated worker does not match hello worker_uuid",
            )
        if hello_message.reachability_mode != WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Worker hello must negotiate outbound_control_ws reachability",
            )

        async with async_session() as db_session:
            command_service = WorkerCommandService(db_session)
            persisted_session = await command_service.open_session(
                worker.id,
                session_id=hello_message.session_id,
                control_channel=WorkerControlChannelEnum.OUTBOUND_CONTROL_WS,
                reachability_mode=hello_message.reachability_mode,
                protocol_version=hello_message.protocol_version,
                now=utcnow(),
            )

        session = await _registry(websocket).register(
            worker_id=worker.id,
            worker_uuid=worker.worker_uuid,
            websocket=websocket,
            generation=persisted_session.generation,
            session_id=persisted_session.session_id,
            protocol_version=persisted_session.protocol_version,
        )
        logger.info(
            "Worker control session connected.",
            extra=control_log_extra(
                worker_id=session.worker_id,
                session_id=session.session_id,
                session_generation=session.generation,
            ),
        )
        await websocket.send_json(
            _build_server_hello(session, hello_message).model_dump(mode="json")
        )
        await dispatch_service.dispatch_replay_window(
            session_id=session.session_id,
            session_generation=session.generation,
            resume_requested=hello_message.session_id is not None,
        )

        while True:
            payload = await _receive_text_message(websocket, limiter)
            message = _parse_message(payload)
            await _registry(websocket).touch(session.worker_id, session.generation)
            async with async_session() as db_session:
                command_service = WorkerCommandService(db_session)
                await command_service.touch_session(
                    session.session_id,
                    session.generation,
                    now=utcnow(),
                )

                if isinstance(message, WorkerPingMessage):
                    await websocket.send_json(
                        WorkerPongMessage(
                            session_id=session.session_id,
                            protocol_version=session.protocol_version,
                            sent_at=utcnow(),
                        ).model_dump(mode="json")
                    )
                    continue

                if isinstance(message, WorkerPongMessage):
                    continue

                if isinstance(message, WorkerCommandAckMessage):
                    await command_service.acknowledge_command(
                        message.command_id,
                        session_id=session.session_id,
                        session_generation=session.generation,
                        accepted=message.accepted,
                        state=message.state,
                        error_message=message.error_message,
                        now=utcnow(),
                    )
                    continue

                if isinstance(message, WorkerCommandResultMessage):
                    await command_service.record_command_result(
                        message.command_id,
                        session_id=session.session_id,
                        session_generation=session.generation,
                        state=message.state,
                        result=message.result,
                        error_message=message.error_message,
                        now=utcnow(),
                    )
                    continue

                if isinstance(message, WorkerErrorMessage):
                    await command_service.record_control_error(
                        session_id=session.session_id,
                        session_generation=session.generation,
                        error_code=message.error_code,
                        error_message=message.error_message,
                        now=utcnow(),
                    )
                    continue

            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason=(
                    f"Unsupported inbound control message type: {message.message_type.value}"
                )
            )
    except StaleWorkerSessionError as e:
        disconnect_reason = "stale_session"
        logger.warning(
            "Worker control session rejected as stale: %s",
            sanitize_log_value(str(e)),
            extra=control_log_extra(
                worker_id=worker.id,
                session_id=getattr(session, "session_id", None),
                session_generation=getattr(session, "generation", None),
                failure_source="control",
            ),
        )
        await _close_socket(
            websocket,
            code=status.WS_1008_POLICY_VIOLATION,
            reason=str(e),
        )
    except asyncio.TimeoutError:
        disconnect_reason = "heartbeat_timeout"
        logger.warning(
            "Worker control websocket heartbeat timed out.",
            extra=control_log_extra(
                worker_id=worker.id,
                session_id=getattr(session, "session_id", None),
                session_generation=getattr(session, "generation", None),
                failure_source="transport",
            ),
        )
        await _close_socket(
            websocket,
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Worker control websocket heartbeat timed out",
        )
    except WorkerControlMessageTooLarge as e:
        disconnect_reason = "message_too_large"
        await _close_socket(
            websocket,
            code=status.WS_1009_MESSAGE_TOO_BIG,
            reason=str(e),
        )
    except WorkerControlRateLimitExceeded as e:
        disconnect_reason = "rate_limit_exceeded"
        await _close_socket(
            websocket,
            code=status.WS_1008_POLICY_VIOLATION,
            reason=str(e),
        )
    except (ValidationError, ValueError):
        disconnect_reason = "invalid_message"
        await _close_socket(
            websocket,
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid worker control websocket message",
        )
    except WebSocketException as e:
        disconnect_reason = e.reason or "websocket_rejected"
        logger.warning(
            "Worker control websocket rejected: %s",
            sanitize_log_value(disconnect_reason),
            extra=control_log_extra(
                worker_id=worker.id,
                session_id=getattr(session, "session_id", None),
                session_generation=getattr(session, "generation", None),
                failure_source="control",
            ),
        )
        await _close_socket(websocket, code=e.code, reason=e.reason or "websocket rejected")
    except WebSocketDisconnect:
        disconnect_reason = "websocket_disconnect"
        logger.info(
            "Worker control websocket disconnected.",
            extra=control_log_extra(
                worker_id=worker.id,
                session_id=getattr(session, "session_id", None),
                session_generation=getattr(session, "generation", None),
                failure_source="transport",
            ),
        )
    finally:
        if session is not None:
            await _registry(websocket).unregister(
                worker_id=session.worker_id,
                generation=session.generation,
                reason=disconnect_reason,
            )
            try:
                async with async_session() as db_session:
                    command_service = WorkerCommandService(db_session)
                    await command_service.close_session(
                        session.session_id,
                        session.generation,
                        now=utcnow(),
                    )
            except StaleWorkerSessionError:
                logger.debug(
                    "worker control session %s/%s already superseded before close",
                    session.session_id,
                    session.generation,
                )
            logger.info(
                "Worker control session closed.",
                extra=control_log_extra(
                    worker_id=session.worker_id,
                    session_id=session.session_id,
                    session_generation=session.generation,
                ),
            )
