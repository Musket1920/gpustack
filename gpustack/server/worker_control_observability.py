import re
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram

from gpustack.schemas.workers import WorkerControlCommandStateEnum
from gpustack.utils.name import metric_name


_REDACTION_PATTERNS = (
    (re.compile(r"(?i)bearer\s+[a-z0-9._\-]+"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*)([^\s,;]+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(token\s*[:=]\s*)([^\s,;]+)"), r"\1[REDACTED]"),
)


worker_control_active_sessions = Gauge(
    metric_name("worker_control_active_sessions"),
    "Active worker control sessions by worker.",
    labelnames=("worker_id",),
)

worker_control_session_connections_total = Counter(
    metric_name("worker_control_session_connections_total"),
    "Total worker control session connections.",
    labelnames=("worker_id", "control_channel"),
)

worker_control_session_disconnects_total = Counter(
    metric_name("worker_control_session_disconnects_total"),
    "Total worker control session disconnects.",
    labelnames=("worker_id", "reason"),
)

worker_control_session_replacements_total = Counter(
    metric_name("worker_control_session_replacements_total"),
    "Total worker control sessions replaced by newer sessions.",
    labelnames=("worker_id",),
)

worker_control_session_reconnects_total = Counter(
    metric_name("worker_control_session_reconnects_total"),
    "Total worker control client reconnects after a prior session.",
    labelnames=("worker_id",),
)

worker_control_command_ack_latency_seconds = Histogram(
    metric_name("worker_control_command_ack_latency_seconds"),
    "Latency from command dispatch to worker acknowledgement.",
    labelnames=("worker_id", "command_type", "accepted"),
)

worker_control_command_result_latency_seconds = Histogram(
    metric_name("worker_control_command_result_latency_seconds"),
    "Latency from command dispatch to worker result.",
    labelnames=("worker_id", "command_type", "state", "failure_source"),
)

worker_control_command_failures_total = Counter(
    metric_name("worker_control_command_failures_total"),
    "Total worker control command failures by source and stage.",
    labelnames=("worker_id", "command_type", "failure_source", "stage"),
)

worker_control_replay_fallbacks_total = Counter(
    metric_name("worker_control_replay_fallbacks_total"),
    "Total replay fallback events that require control-session recovery.",
    labelnames=("worker_id", "reason"),
)

worker_control_capability_route_rejects_total = Counter(
    metric_name("worker_control_capability_route_rejects_total"),
    "Total reverse-only route rejects caused by worker reachability capabilities.",
    labelnames=("worker_id", "operation", "reachability_mode"),
)


def sanitize_log_value(value: object) -> object:
    if not isinstance(value, str):
        return value

    sanitized = value
    for pattern, replacement in _REDACTION_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def control_log_extra(
    *,
    worker_id: Optional[int] = None,
    session_id: Optional[str] = None,
    session_generation: Optional[int] = None,
    command_id: Optional[str] = None,
    command_type: Optional[str] = None,
    failure_source: Optional[str] = None,
    error_code: Optional[str] = None,
) -> dict[str, object]:
    extra: dict[str, object] = {}
    if worker_id is not None:
        extra["worker_id"] = worker_id
    if session_id is not None:
        extra["session_id"] = session_id
    if session_generation is not None:
        extra["session_generation"] = session_generation
    if command_id is not None:
        extra["command_id"] = command_id
    if command_type is not None:
        extra["command_type"] = command_type
    if failure_source is not None:
        extra["failure_source"] = failure_source
    if error_code is not None:
        extra["error_code"] = error_code
    return extra


def record_session_connected(worker_id: int, *, control_channel: str):
    worker_control_active_sessions.labels(worker_id=str(worker_id)).set(1)
    worker_control_session_connections_total.labels(
        worker_id=str(worker_id),
        control_channel=control_channel,
    ).inc()


def record_session_disconnected(worker_id: int, *, reason: str, still_active: bool):
    if not still_active:
        worker_control_active_sessions.labels(worker_id=str(worker_id)).set(0)
    worker_control_session_disconnects_total.labels(
        worker_id=str(worker_id),
        reason=reason,
    ).inc()


def record_session_replacement(worker_id: int):
    worker_control_session_replacements_total.labels(worker_id=str(worker_id)).inc()


def record_session_reconnect(worker_id: int):
    worker_control_session_reconnects_total.labels(worker_id=str(worker_id)).inc()


def record_command_ack_latency(
    *,
    worker_id: int,
    command_type: str,
    accepted: bool,
    latency_seconds: float,
):
    worker_control_command_ack_latency_seconds.labels(
        worker_id=str(worker_id),
        command_type=command_type,
        accepted=str(bool(accepted)).lower(),
    ).observe(latency_seconds)


def record_command_result_latency(
    *,
    worker_id: int,
    command_type: str,
    state: WorkerControlCommandStateEnum,
    failure_source: str,
    latency_seconds: float,
):
    worker_control_command_result_latency_seconds.labels(
        worker_id=str(worker_id),
        command_type=command_type,
        state=state.value,
        failure_source=failure_source,
    ).observe(latency_seconds)


def record_command_failure(
    *,
    worker_id: int,
    command_type: str,
    failure_source: str,
    stage: str,
):
    worker_control_command_failures_total.labels(
        worker_id=str(worker_id),
        command_type=command_type,
        failure_source=failure_source,
        stage=stage,
    ).inc()


def record_replay_fallback(*, worker_id: int, reason: str):
    worker_control_replay_fallbacks_total.labels(
        worker_id=str(worker_id),
        reason=reason,
    ).inc()


def record_capability_route_reject(
    *,
    worker_id: int,
    operation: str,
    reachability_mode: str,
):
    worker_control_capability_route_rejects_total.labels(
        worker_id=str(worker_id),
        operation=operation,
        reachability_mode=reachability_mode,
    ).inc()


def classify_result_failure_source(state: WorkerControlCommandStateEnum) -> str:
    if state == WorkerControlCommandStateEnum.SUCCEEDED:
        return "none"
    return "runtime"


def reset_control_observability_metrics_for_tests():
    metrics = [
        worker_control_active_sessions,
        worker_control_session_connections_total,
        worker_control_session_disconnects_total,
        worker_control_session_replacements_total,
        worker_control_session_reconnects_total,
        worker_control_command_ack_latency_seconds,
        worker_control_command_result_latency_seconds,
        worker_control_command_failures_total,
        worker_control_replay_fallbacks_total,
        worker_control_capability_route_rejects_total,
    ]
    for metric in metrics:
        with metric._lock:
            metric._metrics.clear()
