# WORKER KNOWLEDGE BASE

## OVERVIEW
Worker code is ClientSet-driven, not DB-driven. It owns registration, status sync, runtime managers, and model-instance lifecycle.

## WHERE TO LOOK
- `worker.py` — top-level worker startup, registration, uvicorn serving
- `worker_manager.py` — worker registration and heartbeat sync
- `serve_manager.py` — start/stop/restart, health checks, ports, state transitions
- `model_file_manager.py` — file download/watch lifecycle
- `benchmark_manager.py` — benchmark orchestration
- `collector.py` — worker status and resource collection
- `backends/AGENTS.md` — per-backend runtime implementation rules

## CONVENTIONS
- Use `ClientSet` and worker APIs for state changes; do not introduce DB sessions here.
- Keep long-running behavior in manager classes (`ServeManager`, `BenchmarkManager`, `ModelFileManager`).
- Worker lifecycle uses `_create_async_task()` helpers plus periodic thread-based jobs where already established.
- Error paths usually update remote model-instance state to `ERROR` with a message, then log context.
- Preserve existing separation between manager orchestration and backend-specific launch logic.

## ANTI-PATTERNS
- Do not bypass managers and patch instance state ad hoc from unrelated modules.
- Do not add blocking I/O directly into async watcher loops.
- Do not mix server-side service/session patterns into worker code.
- Do not change health, restart, or port allocation behavior casually; `serve_manager.py` is a central hotspot.

## UNIQUE STYLES
- Mixed concurrency model: async tasks + threads + subprocess/process lifecycle.
- Worker APIs are a narrower FastAPI surface than server routes.
- Runtime code is strongly coupled to model-instance states, logs, and health transitions.
