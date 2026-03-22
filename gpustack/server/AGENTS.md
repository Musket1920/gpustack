# SERVER KNOWLEDGE BASE

## OVERVIEW
Server code is database-backed orchestration: migrations, data init, controllers, services, EventBus, gateway sync, and FastAPI app composition.

## WHERE TO LOOK
- `server.py` — startup order, migrations, subsystem boot
- `app.py` — FastAPI app factory and middleware stack
- `controllers.py` — reconciliation/controller logic
- `services.py` — cached service layer
- `bus.py` — EventBus pub/sub
- `deps.py` — FastAPI dependency aliases
- `db.py` — engine and async session factory

## CONVENTIONS
- Use `async_session()` for DB work; keep session/transaction handling explicit.
- Prefer service-layer helpers and existing caching patterns when business logic already exists there.
- Controller flows are reconcile-style and event-driven; preserve publish/subscribe semantics.
- Keep startup work in `_start_*()` helpers and match current boot ordering.
- Reuse dependency aliases from `deps.py` instead of redefining session/auth types.

## ANTI-PATTERNS
- Do not introduce `ClientSet`-style worker patterns into server code.
- Do not bypass cache invalidation paths when editing services.
- Do not add DB writes without rollback/error handling where the file already uses that pattern.
- Do not dump unrelated logic into `controllers.py`; it is already a major hotspot.

## UNIQUE STYLES
- EventBus and controller patterns are more important here than direct route logic.
- Server code is pure async far more often than worker code.
- Middleware, controller startup, and migration sequencing are tightly coupled.
