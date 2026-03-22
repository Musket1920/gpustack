# ROUTES KNOWLEDGE BASE

## OVERVIEW
`gpustack/routes/` is the FastAPI HTTP surface. Files are mostly one-router-per-resource, then assembled centrally in `routes.py`.

## WHERE TO LOOK
- `routes.py` — root router composition and auth grouping
- `models.py`, `workers.py`, `auth.py`, `inference_backend.py` — representative large route modules
- `worker/` — worker-authenticated route subset

## CONVENTIONS
- Define `router = APIRouter()` per module.
- Reuse existing dependency injection aliases and auth deps; do not invent route-local session/auth patterns.
- Use existing response models and paginated wrappers where surrounding code already does so.
- Keep route validation, auth, and error shapes aligned with current exception helpers.
- Follow the existing split between admin, current-user, worker-client, and cluster-client route groups in `routes.py`.

## ANTI-PATTERNS
- Do not open ad hoc DB sessions when dependency aliases already cover the path.
- Do not invent one-off error payloads instead of using repo exceptions/response models.
- Do not casually rename legacy request/response fields like `perPage`.
- Do not move business logic into routes when similar logic already lives in services/controllers/helpers.

## UNIQUE STYLES
- The route tree is broad but mostly structurally consistent.
- Worker-facing routes under `routes/worker/` are narrower and auth-scoped differently from admin/user routes.

## NOTES
- Large route files usually combine validation, pagination, and auth decisions; preserve that ordering when editing.
- Route assembly and auth scoping live in `routes.py`; resource behavior lives in the leaf module.
