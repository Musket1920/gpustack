# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-22
**Commit:** `a7f1338d`
**Branch:** `main`

## OVERVIEW
GPUStack is a Python/FastAPI cluster manager for model serving. Core domains split cleanly into `server`, `worker`, `policies`, `routes`, `schemas`, and mirrored `tests`.

## STRUCTURE
```text
gpustack/
├── gpustack/server/           # DB-backed orchestration, controllers, services
├── gpustack/worker/           # worker runtime, subprocess/container lifecycle
│   └── backends/              # per-backend inference implementations
├── gpustack/policies/         # scheduling selectors, scorers, worker filters
├── gpustack/routes/           # FastAPI route modules and router assembly
├── gpustack/schemas/          # SQLModel/Pydantic schema layer
├── tests/                     # mirrored tests, fixtures, mocks
└── hack/                      # make-driven shell and PowerShell scripts
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Start CLI / process mode | `gpustack/main.py`, `gpustack/cmd/start.py` | console entrypoint + start flow |
| Server lifecycle | `gpustack/server/AGENTS.md` | migrations, controllers, EventBus, services |
| Worker lifecycle | `gpustack/worker/AGENTS.md` | ClientSet, managers, subprocess/runtime behavior |
| Backend integration | `gpustack/worker/backends/AGENTS.md` | `InferenceServer` subclasses and runtime/image logic |
| Scheduling logic | `gpustack/policies/AGENTS.md` | selectors, scorers, filters, resource-fit rules |
| FastAPI endpoints | `gpustack/routes/AGENTS.md` | routers, auth/deps, pagination, response models |
| Schema changes | `gpustack/schemas/AGENTS.md` | SQLModel/Pydantic conventions and compatibility |
| Test additions | `tests/AGENTS.md` | fixtures, async tests, policy test mirrors |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `Server` | class | `gpustack/server/server.py` | server startup, migrations, subsystem orchestration |
| `Worker` | class | `gpustack/worker/worker.py` | worker startup, registration, API serving |
| `InferenceServer` | class | `gpustack/worker/backends/base.py` | base backend abstraction |
| `ScheduleCandidatesSelector` | class | `gpustack/policies/candidate_selectors/base_candidate_selector.py` | backend-specific candidate selection base |
| `Model` / `ModelInstance` | classes | `gpustack/schemas/models.py` | central serving data model |
| `api_router` | router | `gpustack/routes/routes.py` | root route composition |

## CONVENTIONS
- Root rules cover commands, formatting, typing, git hygiene, and verification defaults.
- Child `AGENTS.md` files carry domain-specific rules; prefer them over adding local exceptions here.
- Tests mirror source structure; `tests/policies/` is covered by `tests/AGENTS.md`, not its own child file.
- `hack/` stays under root guidance; it is distinct but too small to justify another local file.

## ANTI-PATTERNS (THIS PROJECT)
- Do not invent a type-check workflow; none is configured.
- Do not touch migrations with style cleanup; they are excluded from Black/Flake8.
- Do not use destructive git commands or amend/push without explicit user request.
- Do not broad-refactor oversized hotspots like `gpustack/server/controllers.py` or policy selectors unless the task demands it.
- Do not assume server and worker use the same state model; server is DB/event-bus centric, worker is ClientSet/process centric.

## UNIQUE STYLES
- Flat package layout (`gpustack/`), not `src/`.
- Async-first service, route, worker, and scheduler code.
- Heavy schema layering: base/create/update/public variants.
- Policy/resource-fit logic is backend-specific and concentrated in large selector files.
- Worker runtime mixes async loops, threads, and subprocess/container orchestration.

## COMMANDS
```bash
make install
make deps
make lint
make test
make build
make ci
uv run pytest tests/path/to/test_file.py::test_name
uv run python -m gpustack.codegen.generate
uv run mkdocs build
```

## NOTES
- CI order is `install -> deps -> lint -> test -> build` on Unix.
- Windows CI differs slightly (`validate` step, no `deps` step).
- Biggest complexity hotspots: `gpustack/policies/`, `gpustack/worker/`, `gpustack/server/controllers.py`, and `tests/policies/`.
