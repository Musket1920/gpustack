# SCHEMAS KNOWLEDGE BASE

## OVERVIEW
`gpustack/schemas/` is the core data-contract layer: SQLModel tables, Pydantic models, public/update/create variants, and shared column/type helpers.

## WHERE TO LOOK
- `common.py` — pagination, generic list wrappers, JSON/UTC type helpers
- `models.py` — central serving/model-instance schema set
- `workers.py`, `model_provider.py`, `config.py` — domain-heavy schema modules

## CONVENTIONS
- Preserve base/create/update/public layering when it already exists.
- Use `SQLModel` plus mixins for DB-backed models.
- Use `@model_validator(mode="after")` for cross-field validation.
- Reuse common helpers like `pydantic_column_type()` and `UTCDateTime` instead of reimplementing serialization logic.
- Maintain backward compatibility for deprecated fields unless the task explicitly removes them.

## ANTI-PATTERNS
- Do not collapse public/update/create variants into one oversized model.
- Do not rename externally visible legacy fields like `perPage` without a coordinated compatibility change.
- Do not move route/service logic into schema methods.
- Do not add untyped free-form dicts where the repo already models the data explicitly.

## UNIQUE STYLES
- Schema files mix SQLModel tables and plain Pydantic helper types.
- Some modules, especially `model_provider.py`, contain many small config classes; keep edits surgical.

## NOTES
- Compatibility fields are intentional in several schemas; remove them only with an explicit migration/API change.
- If a schema change affects routes and services, update the schema first and then follow the public/update/create chain outward.
