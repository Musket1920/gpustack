# POLICIES KNOWLEDGE BASE

## OVERVIEW
`gpustack/policies/` is the scheduling domain: candidate selectors, scorers, worker filters, and event recording. This is one of the densest and most specialized areas in the repo.

## WHERE TO LOOK
- `base.py` — core ABCs and candidate dataclasses
- `candidate_selectors/base_candidate_selector.py` — selector base class and model-parameter parsing
- `candidate_selectors/*_resource_fit_selector.py` — backend-specific resource-fit logic
- `scorers/score_chain.py` — scorer composition pattern
- `worker_filters/` — worker prefilter chain
- `utils.py` — shared resource and worker math

## CONVENTIONS
- Extend existing ABCs/chains instead of inventing new scheduling abstractions.
- Keep backend-specific resource-fit logic in the appropriate selector file.
- Reuse `EventCollector` / event-recorder messaging for operator-visible scheduling diagnostics.
- Keep internal resource math in bytes; convert to GiB only for messages/output helpers.
- Preserve candidate ordering, scoring, and filter-chain semantics unless the task explicitly changes scheduling policy.

## ANTI-PATTERNS
- Do not duplicate GPU/resource math outside `utils.py` or the selector base without strong reason.
- Do not mix backend-specific heuristics into the wrong selector.
- Do not add scheduler-facing behavior directly in tests first and backfill source later; policy code is tightly coupled.
- Do not lightly edit the largest selectors (`gguf`, `sglang`, `vllm`, `ascend_mindie`) without checking mirrored tests.

## UNIQUE STYLES
- One selector per backend, plus shared base selector.
- Chain-based scorers and sequential worker filters.
- `tests/policies/` is the main mirror and must stay aligned with selector/scorer behavior.
