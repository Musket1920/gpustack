# TESTS KNOWLEDGE BASE

## OVERVIEW
`tests/` mirrors source structure and relies on shared fixtures/utilities rather than deep per-directory `conftest.py` files.

## WHERE TO LOOK
- `conftest.py` (repo root) — global `temp_dir` and `config` fixtures
- `tests/conftest.py` — local package import setup
- `tests/utils/mock.py` — async session mock helper
- `tests/utils/model.py` — model/model-instance factories
- `tests/utils/scheduler.py` — candidate comparison helpers
- `tests/fixtures/workers/fixtures.py`, `tests/fixtures/estimates/fixtures.py` — heavy JSON-backed fixture factories

## CONVENTIONS
- Use `@pytest.mark.asyncio` explicitly for async tests.
- Use `@pytest.mark.parametrize` with named cases for scenario matrices.
- Prefer shared fixture factories and test utilities over bespoke inline setup.
- For scheduling/policy tests, reuse worker and estimate fixtures rather than building resource objects by hand.
- Keep test layout mirrored to the source domain you are exercising.

## ANTI-PATTERNS
- Do not require real GPUs or external services unless the test is already explicitly integration-style.
- Do not duplicate large worker/estimate fixture data inside individual tests.
- Do not add subtree `AGENTS.md` files under `tests/policies/`; this parent file covers test conventions well enough.
- Do not hide async behavior inside sync tests when the code path is genuinely async.

## UNIQUE STYLES
- `tests/policies/` is the densest test subtree and mirrors backend-specific selector behavior.
- Fixture factories are named after concrete hardware/model scenarios, not generic sample names.
