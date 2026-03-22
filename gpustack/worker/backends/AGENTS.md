# BACKENDS KNOWLEDGE BASE

## OVERVIEW
`gpustack/worker/backends/` contains `InferenceServer` subclasses for concrete inference engines. This layer is runtime/image/version heavy and should stay consistent across backends.

## WHERE TO LOOK
- `base.py` — `InferenceServer`, shared startup/error helpers, image/version resolution
- `vllm.py`, `sglang.py`, `custom.py`, `vox_box.py`, `ascend_mindie.py` — backend implementations

## CONVENTIONS
- New backends should extend `InferenceServer` and reuse base helpers before adding custom logic.
- Keep command construction separate from workload/container creation.
- Preserve version/image resolution precedence from `base.py`.
- Reuse model/env/port/resource helpers instead of hardcoding values in backend files.
- Backend startup failures should flow through shared error handling and model-instance state updates.

## ANTI-PATTERNS
- Do not bypass `base.py` for image, env, runner, or deployment metadata handling.
- Do not hardcode ports, runtime env, or registry behavior inside one backend unless truly backend-specific.
- Do not copy/paste another backend wholesale when only a few arguments differ.
- Do not add distributed behavior in a backend without matching the existing metadata and port-allocation patterns.

## UNIQUE STYLES
- Container/workload planning is a first-class concept here.
- `ascend_mindie.py` is unusually large and contains vendor-specific edge cases; edit narrowly.
- `custom.py` is still container-oriented even though it allows command overrides.

## NOTES
- Backend files are similar in shape but not interchangeable; follow the nearest backend, not the smallest one.
- When changing shared backend behavior, check both `base.py` and at least one concrete backend before finalizing.
