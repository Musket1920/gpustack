import sys
import types
from pathlib import Path
from typing import Any, cast

# Inject an fcntl stub before importing any gpustack.worker module so that
# gpustack.worker.__init__ -> gpustack.utils.locks -> fcntl does not fail on
# Windows where fcntl is unavailable.
if "fcntl" not in sys.modules:
    _fcntl_stub = types.ModuleType("fcntl")
    _fcntl_stub.LOCK_EX = 1  # type: ignore[attr-defined]
    _fcntl_stub.LOCK_UN = 2  # type: ignore[attr-defined]
    _fcntl_stub.lockf = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    _fcntl_stub.flock = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["fcntl"] = _fcntl_stub

import types

import pytest

from gpustack.utils.config import apply_registry_override_to_image
from gpustack.worker.backends.custom import CustomServer


@pytest.mark.parametrize(
    "image_name, container_registry, expect_image_name, fallback_registry",
    [
        (
            "ghcr.io/ggml-org/llama.cpp:server",
            "test-registry.io",
            "ghcr.io/ggml-org/llama.cpp:server",
            None,
        ),
        (
            "gpustack/runner:cuda12.8-vllm0.10.2",
            "test-registry.io",
            "test-registry.io/gpustack/runner:cuda12.8-vllm0.10.2",
            None,
        ),
        (
            "foo/bar",
            "test-registry.io",
            "test-registry.io/foo/bar",
            None,
        ),
        ("ubuntu:24.04", "test-registry.io", "test-registry.io/ubuntu:24.04", None),
        (
            "gpustack/runner:cuda12.8-vllm0.10.2",
            None,
            "quay.io/gpustack/runner:cuda12.8-vllm0.10.2",
            "quay.io",
        ),
        (
            "lmsysorg/sglang:v0.5.5",
            "",
            "lmsysorg/sglang:v0.5.5",
            None,
        ),
    ],
)
@pytest.mark.asyncio
async def test_apply_registry_override(
    image_name,
    container_registry,
    expect_image_name,
    fallback_registry,
    monkeypatch,
):
    backend = cast(Any, CustomServer.__new__(CustomServer))
    # CustomServer inherits _apply_registry_override from InferenceServer,
    # and _apply_registry_override accesses self._config.system_default_container_registry.
    # Since we constructed the instance via __new__ (without __init__),
    # the _config attribute does not exist. We attach a minimal stub config here.
    backend._config = types.SimpleNamespace(
        system_default_container_registry=container_registry,
    )
    backend._fallback_registry = fallback_registry

    assert (
        apply_registry_override_to_image(
            cast(Any, backend._config), image_name, backend._fallback_registry
        )
        == expect_image_name
    )

    if container_registry:
        backend._config = types.SimpleNamespace(system_default_container_registry=None)
        assert (
            apply_registry_override_to_image(
                cast(Any, backend._config), image_name, backend._fallback_registry
            )
            == image_name
        )


@pytest.mark.parametrize(
    "backend_parameters, expected",
    [
        (
            ["--ctx-size 1024"],
            ["--ctx-size", "1024"],
        ),
        (
            ["--served-model-name foo"],
            ["--served-model-name", "foo"],
        ),
        (
            ['--served-model-name "foo bar"'],
            ["--served-model-name", "foo bar"],
        ),
        (
            ['--arg1', '--arg2 "val with spaces"'],
            ['--arg1', '--arg2', 'val with spaces'],
        ),
        (
            ['--arg1 "val with spaces"', '--arg2="val with spaces"'],
            ['--arg1', 'val with spaces', '--arg2="val with spaces"'],
        ),
        (
            [
                """--hf-overrides '{"architectures": ["NewModel"]}'""",
                """--hf-overrides={"architectures": ["NewModel"]}""",
            ],
            [
                '--hf-overrides',
                '{"architectures": ["NewModel"]}',
                """--hf-overrides={"architectures": ["NewModel"]}""",
            ],
        ),
        # Test cases for whitespace handling
        (
            [" --ctx-size=1024"],
            ["--ctx-size=1024"],
        ),
        (
            ["--ctx-size =1024"],
            ["--ctx-size=1024"],
        ),
        (
            ["  --ctx-size  =1024"],
            ["--ctx-size=1024"],
        ),
        (
            ["--ctx-size  =  1024"],
            ["--ctx-size=1024"],
        ),
        (
            ["  --ctx-size 1024"],
            ["--ctx-size", "1024"],
        ),
        (
            [" --max-model-len=8192"],
            ["--max-model-len=8192"],
        ),
        (
            ["--foo =bar", "  --baz  =  qux"],
            ["--foo=bar", "--baz=qux"],
        ),
        (
            None,
            [],
        ),
    ],
)
def test_flatten_backend_param(backend_parameters, expected):
    backend = cast(Any, CustomServer.__new__(CustomServer))
    backend._model = types.SimpleNamespace(backend_parameters=backend_parameters)
    assert backend._flatten_backend_param() == expected


def test_merge_direct_process_prepared_env_artifacts_preserves_system_path_tail(
    monkeypatch, tmp_path: Path
):
    backend = cast(Any, CustomServer.__new__(CustomServer))
    prepared_env_path = tmp_path / "prepared.env"
    prepared_env_path.write_text(
        "VIRTUAL_ENV=/tmp/prepared/venv\n"
        "PATH=/tmp/prepared/bin:/tmp/prepared/venv/bin:${PATH}\n",
        encoding="utf-8",
    )
    prepared_config_path = tmp_path / "prepared-config.json"
    prepared_provenance_path = tmp_path / "executable-provenance.json"
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    merged_env = backend.merge_direct_process_prepared_env_artifacts(
        {},
        cast(
            Any,
            types.SimpleNamespace(
            prepared_config={"env_artifact": str(prepared_env_path)},
            prepared_env_path=prepared_env_path,
            prepared_config_path=prepared_config_path,
            prepared_provenance_path=prepared_provenance_path,
            manifest_hash="manifest-hash",
            prepared_environment_id="vllm:0.8.0",
            prepared_provenance={"prepared_path": "/tmp/prepared/bin/vllm"},
        ),
        ),
    )

    assert merged_env["VIRTUAL_ENV"] == "/tmp/prepared/venv"
    assert (
        merged_env["PATH"]
        == "/tmp/prepared/bin:/tmp/prepared/venv/bin:/usr/bin:/bin"
    )
    assert merged_env["GPUSTACK_PREPARED_ENV_ARTIFACT"] == str(prepared_env_path)
