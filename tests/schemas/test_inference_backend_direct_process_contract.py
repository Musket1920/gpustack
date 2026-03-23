"""
Schema validation tests for DirectProcessContract and VersionConfig
with direct-process contract support.
"""

import pytest
from pydantic import ValidationError

from gpustack.schemas.inference_backend import (
    DirectProcessContract,
    InferenceBackend,
    InferenceBackendBase,
    InferenceBackendCreate,
    VersionConfig,
    VersionConfigDict,
)
from gpustack.schemas.models import BackendSourceEnum


# ---------------------------------------------------------------------------
# DirectProcessContract model tests
# ---------------------------------------------------------------------------


class TestDirectProcessContractDefaults:
    """Verify default values and required fields."""

    def test_minimal_contract_requires_command_template(self):
        """command_template is the only required field."""
        contract = DirectProcessContract(
            command_template="python -m my_server --port {{port}}"
        )
        assert contract.command_template == "python -m my_server --port {{port}}"
        assert contract.health_path == "/health"
        assert contract.startup_timeout_seconds == 120
        assert contract.stop_signal == "SIGTERM"
        assert contract.stop_timeout_seconds == 30
        assert contract.env_template is None
        assert contract.workdir is None

    def test_missing_command_template_raises(self):
        """Omitting command_template must raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DirectProcessContract()  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "command_template" in field_names

    def test_all_fields_populated(self):
        """All fields can be set explicitly."""
        contract = DirectProcessContract(
            command_template="vllm serve {{model_path}} --port {{port}}",
            env_template={"CUDA_VISIBLE_DEVICES": "0", "HF_HOME": "/models"},
            health_path="/v1/health",
            startup_timeout_seconds=300,
            stop_signal="SIGINT",
            stop_timeout_seconds=60,
            workdir="/opt/vllm",
        )
        assert contract.env_template == {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HOME": "/models",
        }
        assert contract.health_path == "/v1/health"
        assert contract.startup_timeout_seconds == 300
        assert contract.stop_signal == "SIGINT"
        assert contract.stop_timeout_seconds == 60
        assert contract.workdir == "/opt/vllm"


class TestDirectProcessContractValidation:
    """Boundary and type validation."""

    def test_startup_timeout_must_be_positive(self):
        with pytest.raises(ValidationError):
            DirectProcessContract(
                command_template="cmd", startup_timeout_seconds=0
            )

    def test_stop_timeout_must_be_positive(self):
        with pytest.raises(ValidationError):
            DirectProcessContract(
                command_template="cmd", stop_timeout_seconds=0
            )

    def test_startup_timeout_negative_rejected(self):
        with pytest.raises(ValidationError):
            DirectProcessContract(
                command_template="cmd", startup_timeout_seconds=-1
            )

    def test_env_template_must_be_dict(self):
        with pytest.raises(ValidationError):
            DirectProcessContract(
                command_template="cmd", env_template="not_a_dict"  # type: ignore[arg-type]
            )


class TestDirectProcessContractSerialization:
    """Round-trip serialization."""

    def test_dict_round_trip(self):
        original = DirectProcessContract(
            command_template="python serve.py --port {{port}}",
            env_template={"KEY": "val"},
            health_path="/ready",
            startup_timeout_seconds=60,
            stop_signal="SIGTERM",
            stop_timeout_seconds=10,
            workdir="/app",
        )
        data = original.model_dump()
        restored = DirectProcessContract(**data)
        assert restored == original

    def test_json_round_trip(self):
        original = DirectProcessContract(
            command_template="python serve.py --port {{port}}",
        )
        json_str = original.model_dump_json()
        restored = DirectProcessContract.model_validate_json(json_str)
        assert restored == original


# ---------------------------------------------------------------------------
# VersionConfig with direct_process_contract
# ---------------------------------------------------------------------------


class TestVersionConfigDirectProcessContract:
    """VersionConfig integration with DirectProcessContract."""

    def test_version_config_without_contract(self):
        """Existing behavior: VersionConfig without contract is valid."""
        vc = VersionConfig(image_name="my-image:latest")
        assert vc.direct_process_contract is None
        assert vc.image_name == "my-image:latest"

    def test_version_config_with_contract_no_image(self):
        """A version config can have a contract and omit image_name."""
        contract = DirectProcessContract(
            command_template="python -m server --port {{port}}"
        )
        vc = VersionConfig(direct_process_contract=contract)
        assert vc.image_name is None
        assert vc.entrypoint is None
        assert vc.direct_process_contract is not None
        assert (
            vc.direct_process_contract.command_template
            == "python -m server --port {{port}}"
        )

    def test_version_config_with_both_contract_and_image(self):
        """A version config can have both contract and image (dual-mode)."""
        contract = DirectProcessContract(
            command_template="python -m server --port {{port}}"
        )
        vc = VersionConfig(
            image_name="my-image:v1",
            direct_process_contract=contract,
        )
        assert vc.image_name == "my-image:v1"
        assert vc.direct_process_contract is not None

    def test_version_config_contract_from_dict(self):
        """VersionConfig can be constructed from a dict with nested contract."""
        data = {
            "run_command": "python serve.py",
            "direct_process_contract": {
                "command_template": "python serve.py --port {{port}}",
                "health_path": "/healthz",
                "startup_timeout_seconds": 90,
            },
        }
        vc = VersionConfig(**data)
        assert vc.direct_process_contract is not None
        assert vc.direct_process_contract.health_path == "/healthz"
        assert vc.direct_process_contract.startup_timeout_seconds == 90

    def test_version_config_dict_serialization(self):
        """VersionConfigDict round-trips with contract data."""
        contract = DirectProcessContract(
            command_template="cmd --port {{port}}"
        )
        vcd = VersionConfigDict(
            root={
                "v1.0-custom": VersionConfig(
                    direct_process_contract=contract,
                    run_command="cmd",
                ),
            }
        )
        data = vcd.model_dump()
        restored = VersionConfigDict(**data)
        assert (
            restored.root["v1.0-custom"].direct_process_contract.command_template
            == "cmd --port {{port}}"
        )


# ---------------------------------------------------------------------------
# InferenceBackendBase / InferenceBackendCreate with contract
# ---------------------------------------------------------------------------


class TestInferenceBackendDirectProcessMetadata:
    """Backend-level supports_direct_process metadata."""

    def test_supports_direct_process_default_none(self):
        """By default, supports_direct_process is None."""
        backend = InferenceBackendCreate(
            backend_name="test-custom",
        )
        assert backend.supports_direct_process is None

    def test_supports_direct_process_true(self):
        backend = InferenceBackendCreate(
            backend_name="test-custom",
            supports_direct_process=True,
        )
        assert backend.supports_direct_process is True

    def test_supports_direct_process_false(self):
        backend = InferenceBackendCreate(
            backend_name="test-custom",
            supports_direct_process=False,
        )
        assert backend.supports_direct_process is False


class TestInferenceBackendHasDirectProcessContract:
    """Test the has_direct_process_contract() helper method."""

    def test_no_versions_returns_false(self):
        backend = InferenceBackendCreate(backend_name="test-custom")
        assert backend.has_direct_process_contract() is False

    def test_versions_without_contract_returns_false(self):
        backend = InferenceBackendCreate(
            backend_name="test-custom",
            version_configs=VersionConfigDict(
                root={
                    "v1-custom": VersionConfig(image_name="img:v1"),
                }
            ),
        )
        assert backend.has_direct_process_contract() is False

    def test_versions_with_contract_returns_true(self):
        contract = DirectProcessContract(
            command_template="cmd --port {{port}}"
        )
        backend = InferenceBackendCreate(
            backend_name="test-custom",
            version_configs=VersionConfigDict(
                root={
                    "v1-custom": VersionConfig(
                        direct_process_contract=contract,
                    ),
                }
            ),
        )
        assert backend.has_direct_process_contract() is True

    def test_mixed_versions_returns_true(self):
        """If at least one version has a contract, returns True."""
        contract = DirectProcessContract(
            command_template="cmd --port {{port}}"
        )
        backend = InferenceBackendCreate(
            backend_name="test-custom",
            version_configs=VersionConfigDict(
                root={
                    "v1-custom": VersionConfig(image_name="img:v1"),
                    "v2-custom": VersionConfig(
                        direct_process_contract=contract,
                    ),
                }
            ),
        )
        assert backend.has_direct_process_contract() is True


class TestInferenceBackendContractOmitsImage:
    """
    Verify that a custom backend definition can omit image/entrypoint
    when it declares a direct-process contract.
    """

    def test_contract_only_no_image_no_entrypoint(self):
        """A custom backend with only a contract and no image is valid."""
        contract = DirectProcessContract(
            command_template="python -m my_backend --port {{port}} --model {{model_path}}",
            health_path="/health",
        )
        backend = InferenceBackendCreate(
            backend_name="kokoro-fastapi-custom",
            supports_direct_process=True,
            version_configs=VersionConfigDict(
                root={
                    "v1.0-custom": VersionConfig(
                        direct_process_contract=contract,
                    ),
                }
            ),
        )
        vc = backend.version_configs.root["v1.0-custom"]
        assert vc.image_name is None
        assert vc.entrypoint is None
        assert vc.direct_process_contract is not None
        assert backend.supports_direct_process is True

    def test_container_fields_still_work(self):
        """Container fields remain intact and functional."""
        backend = InferenceBackendCreate(
            backend_name="my-backend-custom",
            version_configs=VersionConfigDict(
                root={
                    "v1.0-custom": VersionConfig(
                        image_name="my-backend:v1.0",
                        entrypoint="python serve.py",
                        run_command="python serve.py --port {{port}}",
                    ),
                }
            ),
        )
        vc = backend.version_configs.root["v1.0-custom"]
        assert vc.image_name == "my-backend:v1.0"
        assert vc.entrypoint == "python serve.py"
        assert vc.direct_process_contract is None


class TestInferenceBackendFullSerialization:
    """Full model_dump / model_validate round-trip with contract."""

    def test_full_round_trip(self):
        contract = DirectProcessContract(
            command_template="python -m server --port {{port}}",
            env_template={"CUDA_VISIBLE_DEVICES": "0"},
            health_path="/health",
            startup_timeout_seconds=180,
            stop_signal="SIGTERM",
            stop_timeout_seconds=30,
            workdir="/opt/server",
        )
        backend = InferenceBackendCreate(
            backend_name="community-backend-custom",
            supports_direct_process=True,
            description="A community backend with direct-process support",
            health_check_path="/health",
            backend_source=BackendSourceEnum.CUSTOM,
            version_configs=VersionConfigDict(
                root={
                    "v1.0-custom": VersionConfig(
                        direct_process_contract=contract,
                        run_command="python -m server",
                        env={"LOG_LEVEL": "info"},
                    ),
                }
            ),
        )
        data = backend.model_dump()
        restored = InferenceBackendCreate.model_validate(data)

        assert restored.supports_direct_process is True
        assert restored.backend_name == "community-backend-custom"
        vc = restored.version_configs.root["v1.0-custom"]
        assert vc.direct_process_contract is not None
        assert (
            vc.direct_process_contract.command_template
            == "python -m server --port {{port}}"
        )
        assert vc.direct_process_contract.env_template == {
            "CUDA_VISIBLE_DEVICES": "0"
        }
        assert vc.direct_process_contract.workdir == "/opt/server"
        assert vc.env == {"LOG_LEVEL": "info"}
