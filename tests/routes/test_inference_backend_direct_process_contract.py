"""
Route/API validation tests for custom backend CRUD with direct-process
contract fields.

These tests exercise the validation and data-flow paths that the
inference_backend routes use, without requiring a live database or
running FastAPI server.  They verify that:

- The route-level schema models (InferenceBackendCreate, InferenceBackendUpdate)
  accept and persist direct-process contract fields.
- The YAML import helpers correctly parse direct_process_contract from raw dicts.
- The validate_custom_suffix function still works with contract-bearing configs.
- The InferenceBackendListItem exposes supports_direct_process.
"""

import pytest
from pydantic import ValidationError

from gpustack.schemas.inference_backend import (
    DirectProcessContract,
    InferenceBackend,
    InferenceBackendCreate,
    InferenceBackendListItem,
    InferenceBackendPublic,
    InferenceBackendUpdate,
    VersionConfig,
    VersionConfigDict,
)
from gpustack.schemas.models import BackendSourceEnum

# Import route-level helpers that can be tested without DB
from gpustack.routes.inference_backend import (
    validate_custom_suffix,
    _process_version_configs,
)


# ---------------------------------------------------------------------------
# Route-level create/update schema acceptance
# ---------------------------------------------------------------------------


class TestCreateSchemaAcceptsContract:
    """InferenceBackendCreate accepts direct-process contract fields."""

    def test_create_with_contract_version(self):
        """Create payload with a version that has a direct-process contract."""
        contract = DirectProcessContract(
            command_template="python -m kokoro_fastapi --port {{port}} --model {{model_path}}",
            health_path="/health",
            startup_timeout_seconds=180,
        )
        payload = InferenceBackendCreate(
            backend_name="kokoro-fastapi-custom",
            supports_direct_process=True,
            backend_source=BackendSourceEnum.CUSTOM,
            version_configs=VersionConfigDict(
                root={
                    "v1.0-custom": VersionConfig(
                        direct_process_contract=contract,
                        run_command="python -m kokoro_fastapi --port {{port}}",
                    ),
                }
            ),
        )
        assert payload.supports_direct_process is True
        vc = payload.version_configs.root["v1.0-custom"]
        assert vc.direct_process_contract is not None
        assert vc.image_name is None  # omitted — valid for direct-process

    def test_create_without_contract_still_works(self):
        """Existing container-only create payloads remain valid."""
        payload = InferenceBackendCreate(
            backend_name="my-backend-custom",
            version_configs=VersionConfigDict(
                root={
                    "v1.0-custom": VersionConfig(
                        image_name="my-backend:v1.0",
                        run_command="python serve.py --port {{port}}",
                    ),
                }
            ),
        )
        assert payload.supports_direct_process is None
        vc = payload.version_configs.root["v1.0-custom"]
        assert vc.direct_process_contract is None
        assert vc.image_name == "my-backend:v1.0"

    def test_create_with_supports_direct_process_false(self):
        """Explicitly setting supports_direct_process=False is valid."""
        payload = InferenceBackendCreate(
            backend_name="container-only-custom",
            supports_direct_process=False,
        )
        assert payload.supports_direct_process is False


class TestUpdateSchemaAcceptsContract:
    """InferenceBackendUpdate accepts direct-process contract fields."""

    def test_update_adds_contract_to_existing_version(self):
        contract = DirectProcessContract(
            command_template="python -m server --port {{port}}",
        )
        payload = InferenceBackendUpdate(
            backend_name="my-backend-custom",
            supports_direct_process=True,
            version_configs=VersionConfigDict(
                root={
                    "v1.0-custom": VersionConfig(
                        image_name="my-backend:v1.0",
                        direct_process_contract=contract,
                    ),
                }
            ),
        )
        vc = payload.version_configs.root["v1.0-custom"]
        assert vc.direct_process_contract is not None
        assert vc.image_name == "my-backend:v1.0"
        assert payload.supports_direct_process is True

    def test_update_removes_contract(self):
        """Setting direct_process_contract to None removes it."""
        payload = InferenceBackendUpdate(
            backend_name="my-backend-custom",
            supports_direct_process=False,
            version_configs=VersionConfigDict(
                root={
                    "v1.0-custom": VersionConfig(
                        image_name="my-backend:v1.0",
                        direct_process_contract=None,
                    ),
                }
            ),
        )
        vc = payload.version_configs.root["v1.0-custom"]
        assert vc.direct_process_contract is None


# ---------------------------------------------------------------------------
# YAML import helper: _process_version_configs
# ---------------------------------------------------------------------------


class TestProcessVersionConfigsWithContract:
    """_process_version_configs correctly parses direct_process_contract."""

    def test_yaml_dict_with_contract(self):
        raw = {
            "v1.0-custom": {
                "run_command": "python serve.py --port {{port}}",
                "direct_process_contract": {
                    "command_template": "python serve.py --port {{port}}",
                    "health_path": "/healthz",
                    "startup_timeout_seconds": 90,
                    "stop_signal": "SIGINT",
                    "stop_timeout_seconds": 15,
                    "workdir": "/opt/app",
                },
            },
        }
        result = _process_version_configs(raw)
        assert "v1.0-custom" in result.root
        vc = result.root["v1.0-custom"]
        assert vc.direct_process_contract is not None
        assert vc.direct_process_contract.command_template == "python serve.py --port {{port}}"
        assert vc.direct_process_contract.health_path == "/healthz"
        assert vc.direct_process_contract.startup_timeout_seconds == 90
        assert vc.direct_process_contract.stop_signal == "SIGINT"
        assert vc.direct_process_contract.stop_timeout_seconds == 15
        assert vc.direct_process_contract.workdir == "/opt/app"

    def test_yaml_dict_without_contract(self):
        raw = {
            "v1.0-custom": {
                "image_name": "my-image:v1.0",
                "run_command": "python serve.py --port {{port}}",
            },
        }
        result = _process_version_configs(raw)
        vc = result.root["v1.0-custom"]
        assert vc.direct_process_contract is None
        assert vc.image_name == "my-image:v1.0"

    def test_yaml_dict_with_invalid_contract_raises(self):
        """Invalid contract data (missing command_template) raises."""
        raw = {
            "v1.0-custom": {
                "direct_process_contract": {
                    "health_path": "/health",
                    # command_template is missing — required
                },
            },
        }
        with pytest.raises(ValidationError):
            _process_version_configs(raw)

    def test_yaml_dict_clears_built_in_frameworks(self):
        """built_in_frameworks is cleared even when contract is present."""
        raw = {
            "v1.0-custom": {
                "built_in_frameworks": ["cuda"],
                "direct_process_contract": {
                    "command_template": "cmd --port {{port}}",
                },
            },
        }
        result = _process_version_configs(raw)
        vc = result.root["v1.0-custom"]
        assert vc.built_in_frameworks is None
        assert vc.direct_process_contract is not None


# ---------------------------------------------------------------------------
# validate_custom_suffix with contract-bearing configs
# ---------------------------------------------------------------------------


class TestValidateCustomSuffixWithContract:
    """validate_custom_suffix works correctly with contract-bearing configs."""

    def test_valid_custom_name_with_contract(self):
        """No exception for valid -custom suffix with contract."""
        contract = DirectProcessContract(
            command_template="cmd --port {{port}}"
        )
        # Should not raise
        validate_custom_suffix("my-backend-custom", None)

    def test_invalid_custom_name_raises(self):
        """Missing -custom suffix raises BadRequestException."""
        from gpustack.api.exceptions import BadRequestException

        with pytest.raises(BadRequestException):
            validate_custom_suffix("my-backend", None)

    def test_valid_version_suffix_with_contract(self):
        """Version names with -custom suffix pass validation."""
        contract = DirectProcessContract(
            command_template="cmd --port {{port}}"
        )
        vcd = VersionConfigDict(
            root={
                "v1.0-custom": VersionConfig(
                    direct_process_contract=contract,
                ),
            }
        )
        # Should not raise
        validate_custom_suffix(None, vcd)

    def test_invalid_version_suffix_with_contract_raises(self):
        """Version names without -custom suffix raise BadRequestException."""
        from gpustack.api.exceptions import BadRequestException

        contract = DirectProcessContract(
            command_template="cmd --port {{port}}"
        )
        vcd = VersionConfigDict(
            root={
                "v1.0": VersionConfig(
                    direct_process_contract=contract,
                ),
            }
        )
        with pytest.raises(BadRequestException):
            validate_custom_suffix(None, vcd)


# ---------------------------------------------------------------------------
# InferenceBackendListItem exposes supports_direct_process
# ---------------------------------------------------------------------------


class TestListItemExposesDirectProcessSupport:
    """InferenceBackendListItem includes supports_direct_process."""

    def test_list_item_with_direct_process_true(self):
        item = InferenceBackendListItem(
            backend_name="my-backend-custom",
            supports_direct_process=True,
        )
        assert item.supports_direct_process is True

    def test_list_item_with_direct_process_none(self):
        item = InferenceBackendListItem(
            backend_name="my-backend-custom",
        )
        assert item.supports_direct_process is None

    def test_list_item_serialization_includes_field(self):
        item = InferenceBackendListItem(
            backend_name="my-backend-custom",
            supports_direct_process=True,
        )
        data = item.model_dump()
        assert "supports_direct_process" in data
        assert data["supports_direct_process"] is True


# ---------------------------------------------------------------------------
# InferenceBackendPublic exposes contract data
# ---------------------------------------------------------------------------


class TestPublicModelExposesContract:
    """InferenceBackendPublic preserves contract data through serialization."""

    def test_public_model_with_contract(self):
        contract = DirectProcessContract(
            command_template="python -m server --port {{port}}",
        )
        public = InferenceBackendPublic(
            id=1,
            created_at=None,
            updated_at=None,
            backend_name="test-custom",
            supports_direct_process=True,
            version_configs=VersionConfigDict(
                root={
                    "v1.0-custom": VersionConfig(
                        direct_process_contract=contract,
                    ),
                }
            ),
        )
        data = public.model_dump()
        # VersionConfigDict is a RootModel — model_dump() may nest under
        # "root" or flatten depending on Pydantic/SQLModel version.
        vc_dict = data["version_configs"]
        if "root" in vc_dict:
            vc_dict = vc_dict["root"]
        vc_data = vc_dict["v1.0-custom"]
        assert vc_data["direct_process_contract"] is not None
        assert (
            vc_data["direct_process_contract"]["command_template"]
            == "python -m server --port {{port}}"
        )
        assert data["supports_direct_process"] is True


# ---------------------------------------------------------------------------
# End-to-end: full create payload round-trip
# ---------------------------------------------------------------------------


class TestFullCreatePayloadRoundTrip:
    """Simulate a full create payload as the route would receive it."""

    def test_community_backend_direct_process_payload(self):
        """
        Simulate creating a community-style backend (e.g. Kokoro-FastAPI)
        with a direct-process contract and no container image.
        """
        contract_data = {
            "command_template": "python -m kokoro_fastapi.server --port {{port}} --model-path {{model_path}}",
            "env_template": {
                "CUDA_VISIBLE_DEVICES": "{{GPU_INDEX}}",
                "HF_HOME": "/models",
            },
            "health_path": "/health",
            "startup_timeout_seconds": 240,
            "stop_signal": "SIGTERM",
            "stop_timeout_seconds": 30,
            "workdir": "/opt/kokoro",
        }
        payload_data = {
            "backend_name": "kokoro-fastapi-custom",
            "supports_direct_process": True,
            "description": "Kokoro FastAPI TTS backend via direct-process",
            "health_check_path": "/health",
            "version_configs": {
                "v0.1-custom": {
                    "run_command": "python -m kokoro_fastapi.server --port {{port}}",
                    "direct_process_contract": contract_data,
                },
            },
        }
        backend = InferenceBackendCreate.model_validate(payload_data)
        assert backend.supports_direct_process is True
        assert backend.backend_name == "kokoro-fastapi-custom"

        vc = backend.version_configs.root["v0.1-custom"]
        assert vc.image_name is None
        assert vc.entrypoint is None
        assert vc.direct_process_contract is not None
        assert vc.direct_process_contract.startup_timeout_seconds == 240
        assert vc.direct_process_contract.workdir == "/opt/kokoro"
        assert vc.direct_process_contract.env_template == {
            "CUDA_VISIBLE_DEVICES": "{{GPU_INDEX}}",
            "HF_HOME": "/models",
        }

        # Round-trip
        dumped = backend.model_dump()
        restored = InferenceBackendCreate.model_validate(dumped)
        assert (
            restored.version_configs.root["v0.1-custom"]
            .direct_process_contract.command_template
            == contract_data["command_template"]
        )

