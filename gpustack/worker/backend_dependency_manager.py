from dataclasses import dataclass
import json
import logging
import os
import re
from typing import Dict, List, Optional

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from gpustack.schemas.models import BackendEnum

logger = logging.getLogger(__name__)

_SHA256_PATTERN = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")


def _normalize_sha256(value: str) -> str:
    match = _SHA256_PATTERN.match(value.strip())
    if not match:
        raise ValueError(f"Expected sha256 hash, got {value!r}")
    return match.group(1).lower()


def _is_exact_pinned_requirement(requirement: str) -> bool:
    stripped = requirement.strip()
    if not stripped or "==" not in stripped:
        return False

    return not any(operator in stripped for operator in (">=", "<=", "~=", "!=", "<", ">"))


@dataclass
class BackendDependencySpec:
    """
    Represents a backend dependency specification.

    Attributes:
        backend: The backend name (e.g., 'vox-box', 'vllm')
        dependencies: List of dependency specifications (e.g., ['transformers==4.51.3', 'torch>=2.0.0'])
    """

    backend: str
    dependencies: List[str]

    def to_pip_args(self) -> str:
        """
        Convert dependencies to pip arguments format.

        Returns:
            String in format "--pip-args='dep1 dep2 dep3'"
        """
        if not self.dependencies:
            return ""

        deps_str = " ".join(self.dependencies)
        return f"--pip-args='{deps_str}'"


@dataclass(frozen=True)
class HashPinnedDependencySpec:
    requirement: str
    hashes: List[str]

    def to_requirements_line(self) -> str:
        hashes = " ".join(f"--hash=sha256:{item}" for item in self.hashes)
        return f"{self.requirement} {hashes}".strip()


@dataclass(frozen=True)
class ExecutableSpec:
    name: str
    path: str
    sha256: str


@dataclass(frozen=True)
class BackendProvisioningPlan:
    dependencies: List[HashPinnedDependencySpec]
    executable: Optional[ExecutableSpec]

    def to_audit_payload(self) -> Dict[str, object]:
        return {
            "dependencies": [
                {
                    "requirement": dependency.requirement,
                    "hashes": [f"sha256:{value}" for value in dependency.hashes],
                }
                for dependency in self.dependencies
            ],
            "executable": (
                None
                if self.executable is None
                else {
                    "name": self.executable.name,
                    "path": self.executable.path,
                    "sha256": f"sha256:{self.executable.sha256}",
                }
            ),
        }


class BackendDependencyManager:
    """
    Manages backend dependencies for different inference backends.

    Examples:
        - model_env: {"GPUSTACK_BACKEND_DEPS"="transformers==4.53.3,torch>=2.0.0"}
    """

    def __init__(
        self,
        backend: str,
        version: str,
        model_env: Optional[Dict[str, str]] = None,
    ):
        self.backend = backend
        self.version = version
        self._model_env = model_env or {}
        self._custom_specs: Optional[BackendDependencySpec] = None

        # Initialize default dependencies for each backend using version specifiers
        # Format: {backend: {version_specifier: [dependencies]}}
        self.default_dependencies_specs: Dict[str, Dict[str, List[str]]] = {
            BackendEnum.VOX_BOX: {
                "<=0.0.20": ["transformers==4.51.3"],
            },
            BackendEnum.VLLM: {
                "<=0.10.0": ["transformers==4.53.3"],
            },
        }

        self._load_from_environment(self._model_env)

    def _get_env_value(self, key: str) -> Optional[str]:
        return self._model_env.get(key) or os.getenv(key)

    def _load_from_environment(self, model_env: Optional[Dict[str, str]] = None):
        """
        Load custom dependency specifications from model environment variables.

        Environment variable format:
        GPUSTACK_BACKEND_DEPS="dep1,dep2"
        """
        if not model_env:
            return
        # First try to get from model_env, then fallback to system environment
        env_deps = (model_env or {}).get("GPUSTACK_BACKEND_DEPS") or os.getenv(
            "GPUSTACK_BACKEND_DEPS"
        )

        if not env_deps:
            return

        try:
            dependencies = [dep.strip() for dep in env_deps.split(",") if dep.strip()]
            self._custom_specs = BackendDependencySpec(
                backend=self.backend, dependencies=dependencies
            )
            logger.info(f"Loaded custom dependency spec: {dependencies}")
        except Exception as e:
            logger.warning(f"Failed to parse GPUSTACK_BACKEND_DEPS: {e}")

    def get_dependency_spec(self) -> Optional[BackendDependencySpec]:
        """
        Get dependency specification for a backend and version.

        Returns:
            BackendDependencySpec with custom or default dependencies
        """
        # First check for legacy format (backend:version)
        if self._custom_specs:
            return self._custom_specs

        # Fall back to default dependencies using version specifiers
        default_version_deps = self.default_dependencies_specs.get(self.backend, {})
        if not default_version_deps:
            return None

        # Normalize version by removing 'v' prefix if present
        normalized_version = self.version.lstrip('v')

        try:
            version_obj = Version(normalized_version)
        except Exception as e:
            logger.warning(
                f"Invalid version format '{self.version}' for backend {self.backend}: {e}"
            )
            return None

        # Check each version specifier to find a match
        for version_spec, dependencies in default_version_deps.items():
            specifier_set = SpecifierSet(version_spec)
            if version_obj in specifier_set:
                logger.debug(
                    f"Found matching dependency spec for {self.backend} {self.version}: {version_spec}"
                )
                return BackendDependencySpec(
                    backend=self.backend, dependencies=dependencies
                )

        return None

    def get_pipx_install_args(self) -> List[str]:
        """
        Get pipx installation arguments for a backend.

        Args:
            backend: Backend name
            version: Backend version

        Returns:
            List of additional arguments for pipx install command
        """
        spec = self.get_dependency_spec()
        if not spec or not spec.dependencies:
            return []

        pip_args = spec.to_pip_args()
        return [pip_args] if pip_args else []

    def build_bootstrap_provisioning_plan(self) -> BackendProvisioningPlan:
        return BackendProvisioningPlan(
            dependencies=self._get_hash_pinned_bootstrap_dependencies(),
            executable=self._get_hash_pinned_bootstrap_executable(),
        )

    def _get_hash_pinned_bootstrap_dependencies(self) -> List[HashPinnedDependencySpec]:
        raw_value = self._get_env_value("GPUSTACK_BOOTSTRAP_PYTHON_DEPS")
        if not raw_value:
            return []

        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "GPUSTACK_BOOTSTRAP_PYTHON_DEPS must be valid JSON"
            ) from exc

        if not isinstance(payload, list):
            raise ValueError("GPUSTACK_BOOTSTRAP_PYTHON_DEPS must be a JSON list")

        dependencies: List[HashPinnedDependencySpec] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(
                    "Each bootstrap dependency must be a JSON object with requirement and hashes"
                )

            requirement = str(item.get("requirement", "")).strip()
            if not _is_exact_pinned_requirement(requirement):
                raise ValueError(
                    f"Bootstrap dependency {requirement!r} must use an exact == pin"
                )

            hashes_value = item.get("hashes")
            if not isinstance(hashes_value, list) or not hashes_value:
                raise ValueError(
                    f"Bootstrap dependency {requirement!r} must be hash-pinned"
                )

            hashes = [_normalize_sha256(str(value)) for value in hashes_value]
            dependencies.append(
                HashPinnedDependencySpec(requirement=requirement, hashes=hashes)
            )

        return dependencies

    def _get_hash_pinned_bootstrap_executable(self) -> Optional[ExecutableSpec]:
        raw_value = self._get_env_value("GPUSTACK_BOOTSTRAP_EXECUTABLE")
        if not raw_value:
            return None

        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "GPUSTACK_BOOTSTRAP_EXECUTABLE must be valid JSON"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError("GPUSTACK_BOOTSTRAP_EXECUTABLE must be a JSON object")

        name = str(payload.get("name", "")).strip()
        path = str(payload.get("path", "")).strip()
        sha256 = str(payload.get("sha256", "")).strip()

        if not name:
            raise ValueError("Bootstrap executable input must declare a name")
        if not path:
            raise ValueError("Bootstrap executable input must declare a path")
        if not sha256:
            raise ValueError(
                f"Bootstrap executable {name!r} must be hash-pinned with sha256"
            )

        return ExecutableSpec(
            name=name,
            path=path,
            sha256=_normalize_sha256(sha256),
        )
