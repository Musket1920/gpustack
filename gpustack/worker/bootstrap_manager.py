from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Callable, TypeVar, Union, cast
from urllib.parse import quote
import venv

from gpustack.config.config import Config


PathLikeValue = Union[str, int]
T = TypeVar("T")
_SHA256_PATTERN = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")


@dataclass(frozen=True)
class BootstrapRuntimeRoots:
    workspace: Path
    artifacts: Path
    manifests: Path
    locks: Path


@dataclass(frozen=True)
class PreparedEnvironmentIdentity:
    backend: str
    backend_version: str
    recipe_id: str | None
    prepared_environment_id: str | None
    resolver_version: str
    python_identity: dict[str, str]

    def manifest_payload(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "recipe_id": self.recipe_id,
            "prepared_environment_id": self.prepared_environment_id,
            "resolver_version": self.resolver_version,
            "python_identity": self.python_identity,
        }


@dataclass(frozen=True)
class DirectProcessLaunchArtifacts:
    backend: str
    backend_version: str
    recipe_id: str | None
    prepared_environment_id: str | None
    resolver_version: str
    python_identity: dict[str, str]
    manifest_hash: str
    prepared_cache_root: Path
    prepared_cache_context_path: Path
    runtime_artifact_path: Path
    prepared_env_path: Path
    prepared_config_path: Path
    prepared_launch_path: Path
    prepared_provenance_path: Path
    prepared_context: dict[str, object]
    runtime_artifact: dict[str, object]
    prepared_config: dict[str, object]
    prepared_provenance: dict[str, object]


@dataclass(frozen=True)
class HostBootstrapInput:
    name: str
    source: str
    sha256: str

    def audit_payload(self) -> dict[str, str]:
        return {
            "name": self.name,
            "source": self.source,
            "sha256": f"sha256:{BootstrapManager.normalize_sha256(self.sha256)}",
        }


@dataclass(frozen=True)
class HostBootstrapAction:
    name: str
    details: dict[str, object] | None = None

    def audit_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"name": self.name}
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass(frozen=True)
class HostBootstrapRequest:
    backend: str
    backend_version: str
    recipe_id: str
    recipe_source: str
    inputs: tuple[HostBootstrapInput, ...]
    actions: tuple[HostBootstrapAction, ...]


@dataclass(frozen=True)
class HostBootstrapResult:
    backend: str
    backend_version: str
    recipe_id: str
    dry_run: bool
    mutation_performed: bool
    audit_path: Path
    audit_events: tuple[dict[str, object], ...]
    actions: tuple[dict[str, object], ...]


class BootstrapManager:
    PREPARED_CACHE_CONTEXT_FILENAME = "bootstrap-prepared-context.json"
    BOOTSTRAP_MANIFEST_FILENAME = "bootstrap-manifest.json"
    BOOTSTRAP_ARTIFACT_FILENAME = "bootstrap-artifact.json"
    BOOTSTRAP_CONTEXT_FILENAME = "bootstrap-context.txt"
    BOOTSTRAP_LOCK_FILENAME = "bootstrap.lock"
    PREPARED_CACHE_RESOLVER_VERSION = "direct-process-bootstrap-v1"
    PREPARED_CACHE_VENV_DIRNAME = "venv"
    PREPARED_CACHE_ARTIFACTS_DIRNAME = "artifacts"
    PREPARED_CACHE_BIN_DIRNAME = "bin"
    PREPARED_CACHE_ENV_FILENAME = "prepared.env"
    PREPARED_CACHE_CONFIG_FILENAME = "prepared-config.json"
    PREPARED_CACHE_LAUNCH_FILENAME = "prepared-launch.sh"
    PREPARED_CACHE_PROVISIONING_FILENAME = "bootstrap-provisioning.json"
    PREPARED_CACHE_EXECUTABLE_PROVENANCE_FILENAME = "executable-provenance.json"
    PREPARED_CACHE_REQUIREMENTS_FILENAME = "requirements.lock"
    PREPARED_CACHE_LOCK_SUFFIX = ".lock"
    PREPARED_CACHE_STAGE_PREFIX = ".bootstrap-staging-"
    HOST_BOOTSTRAP_AUDIT_FILENAME = "host-bootstrap-audit.json"
    HOST_BOOTSTRAP_RESOLVER_VERSION = "host-bootstrap-v1"
    HOST_BOOTSTRAP_SUPPORTED_PLATFORMS = ("linux",)

    def __init__(self, cfg: Config):
        self._config = cfg

    @property
    def prepared_cache_dir(self) -> Path:
        return Path(cast(str, self._config.bootstrap_cache_dir))

    @property
    def workspace_dir(self) -> Path:
        return Path(cast(str, self._config.bootstrap_workspace_dir))

    @property
    def artifacts_dir(self) -> Path:
        return Path(cast(str, self._config.bootstrap_artifacts_dir))

    @property
    def manifests_dir(self) -> Path:
        return Path(cast(str, self._config.bootstrap_manifests_dir))

    @property
    def locks_dir(self) -> Path:
        return Path(cast(str, self._config.bootstrap_locks_dir))

    def prepared_cache_root(self, backend_name: str, backend_version: str) -> Path:
        return (
            self.prepared_cache_dir
            / self._path_component(backend_name)
            / self._path_component(backend_version)
        )

    def ensure_prepared_cache_root(
        self, backend_name: str, backend_version: str
    ) -> Path:
        return self._ensure_dir(self.prepared_cache_root(backend_name, backend_version))

    def prepare_prepared_cache_root(self, backend_name: str, backend_version: str) -> Path:
        return self.ensure_prepared_cache_root(backend_name, backend_version)

    def prepared_cache_context_path(self, backend_name: str, backend_version: str) -> Path:
        return self.prepared_cache_root(backend_name, backend_version).joinpath(
            self.PREPARED_CACHE_CONTEXT_FILENAME
        )

    def prepared_cache_lock_path(self, backend_name: str, backend_version: str) -> Path:
        root = self.prepared_cache_root(backend_name, backend_version)
        return root.parent / f"{root.name}{self.PREPARED_CACHE_LOCK_SUFFIX}"

    def prepared_cache_venv_root(self, backend_name: str, backend_version: str) -> Path:
        return self.prepared_cache_root(backend_name, backend_version).joinpath(
            self.PREPARED_CACHE_VENV_DIRNAME
        )

    def prepared_cache_artifacts_root(
        self, backend_name: str, backend_version: str
    ) -> Path:
        return self.prepared_cache_root(backend_name, backend_version).joinpath(
            self.PREPARED_CACHE_ARTIFACTS_DIRNAME
        )

    def prepared_cache_bin_root(self, backend_name: str, backend_version: str) -> Path:
        return self.prepared_cache_root(backend_name, backend_version).joinpath(
            self.PREPARED_CACHE_BIN_DIRNAME
        )

    def prepared_cache_artifact_path(
        self,
        backend_name: str,
        backend_version: str,
        *parts: PathLikeValue,
    ) -> Path:
        return self.prepared_cache_artifacts_root(backend_name, backend_version).joinpath(
            *self._path_components(*parts)
        )

    def prepared_cache_bin_path(
        self,
        backend_name: str,
        backend_version: str,
        *parts: PathLikeValue,
    ) -> Path:
        return self.prepared_cache_bin_root(backend_name, backend_version).joinpath(
            *self._path_components(*parts)
        )

    def prepared_environment_identity(
        self,
        *,
        backend_name: str,
        backend_version: str,
        recipe_id: str | None,
        prepared_environment_id: str | None,
    ) -> PreparedEnvironmentIdentity:
        return PreparedEnvironmentIdentity(
            backend=backend_name,
            backend_version=backend_version,
            recipe_id=recipe_id,
            prepared_environment_id=prepared_environment_id,
            resolver_version=self.PREPARED_CACHE_RESOLVER_VERSION,
            python_identity=self.python_identity(),
        )

    @classmethod
    def python_identity(cls) -> dict[str, str]:
        return {
            "implementation": sys.implementation.name,
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "cache_tag": sys.implementation.cache_tag or "",
        }

    @staticmethod
    def manifest_hash(payload: dict[str, object]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def build_prepared_cache_record(
        self,
        identity: PreparedEnvironmentIdentity,
        *,
        invalidation_state: str = "valid",
        invalidation_reason: str | None = None,
    ) -> dict[str, object]:
        manifest_payload = identity.manifest_payload()
        return {
            **manifest_payload,
            "manifest_hash": self.manifest_hash(manifest_payload),
            "invalidation": {
                "state": invalidation_state,
                "reason": invalidation_reason,
            },
        }

    def resolve_direct_process_launch_artifacts(
        self,
        *,
        deployment_id: int,
        model_instance_id: int,
        backend_name: str,
        backend_version: str,
        recipe_id: str | None,
        prepared_environment_id: str | None,
    ) -> DirectProcessLaunchArtifacts:
        expected_identity = self.prepared_environment_identity(
            backend_name=backend_name,
            backend_version=backend_version,
            recipe_id=recipe_id,
            prepared_environment_id=prepared_environment_id,
        )
        expected_record = self.build_prepared_cache_record(expected_identity)
        runtime_artifact_path = self.artifact_path(
            deployment_id,
            model_instance_id,
            self.BOOTSTRAP_ARTIFACT_FILENAME,
        )
        runtime_artifact = self._read_json_file(
            runtime_artifact_path,
            "direct-process runtime bootstrap artifact",
        )
        prepared_cache_context_path = self.prepared_cache_context_path(
            backend_name,
            backend_version,
        )
        prepared_context = self._read_json_file(
            prepared_cache_context_path,
            "prepared direct-process bootstrap cache metadata",
        )

        self._validate_direct_process_identity_record(
            record=runtime_artifact,
            expected_record=expected_record,
            label="runtime bootstrap artifact",
        )
        self._validate_direct_process_identity_record(
            record=prepared_context,
            expected_record=expected_record,
            label="prepared bootstrap cache metadata",
        )

        prepared_artifacts = runtime_artifact.get("prepared_artifacts")
        if not isinstance(prepared_artifacts, dict):
            raise RuntimeError(
                "Runtime bootstrap artifact is missing prepared artifact references"
            )

        expected_paths = {
            "prepared_cache_context": prepared_cache_context_path,
            "env": self.prepared_cache_artifact_path(
                backend_name,
                backend_version,
                self.PREPARED_CACHE_ENV_FILENAME,
            ),
            "config": self.prepared_cache_artifact_path(
                backend_name,
                backend_version,
                self.PREPARED_CACHE_CONFIG_FILENAME,
            ),
            "launch": self.prepared_cache_artifact_path(
                backend_name,
                backend_version,
                self.PREPARED_CACHE_LAUNCH_FILENAME,
            ),
            "executable_provenance": self.prepared_cache_artifact_path(
                backend_name,
                backend_version,
                self.PREPARED_CACHE_EXECUTABLE_PROVENANCE_FILENAME,
            ),
        }

        resolved_paths: dict[str, Path] = {}
        for key, expected_path in expected_paths.items():
            observed_path = prepared_artifacts.get(key)
            if observed_path != str(expected_path):
                raise RuntimeError(
                    f"Runtime bootstrap artifact {key} mismatch (expected {expected_path!s}, found {observed_path!r})"
                )
            if not expected_path.exists():
                raise RuntimeError(
                    f"Prepared bootstrap artifact '{expected_path}' is missing"
                )
            resolved_paths[key] = expected_path

        prepared_config = self._read_json_file(
            resolved_paths["config"],
            "prepared direct-process bootstrap config artifact",
        )
        if prepared_config.get("backend") != backend_name:
            raise RuntimeError(
                "Prepared bootstrap config artifact backend mismatch "
                f"(expected {backend_name!r}, found {prepared_config.get('backend')!r})"
            )
        if prepared_config.get("backend_version") != backend_version:
            raise RuntimeError(
                "Prepared bootstrap config artifact backend_version mismatch "
                f"(expected {backend_version!r}, found {prepared_config.get('backend_version')!r})"
            )
        if prepared_config.get("manifest_hash") != expected_record["manifest_hash"]:
            raise RuntimeError(
                "Prepared bootstrap config artifact manifest hash mismatch "
                f"(expected {expected_record['manifest_hash']!r}, found {prepared_config.get('manifest_hash')!r})"
            )
        if prepared_config.get("env_artifact") != str(resolved_paths["env"]):
            raise RuntimeError(
                "Prepared bootstrap config artifact env reference mismatch"
            )
        if prepared_config.get("executable_provenance") != str(
            resolved_paths["executable_provenance"]
        ):
            raise RuntimeError(
                "Prepared bootstrap config artifact executable provenance reference mismatch"
            )

        prepared_provenance = self._read_json_file(
            resolved_paths["executable_provenance"],
            "prepared direct-process executable provenance artifact",
        )

        return DirectProcessLaunchArtifacts(
            backend=backend_name,
            backend_version=backend_version,
            recipe_id=recipe_id,
            prepared_environment_id=prepared_environment_id,
            resolver_version=expected_identity.resolver_version,
            python_identity=expected_identity.python_identity,
            manifest_hash=cast(str, expected_record["manifest_hash"]),
            prepared_cache_root=self.prepared_cache_root(backend_name, backend_version),
            prepared_cache_context_path=prepared_cache_context_path,
            runtime_artifact_path=runtime_artifact_path,
            prepared_env_path=resolved_paths["env"],
            prepared_config_path=resolved_paths["config"],
            prepared_launch_path=resolved_paths["launch"],
            prepared_provenance_path=resolved_paths["executable_provenance"],
            prepared_context=prepared_context,
            runtime_artifact=runtime_artifact,
            prepared_config=prepared_config,
            prepared_provenance=prepared_provenance,
        )

    def prepare_prepared_cache_root_for(
        self,
        backend_name: str,
        backend_version: str,
        materialize: Callable[[Path], T],
    ) -> T:
        root = self.prepared_cache_root(backend_name, backend_version)
        lock_path = self.prepared_cache_lock_path(backend_name, backend_version)

        def prepare() -> T:
            with self._prepare_lock(lock_path):
                self._cleanup_prepared_cache_staging_roots(root)
                if root.exists():
                    return cast(T, root)

                staging_root = self._prepared_cache_staging_root(root)
                self._remove_tree(staging_root)
                self._ensure_dir(staging_root.parent)
                self._ensure_dir(staging_root)
                try:
                    materialized = materialize(staging_root)
                    staging_root.replace(root)
                    return cast(T, root if materialized == staging_root else materialized)
                except Exception:
                    self._remove_tree(staging_root)
                    raise

        return self._prepare_with_cleanup(
            prepare=prepare,
            cleanup=lambda: self.cleanup_prepared_cache_root(backend_name, backend_version),
        )

    def materialize_recipe_prepared_cache(
        self,
        backend_name: str,
        backend_version: str,
        prepared_cache_record: dict[str, object],
        audit_events: list[dict[str, object]] | None = None,
    ) -> Path:
        def materialize(root: Path) -> Path:
            self._materialize_recipe_prepared_cache_artifacts(
                root=root,
                backend_name=backend_name,
                backend_version=backend_version,
                prepared_cache_record=prepared_cache_record,
                audit_events=audit_events,
            )
            return root

        return self.prepare_prepared_cache_root_for(
            backend_name,
            backend_version,
            materialize,
        )

    def cleanup_prepared_cache_root(self, backend_name: str, backend_version: str) -> Path:
        root = self.prepared_cache_root(backend_name, backend_version)
        self._cleanup_prepared_cache_staging_roots(root)
        self._remove_tree(root)
        with self._suppress_file_not_found():
            self.prepared_cache_lock_path(backend_name, backend_version).unlink()
        self._prune_empty_parent(root.parent, stop_at=self.prepared_cache_dir)
        return root

    def prepared_cache_repair_reason(self, backend_name: str, backend_version: str) -> str | None:
        root = self.prepared_cache_root(backend_name, backend_version)
        if not root.is_dir():
            return None

        required_paths = {
            "prepared environment": self.prepared_cache_venv_root(
                backend_name, backend_version
            ),
            "prepared artifacts root": self.prepared_cache_artifacts_root(
                backend_name, backend_version
            ),
            "prepared env artifact": self.prepared_cache_artifact_path(
                backend_name,
                backend_version,
                self.PREPARED_CACHE_ENV_FILENAME,
            ),
            "prepared config artifact": self.prepared_cache_artifact_path(
                backend_name,
                backend_version,
                self.PREPARED_CACHE_CONFIG_FILENAME,
            ),
            "prepared launch artifact": self.prepared_cache_artifact_path(
                backend_name,
                backend_version,
                self.PREPARED_CACHE_LAUNCH_FILENAME,
            ),
            "prepared executable provenance": self.prepared_cache_artifact_path(
                backend_name,
                backend_version,
                self.PREPARED_CACHE_EXECUTABLE_PROVENANCE_FILENAME,
            ),
            "prepared provisioning audit": self.prepared_cache_artifact_path(
                backend_name,
                backend_version,
                self.PREPARED_CACHE_PROVISIONING_FILENAME,
            ),
        }
        for label, path in required_paths.items():
            if not path.exists():
                return f"{label} missing"

        return None

    def read_prepared_cache_audit_events(
        self, backend_name: str, backend_version: str
    ) -> list[dict[str, object]]:
        provisioning_path = self.prepared_cache_artifact_path(
            backend_name,
            backend_version,
            self.PREPARED_CACHE_PROVISIONING_FILENAME,
        )
        if not provisioning_path.exists():
            return []

        try:
            payload = json.loads(provisioning_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        events = payload.get("audit_events")
        if not isinstance(events, list):
            return []
        return [event for event in events if isinstance(event, dict)]

    def record_prepared_cache_audit_event(
        self,
        backend_name: str,
        backend_version: str,
        *,
        operation: str,
        outcome: str,
        reason: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        lock_path = self.prepared_cache_lock_path(backend_name, backend_version)
        provisioning_path = self.prepared_cache_artifact_path(
            backend_name,
            backend_version,
            self.PREPARED_CACHE_PROVISIONING_FILENAME,
        )
        if not provisioning_path.exists():
            return

        with self._prepare_lock(lock_path):
            if not provisioning_path.exists():
                return
            payload = json.loads(provisioning_path.read_text(encoding="utf-8"))
            events = payload.get("audit_events")
            if not isinstance(events, list):
                events = []
            events.append(
                self._build_audit_event(
                    operation=operation,
                    outcome=outcome,
                    reason=reason,
                    details=details,
                )
            )
            payload["audit_events"] = events
            provisioning_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def deployment_root(self, deployment_id: int) -> Path:
        return self.workspace_dir / self._deployment_component(deployment_id)

    def runtime_workspace_root(self, deployment_id: int, model_instance_id: int) -> Path:
        return self.deployment_root(deployment_id) / self._instance_component(
            model_instance_id
        )

    def runtime_artifacts_root(self, deployment_id: int, model_instance_id: int) -> Path:
        return self.artifacts_dir / self._runtime_key(deployment_id, model_instance_id)

    def runtime_manifests_root(self, deployment_id: int, model_instance_id: int) -> Path:
        return self.manifests_dir / self._runtime_key(deployment_id, model_instance_id)

    def runtime_locks_root(self, deployment_id: int, model_instance_id: int) -> Path:
        return self.locks_dir / self._runtime_key(deployment_id, model_instance_id)

    def runtime_roots(
        self, deployment_id: int, model_instance_id: int
    ) -> BootstrapRuntimeRoots:
        return BootstrapRuntimeRoots(
            workspace=self.runtime_workspace_root(deployment_id, model_instance_id),
            artifacts=self.runtime_artifacts_root(deployment_id, model_instance_id),
            manifests=self.runtime_manifests_root(deployment_id, model_instance_id),
            locks=self.runtime_locks_root(deployment_id, model_instance_id),
        )

    def prepare_runtime_roots(
        self, deployment_id: int, model_instance_id: int
    ) -> BootstrapRuntimeRoots:
        roots = self.runtime_roots(deployment_id, model_instance_id)
        return BootstrapRuntimeRoots(
            workspace=self._ensure_dir(roots.workspace),
            artifacts=self._ensure_dir(roots.artifacts),
            manifests=self._ensure_dir(roots.manifests),
            locks=self._ensure_dir(roots.locks),
        )

    def prepare_runtime_roots_for(
        self,
        deployment_id: int,
        model_instance_id: int,
        materialize: Callable[[BootstrapRuntimeRoots], T],
    ) -> T:
        roots = self.prepare_runtime_roots(deployment_id, model_instance_id)
        return self._prepare_with_cleanup(
            prepare=lambda: materialize(roots),
            cleanup=lambda: self.cleanup_runtime_roots(deployment_id, model_instance_id),
        )

    def cleanup_runtime_roots(
        self, deployment_id: int, model_instance_id: int
    ) -> BootstrapRuntimeRoots:
        roots = self.runtime_roots(deployment_id, model_instance_id)
        self._remove_tree(roots.workspace)
        self._remove_tree(roots.artifacts)
        self._remove_tree(roots.manifests)
        self._remove_tree(roots.locks)
        self._prune_empty_parent(roots.workspace.parent, stop_at=self.workspace_dir)
        self._prune_empty_parent(roots.artifacts.parent, stop_at=self.artifacts_dir)
        self._prune_empty_parent(roots.manifests.parent, stop_at=self.manifests_dir)
        self._prune_empty_parent(roots.locks.parent, stop_at=self.locks_dir)
        return roots

    def ensure_runtime_workspace_root(
        self, deployment_id: int, model_instance_id: int
    ) -> Path:
        return self._ensure_dir(
            self.runtime_workspace_root(deployment_id, model_instance_id)
        )

    def ensure_runtime_artifacts_root(
        self, deployment_id: int, model_instance_id: int
    ) -> Path:
        return self._ensure_dir(
            self.runtime_artifacts_root(deployment_id, model_instance_id)
        )

    def ensure_runtime_manifests_root(
        self, deployment_id: int, model_instance_id: int
    ) -> Path:
        return self._ensure_dir(
            self.runtime_manifests_root(deployment_id, model_instance_id)
        )

    def ensure_runtime_locks_root(self, deployment_id: int, model_instance_id: int) -> Path:
        return self._ensure_dir(self.runtime_locks_root(deployment_id, model_instance_id))

    def workspace_path(
        self, deployment_id: int, model_instance_id: int, *parts: PathLikeValue
    ) -> Path:
        return self.runtime_workspace_root(deployment_id, model_instance_id).joinpath(
            *self._path_components(*parts)
        )

    def artifact_path(
        self, deployment_id: int, model_instance_id: int, *parts: PathLikeValue
    ) -> Path:
        return self.runtime_artifacts_root(deployment_id, model_instance_id).joinpath(
            *self._path_components(*parts)
        )

    def manifest_path(
        self, deployment_id: int, model_instance_id: int, *parts: PathLikeValue
    ) -> Path:
        return self.runtime_manifests_root(deployment_id, model_instance_id).joinpath(
            *self._path_components(*parts)
        )

    def lock_path(
        self,
        deployment_id: int,
        model_instance_id: int,
        name: PathLikeValue = "bootstrap.lock",
    ) -> Path:
        return self.runtime_locks_root(deployment_id, model_instance_id) / self._path_component(
            name
        )

    def ensure_parent_dir(self, path: Path) -> Path:
        return self._ensure_dir(path.parent)

    def host_bootstrap_audit_path(
        self,
        *,
        backend_name: str,
        backend_version: str,
        recipe_id: str,
    ) -> Path:
        return (
            self.manifests_dir
            / "host-bootstrap"
            / self._path_component(backend_name)
            / self._path_component(backend_version)
            / self._path_component(recipe_id)
            / self.HOST_BOOTSTRAP_AUDIT_FILENAME
        )

    def execute_host_bootstrap(
        self,
        request: HostBootstrapRequest,
        *,
        mutate_action: Callable[[HostBootstrapAction], None] | None = None,
        dry_run: bool | None = None,
    ) -> HostBootstrapResult:
        resolved_dry_run = (
            self._config.host_bootstrap_dry_run if dry_run is None else dry_run
        )
        audit_path = self.host_bootstrap_audit_path(
            backend_name=request.backend,
            backend_version=request.backend_version,
            recipe_id=request.recipe_id,
        )
        audit_events: list[dict[str, object]] = []
        mutation_performed = False

        def reject(reason: str, *, details: dict[str, object] | None = None) -> None:
            audit_events.append(
                self._build_audit_event(
                    operation="host-bootstrap",
                    outcome="rejected",
                    reason=reason,
                    details=details,
                )
            )
            self._write_host_bootstrap_audit_log(
                request=request,
                audit_path=audit_path,
                dry_run=resolved_dry_run,
                mutation_performed=mutation_performed,
                audit_events=audit_events,
            )
            raise RuntimeError(reason)

        if not self._config.enable_host_bootstrap:
            reject("Host bootstrap is disabled on this worker")

        platform_name = sys.platform.lower()
        if not any(
            platform_name == supported or platform_name.startswith(f"{supported}-")
            for supported in self.HOST_BOOTSTRAP_SUPPORTED_PLATFORMS
        ) and not any(platform_name.startswith(supported) for supported in self.HOST_BOOTSTRAP_SUPPORTED_PLATFORMS):
            reject(
                "Host bootstrap is only supported on Linux workers",
                details={"platform": sys.platform},
            )

        if not self._is_allowlisted_host_bootstrap_source(request.recipe_source):
            reject(
                f"Host bootstrap recipe source {request.recipe_source!r} is not allowlisted",
                details={
                    "recipe_source": request.recipe_source,
                    "allowlist": self._host_bootstrap_allowed_recipe_sources(),
                },
            )

        if not request.inputs:
            reject(
                "Host bootstrap inputs must be declared and hash-pinned",
                details={"recipe_source": request.recipe_source},
            )

        normalized_inputs: list[dict[str, str]] = []
        for input_spec in request.inputs:
            try:
                normalized_inputs.append(input_spec.audit_payload())
            except ValueError as exc:
                reject(str(exc), details={"input": input_spec.name, "source": input_spec.source})

        if not request.actions:
            reject("Host bootstrap request must declare at least one action")

        action_payloads = [action.audit_payload() for action in request.actions]
        audit_events.append(
            self._build_audit_event(
                operation="host-bootstrap",
                outcome="planned",
                details={
                    "recipe_source": request.recipe_source,
                    "inputs": normalized_inputs,
                    "actions": action_payloads,
                },
            )
        )

        if resolved_dry_run:
            audit_events.append(
                self._build_audit_event(
                    operation="host-bootstrap",
                    outcome="dry-run",
                    details={
                        "mutations_blocked": True,
                        "action_count": len(action_payloads),
                    },
                )
            )
        else:
            if mutate_action is None:
                reject("Host bootstrap mutation executor is not configured")
            mutation_executor = cast(Callable[[HostBootstrapAction], None], mutate_action)

            for action, action_payload in zip(request.actions, action_payloads):
                audit_events.append(
                    self._build_audit_event(
                        operation="host-bootstrap-mutation",
                        outcome="started",
                        details=action_payload,
                    )
                )
                try:
                    mutation_executor(action)
                except Exception as exc:
                    audit_events.append(
                        self._build_audit_event(
                            operation="host-bootstrap-mutation",
                            outcome="failed",
                            reason=str(exc),
                            details=action_payload,
                        )
                    )
                    self._write_host_bootstrap_audit_log(
                        request=request,
                        audit_path=audit_path,
                        dry_run=resolved_dry_run,
                        mutation_performed=mutation_performed,
                        audit_events=audit_events,
                        normalized_inputs=normalized_inputs,
                        actions=action_payloads,
                    )
                    raise
                audit_events.append(
                    self._build_audit_event(
                        operation="host-bootstrap-mutation",
                        outcome="completed",
                        details=action_payload,
                    )
                )

            mutation_performed = True
            audit_events.append(
                self._build_audit_event(
                    operation="host-bootstrap",
                    outcome="completed",
                    details={"action_count": len(action_payloads)},
                )
            )

        self._write_host_bootstrap_audit_log(
            request=request,
            audit_path=audit_path,
            dry_run=resolved_dry_run,
            mutation_performed=mutation_performed,
            audit_events=audit_events,
            normalized_inputs=normalized_inputs,
            actions=action_payloads,
        )
        return HostBootstrapResult(
            backend=request.backend,
            backend_version=request.backend_version,
            recipe_id=request.recipe_id,
            dry_run=resolved_dry_run,
            mutation_performed=mutation_performed,
            audit_path=audit_path,
            audit_events=tuple(audit_events),
            actions=tuple(action_payloads),
        )

    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        match = _SHA256_PATTERN.match(value.strip())
        if not match:
            raise ValueError(f"Expected sha256 hash, got {value!r}")
        return match.group(1).lower()

    def _host_bootstrap_allowed_recipe_sources(self) -> tuple[str, ...]:
        raw_sources = self._config.host_bootstrap_allowed_recipe_sources or []
        return tuple(source.strip() for source in raw_sources if source.strip())

    def _is_allowlisted_host_bootstrap_source(self, recipe_source: str) -> bool:
        allowlist = self._host_bootstrap_allowed_recipe_sources()
        if not allowlist:
            return False
        normalized_source = recipe_source.strip()
        for allowed_source in allowlist:
            if normalized_source == allowed_source:
                return True
            prefix = allowed_source.rstrip("/") + "/"
            if normalized_source.startswith(prefix):
                return True
        return False

    def _write_host_bootstrap_audit_log(
        self,
        *,
        request: HostBootstrapRequest,
        audit_path: Path,
        dry_run: bool,
        mutation_performed: bool,
        audit_events: list[dict[str, object]],
        normalized_inputs: list[dict[str, str]] | None = None,
        actions: list[dict[str, object]] | None = None,
    ) -> None:
        self._ensure_dir(audit_path.parent)
        audit_path.write_text(
            json.dumps(
                {
                    "backend": request.backend,
                    "backend_version": request.backend_version,
                    "recipe_id": request.recipe_id,
                    "recipe_source": request.recipe_source,
                    "resolver_version": self.HOST_BOOTSTRAP_RESOLVER_VERSION,
                    "platform": sys.platform,
                    "dry_run": dry_run,
                    "mutation_performed": mutation_performed,
                    "inputs": normalized_inputs
                    if normalized_inputs is not None
                    else [
                        {
                            "name": input_spec.name,
                            "source": input_spec.source,
                            "sha256": input_spec.sha256,
                        }
                        for input_spec in request.inputs
                    ],
                    "actions": actions
                    if actions is not None
                    else [action.audit_payload() for action in request.actions],
                    "audit_events": audit_events,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _materialize_recipe_prepared_cache_artifacts(
        self,
        *,
        root: Path,
        backend_name: str,
        backend_version: str,
        prepared_cache_record: dict[str, object],
        audit_events: list[dict[str, object]] | None = None,
    ) -> None:
        published_root = self._published_prepared_cache_root(root)
        venv_root = root / self.PREPARED_CACHE_VENV_DIRNAME
        published_venv_root = published_root / self.PREPARED_CACHE_VENV_DIRNAME
        artifacts_root = root / self.PREPARED_CACHE_ARTIFACTS_DIRNAME
        published_artifacts_root = published_root / self.PREPARED_CACHE_ARTIFACTS_DIRNAME
        bin_root = root / self.PREPARED_CACHE_BIN_DIRNAME
        published_bin_root = published_root / self.PREPARED_CACHE_BIN_DIRNAME
        self._ensure_dir(artifacts_root)
        self._ensure_dir(bin_root)

        self._create_venv(venv_root)
        from gpustack.worker.backend_dependency_manager import BackendDependencyManager

        dependency_manager = BackendDependencyManager(backend_name, backend_version)
        try:
            provisioning_plan = dependency_manager.build_bootstrap_provisioning_plan()
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        requirements_path = artifacts_root / self.PREPARED_CACHE_REQUIREMENTS_FILENAME
        install_command: list[str] = []

        if provisioning_plan.dependencies:
            requirements_path.write_text(
                "\n".join(
                    dependency.to_requirements_line()
                    for dependency in provisioning_plan.dependencies
                )
                + "\n",
                encoding="utf-8",
            )
            install_command = [
                str(self._venv_python_path(venv_root)),
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "-r",
                str(requirements_path),
            ]
            subprocess.run(
                install_command,
                check=True,
                capture_output=True,
                text=True,
            )

        executable_provenance = self._resolve_executable_provenance(
            bin_root=bin_root,
            published_bin_root=published_bin_root,
            executable_spec=provisioning_plan.executable,
        )
        env_path = artifacts_root / self.PREPARED_CACHE_ENV_FILENAME
        config_path = artifacts_root / self.PREPARED_CACHE_CONFIG_FILENAME
        launch_path = artifacts_root / self.PREPARED_CACHE_LAUNCH_FILENAME
        provenance_path = artifacts_root / self.PREPARED_CACHE_EXECUTABLE_PROVENANCE_FILENAME
        provisioning_path = artifacts_root / self.PREPARED_CACHE_PROVISIONING_FILENAME

        env_lines = [
            f"VIRTUAL_ENV={published_venv_root}",
            f"PATH={published_bin_root}{os.pathsep}{self._venv_bin_dir(published_venv_root)}{os.pathsep}${{PATH}}",
        ]
        env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        provenance_path.write_text(
            json.dumps(executable_provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config_path.write_text(
            json.dumps(
                {
                    "backend": backend_name,
                    "backend_version": backend_version,
                    "prepared_cache_root": str(published_root),
                    "venv_root": str(published_venv_root),
                    "env_artifact": str(published_artifacts_root / self.PREPARED_CACHE_ENV_FILENAME),
                    "requirements_lock": (
                        str(published_artifacts_root / self.PREPARED_CACHE_REQUIREMENTS_FILENAME)
                        if provisioning_plan.dependencies
                        else None
                    ),
                    "executable_provenance": str(
                        published_artifacts_root
                        / self.PREPARED_CACHE_EXECUTABLE_PROVENANCE_FILENAME
                    ),
                    "manifest_hash": prepared_cache_record["manifest_hash"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        launch_lines = [
            "#!/usr/bin/env sh",
            "set -eu",
            f". \"{env_path}\"",
            'if [ "$#" -gt 0 ] && [ "$1" = "__GPUSTACK_PREPARED_EXECUTABLE__" ]; then',
            "  shift",
            (
                f"  exec \"{executable_provenance['prepared_path']}\" \"$@\""
                if executable_provenance.get("state") == "resolved"
                else '  printf "%s\n" "No resolved bootstrap executable recorded." >&2; exit 1'
            ),
            "fi",
            'exec "$@"',
        ]
        launch_path.write_text("\n".join(launch_lines) + "\n", encoding="utf-8")
        if os.name != "nt":
            launch_path.chmod(launch_path.stat().st_mode | stat.S_IEXEC)

        provisioning_path.write_text(
            json.dumps(
                {
                    "backend": backend_name,
                    "backend_version": backend_version,
                    "prepared_cache_root": str(published_root),
                    "venv_root": str(published_venv_root),
                    "provisioning_inputs": provisioning_plan.to_audit_payload(),
                    "requirements_lock": (
                        str(published_artifacts_root / self.PREPARED_CACHE_REQUIREMENTS_FILENAME)
                        if provisioning_plan.dependencies
                        else None
                    ),
                    "install_command": install_command,
                    "artifacts": {
                        "env": str(published_artifacts_root / self.PREPARED_CACHE_ENV_FILENAME),
                        "config": str(published_artifacts_root / self.PREPARED_CACHE_CONFIG_FILENAME),
                        "launch": str(published_artifacts_root / self.PREPARED_CACHE_LAUNCH_FILENAME),
                        "executable_provenance": str(
                            published_artifacts_root
                            / self.PREPARED_CACHE_EXECUTABLE_PROVENANCE_FILENAME
                        ),
                    },
                    "prepared_cache_record": prepared_cache_record,
                    "audit_events": [
                        *(audit_events or []),
                        self._build_audit_event(
                            operation=("repair" if audit_events else "provision"),
                            outcome=("repaired" if audit_events else "prepared"),
                            details={
                                "manifest_hash": prepared_cache_record["manifest_hash"],
                                "prepared_cache_root": str(published_root),
                            },
                        ),
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _build_audit_event(
        *,
        operation: str,
        outcome: str,
        reason: str | None = None,
        details: dict[str, object] | None = None,
    ) -> dict[str, object]:
        event: dict[str, object] = {
            "operation": operation,
            "outcome": outcome,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        if reason is not None:
            event["reason"] = reason
        if details:
            event["details"] = details
        return event

    @staticmethod
    def _create_venv(path: Path) -> None:
        builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=os.name != "nt")
        builder.create(path)

    @staticmethod
    def _venv_bin_dir(venv_root: Path) -> Path:
        return venv_root / ("Scripts" if os.name == "nt" else "bin")

    def _venv_python_path(self, venv_root: Path) -> Path:
        executable_name = "python.exe" if os.name == "nt" else "python"
        return self._venv_bin_dir(venv_root) / executable_name

    def _resolve_executable_provenance(
        self,
        *,
        bin_root: Path,
        published_bin_root: Path,
        executable_spec,
    ) -> dict[str, object]:
        if executable_spec is None:
            return {
                "state": "not-declared",
                "reason": "No hash-pinned executable input declared for bootstrap provisioning",
                "prepared_path": None,
            }

        source_path = Path(executable_spec.path)
        if not source_path.exists():
            raise RuntimeError(
                f"Bootstrap executable '{executable_spec.name}' does not exist at {source_path}"
            )

        observed_sha256 = self._sha256_path(source_path)
        if observed_sha256 != executable_spec.sha256:
            raise RuntimeError(
                f"Bootstrap executable '{executable_spec.name}' hash mismatch: expected sha256:{executable_spec.sha256}, found sha256:{observed_sha256}"
            )

        target_path = bin_root / source_path.name
        shutil.copy2(source_path, target_path)
        if os.name != "nt":
            target_path.chmod(target_path.stat().st_mode | stat.S_IEXEC)

        return {
            "state": "resolved",
            "name": executable_spec.name,
            "source_path": str(source_path),
            "prepared_path": str(published_bin_root / source_path.name),
            "sha256": f"sha256:{observed_sha256}",
        }

    @staticmethod
    def _sha256_path(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _read_json_file(path: Path, label: str) -> dict[str, object]:
        if not path.exists():
            raise RuntimeError(f"{label} is missing at {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{label} is unreadable at {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"{label} at {path} must be a JSON object")
        return payload

    @staticmethod
    def _validate_direct_process_identity_record(
        *,
        record: dict[str, object],
        expected_record: dict[str, object],
        label: str,
    ) -> None:
        invalidation = record.get("invalidation")
        if not isinstance(invalidation, dict):
            raise RuntimeError(f"{label} invalidation state is missing")
        if invalidation.get("state") != "valid":
            raise RuntimeError(
                f"{label} invalidation state is {invalidation.get('state')!r}"
            )

        for key in (
            "backend",
            "recipe_id",
            "backend_version",
            "resolver_version",
            "python_identity",
            "prepared_environment_id",
            "manifest_hash",
        ):
            if record.get(key) != expected_record.get(key):
                raise RuntimeError(
                    f"{label} {key} mismatch (expected {expected_record.get(key)!r}, found {record.get(key)!r})"
                )

    @staticmethod
    def _prepare_with_cleanup(prepare: Callable[[], T], cleanup: Callable[[], object]) -> T:
        try:
            return prepare()
        except Exception:
            cleanup()
            raise

    @staticmethod
    def _prepared_cache_staging_root(root: Path) -> Path:
        return root.parent / (
            f"{root.name}{BootstrapManager.PREPARED_CACHE_STAGE_PREFIX}"
            f"{os.getpid()}-{time.time_ns()}"
        )

    def _published_prepared_cache_root(self, root: Path) -> Path:
        marker = f"{self.PREPARED_CACHE_STAGE_PREFIX}"
        if marker not in root.name:
            return root
        return root.with_name(root.name.split(marker, 1)[0])

    def _cleanup_prepared_cache_staging_roots(self, root: Path) -> None:
        for path in root.parent.glob(f"{root.name}{self.PREPARED_CACHE_STAGE_PREFIX}*"):
            self._remove_tree(path)

    def _prepare_lock(self, path: Path):
        self._ensure_dir(path.parent)
        return _BootstrapFileLock(path)

    @staticmethod
    def _suppress_file_not_found():
        from contextlib import suppress

        return suppress(FileNotFoundError)

    def _runtime_key(self, deployment_id: int, model_instance_id: int) -> Path:
        return Path(self._deployment_component(deployment_id)) / self._instance_component(
            model_instance_id
        )

    @staticmethod
    def _ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _remove_tree(path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _prune_empty_parent(path: Path, *, stop_at: Path) -> None:
        if path == stop_at:
            return

        current = path
        while current != stop_at:
            try:
                current.rmdir()
            except FileNotFoundError:
                current = current.parent
                continue
            except OSError:
                break

            current = current.parent

    @staticmethod
    def _path_component(value: PathLikeValue) -> str:
        return quote(str(value), safe="._-")

    def _path_components(self, *parts: PathLikeValue) -> tuple[str, ...]:
        return tuple(self._path_component(part) for part in parts)

    @staticmethod
    def _deployment_component(deployment_id: int) -> str:
        return f"deployment-{deployment_id}"

    @staticmethod
    def _instance_component(model_instance_id: int) -> str:
        return f"model-instance-{model_instance_id}"


class _BootstrapFileLock:
    def __init__(self, path: Path):
        self._path = path
        self._handle = None

    def __enter__(self):
        self._handle = self._path.open("a+", encoding="utf-8")
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.01)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        assert self._handle is not None
        if os.name == "nt":
            import msvcrt

            self._handle.seek(0)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        return False
