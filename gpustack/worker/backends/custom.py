import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from gpustack.schemas.inference_backend import DirectProcessContract
from gpustack.schemas.models import ModelInstanceDeploymentMetadata
from gpustack.utils.envs import sanitize_env
from gpustack.worker.backends.base import InferenceServer
from gpustack.worker.process_registry import (
    DIRECT_PROCESS_RUNTIME_MODE,
    get_process_group_id,
)

from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    ContainerProfileEnum,
    WorkloadPlan,
    create_workload,
    ContainerRestartPolicyEnum,
)

logger = logging.getLogger(__name__)
_UNRESOLVED_TEMPLATE_PATTERN = re.compile(r"\{\{[^{}]+\}\}")


class CustomServer(InferenceServer):
    """
    Generic pluggable inference server backend with container management capabilities.

    This backend allows users to specify any command and automatically handles:
    - Command path resolution
    - Version management
    - Environment variable setup
    - Model path and port configuration
    - Backend parameters passing
    - Error handling and logging
    - Container management operations (logs, stop, status, etc.)

    When a ``DirectProcessContract`` is attached to the backend's version
    config, this server can also launch the backend as a direct host process
    instead of inside a container.

    Usage:
        Set model.backend_command to specify the command name (e.g., "vllm", "custom-server")
        The backend will automatically call get_command_path(command_name) to resolve the path.
    """

    # ------------------------------------------------------------------
    # Shared direct-process contract implementation
    # ------------------------------------------------------------------

    @classmethod
    def supports_direct_process(cls) -> bool:
        return True

    @classmethod
    def supports_distributed_direct_process(cls) -> bool:
        return False

    @classmethod
    def get_direct_process_bootstrap_recipe_id(cls) -> Optional[str]:
        return "custom"

    def _get_direct_process_contract(self) -> DirectProcessContract:
        """Resolve the ``DirectProcessContract`` from the backend's version config.

        Raises:
            ValueError: if no contract is configured for the current version.
        """
        if not self.inference_backend:
            raise ValueError(
                "Custom backend direct-process requires an inference backend "
                "with a DirectProcessContract, but no inference backend is set."
            )
        version = self._model.backend_version
        try:
            version_config, _ = self.inference_backend.get_version_config(version)
        except KeyError:
            raise ValueError(
                f"Custom backend direct-process: version '{version}' not found "
                f"in backend '{self.inference_backend.backend_name}'."
            )
        contract = version_config.direct_process_contract
        if contract is None:
            raise ValueError(
                "Custom backend direct-process requires a DirectProcessContract "
                f"on version '{version}' of backend "
                f"'{self.inference_backend.backend_name}', but none is configured."
            )
        self._validate_direct_process_contract(contract)
        return contract

    def _validate_direct_process_contract(
        self, contract: DirectProcessContract
    ) -> None:
        if not contract.command_template.strip():
            raise ValueError(
                "Custom backend direct-process contract must declare a non-empty "
                "command_template."
            )
        if not contract.health_path.strip() or not contract.health_path.startswith("/"):
            raise ValueError(
                "Custom backend direct-process contract health_path must start "
                "with '/'."
            )
        if not contract.stop_signal.strip():
            raise ValueError(
                "Custom backend direct-process contract must declare a non-empty "
                "stop_signal."
            )
        if contract.workdir is not None and not contract.workdir.strip():
            raise ValueError(
                "Custom backend direct-process contract workdir must be omitted "
                "or non-empty."
            )
        if contract.env_template:
            for key in contract.env_template:
                if not key or not key.strip():
                    raise ValueError(
                        "Custom backend direct-process contract env_template "
                        "keys must be non-empty."
                    )

    def _validate_resolved_contract_value(
        self,
        *,
        field_name: str,
        resolved_value: str,
        allow_empty: bool = False,
    ) -> None:
        if not allow_empty and not resolved_value.strip():
            raise ValueError(
                "Custom backend direct-process contract "
                f"{field_name} resolved to an empty string."
            )

        unresolved = sorted(set(_UNRESOLVED_TEMPLATE_PATTERN.findall(resolved_value)))
        if unresolved:
            raise ValueError(
                "Custom backend direct-process contract "
                f"{field_name} contains unresolved placeholders: {', '.join(unresolved)}"
            )

    def _validate_prepared_executable_matches_contract(
        self,
        command_args: List[str],
        prepared_executable: str,
        provenance_name: Optional[str],
    ) -> None:
        if not command_args:
            raise ValueError(
                "Custom backend direct-process contract command_template must "
                "resolve to at least one executable argument."
            )

        expected_name = Path(command_args[0]).name or command_args[0]
        prepared_name = Path(prepared_executable).name or prepared_executable
        if provenance_name and provenance_name.strip() and provenance_name != expected_name:
            raise RuntimeError(
                "Prepared direct-process executable provenance name mismatch for "
                "custom backend "
                f"(expected {expected_name!r}, found {provenance_name!r})"
            )
        if prepared_name != expected_name:
            raise RuntimeError(
                "Prepared direct-process executable provenance path mismatch for "
                "custom backend "
                f"(expected executable {expected_name!r}, found {prepared_name!r})"
            )

    def build_direct_process_command(self, port: int) -> List[str]:
        contract = self._get_direct_process_contract()
        resolved = self.inference_backend.replace_command_param(
            version=self._model.backend_version,
            model_path=self._model_path,
            port=port,
            worker_ip=self._worker.ip,
            model_name=self._model_instance.model_name,
            command=contract.command_template,
            env=self._model.env,
        )
        if not resolved:
            raise ValueError(
                "Custom backend direct-process: command_template resolved to "
                "an empty string."
            )
        self._validate_resolved_contract_value(
            field_name="command_template",
            resolved_value=resolved,
        )
        try:
            command_args = shlex.split(resolved)
        except ValueError as exc:
            raise ValueError(
                "Custom backend direct-process contract command_template could "
                f"not be parsed into arguments: {exc}"
            ) from exc
        if not command_args:
            raise ValueError(
                "Custom backend direct-process contract command_template must "
                "resolve to at least one executable argument."
            )
        return command_args

    def build_direct_process_env(self) -> Dict[str, str]:
        env = self._get_configured_env()
        contract = self._get_direct_process_contract()
        if contract.env_template:
            # Resolve placeholders in env template values
            for key, value in contract.env_template.items():
                normalized_key = key.strip()
                resolved_value = self.inference_backend.replace_command_param(
                    version=self._model.backend_version,
                    model_path=self._model_path,
                    port=self._get_serving_port(),
                    worker_ip=self._worker.ip,
                    model_name=self._model_instance.model_name,
                    command=value,
                    env=self._model.env,
                )
                self._validate_resolved_contract_value(
                    field_name=f"env_template[{normalized_key}]",
                    resolved_value=resolved_value,
                    allow_empty=True,
                )
                env[normalized_key] = resolved_value
        return env

    def get_direct_process_health_path(self) -> str:
        contract = self._get_direct_process_contract()
        return contract.health_path

    def preflight_direct_process(
        self,
        command_args: List[str],
        env: Dict[str, str],
        port: int,
    ) -> None:
        failures: List[str] = []
        executable = env.get("GPUSTACK_PREPARED_EXECUTABLE") or (
            command_args[0] if command_args else "<unknown>"
        )

        self._check_direct_process_command(
            executable=executable, env=env, failures=failures
        )
        self._check_direct_process_directories(failures)
        self._check_direct_process_port(port=port, failures=failures)

        if failures:
            message = "; ".join(failures)
            logger.error(
                "Direct-process custom backend preflight failed for %s: %s",
                self._model_instance.name,
                message,
            )
            raise RuntimeError(
                f"Direct-process custom backend host prerequisites not met: {message}"
            )

    def start_direct_process(self):
        deployment_metadata = self._get_deployment_metadata()
        if (
            deployment_metadata.distributed
            or deployment_metadata.distributed_leader
            or deployment_metadata.distributed_follower
        ):
            raise ValueError(
                "Direct-process custom backend does not support distributed launches."
            )

        contract = self._get_direct_process_contract()
        launch_artifacts = self.resolve_direct_process_launch_artifacts()
        self.require_resolved_direct_process_executable(launch_artifacts)

        port = self._get_serving_port()
        env = self.merge_direct_process_prepared_env_artifacts(
            self.build_direct_process_env(), launch_artifacts
        )
        command_args = self.build_direct_process_command(port)
        prepared_executable = env.get("GPUSTACK_PREPARED_EXECUTABLE")
        if not prepared_executable:
            raise RuntimeError(
                "Prepared direct-process executable provenance did not provide "
                "GPUSTACK_PREPARED_EXECUTABLE for custom backend launch"
            )
        self._validate_prepared_executable_matches_contract(
            command_args,
            prepared_executable,
            str(launch_artifacts.prepared_provenance.get("name") or ""),
        )
        self.preflight_direct_process(
            command_args=command_args, env=env, port=port
        )
        wrapped_command_args = self.build_direct_process_launch_wrapper_args(
            launch_artifacts.prepared_launch_path,
            command_args,
            use_prepared_executable=True,
        )

        workdir = contract.workdir
        bootstrap_recipe_id = type(self).get_direct_process_bootstrap_recipe_id()
        prepared_environment_id = self.get_direct_process_prepared_environment_identity()
        logger.info(
            f"Starting custom backend direct process: {deployment_metadata.name}"
        )
        logger.info(
            f"With arguments: [{' '.join(wrapped_command_args)}], "
            f"recipe={bootstrap_recipe_id}, "
            f"prepared_environment={prepared_environment_id}, "
            f"workdir: {workdir or '<inherit>'}, "
            f"envs(inconsistent input items mean unchangeable):{os.linesep}"
            f"{os.linesep.join(f'{k}={v}' for k, v in sorted(sanitize_env(env).items()))}"
        )

        process = subprocess.Popen(
            wrapped_command_args,
            env=env,
            cwd=workdir,
            stdin=subprocess.DEVNULL,
            start_new_session=os.name != "nt",
        )

        logger.info(
            f"Started custom backend direct process: {deployment_metadata.name}, "
            f"pid: {process.pid}"
        )
        return {
            "pid": process.pid,
            "process_group_id": get_process_group_id(process.pid),
            "port": port,
            "mode": DIRECT_PROCESS_RUNTIME_MODE,
            "startup_timeout_seconds": contract.startup_timeout_seconds,
            "stop_signal": contract.stop_signal,
            "stop_timeout_seconds": contract.stop_timeout_seconds,
        }

    # ------------------------------------------------------------------
    # Direct-process preflight helpers
    # ------------------------------------------------------------------

    def _check_direct_process_command(
        self,
        executable: str,
        env: Dict[str, str],
        failures: List[str],
    ) -> None:
        if not executable:
            failures.append("prepared executable path is missing")
            return

        executable_path = Path(executable)
        if executable_path.is_absolute() or executable_path.parent != Path("."):
            if executable_path.exists():
                return
        elif shutil.which(
            executable, path=env.get("PATH") or os.environ.get("PATH")
        ):
            return

        failures.append(
            f"`{executable}` is not available on PATH or does not exist"
        )

    def _check_direct_process_directories(self, failures: List[str]) -> None:
        data_dir = Path(str(self._config.data_dir))
        log_dir = Path(str(self._config.log_dir))
        required_dirs = [
            ("data directory", data_dir),
            ("cache directory", Path(str(self._config.cache_dir))),
            ("log directory", log_dir),
            ("serve log directory", log_dir / "serve"),
            ("direct-process registry directory", data_dir / "worker"),
        ]

        for label, directory in required_dirs:
            if not directory.exists():
                failures.append(f"{label} '{directory}' does not exist")
                continue
            if not directory.is_dir():
                failures.append(f"{label} '{directory}' is not a directory")
                continue
            if not os.access(directory, os.W_OK):
                failures.append(f"{label} '{directory}' is not writable")

    def _check_direct_process_port(self, port: int, failures: List[str]) -> None:
        host = self._worker.ip
        if not host:
            failures.append(
                "worker IP is missing; direct-process readiness requires "
                "a bindable host address"
            )
            return

        if port <= 0 or port > 65535:
            failures.append(f"serving port '{port}' is invalid")
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
        except OSError as exc:
            failures.append(
                f"cannot bind direct-process listener to {host}:{port}: {exc}"
            )
        finally:
            sock.close()

    # ------------------------------------------------------------------

    def start(self):
        try:
            self._start()
        except Exception as e:
            self._handle_error(e)

    def _start(self):
        logger.info(
            f"Starting custom backend model instance: {self._model_instance.name}"
        )

        if getattr(self._config, "direct_process_mode", False):
            return self.start_direct_process()

        deployment_metadata = self._get_deployment_metadata()

        env = self._get_configured_env()

        command = None
        if self.inference_backend:
            command = self.inference_backend.get_container_entrypoint(
                self._model.backend_version
            )

        command_args = self._build_command_args()

        self._create_workload(
            deployment_metadata=deployment_metadata,
            command=command,
            command_args=command_args,
            env=env,
        )

    def _create_workload(
        self,
        deployment_metadata: ModelInstanceDeploymentMetadata,
        command: Optional[List[str]],
        command_args: List[str],
        env: Dict[str, str],
    ):
        image = self._get_configured_image()
        if not image:
            raise ValueError("Failed to get Custom backend image")

        resources = self._get_configured_resources()

        mounts = self._get_configured_mounts()

        ports = self._get_configured_ports()

        # Read container config from environment variables
        container_config = self._get_container_env_config(env)

        run_container = Container(
            image=image,
            name="default",
            profile=ContainerProfileEnum.RUN,
            restart_policy=ContainerRestartPolicyEnum.NEVER,
            execution=ContainerExecution(
                privileged=True,
                command=command,
                args=command_args,
                run_as_user=container_config.user,
                run_as_group=container_config.group,
            ),
            envs=[
                ContainerEnv(
                    name=name,
                    value=value,
                )
                for name, value in env.items()
            ],
            mounts=mounts,
            resources=resources,
            ports=ports,
        )

        logger.info(
            f"Creating custom backend container workload: {deployment_metadata.name}"
        )
        logger.info(
            f"With image: {image}, "
            f"command: [{' '.join(command) if command else ''}], "
            f"arguments: [{' '.join(command_args)}], "
            f"ports: [{','.join([str(port.internal) for port in ports])}], "
            f"envs(inconsistent input items mean unchangeable):{os.linesep}"
            f"{os.linesep.join(f'{k}={v}' for k, v in sorted(sanitize_env(env).items()))}"
        )

        workload_plan = WorkloadPlan(
            name=deployment_metadata.name,
            host_network=True,
            shm_size=int(container_config.shm_size_gib * (1 << 30)),
            containers=[run_container],
            run_as_user=container_config.user,
            run_as_group=container_config.group,
        )
        create_workload(self._transform_workload_plan(workload_plan))

        logger.info(
            f"Created custom backend container workload: {deployment_metadata.name}"
        )

    def _build_command_args(self) -> List[str]:
        command_args = []

        command_args_inline = self.inference_backend.replace_command_param(
            version=self._model.backend_version,
            model_path=self._model_path,
            port=self._get_serving_port(),
            worker_ip=self._worker.ip,
            model_name=self._model.name,
            command=self._model.run_command,
            env=self._model.env,
        )
        if command_args_inline:
            command_args = shlex.split(command_args_inline)

        # Add user-defined backend parameters
        command_args.extend(self._flatten_backend_param())

        return command_args
