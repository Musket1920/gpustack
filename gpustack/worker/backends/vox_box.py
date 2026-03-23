import logging
import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from gpustack.schemas.models import ModelInstanceDeploymentMetadata
from gpustack.utils.command import extend_args_no_exist
from gpustack.utils.envs import sanitize_env
from gpustack.worker.process_registry import (
    DIRECT_PROCESS_RUNTIME_MODE,
    get_process_group_id,
)
from gpustack.worker.backends.base import InferenceServer

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


class VoxBoxServer(InferenceServer):

    # ------------------------------------------------------------------
    # Shared direct-process contract implementation
    # ------------------------------------------------------------------

    @classmethod
    def supports_direct_process(cls) -> bool:
        return True

    @classmethod
    def supports_distributed_direct_process(cls) -> bool:
        return False

    def build_direct_process_command(self, port: int) -> List[str]:
        return self._build_command_args(port=port)

    def build_direct_process_env(self) -> Dict[str, str]:
        return self._get_configured_env()

    def get_direct_process_health_path(self) -> str:
        return "/health"

    def preflight_direct_process(
        self,
        command_args: List[str],
        env: Dict[str, str],
        port: int,
    ) -> None:
        return self._preflight_direct_process(
            command_args=command_args, env=env, port=port
        )

    def start_direct_process(self):
        return self._start_direct_process(self._get_deployment_metadata())

    # ------------------------------------------------------------------
    # Direct-process internal helpers
    # ------------------------------------------------------------------

    def _start_direct_process(
        self,
        deployment_metadata: ModelInstanceDeploymentMetadata,
    ) -> Dict[str, int | str | None]:
        if (
            deployment_metadata.distributed
            or deployment_metadata.distributed_leader
            or deployment_metadata.distributed_follower
        ):
            raise ValueError(
                "Direct-process VoxBox does not support distributed launches."
            )

        port = self._get_serving_port()
        env = self._get_configured_env()
        command_args = self._build_command_args(port=port)
        self._preflight_direct_process(command_args=command_args, env=env, port=port)

        logger.info(f"Starting VoxBox direct process: {deployment_metadata.name}")
        logger.info(
            f"With arguments: [{' '.join(command_args)}], "
            f"envs(inconsistent input items mean unchangeable):{os.linesep}"
            f"{os.linesep.join(f'{k}={v}' for k, v in sorted(sanitize_env(env).items()))}"
        )

        process = subprocess.Popen(
            command_args,
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=os.name != "nt",
        )

        logger.info(
            f"Started VoxBox direct process: {deployment_metadata.name}, pid: {process.pid}"
        )
        return {
            "pid": process.pid,
            "process_group_id": get_process_group_id(process.pid),
            "port": port,
            "mode": DIRECT_PROCESS_RUNTIME_MODE,
        }

    def _preflight_direct_process(
        self,
        command_args: List[str],
        env: Dict[str, str],
        port: int,
    ) -> None:
        failures: List[str] = []
        executable = command_args[0] if command_args else "vox-box"

        self._check_direct_process_command(executable=executable, env=env, failures=failures)
        self._check_direct_process_directories(failures)
        self._check_direct_process_port(port=port, failures=failures)

        if failures:
            message = "; ".join(failures)
            logger.error(
                "Direct-process VoxBox preflight failed for %s: %s",
                self._model_instance.name,
                message,
            )
            raise RuntimeError(
                f"Direct-process VoxBox host prerequisites not met: {message}"
            )

    def _check_direct_process_command(
        self,
        executable: str,
        env: Dict[str, str],
        failures: List[str],
    ) -> None:
        executable_path = Path(executable)
        if executable_path.is_absolute() or executable_path.parent != Path("."):
            if executable_path.exists():
                return
        elif shutil.which(executable, path=env.get("PATH") or os.environ.get("PATH")):
            return

        failures.append(
            f"`{executable}` is not available on PATH or does not exist"
        )

    def _check_direct_process_directories(self, failures: List[str]) -> None:
        data_dir = Path(str(self._config.data_dir))
        log_dir = Path(str(self._config.log_dir))
        required_dirs: List[Tuple[str, Path]] = [
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
            failures.append("worker IP is missing; direct-process readiness requires a bindable host address")
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
        logger.info(f"Starting VoxBox model instance: {self._model_instance.name}")

        deployment_metadata = self._get_deployment_metadata()

        if getattr(self._config, "direct_process_mode", False):
            return self._start_direct_process(deployment_metadata)

        env = self._get_configured_env()

        command = None
        if self.inference_backend:
            command = self.inference_backend.get_container_entrypoint(
                self._model.backend_version
            )

        command_script = self._get_serving_command_script(env)

        command_args = self._build_command_args(
            port=self._get_serving_port(),
        )

        self._create_workload(
            deployment_metadata=deployment_metadata,
            command=command,
            command_script=command_script,
            command_args=command_args,
            env=env,
        )

    def _create_workload(
        self,
        deployment_metadata: ModelInstanceDeploymentMetadata,
        command: Optional[List[str]],
        command_script: Optional[str],
        command_args: List[str],
        env: Dict[str, str],
    ):
        image = self._get_configured_image()
        if not image:
            raise ValueError("Failed to get VoxBox backend image")

        # Command script will override the given command,
        # so we need to prepend command to command args.
        if command_script and command:
            command_args = command + command_args
            command = None

        resources = self._get_configured_resources(
            # Pass-through all devices as vox-box handles device itself.
            mount_all_devices=True,
        )

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
                command_script=command_script,
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
            resources=resources,
            mounts=mounts,
            ports=ports,
        )

        logger.info(f"Creating VoxBox container workload: {deployment_metadata.name}")
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

        logger.info(f"Created VoxBox container workload: {deployment_metadata.name}")

    def _build_command_args(self, port: int) -> List[str]:
        arguments = [
            "vox-box",
            "start",
            "--model",
            self._model_path,
            "--data-dir",
            self._config.data_dir,
        ]
        # Allow version-specific command override if configured (before appending extra args)
        arguments = self.build_versioned_command_args(
            arguments,
            model_path=self._model_path,
            port=port,
        )
        arguments.extend(self._flatten_backend_param())
        # Append immutable arguments to ensure proper operation for accessing
        # Only add if not already present in arguments
        extend_args_no_exist(
            arguments, ("--host", self._worker.ip), ("--port", str(port))
        )
        if self._model_instance.gpu_indexes is not None:
            extend_args_no_exist(
                arguments, ("--device", f"cuda:{self._model_instance.gpu_indexes[0]}")
            )

        return arguments
