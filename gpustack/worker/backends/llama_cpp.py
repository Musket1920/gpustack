import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
from typing import Dict, List, Tuple

from gpustack.schemas.models import ModelInstanceDeploymentMetadata
from gpustack.utils.command import extend_args_no_exist
from gpustack.utils.envs import sanitize_env
from gpustack.worker.backends.custom import CustomServer
from gpustack.worker.process_registry import (
    DIRECT_PROCESS_RUNTIME_MODE,
    get_process_group_id,
)

logger = logging.getLogger(__name__)


class LlamaCppServer(CustomServer):
    @classmethod
    def supports_direct_process(cls) -> bool:
        return True

    @classmethod
    def supports_distributed_direct_process(cls) -> bool:
        return False

    def build_direct_process_command(self, port: int) -> List[str]:
        arguments = [
            "llama-server",
            "-m",
            self._model_path,
            "--host",
            self._worker.ip,
            "--port",
            str(port),
            "--alias",
            self._model_instance.model_name,
        ]

        arguments.extend(self._flatten_backend_param())
        extend_args_no_exist(arguments, ("--host", self._worker.ip))
        extend_args_no_exist(arguments, ("--port", str(port)))
        extend_args_no_exist(arguments, ("--alias", self._model_instance.model_name))
        return arguments

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
        failures: List[str] = []
        executable = command_args[0] if command_args else "llama-server"

        self._check_direct_process_command(
            executable=executable, env=env, failures=failures
        )
        self._check_direct_process_directories(failures)
        self._check_direct_process_port(port=port, failures=failures)

        if failures:
            message = "; ".join(failures)
            logger.error(
                "Direct-process llama.cpp preflight failed for %s: %s",
                self._model_instance.name,
                message,
            )
            raise RuntimeError(
                f"Direct-process llama.cpp host prerequisites not met: {message}"
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
            failures.append(
                "worker IP is missing; direct-process readiness requires a bindable host address"
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

    def start_direct_process(self):
        return self._start_direct_process(self._get_deployment_metadata())

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
                "Direct-process llama.cpp does not support distributed launches."
            )

        port = self._get_serving_port()
        env = self.build_direct_process_env()
        command_args = self.build_direct_process_command(port=port)
        self.preflight_direct_process(command_args=command_args, env=env, port=port)

        logger.info(f"Starting llama.cpp direct process: {deployment_metadata.name}")
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
            "Started llama.cpp direct process: %s, pid: %s",
            deployment_metadata.name,
            process.pid,
        )
        return {
            "pid": process.pid,
            "process_group_id": get_process_group_id(process.pid),
            "port": port,
            "mode": DIRECT_PROCESS_RUNTIME_MODE,
        }
