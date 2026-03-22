import asyncio
import contextlib
from datetime import datetime, timezone
import multiprocessing
import signal
import threading

import requests
import setproctitle
import os
from typing import Dict, Optional, Set, List, Callable
import logging

from gpustack_runtime.deployer import (
    get_workload,
    WorkloadStatusStateEnum,
    delete_workload,
)
from gpustack_runtime.deployer.__utils__ import compare_versions

from gpustack.api.exceptions import NotFoundException
from gpustack.config.config import Config
from gpustack.config import registration
from gpustack.logging import (
    RedirectStdoutStderr,
)
from gpustack.schemas.inference_backend import InferenceBackend, is_built_in_backend
from gpustack.utils import network, platform
from gpustack.utils.attrs import set_attr
from gpustack.utils.command import find_int_parameter
from gpustack.utils.process import terminate_process_tree, add_signal_handlers
from gpustack.worker.backends.ascend_mindie import AscendMindIEServer
from gpustack.worker.backends.sglang import SGLangServer
from gpustack.worker.backends.vllm import VLLMServer
from gpustack.worker.backends.vox_box import VoxBoxServer
from gpustack.worker.backends.custom import CustomServer
from gpustack.worker.model_meta import get_meta_from_running_instance
from gpustack.worker.direct_process import (
    ensure_model_instance_direct_process_support,
)
from gpustack.worker.process_registry import (
    DIRECT_PROCESS_RUNTIME_MODE,
    DirectProcessEntryStatus,
    DirectProcessRegistry,
    DirectProcessRegistryEntry,
    DirectProcessStateTransition,
    map_direct_process_state_transition,
)
from gpustack.client import ClientSet
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelUpdate,
    ModelInstance,
    ModelInstanceUpdate,
    ModelInstanceStateEnum,
    get_backend,
    DistributedServerCoordinateModeEnum,
    ModelInstanceSubordinateWorker,
    CategoryEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker.inference_backend_manager import InferenceBackendManager

logger = logging.getLogger(__name__)

# Global lock for port assignment to avoid pickle serialization issues
_port_lock = threading.Lock()

_SERVER_CLASS_MAPPING = {
    BackendEnum.VLLM: VLLMServer,
    BackendEnum.SGLANG: SGLangServer,
    BackendEnum.VOX_BOX: VoxBoxServer,
    BackendEnum.ASCEND_MINDIE: AscendMindIEServer,
}


class ServeManager:
    @property
    def _worker_id(self) -> int:
        return self._worker_id_getter()

    """
    The ID of current worker.
    """
    _config: Config
    """
    Global configuration.
    """
    _serve_log_dir: str
    """
    The directory to store logs of serving model instances(in subprocess).
    """

    @property
    def _clientset(self) -> ClientSet:
        return self._clientset_getter()

    """
    The clientset to access the API server.
    """
    _inference_backend_manager: InferenceBackendManager
    """
    The inference backend manager.
    """
    _provisioning_processes: Dict[int, multiprocessing.Process] = {}
    """
    The mapping of model instance ID to provisioning (sub)process.
    When the (sub)process is alive, the model instance is provisioning.
    If the (sub)process exited, the model instance is either running or failed.
    """
    _error_model_instances: Dict[int, ModelInstance] = {}
    """
    The mapping of model instance ID to error model instances.
    Used to restart error model instances.
    """
    _model_cache_by_instance: Dict[int, Model] = {}
    """
    The cache of models by model instance ID.
    Used to avoid redundant API calls to get model information.
    """
    _model_instance_by_instance_id: Dict[int, ModelInstance] = {}
    _direct_process_registry: Optional[DirectProcessRegistry] = None

    _clientset_getter: Callable[[], ClientSet]
    _worker_id_getter: Callable[[], int]

    def __init__(
        self,
        worker_id_getter: Callable[[], int],
        clientset_getter: Callable[[], ClientSet],
        cfg: Config,
    ):
        self._worker_id_getter = worker_id_getter
        self._config = cfg
        self._serve_log_dir = f"{cfg.log_dir}/serve"
        self._clientset_getter = clientset_getter

        # Instance-level port tracking to avoid conflicts
        self._assigned_ports: Dict[int, Set[int]] = {}
        self._direct_process_registry = DirectProcessRegistry(cfg)

        os.makedirs(self._serve_log_dir, exist_ok=True)

    async def watch_models(self):
        """
        Loop to watch models to keep the cache updated.

        """

        logger.debug("Watching models.")

        while True:
            try:
                # Watch models without callback to keep the cache updated.
                await self._clientset.models.awatch(callback=None)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching models: {e}")
                await asyncio.sleep(5)

    async def watch_model_instances_event(self):
        """
        Loop to watch model instances' event and handle.

        """

        logger.debug("Watching model instances event.")

        while True:
            try:
                await self._clientset.model_instances.awatch(
                    callback=self._handle_model_instance_event
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching model instances: {e}")
                await asyncio.sleep(5)

    async def watch_model_instances(self):
        """
        Loop to post process model instances, for example, restarting error instances.

        """

        logger.debug("Watching model instances.")

        while True:
            try:
                for mi in list(self._error_model_instances.values()):
                    self._restart_error_model_instance(mi)
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Error restarting model instances: {e}")
                await asyncio.sleep(5)

    def sync_model_instances_state(self):  # noqa: C901
        """
        Synchronize model instances' state.

        - If the model instance is scheduled but not initialized, skip.
        - If the provision process is still alive, skip.
        - If the workload is still launching, skip.
        - If the workload is not existed, unhealthy, inactive or failed, update the model instance state to ERROR.
        - If everything is fine, update the model instance state to RUNNING.
        """

        # Get all model instances assigned to this worker.
        #
        # FIXME(thxCode): This may cause performance issues when there are many model instances in the system.
        #                 A mechanism is needed to improve efficiency here.
        model_instances_page = self._clientset.model_instances.list()
        if not model_instances_page.items:
            return
        model_instances = []
        for model_instance in model_instances_page.items:
            # if the model instance is assigned to this worker, it must be scheduled.
            # But we don't need to sync the scheduled model when it is not initialized yet.
            if (
                model_instance.worker_id == self._worker_id
                and model_instance.state != ModelInstanceStateEnum.SCHEDULED
            ):
                model_instances.append(model_instance)
            if (
                model_instance.distributed_servers
                and model_instance.distributed_servers.subordinate_workers
            ):
                for sw in model_instance.distributed_servers.subordinate_workers:
                    if sw.worker_id == self._worker_id:
                        model_instances.append(model_instance)
                        break

        for model_instance in model_instances:
            # Skip if the provision process has not exited yet.
            if self._is_provisioning(model_instance):
                logger.trace(
                    f"Model instance {model_instance.name} is provisioning. Skipping sync."
                )
                continue

            if self._is_direct_process_mode():
                self._sync_direct_process_model_instance(model_instance)
                continue

            # Skip if the workload is still launching.
            workload = get_workload(model_instance.name)
            if workload and workload.state in [
                WorkloadStatusStateEnum.PENDING,
                WorkloadStatusStateEnum.INITIALIZING,
            ]:
                logger.trace(
                    f"Model instance {model_instance.name} workload is still launching. Skipping sync."
                )
                continue

            is_main_worker = model_instance.worker_id == self._worker_id

            # Update model instance state to ERROR if the workload is not existed, unhealthy, inactive or failed.
            if not workload or workload.state in [
                WorkloadStatusStateEnum.UNKNOWN,  # Rare, but possible, for example, leaving pause container.
                WorkloadStatusStateEnum.UNHEALTHY,
                WorkloadStatusStateEnum.INACTIVE,
                WorkloadStatusStateEnum.FAILED,
            ]:
                # NB(thxCode): Since the `sync_model_instances_state` and `watch_model_instances_event` are in different loops,
                # subordinate workers haven't had time to create the workload yet even though the model instance's state is expected.
                # So we skip if the subordinate worker didn't have workload yet.
                #
                # FIXME(thxCode): Another problem caused by skipping this check is that if we actively delete the workload on the subordinate worker,
                #                 we may not be able to correct the state of the subordinate worker.
                if not is_main_worker and not workload:
                    continue
                # Only if not in ERROR state yet.
                if model_instance.state != ModelInstanceStateEnum.ERROR:
                    with contextlib.suppress(NotFoundException):
                        # Get patch dict for main worker.
                        if is_main_worker:
                            patch_dict = {
                                "state": ModelInstanceStateEnum.ERROR,
                                "state_message": "Inference server exited or unhealthy.",
                            }
                        # Get patch dict for subordinate worker.
                        else:
                            sw_pos = next(
                                (
                                    i
                                    for i, sw in enumerate(
                                        model_instance.distributed_servers.subordinate_workers
                                    )
                                    if sw.worker_id == self._worker_id
                                ),
                            )
                            sw = model_instance.distributed_servers.subordinate_workers[
                                sw_pos
                            ]
                            sw.state = ModelInstanceStateEnum.ERROR
                            sw.state_message = "Inference server exited or unhealthy."
                            patch_dict = {
                                f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                            }
                        # Update model instance.
                        self._update_model_instance(model_instance.id, **patch_dict)
                continue

            # Otherwise, update model instance state to RUNNING if everything is fine.
            model = self._get_model(model_instance)
            if not model.backend_version:
                # backend version may be empty on initialization.
                # try to refresh to get updated model info on syncs.
                model = self._refresh_model(model_instance)

            backend = get_backend(model)
            health_check_path = self._get_health_check_path(backend)
            if model.env and 'GPUSTACK_MODEL_HEALTH_CHECK_PATH' in model.env:
                # NOTE: There is no known use case for now. Keep this in case the built-in backends
                # introduce breaking changes and the default health check path no longer works.
                health_check_path = model.env['GPUSTACK_MODEL_HEALTH_CHECK_PATH']

            with contextlib.suppress(NotFoundException):
                # Get patch dict for main worker.
                if is_main_worker:
                    sw_error_msg = None
                    if (
                        model_instance.distributed_servers
                        and model_instance.distributed_servers.subordinate_workers
                    ):
                        for (
                            sw
                        ) in model_instance.distributed_servers.subordinate_workers:
                            if sw.state == ModelInstanceStateEnum.ERROR:
                                sw_error_msg = f"Distributed serving error in subordinate worker {sw.worker_ip}: {sw.state_message}."
                                break
                    # If there is no error message from subordinate workers,
                    # check whether the main worker is healthy.
                    if not sw_error_msg:
                        if model_instance.state == ModelInstanceStateEnum.RUNNING:
                            continue
                        if not is_ready(
                            backend, model_instance, health_check_path, model
                        ):
                            continue

                        patch_dict = {
                            "state": ModelInstanceStateEnum.RUNNING,
                            "restart_count": 0,  # Reset restart count on successful run.
                            "state_message": "",
                        }

                        # Fetch model meta once running.
                        meta = get_meta_from_running_instance(
                            model_instance, backend, model
                        )
                        if meta:
                            # Some meta is set in server evaluation and should be preserved, so we update meta instead of overwrite.
                            merged_meta = dict(model.meta or {})
                            merged_meta.update(meta)
                            if merged_meta != model.meta:
                                self._update_model(model.id, meta=merged_meta)
                    # Otherwise, update the main worker state to ERROR.
                    else:
                        patch_dict = {
                            "state": ModelInstanceStateEnum.ERROR,
                            "state_message": sw_error_msg,
                        }
                # Get patch dict for subordinate worker.
                else:
                    # For initialize later mode, the state is set to RUNNING directly,
                    # which means the subordinate worker doesn't need to wait for the main worker to be healthy.
                    if (
                        model_instance.distributed_servers.mode
                        == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                    ):
                        continue
                    # Otherwise, update subordinate worker state to RUNNING.
                    sw_pos = next(
                        (
                            i
                            for i, sw in enumerate(
                                model_instance.distributed_servers.subordinate_workers
                            )
                            if sw.worker_id == self._worker_id
                        ),
                    )
                    sw = model_instance.distributed_servers.subordinate_workers[sw_pos]
                    if sw.state == ModelInstanceStateEnum.RUNNING:
                        continue
                    sw.state = ModelInstanceStateEnum.RUNNING
                    sw.state_message = ""
                    patch_dict = {
                        f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                    }
                # Update model instance.
                self._update_model_instance(model_instance.id, **patch_dict)

    @staticmethod
    def _serve_model_instance(
        mi: ModelInstance,
        backend: BackendEnum,
        client_headers: dict,
        log_file_path: str,
        cfg: Config,
        worker_id: int,
        inference_backend: InferenceBackend,
        fallback_registry: Optional[str] = None,
        direct_process_registry_path: Optional[str] = None,
    ):
        """
        Serve model instance in a subprocess.
        Exits the subprocess when serving ends.

        Args:
            mi: The model instance to serve.
            backend: The backend of the model instance.
            client_headers: The headers for the clientset.
            log_file_path: The path to the log file.
            cfg: The configuration.
            worker_id: The ID of the worker.
            inference_backend: The inference backend configuration.
            fallback_registry: The fallback container registry to use if needed.
        """

        setproctitle.setproctitle(f"gpustack_model_instance_{mi.id}")
        add_signal_handlers()

        clientset = ClientSet(
            base_url=cfg.get_server_url(),
            headers=client_headers,
        )

        with open(log_file_path, "w", buffering=1, encoding="utf-8") as log_file:
            with RedirectStdoutStderr(log_file):
                try:
                    server_cls = _SERVER_CLASS_MAPPING.get(backend, CustomServer)
                    server_ins = server_cls(
                        clientset,
                        mi,
                        cfg,
                        worker_id,
                        inference_backend,
                        fallback_registry,
                    )
                    logger.info(f"Provisioning model instance {mi.name}")
                    start_result = server_ins.start()
                    if direct_process_registry_path:
                        ServeManager._record_direct_process_runtime(
                            registry_path=direct_process_registry_path,
                            mi=mi,
                            worker_id=worker_id,
                            backend=backend,
                            log_file_path=log_file_path,
                            start_result=start_result,
                        )
                    logger.info(f"Finished provisioning model instance {mi.name}")
                except Exception as e:
                    logger.exception(
                        f"Error provisioning model instance {mi.name}: {e}"
                    )
                    raise e

    def _handle_model_instance_event(self, event: Event):  # noqa: C901
        """
        Handle model instance events.

        Args:
            event: The model instance event to handle.

        """
        mi = ModelInstance.model_validate(event.data)

        logger.trace(
            f"Received event: {str(event.type)}, id: {mi.id}, name: {mi.name}, state: {str(mi.state)}"
        )

        is_main_worker = mi.worker_id == self._worker_id

        if is_main_worker:
            self._model_instance_by_instance_id[mi.id] = mi
            # Return if all subordinate workers aren't running.
            if (
                mi.distributed_servers
                and mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.RUN_FIRST
                and mi.distributed_servers.subordinate_workers
            ):
                ready = all(
                    sw.state == ModelInstanceStateEnum.RUNNING
                    for sw in mi.distributed_servers.subordinate_workers
                )
                if not ready:
                    logger.info(
                        f"Model instance {mi.name} waits for all subordinate workers to be ready."
                    )
                    return
        else:
            # Return if it isn't a distribution serving.
            if not mi.distributed_servers:
                return
            # Return if it's a delegated distribution,
            # which means the main worker is responsible for serving.
            if (
                mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.DELEGATED
            ):
                return
            # Return if it isn't the member of the distribution serving.
            joined = any(
                sw.worker_id == self._worker_id
                for sw in mi.distributed_servers.subordinate_workers or []
            )
            if not joined:
                return
            # Return if the main worker isn't initialized.
            if (
                mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                and (
                    mi.state
                    not in [
                        ModelInstanceStateEnum.STARTING,
                        ModelInstanceStateEnum.RUNNING,
                        ModelInstanceStateEnum.ERROR,
                    ]
                )
            ):
                logger.info(
                    f"Model instance {mi.name} waits for main worker {mi.worker_ip} to be initialized."
                )
                return
            # FIXME: This is a temporary solution to prevent the main worker from being unable to start due to phantom reads.
            #        We confirm whether the operation should be performed by checking the state of the earlier subordinate worker.
            for sw in mi.distributed_servers.subordinate_workers:
                if sw.worker_id == self._worker_id:
                    break
                if sw.state not in [
                    ModelInstanceStateEnum.RUNNING,
                    ModelInstanceStateEnum.ERROR,
                ]:
                    logger.info(
                        f"Model instance {mi.name} waits for previous subordinate worker {sw.worker_ip} to be ready."
                    )
                    return

        if event.type == EventType.DELETED:
            self._stop_model_instance(mi)
            logger.trace(f"DELETED event: stopped deleted model instance {mi.name}.")
            return

        if event.type == EventType.UPDATED:
            # Caching matched ERROR instances for restart handling.
            if mi.state == ModelInstanceStateEnum.ERROR:
                model = self._get_model(mi)
                if model.restart_on_error:
                    self._error_model_instances[mi.id] = mi
                    logger.trace(
                        f"UPDATED event: cached error model instance {mi.name} for restart."
                    )
                return

            # Restart if scheduled.
            if mi.state == ModelInstanceStateEnum.SCHEDULED:
                self._restart_model_instance(mi)
                logger.trace(
                    f"UPDATED event: restarted scheduled model instance {mi.name}."
                )

            # Start on subordinate worker if not started yet.
            if not is_main_worker:
                if self._should_start_model_instance(mi):
                    self._start_model_instance(mi)
                    logger.trace(
                        f"UPDATED event: started model instance {mi.name} on subordinate worker."
                    )

            return

        if event.type == EventType.CREATED:
            if mi.state == ModelInstanceStateEnum.RUNNING:
                logger.warning(
                    f"Model instance {mi.name} is already running. Skipping start."
                )
                return
            self._start_model_instance(mi)
            logger.trace(f"CREATED event: started created model instance {mi.name}.")

    def _start_model_instance(self, mi: ModelInstance):  # noqa: C901
        """
        Start model instance through a subprocess.

        Args:
            mi: The model instance to start.

        """
        if self._is_provisioning(mi):
            logger.warning(f"Model instance {mi.name} is provisioning. Skipping start.")
            return

        model = self._get_model(mi)
        backend = get_backend(model)

        is_main_worker = mi.worker_id == self._worker_id

        log_file_path = f"{self._serve_log_dir}/{mi.id}.log"
        if os.path.exists(log_file_path) and platform.system() != "windows":
            # TODO Windows does not support os.remove() on open files.
            # Investigate file occupation issue.
            os.remove(log_file_path)

        sw_pos: Optional[int] = None
        sw: Optional[ModelInstanceSubordinateWorker] = None
        if not is_main_worker:
            sw_pos = next(
                (
                    i
                    for i, sw in enumerate(mi.distributed_servers.subordinate_workers)
                    if sw.worker_id == self._worker_id
                ),
            )
            sw = mi.distributed_servers.subordinate_workers[sw_pos]

        try:
            ensure_model_instance_direct_process_support(
                self._config,
                mi,
                backend,
                self._worker_id,
            )

            if self._is_direct_process_mode():
                self._get_direct_process_registry().remove_by_model_instance_id(mi.id)

            self._assign_ports(mi, model, backend)

            logger.debug(
                f"Starting model instance {mi.name}"
                f"{'' if not is_main_worker else f' on ports {mi.ports if mi.ports else [mi.port]}'}"
            )

            fallback_registry = (
                registration.determine_default_registry(
                    self._config.system_default_container_registry
                )
                if is_built_in_backend(backend)
                else None
            )

            process = multiprocessing.Process(
                target=ServeManager._serve_model_instance,
                args=(
                    mi,
                    backend,
                    self._clientset.headers,
                    log_file_path,
                    self._config,
                    self._worker_id,
                    self._inference_backend_manager.get_backend_by_name(backend),
                    fallback_registry,
                    (
                        str(self._get_direct_process_registry().path)
                        if self._is_direct_process_mode()
                        else None
                    ),
                ),
            )
            process.daemon = False
            process.start()
            self._provisioning_processes[mi.id] = process

            # Get patch dict for main worker.
            if is_main_worker:
                patch_dict = {
                    "state": ModelInstanceStateEnum.INITIALIZING,
                    "port": mi.port,
                    "ports": mi.ports,
                }
                if not self._is_direct_process_mode():
                    patch_dict["pid"] = process.pid
                else:
                    patch_dict["pid"] = None
            # Get patch dict for subordinate worker.
            else:
                sw.state = ModelInstanceStateEnum.INITIALIZING
                # For initialize later mode, the state is set to RUNNING directly,
                # which means the subordinate worker doesn't need to wait for the main worker to be healthy.
                if (
                    mi.distributed_servers.mode
                    == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                ):
                    sw.state = ModelInstanceStateEnum.RUNNING
                sw.pid = None if self._is_direct_process_mode() else process.pid
                patch_dict = {
                    f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                }

            self._update_model_instance(mi.id, **patch_dict)
            logger.info(
                f"Started model instance {mi.name}"
                f"{'' if not is_main_worker else f' on ports {mi.ports if mi.ports else [mi.port]}'}"
            )

        except Exception as e:
            # Clean up provisioning process if started.
            if mi.id in self._provisioning_processes:
                self._stop_model_instance(mi)

            # Get patch dict for main worker.
            if is_main_worker:
                patch_dict = {
                    "state": ModelInstanceStateEnum.ERROR,
                    "state_message": f"Failed to start model instance: {e}",
                }
            # Get patch dict for subordinate worker.
            else:
                sw.state = ModelInstanceStateEnum.ERROR
                sw.state_message = f"Failed to start model instance: {e}"
                patch_dict = {
                    f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                }

            self._update_model_instance(mi.id, **patch_dict)
            logger.error(f"Failed to start model instance {mi.name}: {e}")

    def _assign_ports(
        self,
        mi: ModelInstance,
        model: Model,
        backend: BackendEnum,
    ) -> None:
        """
        Assign ports to the model instance.

        This method is thread-safe and allocates ports for:
        - Main serving port
        - RPC port for vLLM DP communication (if applicable)
        - Connecting port for subordinate workers (if applicable)

        Args:
            mi: The model instance to assign ports to.
            model: The model associated with the instance.
            backend: The backend type (e.g., vLLM, SGLang).
        """
        if mi.port:
            # Port already assigned, skip.
            return

        with _port_lock:
            if mi.port:
                # Port already assigned, skip.
                return

            if self._assigned_ports:
                unavailable_ports = set.union(*self._assigned_ports.values())
            else:
                unavailable_ports = set()

            # Main serving port
            mi.port = network.get_free_port(
                port_range=self._config.service_port_range,
                unavailable_ports=unavailable_ports,
                host=mi.worker_ip,
            )
            mi.ports = [mi.port]
            unavailable_ports.add(mi.port)

            # Additional ports for distributed servers
            if mi.distributed_servers and mi.distributed_servers.subordinate_workers:
                # RPC port for DP communication in vLLM backend
                if backend == BackendEnum.VLLM:
                    dps = find_int_parameter(
                        model.backend_parameters,
                        ["data-parallel-size", "dp"],
                    )
                    if dps and dps > 1:
                        dp_connecting_port = network.get_free_port(
                            port_range=self._config.service_port_range,
                            unavailable_ports=unavailable_ports,
                            host=mi.worker_ip,
                        )
                        mi.ports.append(dp_connecting_port)
                        unavailable_ports.add(dp_connecting_port)

                # Connecting port for subordinate workers communication
                connecting_port = network.get_free_port(
                    port_range=self._config.service_port_range,
                    unavailable_ports=unavailable_ports,
                    host=mi.worker_ip,
                )
                mi.ports.append(connecting_port)
                unavailable_ports.add(connecting_port)

            self._assigned_ports[mi.id] = set(mi.ports)

    def _restart_model_instance(self, mi: ModelInstance):
        """
        Restart model instance.

        Args:
            mi: The model instance to restart.
        """

        self._stop_model_instance(mi)
        self._start_model_instance(mi)

    def _update_model(self, id: int, **kwargs):
        """
        Update model instance with given fields.

        Args:
            id: The ID of the model instance to update.
            **kwargs: The fields to update, group by field name and value.
        """

        try:
            m_public = self._clientset.models.get(id=id)

            m = ModelUpdate(**m_public.model_dump())
            for key, value in kwargs.items():
                set_attr(m, key, value)

            self._clientset.models.update(id=id, model_update=m)
        except NotFoundException:
            logger.warning(f"Model with ID {id} not found when trying to update.")

    def _update_model_instance(self, id: int, **kwargs):
        """
        Update model instance with given fields.

        Args:
            id: The ID of the model instance to update.
            **kwargs: The fields to update, group by field name and value.
        """

        try:
            mi_public = self._clientset.model_instances.get(id=id)

            mi = ModelInstanceUpdate(**mi_public.model_dump())
            for key, value in kwargs.items():
                set_attr(mi, key, value)

            self._clientset.model_instances.update(id=id, model_update=mi)
        except NotFoundException:
            logger.warning(
                f"Model instance with ID {id} not found when trying to update."
            )

    def _stop_model_instance(self, mi: ModelInstance):
        """
        Stop model instance and clean up.

        Args:
            mi: The model instance to stop.
        """

        logger.debug(f"Stopping model instance {mi.name or mi.id}")

        # Teardown provisioning process if still alive.
        if self._is_provisioning(mi):
            terminate_process_tree(self._provisioning_processes[mi.id].pid)

        if self._is_direct_process_mode():
            direct_process_entry = self._get_direct_process_registry().remove_by_model_instance_id(
                mi.id
            )
            self._terminate_direct_process_entry(direct_process_entry)
        else:
            # Delete workload.
            deployment_metadata = mi.get_deployment_metadata(self._worker_id)
            if deployment_metadata:
                delete_workload(deployment_metadata.name)

        # Cleanup internal states.
        self._provisioning_processes.pop(mi.id, None)
        self._assigned_ports.pop(mi.id, None)
        self._error_model_instances.pop(mi.id, None)
        self._model_cache_by_instance.pop(mi.id, None)
        self._model_instance_by_instance_id.pop(mi.id, None)

        logger.info(f"Stopped model instance {mi.name or mi.id}")

    def reconcile_stale_direct_process_registry(self) -> None:
        if not self._is_direct_process_mode():
            return

        registry = self._get_direct_process_registry()
        entries = registry.list_entries()
        if not entries:
            return

        logger.info(
            "Reconciling %s direct-process registry entries with cleanup-and-recreate policy.",
            len(entries),
        )
        for entry in entries:
            status = registry.inspect_by_model_instance_id(entry.model_instance_id)
            logger.info(
                "Cleaning direct-process registry entry for %s (%s).",
                entry.deployment_name,
                status.reason or status.status.value,
            )
            removed_entry = registry.remove_by_model_instance_id(entry.model_instance_id)
            self._terminate_direct_process_entry(removed_entry or entry)

    def _restart_error_model_instance(self, mi: ModelInstance):
        """
        Restart error model instance with exponential backoff,
        maximum delay 5 minutes.

        When `sync_model_instances_state` catches once RUNNING,
        the accumulated `restart_count` will be reset.

        Args:
            mi: The model instance to restart.
        """
        if self._is_provisioning(mi):
            logger.debug(f"Model instance {mi.name} is provisioning. Skipping restart.")
            return

        restart_count = mi.restart_count or 0
        last_restart_time = mi.last_restart_time or mi.updated_at

        current_time = datetime.now(timezone.utc)
        delay = min(10 * (2 ** (restart_count - 1)), 300)
        if restart_count > 0 and last_restart_time:
            elapsed_time = (current_time - last_restart_time).total_seconds()
            if elapsed_time < delay:
                logger.trace(
                    f"Delaying restart of {mi.name} for {delay - elapsed_time:.2f} seconds."
                )
                return

        logger.info(
            f"Restarting model instance {mi.name} (attempt {restart_count + 1}) after {delay} seconds delay."
        )

        with contextlib.suppress(NotFoundException):
            self._update_model_instance(
                mi.id,
                restart_count=restart_count + 1,
                last_restart_time=current_time,
                state=ModelInstanceStateEnum.SCHEDULED,
                state_message="",
            )

        # Pop from error model instances,
        # if failed to restart next time, it will be added again in watch_model_instance_events().
        self._error_model_instances.pop(mi.id, None)

    def _get_model(self, mi: ModelInstance) -> Model:
        """
        Efficiently get model related to the model instance with caching.

        Args:
            mi: The model instance whose model to get.
        """
        if model := self._model_cache_by_instance.get(mi.id):
            return model

        model = self._clientset.models.get(mi.model_id)
        self._model_cache_by_instance[mi.id] = model
        return model

    def _refresh_model(self, mi: ModelInstance) -> Model:
        """
        Refresh the model information from the server.

        Args:
            mi: The model instance whose model to refresh.

        Returns:
            The refreshed model.
        """
        logger.debug(f"Refreshing model {mi.model_name} information from server.")
        refreshed_model = self._clientset.models.get(mi.model_id)
        self._model_cache_by_instance[mi.id] = refreshed_model
        return refreshed_model

    def _is_provisioning(self, mi: ModelInstance) -> bool:
        """
        Check if the model instance is still provisioning.

        Args:
            mi: The model instance to check.
        """
        if process := self._provisioning_processes.get(mi.id):
            if process.is_alive():
                process.join(timeout=0)
                return process.is_alive()
        return False

    def _is_direct_process_mode(self) -> bool:
        return bool(getattr(self._config, "direct_process_mode", False))

    def _get_direct_process_registry(self) -> DirectProcessRegistry:
        registry = getattr(self, "_direct_process_registry", None)
        if registry is None:
            registry = DirectProcessRegistry(self._config)
            self._direct_process_registry = registry
        return registry

    def _should_start_model_instance(self, mi: ModelInstance) -> bool:
        if not self._is_direct_process_mode():
            workload = get_workload(mi.name)
            return workload is None

        status = self._get_direct_process_registry().inspect_by_model_instance_id(mi.id)
        return status.status != DirectProcessEntryStatus.LIVE

    def _sync_direct_process_model_instance(self, model_instance: ModelInstance):
        is_main_worker = model_instance.worker_id == self._worker_id
        status = self._get_direct_process_registry().inspect_by_model_instance_id(
            model_instance.id
        )
        entry = status.entry

        model = self._get_model(model_instance)
        if not model.backend_version:
            model = self._refresh_model(model_instance)

        backend = get_backend(model)
        health_check_path = self._get_health_check_path(backend)
        if model.env and "GPUSTACK_MODEL_HEALTH_CHECK_PATH" in model.env:
            health_check_path = model.env["GPUSTACK_MODEL_HEALTH_CHECK_PATH"]

        ready = False
        if status.status == DirectProcessEntryStatus.LIVE:
            ready = is_ready(backend, model_instance, health_check_path, model)

        transition = map_direct_process_state_transition(
            model_instance.state,
            status,
            is_ready=ready,
        )

        with contextlib.suppress(NotFoundException):
            if is_main_worker:
                patch_dict = self._build_direct_process_main_worker_patch(
                    model_instance=model_instance,
                    entry=entry,
                    transition_ready=ready,
                    transition_status=transition,
                    model=model,
                    backend=backend,
                )
            else:
                patch_dict = self._build_direct_process_subordinate_patch(
                    model_instance=model_instance,
                    entry=entry,
                    transition_status=transition,
                )

            if patch_dict:
                self._update_model_instance(model_instance.id, **patch_dict)

    def _build_direct_process_main_worker_patch(
        self,
        model_instance: ModelInstance,
        entry: Optional[DirectProcessRegistryEntry],
        transition_ready: bool,
        transition_status: DirectProcessStateTransition,
        model: Model,
        backend: BackendEnum,
    ) -> Dict[str, object]:
        patch_dict: Dict[str, object] = {}

        if entry and model_instance.pid != entry.pid:
            patch_dict["pid"] = entry.pid

        if transition_status.next_state == ModelInstanceStateEnum.RUNNING:
            patch_dict.update(
                {
                    "state": ModelInstanceStateEnum.RUNNING,
                    "restart_count": 0,
                    "state_message": "",
                }
            )

            meta = get_meta_from_running_instance(model_instance, backend, model)
            if meta:
                merged_meta = dict(model.meta or {})
                merged_meta.update(meta)
                if merged_meta != model.meta:
                    self._update_model(model.id, meta=merged_meta)
        elif transition_status.next_state == ModelInstanceStateEnum.ERROR:
            patch_dict.update(
                {
                    "state": ModelInstanceStateEnum.ERROR,
                    "state_message": transition_status.state_message or "",
                }
            )
        elif transition_status.next_state == ModelInstanceStateEnum.STARTING:
            patch_dict.update(
                {
                    "state": ModelInstanceStateEnum.STARTING,
                    "state_message": transition_status.state_message or "",
                }
            )

        if not transition_ready and transition_status.next_state is None and not patch_dict:
            return {}

        return patch_dict

    def _build_direct_process_subordinate_patch(
        self,
        model_instance: ModelInstance,
        entry: Optional[DirectProcessRegistryEntry],
        transition_status: DirectProcessStateTransition,
    ) -> Dict[str, object]:
        if not model_instance.distributed_servers:
            return {}

        subordinate_workers = (
            model_instance.distributed_servers.subordinate_workers or []
        )

        sw_pos = next(
            (
                i
                for i, sw in enumerate(subordinate_workers)
                if sw.worker_id == self._worker_id
            ),
            None,
        )
        if sw_pos is None:
            return {}

        sw = subordinate_workers[sw_pos]
        if entry and sw.pid != entry.pid:
            sw.pid = entry.pid

        if transition_status.next_state == ModelInstanceStateEnum.RUNNING:
            sw.state = ModelInstanceStateEnum.RUNNING
            sw.state_message = ""
        elif transition_status.next_state == ModelInstanceStateEnum.ERROR:
            sw.state = ModelInstanceStateEnum.ERROR
            sw.state_message = transition_status.state_message or ""
        elif transition_status.next_state == ModelInstanceStateEnum.STARTING:
            sw.state = ModelInstanceStateEnum.STARTING
            sw.state_message = transition_status.state_message or ""
        else:
            return {}

        return {f"distributed_servers.subordinate_workers.{sw_pos}": sw}

    def _terminate_direct_process_entry(
        self, entry: Optional[DirectProcessRegistryEntry]
    ) -> None:
        if entry is None:
            return

        killpg = getattr(os, "killpg", None)
        if (
            entry.process_group_id is not None
            and platform.system() != "windows"
            and killpg is not None
        ):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                killpg(entry.process_group_id, signal.SIGTERM)

        terminate_process_tree(entry.pid)

    @staticmethod
    def _record_direct_process_runtime(
        registry_path: str,
        mi: ModelInstance,
        worker_id: int,
        backend: BackendEnum,
        log_file_path: str,
        start_result,
    ) -> None:
        if not isinstance(start_result, dict):
            return

        pid = start_result.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return

        deployment_metadata = mi.get_deployment_metadata(worker_id)
        deployment_name = deployment_metadata.name if deployment_metadata else mi.name
        port = start_result.get("port") or mi.port
        if port is None:
            return

        DirectProcessRegistry(registry_path).upsert(
            model_instance_id=mi.id,
            deployment_name=deployment_name,
            pid=pid,
            process_group_id=start_result.get("process_group_id"),
            port=port,
            log_path=start_result.get("log_path") or log_file_path,
            backend=backend,
            mode=start_result.get("mode") or DIRECT_PROCESS_RUNTIME_MODE,
        )

    def _get_health_check_path(self, backend: str) -> Optional[str]:
        """
        Get health check path for the given backend.

        Args:
            backend: The backend name.
        Returns:
            The health check path if exists, else None.
        """
        inference_backend = self._inference_backend_manager.get_backend_by_name(backend)

        return inference_backend.health_check_path if inference_backend else None

    def get_instance_port_by_model_instance_id(
        self, model_instance_id: int
    ) -> Optional[int]:
        """
        Get the port of the model instance related to the given model instance ID.

        Args:
            model_instance_id: The model instance ID to get the port for.

        Returns:
            The port of the model instance if it exists and is running, else None.
        """
        instance = self._model_instance_by_instance_id.get(
            model_instance_id
        )  # Ensure the model instance is cached.
        return (
            instance.port
            if instance and instance.state == ModelInstanceStateEnum.RUNNING
            else None
        )


def is_ready(
    backend: str,
    mi: ModelInstance,
    health_check_path: Optional[str] = None,
    model: Model = None,
) -> bool:
    """
    Access the health endpoint of the given model instance to check if it is servable.
    """
    is_built_in = is_built_in_backend(backend)
    if (not is_built_in or backend == BackendEnum.CUSTOM) and (not health_check_path):
        # If custom backend does not have health check path, consider it always ready.
        return True

    if backend == BackendEnum.ASCEND_MINDIE and not health_check_path:
        # Ref: https://www.hiascend.com/document/detail/zh/mindie/21RC2/mindieservice/servicedev/mindie_service0066.html
        # /info provides metadata information and requires more time to respond. Use it for health check.
        health_check_path = "/info"
    elif (
        backend == BackendEnum.SGLANG
        and model
        and CategoryEnum.IMAGE in model.categories
    ):
        if not model.backend_version:
            # version may be empty at initialization, consider it not ready.
            return False
        elif compare_versions(model.backend_version, "0.5.5.post3") >= 0:
            # SGLang Diffusion supported health check path at v0.5.5.post3
            health_check_path = "/health"
        else:
            # Older versions do not support health check, consider it always ready.
            return True
    elif is_built_in and backend != BackendEnum.CUSTOM and not health_check_path:
        # Built-in backends (vLLM, SGLang, vox-box) except (Custom, MindIE) use /v1/models as health check path.
        health_check_path = "/v1/models"

    try:
        # Use the worker IP instead of localhost for health check.
        # Reasons:
        # 1. Connectivity to the loopback address does not work with Ascend MindIE.
        # 2. More adaptable to container networks.
        health_check_url = f"http://{mi.worker_ip}:{mi.port}{health_check_path}"
        response = requests.get(health_check_url, timeout=1)
        if response.status_code == 200:
            return True
    except Exception as e:
        logger.debug(f"Error checking model instance {mi.name} health: {e}")
        pass
    return False
