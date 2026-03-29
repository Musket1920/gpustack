import socket
import logging
from typing import Callable, Optional
from gpustack.config.config import Config
from gpustack.client.generated_clientset import ClientSet
from gpustack.detectors.base import GPUDetectExepction
from gpustack.detectors.custom.custom import Custom
from gpustack.detectors.detector_factory import DetectorFactory
from gpustack.envs import WORKER_STATUS_COLLECTION_LOG_SLOW_SECONDS
from gpustack.policies.base import Allocated
from gpustack.schemas.models import ComputedResourceClaim
from gpustack.schemas.workers import (
    MountPoint,
    WorkerStatusPublic,
    WorkerStatus,
    SystemReserved,
    GPUDevicesStatus,
    SystemInfo,
    WorkerReachabilityCapabilities,
    WorkerReachabilityModeEnum,
)
from gpustack.utils.profiling import time_decorator


logger = logging.getLogger(__name__)


class WorkerStatusCollector:
    _cfg: Config
    _worker_id_getter: Callable[[], int]
    _worker_ifname_getter: Callable[[], str]
    _worker_ip_getter: Callable[[], str]
    _worker_uuid_getter: Callable[[], str]
    _capabilities_getter: Callable[[], WorkerReachabilityCapabilities]
    _reachability_mode_getter: Callable[[], WorkerReachabilityModeEnum]
    _gpu_devices: Optional[GPUDevicesStatus]
    _system_info: Optional[SystemInfo]

    @property
    def gpu_devices(self) -> Optional[GPUDevicesStatus]:
        return self._gpu_devices

    @property
    def system_info(self) -> Optional[SystemInfo]:
        return self._system_info

    def __init__(
        self,
        cfg: Config,
        worker_ip_getter: Callable[[], str],
        worker_ifname_getter: Callable[[], str],
        worker_id_getter: Callable[[], int],
        worker_uuid_getter: Callable[[], str],
        capabilities_getter: Callable[[], WorkerReachabilityCapabilities],
        reachability_mode_getter: Callable[[], WorkerReachabilityModeEnum],
    ):
        self._cfg = cfg
        self._worker_ip_getter = worker_ip_getter
        self._worker_ifname_getter = worker_ifname_getter
        self._worker_id_getter = worker_id_getter
        self._worker_uuid_getter = worker_uuid_getter
        self._capabilities_getter = capabilities_getter
        self._reachability_mode_getter = reachability_mode_getter
        self._gpu_devices = cfg.get_gpu_devices()
        self._system_info = cfg.get_system_info()
        if self._gpu_devices and self._system_info:
            self._detector_factory = DetectorFactory(
                device="custom",
                gpu_detectors={"custom": [Custom(gpu_devices=self._gpu_devices)]},
                system_info_detector=Custom(system_info=self._system_info),
            )
        elif self._gpu_devices:
            self._detector_factory = DetectorFactory(
                device="custom",
                gpu_detectors={"custom": [Custom(gpu_devices=self._gpu_devices)]},
            )
        elif self._system_info:
            self._detector_factory = DetectorFactory(
                system_info_detector=Custom(system_info=self._system_info)
            )
        else:
            self._detector_factory = DetectorFactory()

    """A class for collecting worker status information."""

    @time_decorator(log_slow_seconds=WORKER_STATUS_COLLECTION_LOG_SLOW_SECONDS)
    def timed_collect(
        self, clientset: Optional[ClientSet] = None, initial: bool = False
    ):
        return self.collect(clientset=clientset, initial=initial)

    def collect(
        self, clientset: Optional[ClientSet] = None, initial: bool = False
    ) -> WorkerStatusPublic:  # noqa: C901
        """Collect worker status information."""
        status = WorkerStatus.get_default_status()
        state_message = None
        try:
            system_info = self._detector_factory.detect_system_info()
            status = WorkerStatus.model_validate({**system_info.model_dump()})
        except Exception as e:
            logger.error(f"Failed to detect system info: {e}")

        if not initial:
            try:
                gpu_devices = self._detector_factory.detect_gpus()
                status.gpu_devices = gpu_devices
            except GPUDetectExepction as e:
                state_message = str(e)
            except Exception as e:
                logger.error(f"Failed to detect GPU devices: {e}")
        self._inject_unified_memory(status)
        self._inject_computed_filesystem_usage(status)
        self._inject_allocated_resource(clientset, status)

        # If disable_worker_metrics is set, set metrics_port to -1
        metrics_port = self._cfg.worker_metrics_port
        if self._cfg.disable_worker_metrics:
            metrics_port = -1

        return WorkerStatusPublic(
            advertise_address=self._cfg.advertise_address or self._worker_ip_getter(),
            hostname=socket.gethostname(),
            ip=self._worker_ip_getter(),
            ifname=self._worker_ifname_getter(),
            port=self._cfg.worker_port,
            metrics_port=metrics_port,
            system_reserved=SystemReserved(**self._cfg.get_system_reserved()),
            state_message=state_message,
            status=status,
            worker_uuid=self._worker_uuid_getter(),
            capabilities=self._capabilities_getter(),
            reachability_mode=self._reachability_mode_getter(),
            proxy_mode=self._cfg.proxy_mode,
        )

    def _inject_unified_memory(self, status: WorkerStatus):
        is_unified_memory = False
        if status.gpu_devices is not None and len(status.gpu_devices) != 0:
            first_gpu = status.gpu_devices[0]
            memory = first_gpu.memory
            if memory is not None:
                is_unified_memory = memory.is_unified_memory

        if status.memory is not None:
            status.memory.is_unified_memory = is_unified_memory

    def _inject_computed_filesystem_usage(self, status: WorkerStatus):
        if (
            status.os is None
            or "Windows" not in status.os.name
            or status.filesystem is None
        ):
            return

        try:
            computed = MountPoint(
                name="computed",
                mount_point="/",
                total=int(0),
                used=int(0),
                free=int(0),
                available=int(0),
            )
            for mountpoint in status.filesystem:
                total = (computed.total or 0) + (mountpoint.total or 0)
                used = (computed.used or 0) + (mountpoint.used or 0)
                free = (computed.free or 0) + (mountpoint.free or 0)
                available = (computed.available or 0) + (mountpoint.available or 0)
                computed.total = int(total)
                computed.used = int(used)
                computed.free = int(free)
                computed.available = int(available)

            # inject computed filesystem usage
            status.filesystem.append(computed)
        except Exception as e:
            logger.error(f"Failed to inject filesystem usage: {e}")

    def _inject_allocated_resource(  # noqa: C901
        self, clientset: Optional[ClientSet], status: WorkerStatus
    ):
        if clientset is None:
            return
        worker_id = self._worker_id_getter()
        allocated = Allocated(ram=0, vram={})
        try:
            # TODO avoid listing model_instances with clientset.
            # The calculation might not be needed here.
            model_instances = clientset.model_instances.list()
            for model_instance in model_instances.items:
                if (
                    model_instance.distributed_servers
                    and model_instance.distributed_servers.subordinate_workers
                ):
                    for (
                        subworker
                    ) in model_instance.distributed_servers.subordinate_workers:
                        if subworker.worker_id != worker_id:
                            continue

                        aggregate_computed_resource_claim_allocated(
                            allocated, subworker.computed_resource_claim
                        )

                if model_instance.worker_id != worker_id:
                    continue

                aggregate_computed_resource_claim_allocated(
                    allocated, model_instance.computed_resource_claim
                )

            # inject allocated resources
            if status.memory is not None:
                status.memory.allocated = allocated.ram
            if status.gpu_devices is not None:
                for i, device in enumerate(status.gpu_devices):
                    device_memory = device.memory
                    if device_memory is None:
                        continue
                    if device.index in allocated.vram:
                        device_memory.allocated = allocated.vram[device.index]
                    else:
                        device_memory.allocated = 0
        except Exception as e:
            logger.error(f"Failed to inject allocated resources: {e}")


def aggregate_computed_resource_claim_allocated(
    allocated: Allocated, computed_resource_claim: Optional[ComputedResourceClaim]
):
    """Aggregate allocated resources from a ComputedResourceClaim into Allocated."""
    if computed_resource_claim is None:
        return

    if computed_resource_claim.ram:
        allocated.ram += computed_resource_claim.ram

    for gpu_index, vram in (computed_resource_claim.vram or {}).items():
        allocated.vram[gpu_index] = (allocated.vram.get(gpu_index) or 0) + vram
