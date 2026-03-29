from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Dict, Optional, Any
from pydantic import ConfigDict, BaseModel, PrivateAttr, model_validator
from sqlmodel import (
    Field,
    SQLModel,
    JSON,
    Column,
    Text,
    Relationship,
    Integer,
    ForeignKey,
)
import sqlalchemy as sa
from sqlalchemy import String
from gpustack.schemas.common import (
    ListParams,
    PaginatedList,
    UTCDateTime,
    pydantic_column_type,
)
from gpustack import envs
from gpustack.mixins import BaseModelMixin
from typing import List
from sqlalchemy.orm import declarative_base

from gpustack.utils.network import is_offline
from gpustack.schemas.clusters import ClusterProvider, Cluster, WorkerPool
from gpustack.schemas.config import (
    PredefinedConfigNoDefaults,
    ModelInstanceProxyModeEnum,
)

Base = declarative_base()


class UtilizationInfo(BaseModel):
    total: int = Field(default=None)
    utilization_rate: Optional[float] = Field(default=None)  # rang from 0 to 100


class MemoryInfo(UtilizationInfo):
    is_unified_memory: bool = Field(default=False)
    used: Optional[int] = Field(default=None)
    allocated: Optional[int] = Field(default=None)


class CPUInfo(UtilizationInfo):
    pass


class GPUCoreInfo(UtilizationInfo):
    pass


class GPUNetworkInfo(BaseModel):
    status: str = Field(default="down")  # Network status (up/down)
    inet: str = Field(default="")  # IPv4 address
    netmask: str = Field(default="")  # Subnet mask
    mac: str = Field(default="")  # MAC address
    gateway: str = Field(default="")  # Default gateway
    iface: Optional[str] = Field(default=None)  # Network interface name
    mtu: Optional[int] = Field(default=None)  # Maximum Transmission Unit


class SwapInfo(UtilizationInfo):
    used: Optional[int] = Field(default=None)
    pass


class GPUDeviceInfo(BaseModel):
    vendor: Optional[str] = Field(default="")
    """
    Manufacturer of the GPU device, e.g. nvidia, amd, ascend, etc.
    """
    type: Optional[str] = Field(default="")
    """
    Device runtime backend type, e.g. cuda, rocm, cann, etc.
    """
    index: Optional[int] = Field(default=None)
    """
    GPU index, which is the logic ID of the GPU chip,
    which is a human-readable index and counted from 0 generally.
    It might be recognized as the GPU device ID in some cases, when there is no more than one GPU chip on the same card.
    """
    device_index: Optional[int] = Field(default=0)
    """
    GPU device index, which is the index of the onboard GPU device.
    In Linux, it can be retrieved under the /dev/ path.
    For example, /dev/nvidia0 (the first Nvidia card), /dev/davinci2(the third Ascend card), etc.
    """
    device_chip_index: Optional[int] = Field(default=0)
    """
    GPU device chip index, which is the index of the GPU chip on the card.
    It works with `device_index` to identify a GPU chip uniquely.
    For example, the first chip on the first card is 0, and the second chip on the first card is 1.
    """
    arch_family: Optional[str] = Field(default=None)
    """
    Architecture family of the GPU device.
    """
    name: str = Field(default="")
    """
    GPU name, e.g. NVIDIA A100-SXM4-40GB, NVIDIA RTX 3090, AMD MI100, Ascend 310P, etc.
    """
    uuid: Optional[str] = Field(default="")
    """
    UUID is a unique identifier assigned to each GPU device.
    """
    driver_version: Optional[str] = Field(default=None)
    """
    Driver version of the GPU device, e.g. for NVIDIA GPUs.
    """
    runtime_version: Optional[str] = Field(default=None)
    """
    Runtime version of the GPU device, e.g. CUDA version for NVIDIA GPUs.
    """
    compute_capability: Optional[str] = Field(default=None)
    """
    Compute compatibility version of the GPU device, e.g. for NVIDIA GPUs.
    """


class GPUDeviceStatus(GPUDeviceInfo):
    core: Optional[GPUCoreInfo] = Field(sa_column=Column(JSON), default=None)
    """
    Core information of the GPU device.
    """
    memory: Optional[MemoryInfo] = Field(sa_column=Column(JSON), default=None)
    """
    Memory information of the GPU device.
    """
    temperature: Optional[float] = Field(default=None)
    """
    Temperature of the GPU device in Celsius.
    """
    network: Optional[GPUNetworkInfo] = Field(sa_column=Column(JSON), default=None)
    """
    Network information of the GPU device, mainly for Ascend devices.
    """


GPUDevicesStatus = List[GPUDeviceStatus]


class MountPoint(BaseModel):
    name: str = Field(default="")
    mount_point: str = Field(default="")
    mount_from: str = Field(default="")
    total: int = Field(default=None)  # in bytes
    used: Optional[int] = Field(default=None)
    free: Optional[int] = Field(default=None)
    available: Optional[int] = Field(default=None)


FileSystemInfo = List[MountPoint]


class OperatingSystemInfo(BaseModel):
    name: str = Field(default="")
    version: str = Field(default="")


class KernelInfo(BaseModel):
    name: str = Field(default="")
    release: str = Field(default="")
    version: str = Field(default="")
    architecture: str = Field(default="")


class UptimeInfo(BaseModel):
    uptime: float = Field(default=None)  # in seconds
    boot_time: str = Field(default="")


class SystemReserved(BaseModel):
    ram: Optional[int] = Field(default=None)
    vram: Optional[int] = Field(default=None)


class RPCServer(BaseModel):
    pid: Optional[int] = None
    port: Optional[int] = None
    gpu_index: Optional[int] = None


class WorkerControlChannelEnum(str, Enum):
    OUTBOUND_CONTROL_WS = "outbound_control_ws"
    REVERSE_HTTP = "reverse_http"


class WorkerReachabilityModeEnum(str, Enum):
    REVERSE_PROBE = "reverse_probe"
    OUTBOUND_CONTROL_WS = WorkerControlChannelEnum.OUTBOUND_CONTROL_WS.value
    REVERSE_HTTP = WorkerControlChannelEnum.REVERSE_HTTP.value


class WorkerReachabilityCapabilities(BaseModel):
    outbound_control_ws: bool = False
    reverse_http: bool = False

    def supports_mode(self, mode: WorkerReachabilityModeEnum) -> bool:
        if mode == WorkerReachabilityModeEnum.REVERSE_PROBE:
            return True
        if mode == WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS:
            return self.outbound_control_ws
        if mode == WorkerReachabilityModeEnum.REVERSE_HTTP:
            return self.reverse_http
        return False

    @property
    def has_outbound_control(self) -> bool:
        return self.outbound_control_ws or self.reverse_http


def _resolve_default_reachability_mode(
    capabilities: Optional[WorkerReachabilityCapabilities],
) -> WorkerReachabilityModeEnum:
    if not capabilities or not capabilities.has_outbound_control:
        return WorkerReachabilityModeEnum.REVERSE_PROBE

    configured_mode = getattr(
        envs,
        "WORKER_DEFAULT_REACHABILITY_MODE",
        WorkerReachabilityModeEnum.REVERSE_PROBE.value,
    )
    try:
        requested_mode = WorkerReachabilityModeEnum(configured_mode)
    except ValueError:
        requested_mode = WorkerReachabilityModeEnum.REVERSE_PROBE

    if capabilities.supports_mode(requested_mode):
        return requested_mode

    return WorkerReachabilityModeEnum.REVERSE_PROBE


class WorkerStateEnum(str, Enum):
    r"""
    Enum for Worker State

    State Transition Diagram:

    Phase 1: Provisioning Controller          |  Phase 2: Healthcheck Controller
    ------------------------------------------|------------------------------------
    PENDING > PROVISIONING > INITIALIZING > READY < -----|----------->  UNREACHABLE
                |             |         |      ^         |       (Worker Endpoint Unreachable)
                |             |         |      |         |
                |-------------|---------|------|         └----------->   NOT_READY
                \_____________________________/|                 (Worker Stop Posting Status)
                                   ERROR       | (Provisioning failed)       ^
                                     |         |        |                    |
                                     v         |        v                    |
                                 DELETING  <---┘ (provisioning end)          |
                                               |                             |
                                               |                             |
    Phase 3: Upgrade and Maintain              |                             |
    -------------------------------------------|-----------------------------|-----
                                               v                             |
                                           MAINTENANCE <---------------------┘
                                           (Back to Ready/Not Ready after maintenance completed)
    """

    NOT_READY = "not_ready"
    READY = "ready"
    UNREACHABLE = "unreachable"
    PENDING = "pending"
    PROVISIONING = "provisioning"
    INITIALIZING = "initializing"
    DELETING = "deleting"
    ERROR = "error"
    MAINTENANCE = "maintenance"

    @property
    def is_provisioning(self) -> bool:
        return self in [
            WorkerStateEnum.PENDING,
            WorkerStateEnum.PROVISIONING,
            WorkerStateEnum.INITIALIZING,
            WorkerStateEnum.DELETING,
            WorkerStateEnum.ERROR,
        ]


class WorkerTransportHealthEnum(str, Enum):
    UNSUPPORTED = "unsupported"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


class WorkerControlPlaneHealthEnum(str, Enum):
    UNSUPPORTED = "unsupported"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class WorkerRuntimeHealthEnum(str, Enum):
    NOT_READY = WorkerStateEnum.NOT_READY.value
    READY = WorkerStateEnum.READY.value
    PENDING = WorkerStateEnum.PENDING.value
    PROVISIONING = WorkerStateEnum.PROVISIONING.value
    INITIALIZING = WorkerStateEnum.INITIALIZING.value
    DELETING = WorkerStateEnum.DELETING.value
    ERROR = WorkerStateEnum.ERROR.value
    MAINTENANCE = WorkerStateEnum.MAINTENANCE.value


class WorkerReachabilityHealthEnum(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"


class WorkerHealth(BaseModel):
    transport: WorkerTransportHealthEnum = WorkerTransportHealthEnum.UNSUPPORTED
    control_plane: WorkerControlPlaneHealthEnum = (
        WorkerControlPlaneHealthEnum.UNSUPPORTED
    )
    runtime: WorkerRuntimeHealthEnum = WorkerRuntimeHealthEnum.NOT_READY
    reachability: WorkerReachabilityHealthEnum = (
        WorkerReachabilityHealthEnum.NOT_APPLICABLE
    )


class SystemInfo(BaseModel):
    cpu: Optional[CPUInfo] = Field(sa_column=Column(JSON), default=None)
    memory: Optional[MemoryInfo] = Field(sa_column=Column(JSON), default=None)
    swap: Optional[SwapInfo] = Field(sa_column=Column(JSON), default=None)
    filesystem: Optional[FileSystemInfo] = Field(sa_column=Column(JSON), default=None)
    os: Optional[OperatingSystemInfo] = Field(sa_column=Column(JSON), default=None)
    kernel: Optional[KernelInfo] = Field(sa_column=Column(JSON), default=None)
    uptime: Optional[UptimeInfo] = Field(sa_column=Column(JSON), default=None)


class Maintenance(BaseModel):
    enabled: bool = False
    message: Optional[str] = None


class WorkerStatus(SystemInfo):
    """
    rpc_servers: Deprecated
    """

    gpu_devices: Optional[GPUDevicesStatus] = Field(
        sa_column=Column(JSON), default=None
    )
    rpc_servers: Optional[Dict[int, RPCServer]] = Field(
        sa_column=Column(JSON), default=None
    )

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def get_default_status(cls) -> 'WorkerStatus':
        return WorkerStatus(
            cpu=CPUInfo(total=0),
            memory=MemoryInfo(total=0, is_unified_memory=False),
            swap=SwapInfo(total=0),
            filesystem=[],
            os=OperatingSystemInfo(name="", version=""),
            kernel=KernelInfo(name="", release="", version="", architecture=""),
            uptime=UptimeInfo(uptime=0, boot_time=""),
            gpu_devices=[],
        )


class WorkerStatusStored(BaseModel):
    _reachability_mode_inferred: bool = PrivateAttr(default=False)

    advertise_address: Optional[str] = None
    hostname: str
    ip: str
    ifname: str
    port: int
    metrics_port: Optional[int] = None
    capabilities: Optional[WorkerReachabilityCapabilities] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(WorkerReachabilityCapabilities)),
    )
    reachability_mode: Optional[WorkerReachabilityModeEnum] = Field(default=None)

    system_reserved: Optional[SystemReserved] = Field(
        default=None, sa_column=Column(pydantic_column_type(SystemReserved))
    )
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    status: Optional[WorkerStatus] = Field(
        sa_column=Column(pydantic_column_type(WorkerStatus))
    )

    worker_uuid: str = Field(sa_column=Column(String(255), nullable=False))
    machine_id: Optional[str] = Field(
        default=None
    )  # The machine ID of the worker, used for identifying the worker in the cluster

    proxy_mode: Optional[ModelInstanceProxyModeEnum] = Field(
        default=ModelInstanceProxyModeEnum.WORKER,
    )

    @model_validator(mode="after")
    def validate_reachability_mode(self):
        capabilities = self.capabilities or WorkerReachabilityCapabilities()

        if self.reachability_mode is None:
            vars(self)["reachability_mode"] = _resolve_default_reachability_mode(
                capabilities
            )
            object.__setattr__(self, "_reachability_mode_inferred", True)

        reachability_mode = self.reachability_mode
        if reachability_mode is None:
            raise ValueError("reachability_mode could not be resolved")

        if not capabilities.supports_mode(reachability_mode):
            raise ValueError(
                f"reachability_mode '{reachability_mode.value}' requires explicit worker capability advertisement"
            )

        return self

    def _is_reachability_mode_inferred(self) -> bool:
        private_attrs = getattr(self, "__pydantic_private__", None)
        if not private_attrs:
            return False

        return bool(private_attrs.get("_reachability_mode_inferred", False))


class WorkerStatusPublic(WorkerStatusStored):
    gateway_endpoint: Optional[str] = None


class WorkerUpdate(SQLModel):
    """
    WorkerUpdate: updatable fields for Worker
    """

    name: str = Field(index=True, unique=True)
    labels: Dict[str, str] = Field(sa_column=Column(JSON), default={})
    maintenance: Optional[Maintenance] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(Maintenance), default=None),
    )


class WorkerCreate(WorkerStatusStored, WorkerUpdate):
    cluster_id: Optional[int] = Field(
        sa_column=Column(Integer, ForeignKey("clusters.id"), nullable=False),
        default=None,
    )
    external_id: Optional[str] = Field(
        default=None, sa_column=Column(String(255), nullable=True)
    )

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        if self.capabilities is None:
            data.pop("capabilities", None)
        if self._is_reachability_mode_inferred():
            data.pop("reachability_mode", None)
        return data


class WorkerBase(WorkerCreate):
    _active_control_session: Optional["WorkerSession"] = PrivateAttr(default=None)

    state: WorkerStateEnum = WorkerStateEnum.NOT_READY
    heartbeat_time: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    unreachable: bool = False

    def set_active_control_session(
        self,
        worker_session: Optional["WorkerSession"],
    ) -> None:
        self._active_control_session = worker_session

    def uses_legacy_unreachable_instance_coupling(self) -> bool:
        return self.reachability_mode == WorkerReachabilityModeEnum.REVERSE_PROBE

    def should_block_new_placements(self) -> bool:
        if self.state != WorkerStateEnum.READY:
            return True

        if self.reachability_mode != WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS:
            return False

        health = self.health_status
        return (
            health.transport != WorkerTransportHealthEnum.CONNECTED
            or health.control_plane != WorkerControlPlaneHealthEnum.AVAILABLE
            or health.runtime != WorkerRuntimeHealthEnum.READY
        )

    def _compute_runtime_health(self) -> WorkerRuntimeHealthEnum:
        if self.maintenance and self.maintenance.enabled:
            return WorkerRuntimeHealthEnum.MAINTENANCE

        if self.state.is_provisioning:
            return WorkerRuntimeHealthEnum(self.state.value)

        if self.state == WorkerStateEnum.NOT_READY and self.state_message is not None:
            return WorkerRuntimeHealthEnum.NOT_READY

        is_not_ready_flag, _ = is_offline(
            self.heartbeat_time,
            envs.WORKER_HEARTBEAT_GRACE_PERIOD,
            datetime.now(timezone.utc),
        )
        if is_not_ready_flag:
            return WorkerRuntimeHealthEnum.NOT_READY

        return WorkerRuntimeHealthEnum.READY

    def _compute_reachability_health(self) -> WorkerReachabilityHealthEnum:
        if self.reachability_mode != WorkerReachabilityModeEnum.REVERSE_PROBE:
            return WorkerReachabilityHealthEnum.NOT_APPLICABLE

        if self.unreachable:
            return WorkerReachabilityHealthEnum.UNREACHABLE

        return WorkerReachabilityHealthEnum.REACHABLE

    def _compute_transport_health(self) -> WorkerTransportHealthEnum:
        if self.reachability_mode != WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS:
            return WorkerTransportHealthEnum.UNSUPPORTED

        if self._active_control_session is not None:
            return WorkerTransportHealthEnum.CONNECTED

        return WorkerTransportHealthEnum.DISCONNECTED

    def _compute_control_plane_health(self) -> WorkerControlPlaneHealthEnum:
        if self.reachability_mode not in {
            WorkerReachabilityModeEnum.OUTBOUND_CONTROL_WS,
            WorkerReachabilityModeEnum.REVERSE_HTTP,
        }:
            return WorkerControlPlaneHealthEnum.UNSUPPORTED

        if self._active_control_session is not None:
            return WorkerControlPlaneHealthEnum.AVAILABLE

        return WorkerControlPlaneHealthEnum.UNAVAILABLE

    @property
    def health_status(self) -> WorkerHealth:
        return WorkerHealth(
            transport=self._compute_transport_health(),
            control_plane=self._compute_control_plane_health(),
            runtime=self._compute_runtime_health(),
            reachability=self._compute_reachability_health(),
        )

    def compute_state(self):
        runtime_health = self._compute_runtime_health()

        if runtime_health == WorkerRuntimeHealthEnum.MAINTENANCE:
            self.state = WorkerStateEnum.MAINTENANCE
            self.state_message = self.maintenance.message if self.maintenance else None
            return

        if self.state.is_provisioning:
            return
        if runtime_health == WorkerRuntimeHealthEnum.NOT_READY and self.state == WorkerStateEnum.NOT_READY and self.state_message is not None:
            return

        is_not_ready_flag, last_heartbeat_str = is_offline(
            self.heartbeat_time,
            envs.WORKER_HEARTBEAT_GRACE_PERIOD,
            datetime.now(timezone.utc),
        )

        if is_not_ready_flag:
            reschedule_minutes = envs.MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD / 60
            self.state = WorkerStateEnum.NOT_READY
            self.state_message = (
                f"Heartbeat lost (last heartbeat: {last_heartbeat_str}). "
                f"If the worker remains unresponsive for more than {reschedule_minutes:.1f} minutes, "
                "the instances on this worker will be rescheduled automatically. "
                "If this downtime is planned maintenance, please enable maintenance mode. "
                "Otherwise, please <a href='https://docs.gpustack.ai/latest/troubleshooting/#view-gpustack-logs'>check the worker logs</a>."
            )
            return

        if self._compute_reachability_health() == WorkerReachabilityHealthEnum.UNREACHABLE:
            address = self.advertise_address or self.ip
            healthz_url = f"http://{address}:{self.port}/healthz"
            msg = (
                "Server cannot access the "
                f"worker's health check endpoint at {healthz_url}. "
                "Please verify the port requirements in the "
                "<a href='https://docs.gpustack.ai/latest/installation/requirements/#port-requirements'>documentation</a>"
            )
            self.state = WorkerStateEnum.UNREACHABLE
            self.state_message = msg
            return

        self.state = WorkerStateEnum.READY
        self.state_message = None

    provider: ClusterProvider = Field(default=ClusterProvider.Docker)
    worker_pool_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("worker_pools.id"), nullable=True),
    )  # The worker pool this worker belongs to

    # Not setting foreign key to manage lifecycle
    ssh_key_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    provider_config: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )


class Worker(WorkerBase, BaseModelMixin, table=True):
    __tablename__ = 'workers'  # type: ignore
    id: Optional[int] = Field(default=None, primary_key=True)

    cluster: Cluster = Relationship(
        back_populates="cluster_workers", sa_relationship_kwargs={"lazy": "noload"}
    )
    worker_pool: Optional[WorkerPool] = Relationship(
        back_populates="pool_workers", sa_relationship_kwargs={"lazy": "noload"}
    )
    worker_sessions: List["WorkerSession"] = Relationship(
        back_populates="worker",
        sa_relationship_kwargs={"lazy": "noload", "cascade": "delete"},
    )
    worker_commands: List["WorkerCommand"] = Relationship(
        back_populates="worker",
        sa_relationship_kwargs={"lazy": "noload", "cascade": "delete"},
    )

    # This field should be replaced by x509 credential if mTLS is supported.
    token: Optional[str] = Field(default=None, nullable=True)

    @property
    def provision_progress(self) -> Optional[str]:
        """
        The provisioning progress should have following steps:
        1. create_ssh_key
        2. create_instance with created ssh_key
        3. wait_for_started
        4. wait_for_public_ip
        5. [optional] create_volumes_and_attach
        """
        if self.state == WorkerStateEnum.INITIALIZING:
            return "5/5"
        if (
            self.state != WorkerStateEnum.PROVISIONING
            and self.state != WorkerStateEnum.PENDING
        ):
            return None
        format = "{}/{}"
        total = 5
        current = sum(
            [
                self.state == WorkerStateEnum.PROVISIONING,
                self.ssh_key_id is not None,
                self.external_id is not None,
                self.ip is not None and self.ip != "",
                "volume_ids" in (self.provider_config or {}),
            ]
        )
        return format.format(current, total)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, Worker):
            return self.id == other.id
        return False


class WorkerListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "state",
        "ip",
        "status.cpu.utilization_rate",
        "status.memory.utilization_rate",
        "gpus",  # gpu count, the same naming pattern as in Clusters
        "created_at",
        "updated_at",
    ]


class WorkerPublic(
    WorkerBase,
):
    id: int
    created_at: datetime
    updated_at: datetime
    health: WorkerHealth = Field(default_factory=WorkerHealth)
    reverse_http_available: bool = True
    reverse_http_unavailable_message: Optional[str] = None
    me: Optional[bool] = None  # Indicates if the worker is the current worker
    provision_progress: Optional[str] = None  # Indicates the provisioning progress

    machine_id: Optional[str] = Field(default=None, exclude=True)


class WorkerRegistrationPublic(WorkerPublic):
    token: str
    worker_config: Optional["PredefinedConfigNoDefaults"] = None


WorkersPublic = PaginatedList[WorkerPublic]


class WorkerControlMessageTypeEnum(str, Enum):
    HELLO = "hello"
    COMMAND = "command"
    COMMAND_ACK = "command_ack"
    COMMAND_RESULT = "command_result"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"


class WorkerControlCommandStateEnum(str, Enum):
    PENDING = "pending"
    LEASED = "leased"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    TIMED_OUT = "timed_out"
    SUPERSEDED = "superseded"


class WorkerSessionStateEnum(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    CLOSED = "closed"
    EXPIRED = "expired"


class WorkerControlMessageBase(BaseModel):
    message_type: WorkerControlMessageTypeEnum
    session_id: Optional[str] = None
    protocol_version: int = 1
    sent_at: Optional[datetime] = None


class WorkerHelloMessage(WorkerControlMessageBase):
    message_type: WorkerControlMessageTypeEnum = WorkerControlMessageTypeEnum.HELLO
    worker_uuid: str
    capabilities: WorkerReachabilityCapabilities = Field(
        default_factory=WorkerReachabilityCapabilities
    )
    reachability_mode: Optional[WorkerReachabilityModeEnum] = None

    @model_validator(mode="after")
    def validate_hello_reachability_mode(self):
        if self.reachability_mode is None:
            self.reachability_mode = _resolve_default_reachability_mode(
                self.capabilities
            )

        if not self.capabilities.supports_mode(self.reachability_mode):
            raise ValueError(
                f"reachability_mode '{self.reachability_mode.value}' requires explicit worker capability advertisement"
            )

        return self


class WorkerCommandMessage(WorkerControlMessageBase):
    message_type: WorkerControlMessageTypeEnum = WorkerControlMessageTypeEnum.COMMAND
    command_id: str
    command_type: str
    payload: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    desired_worker_state_revision: Optional[int] = None


class WorkerCommandAckMessage(WorkerControlMessageBase):
    message_type: WorkerControlMessageTypeEnum = (
        WorkerControlMessageTypeEnum.COMMAND_ACK
    )
    command_id: str
    accepted: bool = True
    state: WorkerControlCommandStateEnum = WorkerControlCommandStateEnum.ACKNOWLEDGED
    error_message: Optional[str] = None

    @model_validator(mode="after")
    def validate_ack_message(self):
        if not self.accepted and not self.error_message:
            raise ValueError("error_message is required when a command is rejected")
        return self


class WorkerCommandResultMessage(WorkerControlMessageBase):
    message_type: WorkerControlMessageTypeEnum = (
        WorkerControlMessageTypeEnum.COMMAND_RESULT
    )
    command_id: str
    state: WorkerControlCommandStateEnum
    result: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    error_message: Optional[str] = None

    @model_validator(mode="after")
    def validate_result_message(self):
        failure_states = {
            WorkerControlCommandStateEnum.FAILED,
            WorkerControlCommandStateEnum.CANCELLED,
            WorkerControlCommandStateEnum.EXPIRED,
        }
        if self.state in failure_states and not self.error_message:
            raise ValueError(
                "error_message is required for failed, cancelled, or expired command results"
            )
        return self


class WorkerPingMessage(WorkerControlMessageBase):
    message_type: WorkerControlMessageTypeEnum = WorkerControlMessageTypeEnum.PING


class WorkerPongMessage(WorkerControlMessageBase):
    message_type: WorkerControlMessageTypeEnum = WorkerControlMessageTypeEnum.PONG


class WorkerErrorMessage(WorkerControlMessageBase):
    message_type: WorkerControlMessageTypeEnum = WorkerControlMessageTypeEnum.ERROR
    error_code: str
    error_message: str


class WorkerSessionBase(SQLModel):
    session_id: str = Field(index=True, unique=True)
    worker_id: int = Field(default=None, foreign_key="workers.id")
    generation: int = Field(default=1)
    control_channel: WorkerControlChannelEnum
    reachability_mode: WorkerReachabilityModeEnum
    state: WorkerSessionStateEnum = WorkerSessionStateEnum.ACTIVE
    protocol_version: int = Field(default=1)
    connected_at: Optional[datetime] = Field(sa_column=Column(UTCDateTime), default=None)
    last_seen_at: Optional[datetime] = Field(sa_column=Column(UTCDateTime), default=None)
    disconnected_at: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    expires_at: Optional[datetime] = Field(sa_column=Column(UTCDateTime), default=None)
    last_command_sequence: int = Field(default=0)
    last_acknowledged_command_sequence: int = Field(default=0)
    last_completed_command_sequence: int = Field(default=0)
    replay_cursor: int = Field(default=0)
    requires_full_reconcile: bool = Field(default=False)
    full_reconcile_reason: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    details: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))


class WorkerSession(WorkerSessionBase, BaseModelMixin, table=True):
    __tablename__ = 'worker_sessions'  # type: ignore
    __table_args__ = (
        sa.Index("ix_worker_sessions_worker_id_generation", "worker_id", "generation"),
        sa.UniqueConstraint(
            "worker_id",
            "generation",
            name="uq_worker_sessions_worker_id_generation",
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)

    worker: Worker = Relationship(
        back_populates="worker_sessions",
        sa_relationship_kwargs={"lazy": "noload"},
    )
    worker_commands: List["WorkerCommand"] = Relationship(
        back_populates="worker_session",
        sa_relationship_kwargs={"lazy": "noload"},
    )


class WorkerCommandBase(SQLModel):
    command_id: str = Field(index=True, unique=True)
    worker_id: int = Field(default=None, foreign_key="workers.id")
    sequence: int = Field(default=0)
    worker_session_id: Optional[int] = Field(
        default=None, foreign_key="worker_sessions.id"
    )
    worker_session_generation: Optional[int] = None
    command_type: str
    payload: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    state: WorkerControlCommandStateEnum = WorkerControlCommandStateEnum.PENDING
    idempotency_key: Optional[str] = None
    dispatch_attempts: int = Field(default=0)
    desired_worker_state_revision: Optional[int] = None
    lease_expires_at: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    not_before: Optional[datetime] = Field(sa_column=Column(UTCDateTime), default=None)
    dispatched_at: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    acknowledged_at: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    completed_at: Optional[datetime] = Field(sa_column=Column(UTCDateTime), default=None)
    expires_at: Optional[datetime] = Field(sa_column=Column(UTCDateTime), default=None)
    result: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    error_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class WorkerCommand(WorkerCommandBase, BaseModelMixin, table=True):
    __tablename__ = 'worker_commands'  # type: ignore
    __table_args__ = (
        sa.Index("ix_worker_commands_worker_id_sequence", "worker_id", "sequence"),
        sa.Index(
            "ix_worker_commands_worker_id_idempotency_key",
            "worker_id",
            "idempotency_key",
            unique=True,
        ),
        sa.UniqueConstraint(
            "worker_id",
            "sequence",
            name="uq_worker_commands_worker_id_sequence",
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)

    worker: Worker = Relationship(
        back_populates="worker_commands",
        sa_relationship_kwargs={"lazy": "noload"},
    )
    worker_session: Optional[WorkerSession] = Relationship(
        back_populates="worker_commands",
        sa_relationship_kwargs={"lazy": "noload"},
    )


# ---------------------------------------------------------------------------
# Direct-process capability advertisement helpers
#
# Workers advertise direct-process capabilities through labels (flat
# Dict[str, str]).  This helper class provides structured access to the
# capability labels so the scheduler/filter layer does not need to know
# the raw label keys or parsing rules.
# ---------------------------------------------------------------------------

DIRECT_PROCESS_BACKENDS_LABEL = "gpustack.direct-process-backends"
DIRECT_PROCESS_BOOTSTRAP_BACKENDS_LABEL = (
    "gpustack.direct-process-bootstrap-backends"
)
DIRECT_PROCESS_DISTRIBUTED_BACKENDS_LABEL = (
    "gpustack.direct-process-distributed-backends"
)
DIRECT_PROCESS_HOST_BOOTSTRAP_BACKENDS_LABEL = (
    "gpustack.direct-process-host-bootstrap-backends"
)
DIRECT_PROCESS_CUSTOM_CONTRACT_LABEL = "gpustack.direct-process-custom-contract"


class DirectProcessCapabilities(BaseModel):
    """Structured view of a worker's direct-process capability labels.

    This is a read-only helper — not a DB column.  It is built from worker
    labels at scheduling/filter time and produced by ``WorkerManager`` at
    registration time.
    """

    enabled: bool = False
    """Whether the worker is in direct-process mode at all."""

    single_worker_backends: List[str] = Field(default_factory=list)
    """Backend names that support single-worker direct-process on this worker."""

    bootstrap_ready_backends: List[str] = Field(default_factory=list)
    """Backend names whose direct-process bootstrap prerequisites are ready on this
    worker."""

    distributed_backends: List[str] = Field(default_factory=list)
    """Backend names that support distributed (multi-worker) direct-process."""

    host_bootstrap_backends: List[str] = Field(default_factory=list)
    """Backend names that support an explicit host-bootstrap control path on this
    worker."""

    custom_contract_support: bool = False
    """Whether the worker supports the generic custom/community direct-process
    contract (task 6 will populate this)."""

    # -- factory from raw labels ------------------------------------------

    @classmethod
    def from_labels(cls, labels: Optional[Dict[str, str]]) -> "DirectProcessCapabilities":
        """Parse worker labels into a ``DirectProcessCapabilities`` instance.

        Missing or empty labels are treated as *no capability* (fail-safe).
        """
        if not labels:
            return cls()

        from gpustack.worker.worker_manager import DIRECT_PROCESS_MODE_LABEL

        mode_val = labels.get(DIRECT_PROCESS_MODE_LABEL, "")
        enabled = str(mode_val).lower() in {"1", "true", "yes", "on"}
        if not enabled:
            return cls()

        single_raw = labels.get(DIRECT_PROCESS_BACKENDS_LABEL, "")
        single_backends = [
            b.strip() for b in single_raw.split(",") if b.strip()
        ]

        bootstrap_raw = labels.get(DIRECT_PROCESS_BOOTSTRAP_BACKENDS_LABEL, "")
        bootstrap_backends = [
            b.strip() for b in bootstrap_raw.split(",") if b.strip()
        ]

        dist_raw = labels.get(DIRECT_PROCESS_DISTRIBUTED_BACKENDS_LABEL, "")
        distributed_backends = [
            b.strip() for b in dist_raw.split(",") if b.strip()
        ]

        host_bootstrap_raw = labels.get(DIRECT_PROCESS_HOST_BOOTSTRAP_BACKENDS_LABEL, "")
        host_bootstrap_backends = [
            b.strip() for b in host_bootstrap_raw.split(",") if b.strip()
        ]

        custom_val = labels.get(DIRECT_PROCESS_CUSTOM_CONTRACT_LABEL, "")
        custom_support = str(custom_val).lower() in {"1", "true", "yes", "on"}

        return cls(
            enabled=True,
            single_worker_backends=single_backends,
            bootstrap_ready_backends=bootstrap_backends,
            distributed_backends=distributed_backends,
            host_bootstrap_backends=host_bootstrap_backends,
            custom_contract_support=custom_support,
        )

    # -- convenience queries ----------------------------------------------

    def supports_backend(self, backend: str) -> bool:
        """Check if this worker supports single-worker direct-process for *backend*."""
        return self.enabled and backend in self.single_worker_backends

    def supports_bootstrap_backend(self, backend: str) -> bool:
        """Check if this worker is bootstrap-ready for *backend*."""
        return self.enabled and backend in self.bootstrap_ready_backends

    def supports_distributed_backend(self, backend: str) -> bool:
        """Check if this worker supports distributed direct-process for *backend*."""
        return self.enabled and backend in self.distributed_backends

    def supports_host_bootstrap_backend(self, backend: str) -> bool:
        """Check if this worker supports explicit host-bootstrap for *backend*."""
        return self.enabled and backend in self.host_bootstrap_backends

    # -- label production -------------------------------------------------

    def to_labels(self) -> Dict[str, str]:
        """Produce the flat label dict for worker registration.

        The caller is responsible for merging these into the full label set.
        The coarse ``gpustack.direct-process-mode`` label is NOT included
        here — it is managed by the existing ``_ensure_builtin_labels`` path.
        """
        labels: Dict[str, str] = {}
        if self.single_worker_backends:
            labels[DIRECT_PROCESS_BACKENDS_LABEL] = ",".join(
                sorted(self.single_worker_backends)
            )
        if self.bootstrap_ready_backends:
            labels[DIRECT_PROCESS_BOOTSTRAP_BACKENDS_LABEL] = ",".join(
                sorted(self.bootstrap_ready_backends)
            )
        if self.distributed_backends:
            labels[DIRECT_PROCESS_DISTRIBUTED_BACKENDS_LABEL] = ",".join(
                sorted(self.distributed_backends)
            )
        if self.host_bootstrap_backends:
            labels[DIRECT_PROCESS_HOST_BOOTSTRAP_BACKENDS_LABEL] = ",".join(
                sorted(self.host_bootstrap_backends)
            )
        if self.custom_contract_support:
            labels[DIRECT_PROCESS_CUSTOM_CONTRACT_LABEL] = "true"
        return labels
