# Architecture

The diagram below provides a high-level view of the GPUStack architecture.

![gpustack-v2-architecture](assets/gpustack-v2-architecture.png)

The diagram below details the internal components and their interactions.

![gpustack-v2-components](assets/gpustack-v2-components.png)

### Server

The GPUStack server consists of the following components:

- **API Server:** Provides a RESTful interface for clients to interact with the system. It handles authentication and authorization.
- **Scheduler:** Responsible for assigning model instances to workers.
- **Controllers:** Manages the state of resources in the system. For example, they handle the rollout and scaling of model instances to match the desired number of replicas.

### Worker

The GPUStack worker consists of the following components:

- **GPUStack Runtime:** Detects GPU devices and interacts with the container runtime to deploy model instances.
- **ServeManager:** Manages the lifecycle of model instances on the worker.
- **Metric Exporter:** Exports metrics about the model instances and their performance.

Container deployment is the standard GPUStack worker architecture and remains supported in this fork. In this fork only, a worker can also be started with `GPUSTACK_DIRECT_PROCESS_MODE=true` to launch supported backends as direct Linux processes instead of creating model containers.

The direct-process path remains worker-global in this phase. Workers advertise which direct-process backends and topologies they support, and scheduling is expected to place workloads only on workers whose advertised capabilities match the requested backend and topology.

This phase makes recipe-backed provisioning the preparation layer for supported direct-process backends. `bootstrap` resolves the backend recipe, materializes or reuses the prepared cache for that backend recipe and version, and records the consumed launch artifacts such as `prepared.env`, `prepared-config.json`, `prepared-launch.sh`, executable provenance, and provisioning audit data.

`bootstrap` remains preparatory only. It does not own process lifecycle state and it does not become the runtime control plane. `ServeManager` remains the direct-process lifecycle authority on the worker. It orchestrates launch, cleanup, restart, and reconciliation for each model instance. `process_registry` remains the runtime metadata authority for direct-process serving, tracking the active process identity and worker-side runtime state that `ServeManager` reconciles.

This boundary also separates reusable prepared state from per-instance runtime state. The prepared cache is keyed by backend recipe and version so compatible launches can reuse artifacts. The runtime workspace stays keyed by deployment and model instance so logs, ports, pid data, manifests, and other live process state remain isolated to the instance that `ServeManager` is managing.

Host bootstrap is a separate framework-only control path. It is disabled by default, limited to Linux workers, supports dry-run, requires allowlisted recipe sources and hash-pinned declared inputs, and writes audit records for each execution state. The normal direct-process serve path does not silently mutate the host.

Distributed direct-process scope is limited to `vLLM`, `SGLang`, and `MindIE`. That supported subset still uses the existing worker-wide distributed direct-process enablement gate, and its current config and env naming are legacy `vLLM`-oriented names. Distributed direct-process for `llama.cpp`, `VoxBox`, and `custom backend` is unsupported in this phase and must fail closed rather than implying broader backend parity. Those backends have single-worker recipe-backed parity only.

The fork keeps the same worker control flow, but direct-process instances use file logs at `gpustack/logs/serve/<model-instance-id>.log` and follow a cleanup-and-recreate restart policy. On worker restart, stale direct-process entries are removed and recreated instead of being reattached.

### AI Gateway

The AI Gateway handles incoming API requests from clients. It routes requests to the appropriate model instances based on the requested model. GPUStack uses [Higress](https://github.com/alibaba/higress) for API routing and load balancing.

### SQL Database

The GPUStack server connects to a SQL database as the datastore. GPUStack uses an Embedded PostgreSQL by default, but you can configure it to use an external PostgreSQL or MySQL as well.

### Inference Server

Inference servers are the backends that perform the inference tasks. GPUStack supports [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [Ascend MindIE](https://www.hiascend.com/en/software/mindie) and [VoxBox](https://github.com/gpustack/vox-box) as the built-in inference server. In this fork's direct-process path, distributed recipe-backed support is limited to `vLLM`, `SGLang`, and `MindIE`. `llama.cpp`, `VoxBox`, and `custom backend` have single-worker recipe-backed parity only when the backend fits the documented direct-process contract.

### Ray

[Ray](https://ray.io) is the distributed computing framework used by the shipped `vLLM` distributed direct-process path in this phase. GPUStack bootstraps Ray cluster resources on demand for those `vLLM` multi-worker launches. Other in-scope distributed direct-process backends, including `SGLang` and `MindIE`, are not described here as sharing that same Ray-backed substrate.
