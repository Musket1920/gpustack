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
- **Serving Manager:** Manages the lifecycle of model instances on the worker.
- **Metric Exporter:** Exports metrics about the model instances and their performance.

Container deployment is the standard GPUStack worker architecture and remains supported in this fork. In this fork only, a worker can also be started with `GPUSTACK_DIRECT_PROCESS_MODE=true` to launch supported backends as direct Linux processes instead of creating model containers.

The direct-process path stays worker-global in Part 2. Workers advertise which direct-process backends and topologies they support, and scheduling is expected to place workloads only on workers whose advertised capabilities match the requested backend and topology.

Part 2 direct-process coverage is built-in single-worker `vLLM`, `SGLang`, `MindIE`, and `VoxBox`, plus first-class single-worker `llama.cpp` and a generic `custom backend` path for community backends that satisfy the direct-process contract.

Distributed direct-process is vLLM-only in this phase. Non-`vLLM` distributed direct-process, benchmark direct-process runs, and claims of upstream support are out of scope.

The fork keeps the same worker control flow, but direct-process instances use file logs at `gpustack/logs/serve/<model-instance-id>.log` and follow a cleanup-and-recreate restart policy. On worker restart, stale direct-process entries are removed and recreated instead of being reattached.

### AI Gateway

The AI Gateway handles incoming API requests from clients. It routes requests to the appropriate model instances based on the requested model. GPUStack uses [Higress](https://github.com/alibaba/higress) for API routing and load balancing.

### SQL Database

The GPUStack server connects to a SQL database as the datastore. GPUStack uses an Embedded PostgreSQL by default, but you can configure it to use an external PostgreSQL or MySQL as well.

### Inference Server

Inference servers are the backends that perform the inference tasks. GPUStack supports [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [Ascend MindIE](https://www.hiascend.com/en/software/mindie) and [VoxBox](https://github.com/gpustack/vox-box) as the built-in inference server. In this fork's Part 2 direct-process path, `llama.cpp` is also first-class, and `custom backend` is the generic community path when a backend fits the direct-process contract.

### Ray

[Ray](https://ray.io) is a distributed computing framework that GPUStack utilizes to run distributed vLLM. GPUStack bootstraps Ray cluster on-demand to run distributed vLLM across multiple workers.
