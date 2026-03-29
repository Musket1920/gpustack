<br>

<p align="center">
    <img alt="GPUStack" src="https://raw.githubusercontent.com/gpustack/gpustack/main/docs/assets/gpustack-logo.png" width="300px"/>
</p>
<br>

<p align="center">
    <a href="https://docs.gpustack.ai" target="_blank">
        <img alt="Documentation" src="https://img.shields.io/badge/Docs-GPUStack-blue?logo=readthedocs&logoColor=white"></a>
    <a href="./LICENSE" target="_blank">
        <img alt="License" src="https://img.shields.io/github/license/gpustack/gpustack?logo=github&logoColor=white&label=License&color=blue"></a>
    <a href="./docs/assets/wechat-group-qrcode.jpg" target="_blank">
        <img alt="WeChat" src="https://img.shields.io/badge/WeChat-GPUStack-blue?logo=wechat&logoColor=white"></a>
    <a href="https://discord.gg/VXYJzuaqwD" target="_blank">
        <img alt="Discord" src="https://img.shields.io/badge/Discord-GPUStack-blue?logo=discord&logoColor=white"></a>
    <a href="https://twitter.com/intent/follow?screen_name=gpustack_ai" target="_blank">
        <img alt="Follow on X(Twitter)" src="https://img.shields.io/twitter/follow/gpustack_ai?logo=X"></a>
</p>
<br>

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">简体中文</a> |
  <a href="./README_JP.md">日本語</a>
</p>

<br>

## Overview

GPUStack is an open-source GPU cluster manager designed for efficient AI model deployment. It configures and orchestrates inference engines — vLLM, SGLang, TensorRT-LLM, or your own — to optimize performance across GPU clusters. Its core features include:
- **Multi-Cluster GPU Management.** Manages GPU clusters across multiple environments. This includes on-premises servers, Kubernetes clusters, and cloud providers.
- **Pluggable Inference Engines.** Automatically configures high-performance inference engines such as vLLM, SGLang, and TensorRT-LLM. You can also add custom inference engines as needed.
- **Day 0 Model Support.** GPUStack's pluggable engine architecture enables you to deploy new models on the day they are released.
- **Performance-Optimized Configurations.** Offers pre-tuned modes for low latency or high throughput. GPUStack supports extended KV cache systems like LMCache and HiCache to reduce TTFT. It also includes built-in support for speculative decoding methods such as EAGLE3, MTP, and N-grams.
- **Enterprise-Grade Operations.** Offers support for automated failure recovery, load balancing, monitoring, authentication, and access control.

## Architecture

GPUStack enables development teams, IT organizations, and service providers to deliver Model-as-a-Service at scale. It supports industry-standard APIs for LLM, voice, image, and video models. The platform includes built-in user authentication and access control, real-time monitoring of GPU performance and utilization, and detailed metering of token usage and API request rates.

The figure below illustrates how a single GPUStack server can manage multiple GPU clusters across both on-premises and cloud environments. The GPUStack scheduler allocates GPUs to maximize resource utilization and selects the appropriate inference engines for optimal performance. Administrators also gain full visibility into system health and metrics through integrated Grafana and Prometheus dashboards.

![gpustack-v2-architecture](docs/assets/gpustack-v2-architecture.png)

## Optimized Inference Performance

GPUStack's automated engine selection and parameter optimization deliver strong inference performance out of the box. The following figure shows throughput improvements over default vLLM configurations:

![a100-throughput-comparison](docs/assets/a100-throughput-comparison.png)

For detailed benchmarking methods and results, visit our [Inference Performance Lab](https://docs.gpustack.ai/latest/performance-lab/overview/).

## Quick Start

### Prerequisites

1. A node with at least one NVIDIA GPU. For other GPU types, please check the guidelines in the GPUStack UI when adding a worker, or refer to the [Installation documentation](https://docs.gpustack.ai/latest/installation/requirements/) for more details.
2. Ensure the NVIDIA driver, [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) are installed on the worker node.
3. (Optional) A CPU node for hosting the GPUStack server. The GPUStack server does not require a GPU and can run on a CPU-only machine. [Docker](https://docs.docker.com/engine/install/) must be installed. Docker Desktop (for Windows and macOS) is also supported. If no dedicated CPU node is available, the GPUStack server can be installed on the same machine as a GPU worker node.
4. Only Linux is supported for GPUStack worker nodes. If you use Windows, consider using WSL2 and avoid using Docker Desktop. macOS is not supported for GPUStack worker nodes.

### Fork-only direct-process note

This fork also includes an experimental worker option for operators who already run the GPUStack worker inside a Linux container or directly on a Linux host and want model servers to start as direct host processes instead of creating nested model containers.

Container deployment remains supported, and it is still the default and recommended GPUStack path. Treat direct-process mode as a fork-only operator workflow, not an official upstream GPUStack feature.

Direct-process mode is enabled per worker with `GPUSTACK_DIRECT_PROCESS_MODE=true`. The worker advertises direct-process capabilities for the backends it can launch, but the runtime mode is still worker-global. A worker is either in container mode or in direct-process mode for all deployments placed on that worker.

Recipe-backed provisioning is now the bootstrap path for supported direct-process backends. Before launch, bootstrap resolves a recipe-backed prepared environment, installs or validates its consumed artifacts under a reusable prepared cache, and records live launch inputs such as `prepared.env`, `prepared-config.json`, `prepared-launch.sh`, executable provenance, and provisioning audit data.

Keep prepared cache ownership and runtime workspace ownership separate. The prepared cache is keyed by backend recipe and version so compatible launches can reuse prepared artifacts safely. The runtime workspace is keyed by deployment and model instance so `ServeManager` can keep logs, ports, runtime manifests, and process state isolated for each direct-process instance.

Bootstrap remains preparatory only. It does not replace `ServeManager`, it does not own runtime lifecycle state, and it does not become the runtime control plane. `ServeManager` and `process_registry` remain the runtime authority for direct-process serving.

Host bootstrap is a separate control path for operators who need audited host preparation. It is disabled by default, Linux-only, dry-run capable, allowlisted by recipe source, and hash-pinned for declared inputs. Normal direct-process serving does not silently mutate the host.

Current backend matrix in this phase:

- distributed direct-process support for `vLLM`, `SGLang`, and `MindIE` only
- single-worker recipe-backed parity for `llama.cpp`, `VoxBox`, and `custom backend`
- the distributed direct-process subset still uses the existing worker-wide enablement gate, and its current config and env naming remain legacy `vLLM`-oriented names

Current limits and non-goals:

- Linux worker only
- container mode remains supported alongside this worker option, but not mixed on the same worker
- distributed direct-process for `llama.cpp`, `VoxBox`, and `custom backend` is out of scope in this phase and must fail closed instead of degrading or implying broader support
- benchmark direct-process runs are out of scope
- `custom backend` does not imply support for every external backend, only for backends that satisfy the documented contract

When this mode is enabled, the worker still needs the expected host-side prerequisites. That includes the backend executable for the selected direct-process path, the required Python and GPU runtime libraries, writable worker directories, and the localhost ports needed for readiness checks.

Direct-process instances use file logs as the canonical log source at `gpustack/logs/serve/<model-instance-id>.log`. On worker restart, this fork uses a cleanup-and-recreate policy. It kills stale direct-process entries and starts clean instead of trying to reattach to an existing serving process.

### Install GPUStack

Run the following command to install and start the GPUStack server using Docker:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

<details>
<summary>Alternative: Use Quay Container Registry Mirror</summary>

If you cannot pull images from `Docker Hub` or the download is very slow, you can use our `Quay.io` mirror by pointing your registry to `quay.io`:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    quay.io/gpustack/gpustack \
    --system-default-container-registry quay.io
```
</details>

Check the GPUStack startup logs:

```bash
sudo docker logs -f gpustack
```

After GPUStack starts, run the following command to get the default admin password:

```bash
sudo docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

Open your browser and navigate to `http://your_host_ip` to access the GPUStack UI. Use the default username `admin` and the password you retrieved above to log in.

### Set Up a GPU Cluster

1. On the GPUStack UI, navigate to the `Clusters` page.

2. Click the `Add Cluster` button.

3. Select `Docker` as the cluster provider.

4. Fill in the `Name` and `Description` fields for the new cluster, then click the `Save` button.

5. Follow the UI guidelines to configure the new worker node. You will need to run a Docker command on the worker node to connect it to the GPUStack server. The command will look similar to the following:

    ```bash
    sudo docker run -d --name gpustack-worker \
          --restart=unless-stopped \
          --privileged \
          --network=host \
          --volume /var/run/docker.sock:/var/run/docker.sock \
          --volume gpustack-data:/var/lib/gpustack \
          --runtime nvidia \
          gpustack/gpustack \
          --server-url http://your_gpustack_server_url \
          --token your_worker_token \
          --advertise-address 192.168.1.2
    ```

6. Execute the command on the worker node to connect it to the GPUStack server.

7. After the worker node connects successfully, it will appear on the `Workers` page in the GPUStack UI.

### Outbound worker control rollout

This rollout is scoped to a single GPUStack server. It does not add HA routing, shared websocket session state, or cross-server failover.

Use `GPUSTACK_WORKER_CONTROL_ROLLOUT_MODE` on the worker to choose one of three rollout paths:

- `legacy_only` disables outbound websocket control and keeps the legacy reverse-probe path.
- `hybrid` allows outbound websocket control sessions, but keeps `reverse_probe` as the default reachability mode for websocket-capable workers.
- `ws_preferred` keeps the same registration and status flow, but changes the default reachability mode for websocket-capable workers to `outbound_control_ws`.

Rollback is a worker config change, not a migration. Set `GPUSTACK_WORKER_CONTROL_ROLLOUT_MODE=legacy_only`, restart the worker, and it returns to the legacy reverse-probe path.

`GPUSTACK_WORKER_DEFAULT_REACHABILITY_MODE` still works as a compatibility override, but new rollouts should prefer `GPUSTACK_WORKER_CONTROL_ROLLOUT_MODE`.

Use `GPUSTACK_WORKER_REVERSE_HTTP_ENABLED=false` on websocket-capable NAT workers that cannot serve server-initiated reverse HTTP routes. Leave it at the default `true` for websocket workers that still expose those reverse HTTP paths.

Reverse-only NAT limits still apply. A websocket-only worker can receive outbound control commands, but server-initiated reverse HTTP operations such as worker filesystem checks, worker log fetches, and OpenAI reverse proxy hops still need reverse HTTP reachability. Those operations remain unsupported for reverse-only NAT workers.

Distributed runtime behavior also stays fail closed. If peer workers still need direct reachability and that reachability is missing, GPUStack must reject the placement instead of treating websocket control as a substitute for runtime data-plane connectivity.

For timeout tuning, adjust `GPUSTACK_WORKER_CONTROL_WS_HEARTBEAT_TIMEOUT_SECONDS` and `GPUSTACK_WORKER_CONTROL_SESSION_LOSS_TIMEOUT_SECONDS` together. Keep the session-loss timeout at or above the heartbeat timeout, then raise both only enough to avoid false stale-session transitions on slower links.

Fork-only direct-process workflow for a worker that is already running in a Linux container:

1. Start the worker with `GPUSTACK_DIRECT_PROCESS_MODE=true` in its environment.
2. Make sure the worker environment already has the backend executable you plan to use, such as `vllm`, `sglang`, `mindie`, `vox-box`, `llama-server`, or a `custom backend` command that matches the documented contract, plus the required host GPU libraries.
3. Keep the worker on the normal container-based registration path so it still connects to the GPUStack server in the usual way.
4. Let bootstrap prepare or reuse the recipe-backed prepared cache for that backend and version, then let `ServeManager` launch the instance in its own runtime workspace.
5. Deploy direct-process workloads only for the backends and topologies that worker advertises. Distributed direct-process covers `vLLM`, `SGLang`, and `MindIE` only. `llama.cpp`, `VoxBox`, and `custom backend` remain single-worker only in this phase, and unsupported distributed requests must fail closed.
   That distributed subset still depends on the existing worker-wide distributed direct-process enablement switch, even though its current config and env name is legacy `vLLM` naming.
6. Use host bootstrap only through its explicit operator control path when you need audited host preparation. It stays default-off, supports dry-run, and rejects recipe sources or inputs that are not allowlisted and hash-pinned.
7. Read direct-process serve logs from `gpustack/logs/serve/<model-instance-id>.log` if launch or readiness fails.

### Deploy a Model

1. Navigate to the `Catalog` page in the GPUStack UI.

2. Select the `Qwen3 0.6B` model from the list of available models.

3. After the deployment compatibility checks pass, click the `Save` button to deploy the model.

![deploy qwen3 from catalog](docs/assets/quick-start/quick-start-qwen3.png)

4. GPUStack will start downloading the model files and deploying the model. When the deployment status shows `Running`, the model has been deployed successfully.

![model is running](docs/assets/quick-start/model-running.png)

5. Click `Playground - Chat` in the navigation menu, check that the model `qwen3-0.6b` is selected from the top-right `Model` dropdown. Now you can chat with the model in the UI playground.

![quick chat](docs/assets/quick-start/quick-chat.png)

### Use the model via API

1. Hover over the user avatar and navigate to the `API Keys` page, then click the `New API Key` button.

2. Fill in the `Name` and click the `Save` button.

3. Copy the generated API key and save it somewhere safe. Please note that you can only see it once on creation.

4. You can now use the API key to access the OpenAI-compatible API endpoints provided by GPUStack. For example, use curl as the following:

```bash
# Replace `your_api_key` and `your_gpustack_server_url`
# with your actual API key and GPUStack server URL.
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Tell me a joke."
      }
    ],
    "stream": true
  }'
```

## Documentation

Please see the [official docs site](https://docs.gpustack.ai) for complete documentation.

## Build

1. Install Python (version 3.10 to 3.12).

2. Run `make build`.

You can find the built wheel package in `dist` directory.

## Contributing

Please read the [Contributing Guide](./docs/contributing.md) if you're interested in contributing to GPUStack.

## Join Community

Any issues or have suggestions, feel free to join our [Community](https://discord.gg/VXYJzuaqwD) for support.

## License

Copyright (c) 2024-2025 The GPUStack authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at [LICENSE](./LICENSE) file for details.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
