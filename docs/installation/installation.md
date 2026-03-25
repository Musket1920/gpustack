# Installation

## Prerequisites

**GPUStack server:**

- [Docker](https://docs.docker.com/engine/install/) must be installed. Docker Desktop (Windows and macOS) is also supported.

**GPUStack workers:**

- [Docker](https://docs.docker.com/engine/install/) must be installed. Docker Desktop is **not** supported.
- Only Linux is supported for GPUStack worker nodes. If you use Windows, consider using WSL2 and avoid using Docker Desktop. macOS is not supported for GPUStack worker nodes.
- Ensure the appropriate GPU drivers and container toolkits are installed for your hardware. See the [Installation Requirements](./requirements.md) for details.

!!! note

    Container deployment is the default and recommended GPUStack installation path.

!!! warning

    This fork adds a direct-process worker mode for a narrow operator workflow. It is not an official upstream GPUStack feature.

## Fork-only direct-process worker mode

Use this mode only when a worker already runs inside a Linux container or directly on a Linux host and you want GPUStack to launch model servers as direct processes instead of creating nested model containers.

Enable it on the worker with `GPUSTACK_DIRECT_PROCESS_MODE=true`.

Container deployment is still supported and remains the default path. This direct-process mode is fork-only, and the runtime choice stays worker-global for now. A worker runs in container mode or direct-process mode, not both at once.

Recipe-backed provisioning flow in this phase:

- bootstrap resolves the backend recipe and prepared-environment identity for the selected backend version
- bootstrap prepares or reuses a prepared cache that contains the consumed artifacts for launch, including `prepared.env`, `prepared-config.json`, `prepared-launch.sh`, executable provenance, and provisioning audit data
- `ServeManager` then launches the model instance from that prepared state into a per-instance runtime workspace
- `process_registry` remains the runtime record for active direct-process state

Supported scope in this phase:

- Linux only
- distributed direct-process for `vLLM`, `SGLang`, and `MindIE` only
- single-worker recipe-backed parity for `llama.cpp`, `VoxBox`, and `custom backend`

The distributed direct-process subset still uses the existing worker-wide enablement gate. Its current config and env naming are legacy `vLLM`-oriented names.

Unsupported scope:

- distributed direct-process for `llama.cpp`, `VoxBox`, and `custom backend`, which must fail closed instead of degrading into single-worker behavior
- benchmark direct-process runs
- undocumented upstream parity or support claims

Capability and contract assumptions:

- workers advertise which direct-process backends and topologies they support
- scheduling is expected to place direct-process deployments only on workers that advertise the required backend and topology support
- `custom backend` is the community path for backends that can be described with command, env, health, timeout, workdir, and stop controls
- `custom backend` does not mean every external backend works automatically

Host prerequisites for this fork mode:

- the backend executable must already be installed in the worker environment, for example `vllm`, `sglang`, `mindie`, `vox-box`, `llama-server`, or the executable used by a `custom backend`
- required GPU drivers, runtime libraries, and Python dependencies must already be available on the host or inside the worker container
- worker-managed directories must be writable
- localhost ports required by readiness checks must be available

`bootstrap` remains preparatory only. It prepares recipe-backed artifacts and the prepared cache for a backend version, but it does not replace `ServeManager`, it does not own runtime process lifecycle state, and it does not act as the runtime controller for direct-process serving.

Keep prepared cache ownership and runtime workspace ownership separate when you operate workers. The prepared cache is keyed per backend recipe and version so the worker can reuse prepared artifacts safely. The runtime workspace is keyed per deployment and model instance so runtime files, logs, and process state stay isolated under `ServeManager` control.

## Host bootstrap boundaries

Host bootstrap is a separate operator control path. It is not part of the normal serve path.

- host bootstrap is disabled by default
- only Linux workers are eligible
- `dry-run` is supported so operators can inspect planned host changes before any mutation
- recipe sources must be allowlisted
- declared inputs must be hash-pinned
- audit records are written for planned, rejected, dry-run, and completed execution states

This means direct-process serving does not silently install packages or mutate the host during normal launch. Legacy container-based worker startup stays unchanged, and the legacy worker-wide distributed direct-process gate still applies to the supported distributed direct-process subset.

Operator flow for a worker that is already running inside a Linux container:

1. Start the worker with `GPUSTACK_DIRECT_PROCESS_MODE=true`.
2. Confirm the worker environment can run the backend command you plan to use before registering the worker.
3. Register the worker with the GPUStack server through the normal worker startup flow.
4. Let bootstrap prepare or reuse the recipe-backed prepared cache, then let `ServeManager` launch into the model instance runtime workspace.
5. Deploy direct-process workloads only for the backends and topologies that worker advertises. Distributed direct-process is limited to `vLLM`, `SGLang`, and `MindIE`. `llama.cpp`, `VoxBox`, and `custom backend` remain single-worker only, and unsupported distributed requests must fail closed.
   That distributed subset still depends on the existing worker-wide distributed direct-process enablement switch, even though its current config and env name are legacy `vLLM` naming.
6. Use host bootstrap only through its explicit control path when you need audited preparation, and prefer `dry-run` first.
7. Check `gpustack/logs/serve/<model-instance-id>.log` for launch and runtime logs.

Restart behavior is cleanup-and-recreate only. On worker restart, this fork cleans stale direct-process entries and starts a new process when needed. It does not reattach to an already running serving process.

## Install GPUStack Server

Run the following command to install and start the GPUStack server using Docker:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

!!! note

    GPUStack v2 uses a single unified container image for all GPU device types.

## Startup

Check the GPUStack container logs:

```bash
sudo docker logs -f gpustack
```

If everything is normal, open `http://your_host_ip` in a browser to access the GPUStack UI.

Log in with username `admin` and the default password. Retrieve the initial password with:

```bash
sudo docker exec -it gpustack \
    cat /var/lib/gpustack/initial_admin_password
```

## Add GPU Clusters and Worker Nodes

Please follow the UI instructions on the `Clusters` and `Workers` pages to add GPU clusters and worker nodes.

## Custom Configuration

The following sections describe examples of custom configuration options when starting the GPUStack server container. For a full list of available options, refer to the [CLI Reference](../cli-reference/start.md).

### Enable HTTPS with Custom Certificate


```diff
 sudo docker run -d --name gpustack \
     ...
     -p 80:80 \
+    -p 443:443 \
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/cert_files:/path/to/cert_files:ro \
+    -e GPUSTACK_SSL_KEYFILE=/path/to/cert_files/your_domain.key \
+    -e GPUSTACK_SSL_CERTFILE=/path/to/cert_files/your_domain.crt \
     gpustack/gpustack
     ...
```

### Using an External Database

By default, GPUStack uses an embedded PostgreSQL database. To use an external database such as PostgreSQL or MySQL, set the `GPUSTACK_DATABASE_URL` environment variable or use the `--database-url` argument when starting the GPUStack container:

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    -e GPUSTACK_DATABASE_URL="postgresql://username:password@host:port/dbname" \
     gpustack/gpustack
     ...
```

### Configure External Server URL

If you use a cloud provider to provision workers, set the external server URL for worker registration to ensure that workers can connect to the server correctly.

```diff
sudo docker run -d --name gpustack \
    ...
+   -e GPUSTACK_SERVER_EXTERNAL_URL="https://your_external_server_url" \
    gpustack/gpustack
    ...
```

## Installation via Docker Compose

### Prerequisites

- [Docker Compose](https://docs.docker.com/compose/install/) must be installed.
- [Required ports](./requirements.md#port-requirements) must be available.

### Deployment

The Docker Compose files and configuration files are maintained in the [GPUStack repository](https://github.com/gpustack/gpustack/tree/main/docker-compose).

Run the following commands to clone the latest stable release:

```bash
LATEST_TAG=$(
    curl -s "https://api.github.com/repos/gpustack/gpustack/releases" \
    | grep '"tag_name"' \
    | sed -E 's/.*"tag_name": "([^"]+)".*/\1/' \
    | grep -Ev 'rc|beta|alpha|preview' \
    | head -1
)
echo "Latest stable release: $LATEST_TAG"
git clone -b "$LATEST_TAG" https://github.com/gpustack/gpustack.git
cd gpustack/docker-compose
```

Start the GPUStack server:

```bash
sudo docker compose -f docker-compose.server.yaml up -d
```

If everything is normal, open `http://your_host_ip` in a browser to access the GPUStack UI.

Log in with username `admin` and the default password. Retrieve the initial password with:

```bash
sudo docker exec -it gpustack-server cat /var/lib/gpustack/initial_admin_password
```

For built-in and external observability options, see [Observability](../user-guide/observability.md).
