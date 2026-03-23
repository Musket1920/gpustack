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

Supported scope in Part 2:

- Linux only
- single-worker direct-process for built-in `vLLM`, `SGLang`, `MindIE`, and `VoxBox`
- single-worker direct-process for first-class `llama.cpp`
- generic `custom backend` direct-process for community backends that fit the documented contract
- distributed direct-process for `vLLM` only

Unsupported scope:

- non-`vLLM` distributed direct-process
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

Operator flow for a worker that is already running inside a Linux container:

1. Start the worker with `GPUSTACK_DIRECT_PROCESS_MODE=true`.
2. Confirm the worker environment can run the backend command you plan to use before registering the worker.
3. Register the worker with the GPUStack server through the normal worker startup flow.
4. Deploy direct-process workloads only for the backends and topologies that worker advertises. In Part 2, distributed direct-process is `vLLM`-only.
5. Check `gpustack/logs/serve/<model-instance-id>.log` for launch and runtime logs.

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
