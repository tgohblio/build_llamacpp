# build_llamacpp
Build a custom llamacpp (server and cli) using Github Runner

## What This Workflow Does

A GitHub-hosted controller (`ubuntu-24.04`) calls the Runpod REST v2 API to:

1. Query the CPU catalog and select a flavor that provides the configured vCPU
	count and RAM.
2. Create a CPU-only pod with the configured resources and container image.
3. Boot it with a self-hosted GitHub Actions runner. Creating the pod starts it automatically.
4. Run the build on that pod with `runs-on: [self-hosted, runpod]`.
5. Terminate the pod with REST v2 once the artifacts are uploaded or the build
	fails.

### CUDA-Enabled Build on a CPU Instance

The pod image (`RUNPOD_DOCKER_IMAGE`) is a custom image built from
`nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04`, with the GitHub Actions runner
layered on top. See [`self_runner/Dockerfile`](self_runner/Dockerfile) and
[`self_runner/runner-entrypoint.sh`](self_runner/runner-entrypoint.sh).

Because the CUDA toolkit and cuDNN are already included, compiling `llama.cpp`
with `GGML_CUDA=ON` requires no GPU and no runtime CUDA installation. Only
`nvcc` is needed during compilation, and it is already present. The produced
binaries require a GPU only at runtime.

## One-Time Setup

Build and push the runner image:

```sh
docker build -f self_runner/Dockerfile -t dockerdl2018/llama-runner:cuda13 .
docker push dockerdl2018/llama-runner:cuda13
```

The workflow uses `dockerdl2018/llama-runner:cuda13` by default. Override the
image with the `runpod_image` manual-dispatch input when needed.

## Prerequisites

- Set `RUNPOD_API_KEY` in the repository's **Settings > Secrets and variables**.
- Set `PERSONAL_ACCESS_TOKEN` in GitHub for permission to create the temporary
  repository runner registration token.
- Allow this repository's workflows to use self-hosted runners. In **Settings >
	Actions > General**, the self-hosted runner policy must permit repository runners.

## Runpod Configuration

The following workflow-level environment variables define the pod resources:

| Variable | Default | Description |
| --- | ---: | --- |
| `RUNPOD_VCPU_COUNT` | `8` | Number of vCPUs |
| `RUNPOD_RAM_GB` | `16` | Required RAM in GB |
| `RUNPOD_DISK_GB` | `80` | Container disk in GB |
| `RUNPOD_SSH_PORT` | `22` | Published SSH TCP port |
| `RUNPOD_SSH_ENABLED` | `false` | Whether SSH is provisioned and published |
| `RUNPOD_DOCKER_IMAGE` | `dockerdl2018/llama-runner:cuda13` | Runner image |

RAM is derived by Runpod from the selected CPU flavor. The workflow fails
before pod creation if no catalog flavor matches both `RUNPOD_VCPU_COUNT` and
`RUNPOD_RAM_GB`. When SSH is enabled, the account must have a registered SSH
key and the selected image must support Runpod's SSH startup convention.

## Resource Notes

- The default pod uses 8 vCPUs, 16 GB RAM, and an 80 GB container disk.
- The workflow builds on CPU because the CUDA toolkit is baked into the image;
  the resulting binaries require a GPU only at runtime.
