# build_llamacpp
Build a custom llamacpp (server and cli) using Github Runner

This repository provides GitHub Actions workflows that compile `llama.cpp`
with CUDA support on a CPU-only machine and package the result into a
Runpod serverless Docker image. Both build workflows produce the same artifacts
(`llama-server`, `llama-cli`, `llama-quantize`) and accept the same manual
dispatch inputs for `cuda_architectures` and an optional `pr_number` to build
from a pull request instead of master.

## Workflows

### `ghr-build` — GitHub-Hosted Runner

[`.github/workflows/ghr-build.yaml`](.github/workflows/ghr-build.yaml)

Runs entirely on a GitHub-hosted `ubuntu-24.04` runner inside an
`nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` container. The workflow:

1. Installs build dependencies (`cmake`, `ninja-build`, `ccache`, etc.) inside
   the container.
2. Checks out `ggml-org/llama.cpp` (master, or a specific PR).
3. Configures and builds with `GGML_CUDA=ON` using Ninja.
4. Strips debug symbols and uploads the binaries as artifacts.

No external infrastructure is required — the CUDA toolkit is provided by the
container image. The workflow also triggers on pushes and PRs that modify its
own file.

### `build-flow` — Runpod Self-Hosted Runner

[`.github/workflows/build-flow.yaml`](.github/workflows/build-flow.yaml)

Uses a three-job pipeline (provision → build → teardown) that spins up a
Runpod CPU pod as a self-hosted GitHub Actions runner:

1. **Provision** — A GitHub-hosted controller calls the Runpod REST v2 API to
   query the CPU catalog, select a flavor matching the configured vCPU/RAM,
   create a CPU-only pod, and wait for the runner to register.
2. **Build** — Runs on the Runpod pod (`runs-on: [self-hosted, runpod]`).
   Checks out `ggml-org/llama.cpp`, configures with `GGML_CUDA=ON`, builds
   with Ninja, strips the binaries, and uploads artifacts.
3. **Teardown** — Terminates the pod via REST v2 regardless of build outcome.

The pod image (`RUNPOD_DOCKER_IMAGE`) is a custom image built from
`nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` with the GitHub Actions runner
layered on top. See [`self_runner/Dockerfile`](self_runner/Dockerfile) and
[`self_runner/runner-entrypoint.sh`](self_runner/runner-entrypoint.sh).

### CUDA-Enabled Build on a CPU Instance

Because the CUDA toolkit and cuDNN are already included in both container
images, compiling `llama.cpp` with `GGML_CUDA=ON` requires no GPU and no
runtime CUDA installation. Only `nvcc` is needed during compilation, and it is
already present. The produced binaries require a GPU only at runtime.

### `ghr-build-deploy-image` — Runpod Serverless Deployment

[`.github/workflows/ghr-build-deploy-image.yaml`](.github/workflows/ghr-build-deploy-image.yaml)

Builds and pushes the Runpod serverless Docker image. Triggered automatically
after a successful `ghr-build` or `build-flow` run, or manually with a run ID.
The workflow:

1. Downloads the `llama_apps` artifact from the specified (or triggering) build
   workflow run.
2. Bundles `llama-server` with the Runpod handler scripts from
   [`llama_runpod/`](llama_runpod/).
3. Builds and pushes the Docker image to Docker Hub.

The resulting image starts `llama-server` with DFlash 2 speculative decoding
and runs a Runpod serverless handler that forwards requests to llama-server's
OpenAI-compatible API. Runtime configuration (model, context size, etc.) is
set via environment variables on the Runpod endpoint.

| Env Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `MODEL` | yes | — | HF model, e.g. `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M` |
| `DRAFT_MODEL` | yes | — | HF draft model for speculative decoding |
| `N_GPU_LAYERS` | no | `99` | GPU layers to offload |
| `CTX_SIZE` | no | `8192` | Context size |
| `PARALLEL` | no | `1` | Number of parallel sequences |
| `PORT` | no | `8080` | llama-server listen port |
| `SPEC_TYPE` | no | `draft-dflash` | Speculative decoding type |
| `SPEC_DRAFT_N_MAX` | no | `7` | Max draft tokens per step |

## One-Time Setup

Build and push the runner image:

```sh
docker build -f self_runner/Dockerfile -t dockerdl2018/llama-builder:cuda13 ./self_runner
docker push dockerdl2018/llama-builder:cuda13
```

The workflow uses `dockerdl2018/llama-builder:cuda13` by default. Override the
image with the `runpod_image` manual-dispatch input when needed.

## Prerequisites

`ghr-build` works out of the box on any repository with GitHub Actions enabled.

`build-flow` additionally requires:

- `RUNPOD_API_KEY` in the repository's **Settings > Secrets and variables > Actions**.
- A **GitHub App** with permissions stated in [Github App Permissions](#github-app-permissions) section below.
  Store `APP_CLIENT_ID` as a repository **variable** and `APP_PRIVATE_KEY` as a repository **secret**.\
  Refer how to setup here: [Making authenticated API requests with a GitHub App.](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/making-authenticated-api-requests-with-a-github-app-in-a-github-actions-workflow) 
- Self-hosted runners allowed in **Settings > Actions > General**.

`ghr-build-deploy-image` additionally requires:

- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` as repository **secrets** for pushing images.
- The same `APP_CLIENT_ID` / `APP_PRIVATE_KEY` GitHub App credentials (used to
  download artifacts from other workflow runs).

### Github App Permissions

| Workflow | Administration | Actions | Contents | Why |
| --- | --- | --- | --- | --- |
| `ghr-build.yaml` | ❌No | ❌No | ❌No | Builds code and uploads artifacts using the default `GITHUB_TOKEN`. No GitHub App needed. |
| `ghr-build-deploy-image.yaml` | ❌No | ✅Read & write | ✅Read | Downloads artifacts from other workflow runs and builds/pushes Docker images. |
| `build-flow.yaml` | ✅Write | ❌No | ✅Read | Creates self-hosted runners (requires admin API access) and checks out the repository. |

#### Quick Setup Checklist

✅ Administration - Read and write\
✅ Actions - Read and write\
✅ Contents - Read only\
This configuration will satisfy all three workflows. You can safely leave other permissions disabled.

## Runpod Configuration (llama build)

The following workflow-level environment variables define the pod resources:

| Variable | Default | Description |
| --- | ---: | --- |
| `RUNPOD_VCPU_COUNT` | `8` | Number of vCPUs |
| `RUNPOD_RAM_GB` | `16` | Required RAM in GB |
| `RUNPOD_DISK_GB` | `20` | Container disk in GB |
| `RUNPOD_SSH_PORT` | `22` | Published SSH TCP port |
| `RUNPOD_SSH_ENABLED` | `false` | Whether SSH is provisioned and published |
| `RUNPOD_DOCKER_IMAGE` | `dockerdl2018/llama-builder:cuda13` | Runner image |

RAM is derived by Runpod from the selected CPU flavor. The workflow fails
before pod creation if no catalog flavor matches both `RUNPOD_VCPU_COUNT` and
`RUNPOD_RAM_GB`. When SSH is enabled, the account must have a registered SSH
key and the selected image must support Runpod's SSH startup convention.

## Resource Notes

- The default pod uses 8 vCPUs, 16 GB RAM, and an 20 GB container disk.
- The workflow builds on CPU because the CUDA toolkit is baked into the image;
  the resulting binaries require a GPU only at runtime.

## Runpod Configuration (llama serving)

TBD
