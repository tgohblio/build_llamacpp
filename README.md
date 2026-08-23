# build_llamacpp
Build a custom llamacpp (server and cli) using Github Runner

## What This Workflow Does

A GitHub-hosted runner (`ubuntu-24.04`) calls the Runpod REST v2 API to:

1. Create a CPU-only pod with 8 vCPUs, 16 GB RAM, an 80 GB container disk,
   and SSH exposed on `22/tcp`.
2. Boot it with a self-hosted GitHub Actions runner. Creating the pod starts it automatically.
3. Run the build on that pod with `runs-on: [self-hosted, runpod]`.
4. Terminate the pod once the artifacts are uploaded.

### CUDA-Enabled Build on a CPU Instance

The pod image (`RUNPOD_CONTAINER_IMAGE`) is a custom image built from
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

Then set `RUNPOD_CONTAINER_IMAGE` (or the `runpod_image` dispatch input) to the
image.

## Prerequisites

- Set `RUNPOD_API_KEY` in the repository's **Settings > Secrets and variables**.
	The workflow sends it as a bearer token to the Runpod REST v2 API.
- Allow this repository's workflows to use self-hosted runners. In **Settings >
	Actions > General**, the self-hosted runner policy must permit repository runners.

## Resource Notes

- The workflow uses the `cpu5g` CPU flavor at 8 vCPUs, which provides 16 GB RAM.
- The CUDA development image is large, so the workflow defaults to an 80 GB
	container disk.
