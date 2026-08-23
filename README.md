# build_llamacpp
Build a custom llamacpp (server and cli) using Github Runner

## What This Workflow Does

A GitHub-hosted controller (`ubuntu-24.04`) calls the Runpod API to:

1. Create the cheapest CPU-only pod with a configurable container disk.
2. Boot it with a self-hosted GitHub Actions runner. Creating the pod starts it automatically.
3. Run the build on that pod with `runs-on: [self-hosted, runpod]`.
4. Pause the pod with `runpodctl pod stop` once the artifacts are uploaded.

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
	`runpodctl` reads the API key from the environment.
- Set `PERSONAL_ACCESS_TOKEN` in Github for read/write permissions for `gh cli` usage.
- Allow this repository's workflows to use self-hosted runners. In **Settings >
	Actions > General**, the self-hosted runner policy must permit repository runners.

## Resource Notes

- The cheapest CPU pod (2 vCPU / approximately 4 GB RAM) may run out of memory
	while `nvcc` compiles `llama.cpp`. If the build is killed, use a larger CPU
	flavor and/or increase `RUNPOD_DISK_GB`.
- The CUDA development image is large, so 20 GB is tight once the source and
	build artifacts are added. Prefer 40-60 GB.
