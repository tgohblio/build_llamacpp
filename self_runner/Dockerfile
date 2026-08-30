### Docker image for building llama.cpp using RunPod pods ###
#
# Built FROM the CUDA 13.0.2 + cuDNN devel image so llama.cpp can be compiled
# with GGML_CUDA=ON on a GPU-less CPU pod -- the CUDA toolkit is already baked
# in, so the workflow does NOT need to "Install CUDA toolkit" at runtime.
#
# Build & push ONCE (replace <your-registry> with your Docker Hub / GHCR namespace):
#   docker build -f self_runner/Dockerfile -t <your-registry>/llama-builder:cuda13 .
#   docker push <your-registry>/llama-builder:cuda13
#
# Then point the workflow at it via RUNPOD_DOCKER_IMAGE
# (or the runpod_image workflow_dispatch input).
FROM --platform=linux/amd64 nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV RUNNER_VERSION=2.336.0 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# Install runner dependencies AND build toolchain once during image build
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git jq tar python3-pip \
        ninja-build build-essential libssl-dev libcurl4-openssl-dev ccache file \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages --upgrade "cmake>=3.30" \
    && mkdir -p /runner && cd /runner \
    && curl -sSL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" -o r.tgz \
    && tar xzf r.tgz && rm r.tgz \
    && ./bin/installdependencies.sh

COPY self_runner/runner-entrypoint.sh /runner/runner-entrypoint.sh
RUN chmod +x /runner/runner-entrypoint.sh \
    && useradd --uid 1001 --create-home --user-group --shell /bin/bash runner \
    && chown -R runner:runner /runner

WORKDIR /runner
ENV HOME=/home/runner
USER runner
ENTRYPOINT ["/runner/runner-entrypoint.sh"]
