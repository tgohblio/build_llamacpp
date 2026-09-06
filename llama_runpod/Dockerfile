# Runpod serverless worker for llama.cpp with DFlash 2 speculative decoding.
#
# Build:
#   docker build --platform linux/amd64 -t <your registry>/llama-runpod:latest .
#

FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=https://github.com/ggml-org/llama.cpp
ARG IMAGE_SOURCE=https://github.com/ggml-org/llama.cpp
LABEL org.opencontainers.image.created=$BUILD_DATE \
    org.opencontainers.image.version=$APP_VERSION \
    org.opencontainers.image.revision=$APP_REVISION \
    org.opencontainers.image.title="llama.cpp" \
    org.opencontainers.image.description="LLM inference in C/C++" \
    org.opencontainers.image.url=$IMAGE_URL \
    org.opencontainers.image.source=$IMAGE_SOURCE

# Expose port so healthcheck and external requests work
EXPOSE 8080

# Install python, runpod for Runpod handler, and requests for HTTP calls.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 curl ffmpeg unzip \
    && pip3 install --break-system-packages --no-cache-dir "runpod>=1.12.0" requests \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

# Create the folder, copy the zip into the container, unzip it
WORKDIR /app

# COPY everything from the 'staging' folder to '/app'
# Github action will put the python and bash scripts into the staging folder
COPY staging/ .

RUN chmod +x /app/start_llama.sh /app/llama-server

# start_llama.sh launches llama-server, then runs the handler
ENTRYPOINT ["/app/start_llama.sh"]
