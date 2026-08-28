# Llama runtime image with llama-server
#
# Run the command :
#  docker build \
#  --build-arg BUILD_DATE=date(optional) \
#  --build-arg LLAMA_ZIP_FILE=llama_apps.zip \
#  --platform linux/amd64 \
#  -t dockerdl2018/llama-runner:cuda13 .
#

FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# llama server, cli files in compressed zip
ARG LLAMA_ZIP_FILE="dummy.zip"

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

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl ffmpeg unzip \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

# Create the folder, copy the zip into the container, unzip it
WORKDIR /app
COPY $LLAMA_ZIP_FILE .
RUN unzip -o $LLAMA_ZIP_FILE \
    && rm $LLAMA_ZIP_FILE ./llama-cli ./llama-quantize


# Expose port so healthcheck and external requests work
EXPOSE 8080

# Server, Server only
ENV LLAMA_ARG_HOST=0.0.0.0

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
