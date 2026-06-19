ARG CUDA_IMAGE
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        file \
        git \
        ninja-build \
        rsync \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
