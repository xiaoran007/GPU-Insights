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

RUN if [ -f /usr/local/cuda/lib64/stubs/libcuda.so ] \
        && [ ! -e /usr/local/cuda/lib64/stubs/libcuda.so.1 ]; then \
      ln -s libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1; \
    fi

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

WORKDIR /work
