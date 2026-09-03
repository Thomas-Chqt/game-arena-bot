FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GAME_ARENA_SDK_URL=https://game-arena-assets.irvine.jp/sdk/latest/game-arena-c-sdk.zip
ARG MLX_REF=v0.32.0
# Build without requiring BuildKit to have access to a GPU. These cover the
# T4, A100, RTX 30/A40, L40/RTX 4090, H100, and RTX 5090 classes commonly on Vast.
ARG MLX_CUDA_ARCHITECTURES="75;80;86;89;90;120"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libblas-dev \
        liblapack-dev \
        liblapacke-dev \
        ninja-build \
        pkg-config \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# The C SDK contains its TLS, JSON, and WebSocket dependencies, so it needs no
# additional system libraries.  Install it where CMake will find arena::arena.
RUN curl --fail --location --silent --show-error "$GAME_ARENA_SDK_URL" --output /tmp/game-arena-c-sdk.zip \
    && unzip -q /tmp/game-arena-c-sdk.zip -d /tmp/game-arena-c-sdk \
    && sdk_root=$(find /tmp/game-arena-c-sdk -path '*/sdk/c/CMakeLists.txt' -exec dirname {} \; -quit) \
    && if [ -z "$sdk_root" ]; then sdk_root=$(find /tmp/game-arena-c-sdk -maxdepth 2 -name CMakeLists.txt -exec dirname {} \; -quit); fi \
    && test -n "$sdk_root" \
    && cmake -S "$sdk_root" -B /tmp/game-arena-c-sdk/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/arena \
    && cmake --build /tmp/game-arena-c-sdk/build \
    && cmake --install /tmp/game-arena-c-sdk/build \
    && rm -rf /tmp/game-arena-c-sdk /tmp/game-arena-c-sdk.zip

# MLX is built with the CUDA backend because the project's CMake target links
# its C++ library directly.  Build it once in the image rather than requiring
# a Python wheel or an Apple-only Metal installation on the Vast instance.
RUN git clone --branch "$MLX_REF" --depth 1 https://github.com/ml-explore/mlx.git /tmp/mlx \
    && cmake -S /tmp/mlx -B /tmp/mlx/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/mlx \
        -DMLX_BUILD_CUDA=ON \
        -DMLX_CUDA_ARCHITECTURES="$MLX_CUDA_ARCHITECTURES" \
        -DMLX_BUILD_EXAMPLES=OFF \
        -DMLX_BUILD_TESTS=OFF \
        -DMLX_BUILD_PYTHON_BINDINGS=OFF \
    && cmake --build /tmp/mlx/build \
    && cmake --install /tmp/mlx/build \
    && rm -rf /tmp/mlx

ENV CMAKE_GENERATOR=Ninja \
    CMAKE_PREFIX_PATH=/opt/arena:/opt/mlx \
    GAME_ARENA_SDK_ROOT=/opt/arena \
    MLX_ROOT=/opt/mlx \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/mlx/lib

# Ubuntu 24.04's default GCC 13 does not provide the C++23 <print> header used
# by the bot. Install GCC 14 after MLX so changing this toolchain cannot
# invalidate the expensive MLX build layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++-14 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-14 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-14 100 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY CMakeLists.txt ./
COPY cmake ./cmake
COPY amoeba ./amoeba
COPY *.safetensors ./

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DAMOEBA_BUILD_MLX_FROM_SOURCE=ON \
        -DAMOEBA_BIG_GPU_TRAINING=ON \
        -DBUID_AMOEBA_2=OFF \
    && cmake --build build

ENTRYPOINT ["bash"]

LABEL org.opencontainers.image.source=https://github.com/Thomas-Chqt/game-arena-bot
