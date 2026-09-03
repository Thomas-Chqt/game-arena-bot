ARG MLX_IMAGE=game-arena-bot:mlx-v0.32.2
FROM ${MLX_IMAGE} AS mlx-cuda

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS amoeba-cuda

ARG DEBIAN_FRONTEND=noninteractive
ARG GAME_ARENA_SDK_URL=https://game-arena-assets.irvine.jp/sdk/latest/game-arena-c-sdk.zip

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        g++-14 \
        libblas-dev \
        liblapack-dev \
        liblapacke-dev \
        ninja-build \
        pkg-config \
        unzip \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-14 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-14 100 \
    && rm -rf /var/lib/apt/lists/*

# The C SDK contains its TLS, JSON, and WebSocket dependencies, so it needs no
# additional system libraries. Install it where CMake will find arena::arena.
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

COPY --from=mlx-cuda /opt/mlx /opt/mlx

ENV CMAKE_GENERATOR=Ninja \
    CMAKE_PREFIX_PATH=/opt/arena:/opt/mlx \
    GAME_ARENA_SDK_ROOT=/opt/arena \
    MLX_ROOT=/opt/mlx \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/mlx/lib

WORKDIR /workspace

COPY CMakeLists.txt ./
COPY cmake/ ./cmake/
COPY amoeba/ ./amoeba/
COPY *.safetensors ./

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DAMOEBA_BUILD_MLX_FROM_SOURCE=OFF \
        -DAMOEBA_BIG_GPU_TRAINING=ON \
        -DBUID_AMOEBA_2=OFF \
    && cmake --build build

ENTRYPOINT ["bash"]

LABEL org.opencontainers.image.source=https://github.com/Thomas-Chqt/game-arena-bot
