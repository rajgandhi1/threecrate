# Single Linux environment to benchmark PCL, Open3D, and ThreeCrate on the SAME
# machine/OS/compiler, so the cross-library comparison is apples-to-apples.
#
# Build (context = this directory, which is small — avoids shipping the repo):
#   docker build -t threecrate-bench scripts/pcl_bench
#
# Run (bind-mount the repo and the datasets; nothing is copied):
#   docker run --rm \
#     -v "<repo>":/repo -v "<downloads>":/data:ro \
#     -e CARGO_TARGET_DIR=/tmp/target -w /repo threecrate-bench \
#     python3 scripts/bench_cross_library.py --pcl-bench-exe /opt/pcl_bench/build/pcl_bench ...
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config curl ca-certificates git \
        libpcl-dev libopenmpi-dev \
        python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Open3D + numpy in an isolated venv.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir open3d numpy

# Rust toolchain (for building the ThreeCrate benchmark binary in-container).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

ENV PATH="/root/.cargo/bin:/opt/venv/bin:${PATH}"

# Build the PCL benchmark executable into the image.
COPY . /opt/pcl_bench
RUN cmake -S /opt/pcl_bench -B /opt/pcl_bench/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/pcl_bench/build -j "$(nproc)"

ENV PCL_BENCH_EXE=/opt/pcl_bench/build/pcl_bench
