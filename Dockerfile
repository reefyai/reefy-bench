# reefy-bench - CPU / memory / disk / GPU stress tests.
#
# Two-stage build:
#   1. rust toolchain compiles gpu-fryer (Rust binary from HuggingFace)
#      so the final image doesn't carry cargo/rustc.
#   2. CUDA runtime base (so gpu-fryer's CUDA matmul kernel has its
#      runtime libs available) plus sysbench, fio, Flask, gunicorn.
#
# The CUDA base is ~1.6 GB but it's a one-time pull and lets the image
# work on every device class - on non-NVIDIA hosts the GPU card just
# hides itself, sysbench and fio keep working.

# ── stage 1: build gpu-fryer ──────────────────────────────────────
# gpu-fryer links directly against libcuda/cudart/cublas/cublasLt/curand/
# nvrtc, so the builder needs the CUDA toolkit. The devel image ships
# the stub libcuda.so the linker needs (the real driver lib is supplied
# by CDI at runtime); runtime libs (cudart/cublas/...) are also here.
# This stage is discarded after COPY - the published image stays runtime-
# only. Rust 1.86+ is required for gpu-fryer's edition2024 deps.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS fryer
ENV DEBIAN_FRONTEND=noninteractive \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64 \
    PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential pkg-config libssl-dev \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain 1.86.0 --profile minimal
RUN cargo install --git https://github.com/huggingface/gpu-fryer \
        --locked --root /opt

# ── stage 2: runtime ──────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        sysbench fio \
        python3 python3-flask python3-gunicorn \
        pciutils procps ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=fryer /opt/bin/gpu-fryer /usr/local/bin/gpu-fryer

WORKDIR /app
COPY server.py /app/
COPY templates /app/templates/
COPY static /app/static/

EXPOSE 8500

# 2 worker procs × 8 threads each is plenty for polling-driven UI;
# spawning subprocesses is what the workers mostly do.
CMD ["gunicorn", "-b", "0.0.0.0:8500", \
     "-w", "2", "-k", "gthread", "--threads", "8", \
     "--access-logfile", "-", "server:app"]
