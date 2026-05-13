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
        sysbench fio stress-ng \
        python3 python3-flask python3-gunicorn \
        pciutils procps ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=fryer /opt/bin/gpu-fryer /usr/local/bin/gpu-fryer

WORKDIR /app
COPY server.py workload.py /app/
COPY templates /app/templates/
COPY static /app/static/

EXPOSE 8500

# Single worker is intentional: the job registry is an in-process
# dict, so a poll landing on a different worker than the one that
# spawned the job 404s. `--threads 8` gives enough HTTP concurrency
# (each subprocess streams its own stdout from a daemon thread, not
# the worker thread), and one bench container is one user anyway.
# `python3 -m gunicorn` instead of the bare binary - Ubuntu 24.04's
# python3-gunicorn package no longer installs /usr/bin/gunicorn, but
# the Python module is always there.
CMD ["python3", "-m", "gunicorn", "-b", "0.0.0.0:8500", \
     "-w", "1", "-k", "gthread", "--threads", "8", \
     "--access-logfile", "-", "server:app"]
