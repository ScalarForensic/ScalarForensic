# ScalarForensic — airgapped deployment image
#
# Prerequisites (run on the internet-connected machine before docker build):
#   uv run python scripts/download_models.py
#   bash scripts/download_deps.sh --web
#
# Or run the all-in-one helper:
#   bash scripts/build_airgap_image.sh
#
# GPU at runtime:
#   NVIDIA — add `--gpus all` to `docker run` / see docker-compose.yml
#   AMD ROCm — mount /dev/kfd and /dev/dri / see docker-compose.yml
#
# PyTorch ships its own CUDA runtime as Python packages, so no CUDA base image
# is needed for NVIDIA. For ROCm, the host ROCm driver must match the version
# of the torch+rocm wheel bundled at build time.

FROM python:3.12-slim

# libgomp1: OpenMP runtime required by PyTorch CPU and GPU operations.
# gcc: required by torch.compile / Triton to JIT-compile GPU kernels.
#      Without it, torch.compile raises "Failed to find C compiler" at the first
#      forward pass and silently falls back (or errors) for every batch.
# libdrm-amdgpu1 + symlink: suppresses the ROCm runtime warning
#      "/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory" that
#      appears during model loading when the AMD GPU device IDs file is missing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 gcc libdrm-amdgpu1 \
    && mkdir -p /opt/amdgpu/share/libdrm \
    && ln -s /usr/share/libdrm/amdgpu.ids /opt/amdgpu/share/libdrm/amdgpu.ids \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer 1: Python dependencies ─────────────────────────────────────────────
# Cached until vendor/ or requirements.txt change.
# Both are produced by scripts/download_deps.sh --web (or build_airgap_image.sh).
COPY vendor/ /tmp/vendor/
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-index --find-links /tmp/vendor/ \
        -r /tmp/requirements.txt

# ── Layer 2: Model weights ────────────────────────────────────────────────────
# Cached until models/ changes (~2.4 GB total).
# Produced by scripts/download_models.py (or build_airgap_image.sh).
COPY models/ /app/models/

# ── Layer 3: Application source ───────────────────────────────────────────────
# Invalidated only on code changes; the pip install step runs quickly.
COPY src/ /app/src/
COPY pyproject.toml LICENSE README.md /app/
RUN pip install --no-cache-dir --no-index --find-links /tmp/vendor/ \
        --no-deps . \
    && rm -rf /tmp/vendor/ /tmp/requirements.txt

# Fallback config used when no .env is mounted from the host at runtime.
# The real .env is always supplied via a volume mount — never baked in.
COPY .env.example /app/.env.example

# Default: web UI.  Override with 'sfn' for the CLI indexer:
#   docker compose run --rm sfn-web sfn /images --dino --sscd
CMD ["sfn-web"]
