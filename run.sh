#!/usr/bin/env bash
# Wrapper that exposes the venv's NVIDIA CUDA libs before running any uv command.
# Usage: ./run.sh sfn data/images --dino --sscd
#        ./run.sh sfn-web
#        ./run.sh python myscript.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVIDIA_BASE="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/nvidia"

if [ -d "$NVIDIA_BASE" ]; then
    _cuda_paths=$(find "$NVIDIA_BASE" -maxdepth 2 -name "lib" -type d | tr '\n' ':')
    export LD_LIBRARY_PATH="${_cuda_paths}${LD_LIBRARY_PATH}"
fi

exec uv run "$@"
