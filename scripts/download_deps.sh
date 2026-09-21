#!/usr/bin/env bash
# Download all Python dependency wheels for offline / airgapped deployment.
#
# Run on an internet-connected machine with uv and python3 -m pip available.
# Transfer the generated vendor/ directory and requirements.txt to the
# offline system alongside the rest of the project folder.
#
# IMPORTANT: wheels are platform- and Python-version-specific.
# Run this script on a machine with the same OS, CPU architecture, and
# Python version as the airgapped target (e.g. linux/x86_64, Python 3.12).
# Downloading on macOS or a different Python version and transferring to a
# Linux/x86_64 target will result in install failures on the offline machine.
#
# Usage:
#   bash scripts/download_deps.sh              # core deps → vendor/
#   bash scripts/download_deps.sh --web        # include web-UI group
#   bash scripts/download_deps.sh --heif       # include HEIF/HEIC group
#   bash scripts/download_deps.sh --web --faces --heif # all optional groups
#   bash scripts/download_deps.sh --dest=pkg   # write wheels to pkg/ instead of vendor/
#
# The offline install commands are printed at the end of the script using the
# actual destination path chosen at runtime (vendor/ by default, or --dest value).

set -euo pipefail

DEST="vendor"
UV_GROUPS=()

for arg in "$@"; do
    case "$arg" in
        --web)  UV_GROUPS+=("--group" "web") ;;
        --faces) UV_GROUPS+=("--group" "faces") ;;
        --heif) UV_GROUPS+=("--group" "heif") ;;
        --dest=*) DEST="${arg#--dest=}" ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

if [[ -d "$DEST" && -n "$(ls -A "$DEST")" ]]; then
    echo "Refusing non-empty wheel directory: $DEST. Choose a fresh --dest." >&2
    exit 1
fi

echo "==> Exporting locked dependency list ..."
uv export --locked --no-default-groups --no-emit-project "${UV_GROUPS[@]}" -o requirements.txt
echo "    Written: requirements.txt"

echo "==> Downloading wheels → ${DEST}/ ..."
echo "    (torch and torchvision are fetched from the configured PyTorch index — this may take a while)"
mkdir -p "${DEST}"

# Download with the host pip and
# explicitly pass any extra indexes declared in [[tool.uv.index]] so that
# torch/torchvision are resolved from whichever PyTorch index is currently active.
#
# We also pin --python-version to the version declared in requires-python so that
# pip downloads wheels matching the hashes in the uv-exported requirements.txt
# (pip defaults to the running interpreter, which may differ from the locked version).
read -r PY_VERSION EXTRA_INDEX_URLS <<< "$(python3 -c "
import tomllib, re
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
rp = data['project']['requires-python']
py = re.search(r'(\d+\.\d+)', rp).group(1)
urls = ' '.join(idx['url'] for idx in data.get('tool', {}).get('uv', {}).get('index', []))
print(py, urls)
")"

EXTRA_INDEX_ARGS=()
for url in ${EXTRA_INDEX_URLS}; do
    EXTRA_INDEX_ARGS+=("--extra-index-url" "$url")
done

python3 -m pip download --require-hashes -r requirements.txt -d "${DEST}" \
    --python-version "${PY_VERSION}" --implementation cp \
    --only-binary=:all: \
    "${EXTRA_INDEX_ARGS[@]}"

# Build tooling has its own locked export, including transitive dependencies.
uv export --locked --only-group build --no-emit-project -o requirements-build.txt
python3 -m pip download --require-hashes \
    -r requirements-build.txt -d "${DEST}" \
    --python-version "${PY_VERSION}" --implementation cp --only-binary=:all:

echo ""
echo "Done. Files to transfer to the offline machine:"
echo "  ${DEST}/          ($(ls "${DEST}" | wc -l) wheels)"
echo "  requirements.txt and requirements-build.txt"
echo "  (and the rest of the project folder)"
echo ""
echo "On the offline machine:"
echo "  uv venv"
echo "  uv pip install --no-index --find-links \"${DEST}/\" --require-hashes -r requirements.txt"
echo "  uv pip install --no-index --find-links \"${DEST}/\" --require-hashes -r requirements-build.txt"
echo "  uv pip install --no-index --find-links \"${DEST}/\" --no-deps --no-build-isolation ."
