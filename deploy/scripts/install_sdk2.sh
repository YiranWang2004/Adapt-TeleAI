#!/usr/bin/env bash
# Install the official SDK into the existing project environment.
set -euo pipefail

deploy_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$deploy_repo_root"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/adapt-teleai-uv-cache}"
sdk_source="$deploy_repo_root/deploy/third_party/unitree_sdk2_python"
dds_source="$deploy_repo_root/deploy/third_party/cyclonedds"
mkdir -p deploy/third_party

if [[ ! -d "$sdk_source" ]]; then
  git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python.git "$sdk_source"
fi
if [[ ! -d "$dds_source" ]]; then
  git clone --depth 1 --branch releases/0.10.x \
    https://github.com/eclipse-cyclonedds/cyclonedds.git "$dds_source"
fi

cmake -S "$dds_source" -B "$dds_source/build" \
  -DCMAKE_INSTALL_PREFIX="$dds_source/install" \
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DENABLE_SSL=OFF
cmake --build "$dds_source/build" --target install -j 8
export CYCLONEDDS_HOME="$dds_source/install"
uv pip install --python "$deploy_repo_root/.venv/bin/python" -e "$sdk_source"
uv run --no-sync python -c \
  'from deploy.backends.sdk import require_sdk; require_sdk(); print("Unitree SDK2 HG imports, CRC and serialization: OK")'
