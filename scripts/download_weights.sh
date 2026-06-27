#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-Ruihang/Wan-Move-14B-480P}"
LOCAL_DIR="${LOCAL_DIR:-./Wan-Move-14B-480P}"
FORCE_INSTALL_CLI="${FORCE_INSTALL_CLI:-0}"

cd "$(dirname "$0")/.."

if [[ "$FORCE_INSTALL_CLI" == "1" || "$FORCE_INSTALL_CLI" == "true" ]] || ! command -v huggingface-cli >/dev/null 2>&1; then
  python -m pip install --upgrade "huggingface_hub[cli]"
fi

mkdir -p "$LOCAL_DIR"

huggingface-cli download "$REPO_ID" --local-dir "$LOCAL_DIR"
