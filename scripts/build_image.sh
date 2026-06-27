#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-wan-move-api:latest}"
NO_CACHE="${NO_CACHE:-0}"

cd "$(dirname "$0")/.."

export WAN_MOVE_IMAGE="$IMAGE_NAME"

args=(compose build)
if [[ "$NO_CACHE" == "1" || "$NO_CACHE" == "true" ]]; then
  args+=(--no-cache)
fi
args+=(wan-move-api)

docker "${args[@]}"
