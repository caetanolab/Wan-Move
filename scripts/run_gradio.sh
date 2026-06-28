#!/usr/bin/env bash
set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:-./Wan-Move-14B-480P}"
OUTPUTS_DIR="${OUTPUTS_DIR:-./outputs}"
PORT="${PORT:-7860}"
IMAGE_NAME="${IMAGE_NAME:-wan-move-api:latest}"
BUILD="${BUILD:-0}"

cd "$(dirname "$0")/.."

if [[ ! -d "$WEIGHTS_DIR" ]]; then
  echo "Weights directory not found: $WEIGHTS_DIR. Run scripts/download_weights.sh first." >&2
  exit 1
fi

mkdir -p "$OUTPUTS_DIR"

export WAN_MOVE_IMAGE="$IMAGE_NAME"
export WAN_MOVE_WEIGHTS_DIR="$(cd "$WEIGHTS_DIR" && pwd)"
export WAN_MOVE_OUTPUTS_DIR="$(cd "$OUTPUTS_DIR" && pwd)"
export WAN_MOVE_PORT="$PORT"

if [[ "$BUILD" == "1" || "$BUILD" == "true" ]]; then
  docker compose build wan-move-api
fi

docker compose run --rm \
  --publish "${PORT}:${PORT}" \
  --entrypoint python \
  wan-move-api \
  gradio_app.py \
    --task wan-move-i2v \
    --size 480*832 \
    --ckpt_dir /models/Wan-Move-14B-480P \
    --t5_cpu \
    --offload_model True \
    --dtype bf16 \
    --port "$PORT"
